from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Callable, Iterator

import cv2
from loguru import logger
import numpy as np

from .config import SourcesConfig
from .types import Frame
from .v4l2 import CameraMode, list_camera_modes, select_camera_mode, v4l2_available

CaptureFactory = Callable[[int | str, SourcesConfig], cv2.VideoCapture]
CAMERA_STALE_AFTER_SECONDS = 2.0
CAMERA_STALL_SECONDS = 2.0
CAMERA_REOPEN_FAILURES = 3
CAMERA_REOPEN_BACKOFF_SECONDS = (0.5, 1.0, 2.0, 5.0)


@dataclass(frozen=True)
class SourceSpec:
    raw: str

    @property
    def capture_target(self) -> int | str:
        if self.raw.isdigit():
            device = camera_device_path(int(self.raw))
            if device is not None:
                return device
            return int(self.raw)
        return self.raw

    @property
    def name(self) -> str:
        if self.raw.isdigit():
            return f"camera:{self.raw}"
        target = self.capture_target
        return Path(str(target)).name or str(target)

    @property
    def is_file_like(self) -> bool:
        target = self.capture_target
        return isinstance(target, str) and not target.startswith("/dev/video")


class VideoSource:
    def __init__(self, spec: SourceSpec, config: SourcesConfig, capture_factory: CaptureFactory | None = None) -> None:
        self.spec = spec
        self.name = spec.name
        self._target = spec.capture_target
        self._is_live_camera = _is_live_camera_target(self._target)
        self._capture_configs = _capture_config_candidates(self._target, config) if self._is_live_camera else [config]
        self._capture_config_index = 0
        self._config = self._capture_configs[self._capture_config_index]
        self._capture_factory = capture_factory or _open_direct_capture
        self._latest_lock = threading.Lock()
        self._latest_frame: Frame | None = None
        self._latest_frame_at = 0.0
        self._is_open = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        logger.debug("opening source name={} target={}", self.name, spec.capture_target)
        started = time.monotonic()
        self._capture = self._capture_factory(self._target, self._config)
        self._is_open = self._capture.isOpened()
        logger.debug(
            "opened source name={} is_open={} elapsed={:.3f}s",
            self.name,
            self._is_open,
            time.monotonic() - started,
        )
        if self._is_live_camera and self._is_open:
            self._thread = threading.Thread(target=self._capture_loop, name=f"ybwatch-capture-{self.name}", daemon=True)
            self._thread.start()

    @property
    def is_open(self) -> bool:
        return self._is_open

    def read(self) -> Frame | None:
        if self._is_live_camera:
            return self._read_latest_camera_frame()

        started = time.monotonic()
        ok, frame = self._capture.read()
        if ok:
            self._log_slow_read(started, rewind=False)
            return frame

        logger.debug("source read failed name={}", self.name)
        if self.spec.is_file_like:
            rewind_started = time.monotonic()
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if ok:
                logger.debug(
                    "source rewound and read name={} elapsed={:.3f}s total_elapsed={:.3f}s",
                    self.name,
                    time.monotonic() - rewind_started,
                    time.monotonic() - started,
                )
                return frame

        self._log_slow_read(started, rewind=False)
        return None

    def release(self) -> None:
        logger.debug("releasing source name={}", self.name)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._capture.release()

    def _log_slow_read(self, started: float, *, rewind: bool) -> None:
        elapsed = time.monotonic() - started
        if elapsed >= 0.25:
            logger.warning("slow source read name={} rewind={} elapsed={:.3f}s", self.name, rewind, elapsed)

    def _read_latest_camera_frame(self) -> Frame | None:
        with self._latest_lock:
            if self._latest_frame is None:
                return None
            age = time.monotonic() - self._latest_frame_at
            if age > CAMERA_STALE_AFTER_SECONDS:
                logger.debug("camera frame stale source={} age={:.3f}s", self.name, age)
                return None
            return self._latest_frame.copy()

    def _capture_loop(self) -> None:
        failures = 0
        backoff_index = 0

        while not self._stop.is_set():
            if not self._capture.isOpened():
                self._is_open = False
                backoff_index = self._reopen_after_backoff(backoff_index, "closed")
                failures = 0
                continue

            started = time.monotonic()
            ok, frame = self._capture.read()
            elapsed = time.monotonic() - started
            if self._stop.is_set():
                break

            if ok:
                failures = 0
                backoff_index = 0
                self._is_open = True
                with self._latest_lock:
                    self._latest_frame = frame
                    self._latest_frame_at = time.monotonic()
                if elapsed >= CAMERA_STALL_SECONDS:
                    logger.warning("camera read stalled source={} elapsed={:.3f}s reopening", self.name, elapsed)
                    self._reopen("stalled read")
                elif elapsed >= 0.25:
                    logger.warning("slow camera read source={} elapsed={:.3f}s", self.name, elapsed)
                continue

            failures += 1
            logger.debug("camera read failed source={} failures={} elapsed={:.3f}s", self.name, failures, elapsed)
            if failures >= CAMERA_REOPEN_FAILURES or elapsed >= CAMERA_STALL_SECONDS:
                self._reopen(f"read failure failures={failures} elapsed={elapsed:.3f}s")
                backoff_index = self._sleep_backoff(backoff_index)
                failures = 0

        logger.debug("camera capture worker stopped source={}", self.name)

    def _reopen_after_backoff(self, backoff_index: int, reason: str) -> int:
        next_index = self._sleep_backoff(backoff_index)
        self._reopen(reason)
        return next_index

    def _sleep_backoff(self, backoff_index: int) -> int:
        delay = CAMERA_REOPEN_BACKOFF_SECONDS[min(backoff_index, len(CAMERA_REOPEN_BACKOFF_SECONDS) - 1)]
        logger.warning("camera recovery backoff source={} delay={:.1f}s", self.name, delay)
        self._stop.wait(delay)
        return min(backoff_index + 1, len(CAMERA_REOPEN_BACKOFF_SECONDS) - 1)

    def _reopen(self, reason: str) -> None:
        logger.warning("reopening camera source={} reason={}", self.name, reason)
        self._capture.release()
        if len(self._capture_configs) > 1:
            self._capture_config_index = min(self._capture_config_index + 1, len(self._capture_configs) - 1)
            self._config = self._capture_configs[self._capture_config_index]
            logger.warning(
                "camera recovery mode source={} resolution={}x{} fps={} strategy={}",
                self.name,
                self._config.capture_width,
                self._config.capture_height,
                self._config.capture_fps,
                self._config.mode_strategy,
            )
        self._capture = self._capture_factory(self._target, self._config)
        self._is_open = self._capture.isOpened()
        if self._is_open:
            logger.info("camera reopened source={}", self.name)
        else:
            logger.warning("camera reopen failed source={}", self.name)


class MockSource:
    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.name = "mock"
        self.width = width
        self.height = height
        self._started = time.monotonic()

    @property
    def is_open(self) -> bool:
        return True

    def read(self) -> Frame:
        elapsed = time.monotonic() - self._started
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = (24, 26, 30)

        stripe_x = int((elapsed * 120) % (self.width + 240)) - 240
        cv2.rectangle(frame, (stripe_x, 0), (stripe_x + 120, self.height), (45, 55, 70), -1)
        cv2.putText(
            frame,
            "mock source",
            (48, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return frame

    def release(self) -> None:
        logger.debug("releasing mock source")
        return None


Source = VideoSource | MockSource


class SourceManager:
    def __init__(self, sources: list[Source]) -> None:
        self.sources = sources
        self._last_no_fallback_warning = 0.0

    @classmethod
    def from_config(cls, config: SourcesConfig, *, mock: bool, width: int, height: int) -> "SourceManager":
        if mock:
            logger.info("using mock source width={} height={}", width, height)
            return cls([MockSource(width=width, height=height)])

        specs = configured_specs(config)
        logger.info("configured source specs count={} specs={}", len(specs), [spec.raw for spec in specs])
        sources = [VideoSource(spec, config) for spec in specs]
        open_sources = [source for source in sources if source.is_open]
        closed_sources = [source.name for source in sources if not source.is_open]
        logger.info("open sources count={} sources={}", len(open_sources), [source.name for source in open_sources])
        if closed_sources:
            logger.warning("closed sources ignored sources={}", closed_sources)
        return cls(open_sources)

    def frames_for_selection(self) -> list[tuple[Source, Frame]]:
        started = time.monotonic()
        frames: list[tuple[Source, Frame]] = []
        for source in self.sources:
            frame = source.read()
            if frame is not None:
                frames.append((source, frame))
        logger.debug(
            "selection frames read readable={} total={} elapsed={:.3f}s",
            [source.name for source, _frame in frames],
            len(self.sources),
            time.monotonic() - started,
        )
        return frames

    def first_readable_frame(self) -> tuple[Source, Frame] | None:
        started = time.monotonic()
        for source in self.sources:
            frame = source.read()
            if frame is not None:
                logger.debug("fallback source frame name={} elapsed={:.3f}s", source.name, time.monotonic() - started)
                return source, frame
        now = time.monotonic()
        if now - self._last_no_fallback_warning >= 1.0:
            logger.warning("no readable fallback source elapsed={:.3f}s", now - started)
            self._last_no_fallback_warning = now
        return None

    def release(self) -> None:
        for source in self.sources:
            source.release()


def configured_specs(config: SourcesConfig) -> list[SourceSpec]:
    seen: set[str] = set()
    specs: list[SourceSpec] = []

    for entry in config.entries:
        _append_spec(specs, seen, str(entry))

    if config.auto_discover:
        for index in config.scan_indices:
            logger.debug("probing camera index={}", index)
            if _can_open_camera(index, config):
                _append_spec(specs, seen, str(index))

    return specs


def camera_device_path(index: int) -> str | None:
    path = Path(f"/dev/video{index}")
    if path.exists():
        return str(path)
    return None


def probe_source(spec: SourceSpec, config: SourcesConfig | None = None) -> tuple[bool, bool]:
    config = config or SourcesConfig()
    with quiet_opencv_stderr():
        capture = _open_capture(spec.capture_target, config)
        try:
            if not capture.isOpened():
                return False, False
            ok, _frame = capture.read()
            return True, ok
        finally:
            capture.release()


def _append_spec(specs: list[SourceSpec], seen: set[str], raw: str) -> None:
    if raw in seen:
        return
    seen.add(raw)
    specs.append(SourceSpec(raw=raw))


def _can_open_camera(index: int, config: SourcesConfig) -> bool:
    device = camera_device_path(index)
    if device is not None and v4l2_available():
        minimum_fps = _minimum_fps(config)
        modes = list_camera_modes(Path(device))
        eligible_modes = _auto_discovery_modes(modes, config)
        selected = _select_configured_camera_mode(eligible_modes, config, minimum_fps)
        if selected is not None:
            logger.debug(
                "camera accepted by v4l2 modes index={} device={} selected_mode={}",
                index,
                device,
                selected.format_line(),
            )
            return True
        logger.debug(
            "camera rejected by v4l2 modes index={} device={} minimum_fps={} modes={}",
            index,
            device,
            minimum_fps,
            [mode.format_line() for mode in modes],
        )
        return False

    opened, readable = probe_source(SourceSpec(raw=str(index)), config)
    return opened and readable


def _auto_discovery_modes(modes: list[CameraMode], config: SourcesConfig) -> list[CameraMode]:
    if config.capture_fourcc is None:
        return modes
    return [mode for mode in modes if mode.pixel_format == config.capture_fourcc]


def _open_capture(target: int | str, config: SourcesConfig) -> cv2.VideoCapture:
    if isinstance(target, int):
        return _open_direct_capture(target, config)
    if target.startswith("/dev/video"):
        config = _with_auto_selected_camera_mode(Path(target), config)
        return _open_direct_capture(target, config)
    return cv2.VideoCapture(target)


def _open_direct_capture(target: int | str, config: SourcesConfig) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(target, cv2.CAP_V4L2)
    _configure_capture(capture, config)
    return capture


def _capture_config_candidates(target: int | str, config: SourcesConfig) -> list[SourcesConfig]:
    if isinstance(target, int) or not target.startswith("/dev/video") or not v4l2_available():
        return [config]

    modes = list_camera_modes(Path(target))
    selected = _select_configured_camera_mode(modes, config, _minimum_fps(config))
    if selected is None:
        return [config]

    logger.info(
        "selected camera mode device={} format={} resolution={}x{} fps={} minimum_fps={}",
        target,
        selected.pixel_format,
        selected.width,
        selected.height,
        selected.max_fps,
        _minimum_fps(config),
    )
    candidates = [_config_for_mode(config, selected)]
    if config.mode_strategy in ("max_fps", "max_resolution"):
        seen = {(selected.pixel_format, selected.width, selected.height, selected.max_fps)}
        for mode in _fallback_modes(modes, config):
            key = (mode.pixel_format, mode.width, mode.height, mode.max_fps)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(_config_for_mode(config, mode))
    return candidates


def _with_auto_selected_camera_mode(device: Path, config: SourcesConfig) -> SourcesConfig:
    minimum_fps = _minimum_fps(config)
    if not v4l2_available():
        logger.debug("v4l2-ctl unavailable, using configured capture mode device={}", device)
        return config

    modes = list_camera_modes(device)
    selected = _select_configured_camera_mode(modes, config, minimum_fps)
    if selected is None:
        logger.warning(
            "no camera mode at or above requested fps device={} minimum_fps={} modes={}",
            device,
            minimum_fps,
            [mode.format_line() for mode in modes],
        )
        return config

    logger.info(
        "selected camera mode device={} format={} resolution={}x{} fps={} minimum_fps={}",
        device,
        selected.pixel_format,
        selected.width,
        selected.height,
        selected.max_fps,
        minimum_fps,
    )
    return _config_for_mode(config, selected)


def _config_for_mode(config: SourcesConfig, mode: CameraMode) -> SourcesConfig:
    return SourcesConfig(
        auto_discover=config.auto_discover,
        scan_indices=config.scan_indices,
        entries=config.entries,
        capture_width=mode.width,
        capture_height=mode.height,
        capture_fps=mode.max_fps,
        capture_fourcc=mode.pixel_format,
        mode_strategy=config.mode_strategy,
    )


def _fallback_modes(modes: list[CameraMode], config: SourcesConfig) -> list[CameraMode]:
    minimum_fps = 25.0 if config.mode_strategy == "max_resolution" else _minimum_fps(config)
    candidates = [
        mode
        for mode in modes
        if mode.max_fps >= minimum_fps and (config.capture_fourcc is None or mode.pixel_format == config.capture_fourcc)
    ]
    if config.mode_strategy == "max_fps":
        return sorted(candidates, key=lambda mode: (-mode.max_fps, mode.area))
    return sorted(candidates, key=lambda mode: (-mode.area, -mode.max_fps))


def _select_configured_camera_mode(
    modes: list[CameraMode],
    config: SourcesConfig,
    minimum_fps: float,
) -> CameraMode | None:
    return select_camera_mode(
        modes,
        minimum_fps=minimum_fps,
        preferred_format=config.capture_fourcc,
        preferred_width=config.capture_width,
        preferred_height=config.capture_height,
        mode_strategy=config.mode_strategy,
    )


def _configure_capture(capture: cv2.VideoCapture, config: SourcesConfig) -> None:
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1000)
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
    if config.capture_fourcc:
        fourcc = config.capture_fourcc[:4].ljust(4)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if config.capture_width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_width)
    if config.capture_height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_height)
    if config.capture_fps is not None:
        capture.set(cv2.CAP_PROP_FPS, config.capture_fps)


def _minimum_fps(config: SourcesConfig) -> float:
    if config.mode_strategy == "max_resolution":
        return 0.0
    return config.capture_fps or 25.0


def _is_live_camera_target(target: int | str) -> bool:
    return isinstance(target, int) or target.startswith("/dev/video")


@contextmanager
def quiet_opencv_stderr() -> Iterator[None]:
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
