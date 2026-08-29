from __future__ import annotations

from collections import deque
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from youarebeingwatched.config import SourcesConfig
from youarebeingwatched.sources import SourceManager, SourceSpec, VideoSource, _capture_config_candidates
from youarebeingwatched.v4l2 import CameraMode


class FakeCapture:
    def __init__(self, reads: list[object] | None = None) -> None:
        self.reads = deque(reads or [])
        self.opened = True
        self.release_count = 0
        self.set_calls: list[tuple[int, object]] = []

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.reads:
            time.sleep(0.01)
            return False, None
        item = self.reads.popleft()
        if isinstance(item, float):
            time.sleep(item)
            return False, None
        if item is False:
            return False, None
        return True, item  # type: ignore[return-value]

    def set(self, prop: int, value: object) -> bool:
        self.set_calls.append((prop, value))
        return True

    def release(self) -> None:
        self.release_count += 1
        self.opened = False


def wait_for(predicate: object, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class VideoSourceCaptureWorkerTest(unittest.TestCase):
    def test_live_camera_read_returns_latest_frame_without_blocking(self) -> None:
        first = np.full((2, 2, 3), 10, dtype=np.uint8)
        second = np.full((2, 2, 3), 20, dtype=np.uint8)
        capture = FakeCapture([first, second])
        source = VideoSource(SourceSpec("0"), SourcesConfig(), capture_factory=lambda _target, _config: capture)
        self.addCleanup(source.release)

        self.assertTrue(wait_for(lambda: source.read() is not None))
        started = time.monotonic()
        frame = source.read()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.05)
        self.assertIsNotNone(frame)
        self.assertTrue(np.all(frame == 20))

    def test_live_camera_read_does_not_wait_for_blocking_capture(self) -> None:
        capture = FakeCapture([0.25])
        source = VideoSource(SourceSpec("0"), SourcesConfig(), capture_factory=lambda _target, _config: capture)
        self.addCleanup(source.release)

        started = time.monotonic()
        frame = source.read()
        elapsed = time.monotonic() - started

        self.assertIsNone(frame)
        self.assertLess(elapsed, 0.05)

    def test_repeated_camera_failures_reopen_capture(self) -> None:
        captures = [FakeCapture([False, False]), FakeCapture([np.full((2, 2, 3), 30, dtype=np.uint8)])]
        lock = threading.Lock()

        def factory(_target: int | str, _config: SourcesConfig) -> FakeCapture:
            with lock:
                return captures[min(len([capture for capture in captures if capture.release_count > 0]), len(captures) - 1)]

        with (
            patch("youarebeingwatched.sources.CAMERA_REOPEN_FAILURES", 2),
            patch("youarebeingwatched.sources.CAMERA_REOPEN_BACKOFF_SECONDS", (0.01,)),
        ):
            source = VideoSource(SourceSpec("0"), SourcesConfig(), capture_factory=factory)
            self.addCleanup(source.release)

            self.assertTrue(wait_for(lambda: captures[0].release_count >= 1))

    def test_source_manager_reads_do_not_block_when_latest_frame_is_unavailable(self) -> None:
        capture = FakeCapture([0.25])
        source = VideoSource(SourceSpec("0"), SourcesConfig(), capture_factory=lambda _target, _config: capture)
        self.addCleanup(source.release)
        manager = SourceManager([source])

        started = time.monotonic()
        frames = manager.frames_for_selection()
        fallback = manager.first_readable_frame()
        elapsed = time.monotonic() - started

        self.assertEqual(frames, [])
        self.assertIsNone(fallback)
        self.assertLess(elapsed, 0.05)

    def test_max_resolution_camera_has_safe_fallback_modes(self) -> None:
        modes = [
            CameraMode("MJPG", "Motion-JPEG", 1920, 1080, (10.0,)),
            CameraMode("MJPG", "Motion-JPEG", 1280, 720, (30.0,)),
            CameraMode("MJPG", "Motion-JPEG", 640, 480, (30.0,)),
            CameraMode("YUYV", "YUYV", 640, 480, (60.0,)),
        ]
        config = SourcesConfig(mode_strategy="max_resolution")

        with patch("youarebeingwatched.sources.v4l2_available", return_value=True), patch(
            "youarebeingwatched.sources.list_camera_modes", return_value=modes
        ):
            configs = _capture_config_candidates("/dev/video0", config)

        self.assertEqual((configs[0].capture_width, configs[0].capture_height, configs[0].capture_fps), (1920, 1080, 10.0))
        self.assertEqual((configs[1].capture_width, configs[1].capture_height, configs[1].capture_fps), (1280, 720, 30.0))
        self.assertEqual((configs[2].capture_width, configs[2].capture_height, configs[2].capture_fps), (640, 480, 30.0))

    def test_camera_reopen_advances_to_fallback_mode(self) -> None:
        captures = [FakeCapture([False, False]), FakeCapture([np.full((2, 2, 3), 30, dtype=np.uint8)])]
        configs_seen: list[SourcesConfig] = []

        def factory(_target: int | str, config: SourcesConfig) -> FakeCapture:
            configs_seen.append(config)
            return captures[min(len(configs_seen) - 1, len(captures) - 1)]

        modes = [
            CameraMode("MJPG", "Motion-JPEG", 1920, 1080, (10.0,)),
            CameraMode("MJPG", "Motion-JPEG", 1280, 720, (30.0,)),
            CameraMode("MJPG", "Motion-JPEG", 640, 480, (30.0,)),
        ]
        with (
            patch("youarebeingwatched.sources.v4l2_available", return_value=True),
            patch("youarebeingwatched.sources.list_camera_modes", return_value=modes),
            patch("youarebeingwatched.sources.CAMERA_REOPEN_FAILURES", 2),
            patch("youarebeingwatched.sources.CAMERA_REOPEN_BACKOFF_SECONDS", (0.01,)),
        ):
            source = VideoSource(SourceSpec("/dev/video0"), SourcesConfig(mode_strategy="max_resolution"), capture_factory=factory)
            self.addCleanup(source.release)

            self.assertTrue(wait_for(lambda: len(configs_seen) >= 2))

        self.assertEqual((configs_seen[0].capture_width, configs_seen[0].capture_height), (1920, 1080))
        self.assertEqual((configs_seen[1].capture_width, configs_seen[1].capture_height), (1280, 720))


if __name__ == "__main__":
    unittest.main()
