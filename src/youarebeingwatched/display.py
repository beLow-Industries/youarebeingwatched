from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import time

import cv2
from loguru import logger
import numpy as np

from .config import DisplayConfig
from .types import Detection, Frame

PERSON_COLOR = (0, 255, 80)
DOG_COLOR = (255, 180, 0)
TEXT_COLOR = (245, 245, 245)


class Display:
    def __init__(self, config: DisplayConfig) -> None:
        self.config = config
        logger.info(
            "opening display window={} fullscreen={} size={}x{}",
            config.window_name,
            config.fullscreen,
            config.width,
            config.height,
        )
        _assert_x_display_available()
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.window_name, config.width, config.height)
        if config.fullscreen:
            cv2.setWindowProperty(config.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        logger.info("display ready window={}", config.window_name)

    def show(self, frame: Frame, detections: list[Detection], source_name: str, *, show_source_label: bool = True) -> bool:
        started = time.monotonic()
        rendered = draw_detections(frame, detections)
        render_elapsed = time.monotonic() - started
        if show_source_label:
            cv2.putText(
                rendered,
                source_name,
                (24, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
        imshow_started = time.monotonic()
        cv2.imshow(self.config.window_name, rendered)
        imshow_elapsed = time.monotonic() - imshow_started
        wait_started = time.monotonic()
        key = cv2.waitKey(1) & 0xFF
        wait_elapsed = time.monotonic() - wait_started
        if render_elapsed >= 0.1 or imshow_elapsed >= 0.1 or wait_elapsed >= 0.1:
            logger.warning(
                "slow display stage source={} render={:.3f}s imshow={:.3f}s wait_key={:.3f}s",
                source_name,
                render_elapsed,
                imshow_elapsed,
                wait_elapsed,
            )
        return key not in (ord("q"), 27)

    def close(self) -> None:
        logger.info("closing display window={}", self.config.window_name)
        cv2.destroyWindow(self.config.window_name)


class HeadlessDisplay:
    def __init__(self) -> None:
        self._last_printed = 0.0

    def show(self, frame: Frame, detections: list[Detection], source_name: str, *, show_source_label: bool = True) -> bool:
        now = time.monotonic()
        if now - self._last_printed >= 1.0:
            height, width = frame.shape[:2]
            logger.info("headless frame {}x{} source={} detections={}", width, height, source_name, len(detections))
            self._last_printed = now
        return True

    def close(self) -> None:
        return None


def draw_detections(frame: Frame, detections: list[Detection]) -> Frame:
    rendered = frame.copy()
    for detection in detections:
        color = PERSON_COLOR if detection.class_name == "person" else DOG_COLOR
        box = detection.box
        cv2.rectangle(rendered, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_origin = (box.x1, max(22, box.y1 - 8))
        cv2.putText(rendered, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return rendered


def waiting_frame(width: int, height: int, message: str = "waiting for source") -> Frame:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (18, 18, 18)
    cv2.putText(
        frame,
        message,
        (max(24, width // 12), height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return frame


def _assert_x_display_available() -> None:
    if platform.system() != "Linux":
        return

    display = os.environ.get("DISPLAY")
    logger.debug("checking X display value={}", display)
    if not display:
        raise RuntimeError("OpenCV HighGUI needs an X display; use --headless for smoke checks")

    lib_name = ctypes.util.find_library("X11")
    if lib_name is None:
        return

    lib_x11 = ctypes.cdll.LoadLibrary(lib_name)
    lib_x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib_x11.XOpenDisplay.restype = ctypes.c_void_p
    lib_x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    handle = lib_x11.XOpenDisplay(display.encode())
    if not handle:
        raise RuntimeError(f"cannot connect to X display {display!r}; use --headless for smoke checks")
    lib_x11.XCloseDisplay(handle)
