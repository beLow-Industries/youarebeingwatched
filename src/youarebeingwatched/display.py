from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import platform
import time

import cv2
from loguru import logger
import numpy as np

from .config import DisplayConfig
from .types import Detection, Frame

BOX_COLOR = (255, 255, 255)
LABEL_BACKGROUND = (255, 255, 255)
LABEL_COLOR = (0, 0, 0)
TEXT_COLOR = (245, 245, 245)
LABEL_FONT = cv2.FONT_HERSHEY_PLAIN
LABEL_THICKNESS = 1


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
        self._fullscreen_size = _screen_size() if config.fullscreen else None
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.window_name, config.width, config.height)
        if config.fullscreen:
            cv2.setWindowProperty(config.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        logger.info("display ready window={}", config.window_name)

    def show(
        self,
        frame: Frame,
        detections: list[Detection],
        source_name: str,
        *,
        show_source_label: bool = True,
        show_boxes: bool = False,
        font_size: float = 0.65,
    ) -> bool:
        started = time.monotonic()
        target_size = self._fullscreen_size if self._fullscreen_size is not None else None
        rendered = render_frame(
            frame,
            detections,
            source_name,
            show_source_label=show_source_label,
            show_boxes=show_boxes,
            font_size=font_size,
            target_size=target_size,
        )
        render_elapsed = time.monotonic() - started
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
        return _should_continue(key)

    def close(self) -> None:
        logger.info("closing display window={}", self.config.window_name)
        cv2.destroyWindow(self.config.window_name)


class HeadlessDisplay:
    def __init__(self) -> None:
        self._last_printed = 0.0

    def show(
        self,
        frame: Frame,
        detections: list[Detection],
        source_name: str,
        *,
        show_source_label: bool = True,
        show_boxes: bool = False,
        font_size: float = 0.65,
    ) -> bool:
        now = time.monotonic()
        if now - self._last_printed >= 1.0:
            height, width = frame.shape[:2]
            logger.info("headless frame {}x{} source={} detections={}", width, height, source_name, len(detections))
            self._last_printed = now
        return True

    def close(self) -> None:
        return None


def render_frame(
    frame: Frame,
    detections: list[Detection],
    source_name: str,
    *,
    show_source_label: bool = True,
    show_boxes: bool = False,
    font_size: float = 0.65,
    target_size: tuple[int, int] | None = None,
) -> Frame:
    rendered = draw_detections(frame, detections, font_size=font_size) if show_boxes else frame.copy()
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
    if target_size is not None:
        rendered = _cover_frame(rendered, *target_size)
    return rendered


def draw_detections(frame: Frame, detections: list[Detection], *, font_size: float = 0.65) -> Frame:
    rendered = frame.copy()
    for detection in detections:
        box = detection.box
        cv2.rectangle(rendered, (box.x1, box.y1), (box.x2, box.y2), BOX_COLOR, 2)
        label = _detection_label(detection)
        text_origin = (box.x1, max(22, box.y1 - 8))
        (text_width, text_height), baseline = cv2.getTextSize(label, LABEL_FONT, font_size, LABEL_THICKNESS)
        padding = 3
        label_top = max(0, text_origin[1] - text_height - padding)
        label_bottom = min(rendered.shape[0], text_origin[1] + baseline + padding)
        label_right = min(rendered.shape[1], text_origin[0] + text_width + padding)
        cv2.rectangle(
            rendered,
            (text_origin[0], label_top),
            (label_right, label_bottom),
            LABEL_BACKGROUND,
            cv2.FILLED,
        )
        cv2.putText(rendered, label, text_origin, LABEL_FONT, font_size, LABEL_COLOR, LABEL_THICKNESS, cv2.LINE_AA)
    return rendered


def _detection_label(detection: Detection) -> str:
    name = {"person": "human", "dog": "doggo"}.get(detection.class_name, detection.class_name)
    return f"{name} ({detection.confidence:.0%})"


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


def _should_continue(key: int) -> bool:
    return key not in (ord("q"), ord("Q"), 27)


def _cover_frame(frame: Frame, target_width: int, target_height: int) -> Frame:
    """Scale a frame to fill the target and center-crop any excess."""
    frame_height, frame_width = frame.shape[:2]
    if (frame_width, frame_height) == (target_width, target_height):
        return frame

    scale = max(target_width / frame_width, target_height / frame_height)
    scaled_width = max(target_width, math.ceil(frame_width * scale))
    scaled_height = max(target_height, math.ceil(frame_height * scale))
    resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
    x = (scaled_width - target_width) // 2
    y = (scaled_height - target_height) // 2
    return resized[y : y + target_height, x : x + target_width]


def _screen_size() -> tuple[int, int] | None:
    if platform.system() != "Linux":
        return None

    display = os.environ.get("DISPLAY")
    lib_name = ctypes.util.find_library("X11")
    if not display or lib_name is None:
        return None

    lib_x11 = ctypes.cdll.LoadLibrary(lib_name)
    lib_x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib_x11.XOpenDisplay.restype = ctypes.c_void_p
    lib_x11.XDefaultScreen.argtypes = [ctypes.c_void_p]
    lib_x11.XDefaultScreen.restype = ctypes.c_int
    lib_x11.XDisplayWidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib_x11.XDisplayWidth.restype = ctypes.c_int
    lib_x11.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib_x11.XDisplayHeight.restype = ctypes.c_int
    lib_x11.XCloseDisplay.argtypes = [ctypes.c_void_p]

    handle = lib_x11.XOpenDisplay(display.encode())
    if not handle:
        return None
    try:
        screen = lib_x11.XDefaultScreen(handle)
        width = lib_x11.XDisplayWidth(handle, screen)
        height = lib_x11.XDisplayHeight(handle, screen)
    finally:
        lib_x11.XCloseDisplay(handle)
    if width <= 0 or height <= 0:
        return None
    logger.info("fullscreen display size={}x{}", width, height)
    return width, height


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
