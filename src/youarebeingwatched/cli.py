from __future__ import annotations

import argparse
import os
import sys

from loguru import logger

from .app import run_app
from .config import AppConfig, with_display_overrides, with_source_overrides
from .doctor import run_doctor
from .image import run_image


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "doctor":
        return _doctor(args[1:])
    if args and args[0] == "image":
        return _image(args[1:])
    return _run(args)


def _run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ybwatch")
    parser.add_argument("--source", action="append", help="OpenCV camera index or video/stream path. Can be repeated.")
    parser.add_argument("--mock", action="store_true", help="Use a generated mock source and synthetic detections.")
    parser.add_argument("--fullscreen", action="store_true", default=None, help="Force fullscreen output.")
    parser.add_argument("--windowed", action="store_false", dest="fullscreen", help="Force windowed output.")
    parser.add_argument("--headless", action="store_true", help="Run the processing loop without opening a window.")
    parser.add_argument("--max-frames", type=int, help="Exit after showing this many frames.")
    parser.add_argument("--scan-max", type=int, help="When auto-discovering cameras, scan indices 0 through N-1.")
    parser.add_argument("--segmentation", action="store_true", help="Use YOLO segmentation masks to black out crop backgrounds.")
    parser.add_argument("--threshold", type=_threshold, default=None, help="Detection confidence threshold (default: 0.4).")
    parser.add_argument("--show-box", action="store_true", help="Draw detection boxes and labels.")
    parser.add_argument("--font-size", type=_positive_float, default=0.65, help="Detection box label font scale.")
    parser.add_argument("--margin", type=_nonnegative_int, default=0, help="Add this many source pixels around each crop.")
    parser.add_argument("--stabilize-box", type=_nonnegative_float, default=None, help="Only update crop targets when a box edge moves this many source pixels.")
    parser.add_argument("--do-not-track", action="store_true", help="Show the full selected source frame with detection boxes.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--max-fps", action="store_true", help="Use the highest-FPS MJPG camera mode above the default minimum FPS.")
    mode_group.add_argument("--max-resolution", action="store_true", help="Use the largest MJPG camera mode, regardless of FPS.")
    parser.add_argument("--log-level", default=os.environ.get("YBWATCH_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    config = AppConfig()
    config = with_source_overrides(config, args.source)
    source_mode_strategy = "max_fps" if args.max_fps else "max_resolution" if args.max_resolution else None
    config = with_display_overrides(
        config,
        fullscreen=args.fullscreen,
        scan_max=args.scan_max,
        source_mode_strategy=source_mode_strategy,
    )
    return run_app(
        config,
        mock=args.mock,
        headless=args.headless,
        max_frames=args.max_frames,
        segmentation=args.segmentation,
        threshold=args.threshold,
        show_box=args.show_box,
        font_size=args.font_size,
        margin=args.margin,
        stabilize_box=args.stabilize_box,
        do_not_track=args.do_not_track,
    )


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _nonnegative_float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _threshold(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _doctor(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ybwatch doctor")
    parser.add_argument("--scan-max", type=int, default=10)
    parser.add_argument("--log-level", default=os.environ.get("YBWATCH_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    return run_doctor(args.scan_max)


def _image(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ybwatch image")
    parser.add_argument("--source", required=True, help="JPG or PNG image path.")
    parser.add_argument("--segmentation", action="store_true", help="Use YOLO segmentation masks to black out crop backgrounds.")
    parser.add_argument("--threshold", type=_threshold, default=None, help="Detection confidence threshold (default: 0.4).")
    parser.add_argument("--show-box", action="store_true", help="Draw detection boxes and labels.")
    parser.add_argument("--font-size", type=_positive_float, default=0.65, help="Detection box label font scale.")
    parser.add_argument("--margin", type=_nonnegative_int, default=0, help="Add source pixels around each crop.")
    parser.add_argument("--do-not-track", action="store_true", help="Save the full source image with detection boxes.")
    parser.add_argument("--log-level", default=os.environ.get("YBWATCH_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    return run_image(
        AppConfig(),
        source=args.source,
        segmentation=args.segmentation,
        threshold=args.threshold,
        show_box=args.show_box,
        font_size=args.font_size,
        margin=args.margin,
        do_not_track=args.do_not_track,
    )


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {elapsed} | {level: <8} | {message}",
        backtrace=False,
        diagnose=False,
    )
