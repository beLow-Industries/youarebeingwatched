from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from loguru import logger

from .app import run_app
from .config import load_config, with_display_overrides, with_source_overrides
from .doctor import run_doctor


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "doctor":
        return _doctor(args[1:])
    return _run(args)


def _run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ybwatch")
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--source", action="append", help="OpenCV camera index or video/stream path. Can be repeated.")
    parser.add_argument("--mock", action="store_true", help="Use a generated mock source and synthetic detections.")
    parser.add_argument("--fullscreen", action="store_true", default=None, help="Force fullscreen output.")
    parser.add_argument("--windowed", action="store_false", dest="fullscreen", help="Force windowed output.")
    parser.add_argument("--headless", action="store_true", help="Run the processing loop without opening a window.")
    parser.add_argument("--max-frames", type=int, help="Exit after showing this many frames.")
    parser.add_argument("--scan-max", type=int, help="When auto-discovering cameras, scan indices 0 through N-1.")
    parser.add_argument("--segmentation", action="store_true", help="Use YOLO segmentation masks to black out crop backgrounds.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--max-fps", action="store_true", help="Use the highest-FPS MJPG camera mode above the configured minimum FPS.")
    mode_group.add_argument("--max-resolution", action="store_true", help="Use the largest MJPG camera mode, regardless of FPS.")
    parser.add_argument("--log-level", default=os.environ.get("YBWATCH_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    config = load_config(args.config)
    logger.debug("loaded config from {}", args.config)
    config = with_source_overrides(config, args.source)
    source_mode_strategy = "max_fps" if args.max_fps else "max_resolution" if args.max_resolution else None
    config = with_display_overrides(
        config,
        fullscreen=args.fullscreen,
        scan_max=args.scan_max,
        source_mode_strategy=source_mode_strategy,
    )
    return run_app(config, mock=args.mock, headless=args.headless, max_frames=args.max_frames, segmentation=args.segmentation)


def _doctor(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ybwatch doctor")
    parser.add_argument("--scan-max", type=int, default=5)
    parser.add_argument("--log-level", default=os.environ.get("YBWATCH_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    return run_doctor(args.scan_max)


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {elapsed} | {level: <8} | {message}",
        backtrace=False,
        diagnose=False,
    )
