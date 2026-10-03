from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
from loguru import logger

from .config import OVERLAY_CLASSES, AppConfig
from .detector import YoloDetector
from .display import render_frame
from .postprocess import PostProcessor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


def run_image(
    config: AppConfig,
    *,
    source: str,
    segmentation: bool = False,
    threshold: float | None = None,
    show_box: bool = False,
    font_size: float = 0.65,
    margin: int = 0,
    do_not_track: bool = False,
) -> int:
    source_path = Path(source)
    if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
        logger.error("unsupported image source extension source={}", source)
        return 1

    frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if frame is None:
        logger.error("image source unreadable source={}", source)
        return 1

    weights = config.model.segmentation_weights if segmentation else config.model.weights
    detector = YoloDetector(
        weights,
        confidence=threshold if threshold is not None else config.model.confidence,
        segmentation=segmentation,
    )
    detections = detector.detect(frame, class_names=OVERLAY_CLASSES)
    display = replace(config.display, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fullscreen=False)
    effective_show_box = show_box or do_not_track
    postprocessor = PostProcessor(
        display,
        selection_classes=OVERLAY_CLASSES,
        selection_interval_seconds=config.tracking.reselect_interval_seconds,
        missing_linger_seconds=config.tracking.missing_linger_seconds,
        stabilize_box_pixels=config.tracking.stabilize_box_pixels,
        show_boxes=effective_show_box,
        margin=margin,
    )
    result = postprocessor.process(
        frame,
        detections,
        source_name=source_path.name,
        should_crop=not do_not_track,
        now=0.0,
    )
    rendered = render_frame(
        result.frame,
        result.detections,
        source_path.name,
        show_source_label=result.show_source_label,
        show_boxes=effective_show_box,
        font_size=font_size,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    output_path = next_output_path()
    if not cv2.imwrite(str(output_path), rendered):
        logger.error("image output write failed path={}", output_path)
        return 1

    print(output_path)
    return 0


def next_output_path(directory: Path = Path(".")) -> Path:
    index = 0
    while True:
        path = directory / f"out-{index}.jpg"
        if not path.exists():
            return path
        index += 1
