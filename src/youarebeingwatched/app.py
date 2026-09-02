from __future__ import annotations

import random
import time

from loguru import logger

from .config import OVERLAY_CLASSES, AppConfig
from .detector import Detector, MockDetector, YoloDetector
from .display import Display, HeadlessDisplay, waiting_frame
from .postprocess import PostProcessor
from .sources import Source, SourceManager
from .types import Detection, Frame


def run_app(
    config: AppConfig,
    *,
    mock: bool,
    headless: bool = False,
    max_frames: int | None = None,
    segmentation: bool = False,
    threshold: float | None = None,
    show_box: bool = False,
    margin: int = 0,
    stabilize_box: float | None = None,
    do_not_track: bool = False,
) -> int:
    logger.info(
        "app start mock={} headless={} max_frames={} segmentation={} threshold={} show_box={} margin={} stabilize_box={} do_not_track={} selection_interval={} max_fps={}",
        mock,
        headless,
        max_frames,
        segmentation,
        threshold if threshold is not None else config.model.confidence,
        show_box,
        margin,
        stabilize_box if stabilize_box is not None else config.tracking.stabilize_box_pixels,
        do_not_track,
        config.selection.interval_seconds,
        config.display.max_fps,
    )
    detector: Detector
    if mock:
        detector = MockDetector()
    else:
        weights = config.model.segmentation_weights if segmentation else config.model.weights
        detector = YoloDetector(
            weights,
            confidence=threshold if threshold is not None else config.model.confidence,
            segmentation=segmentation,
        )

    source_manager = SourceManager.from_config(
        config.sources,
        mock=mock,
        width=config.display.width,
        height=config.display.height,
    )
    try:
        display = HeadlessDisplay() if headless else Display(config.display)
    except RuntimeError as exc:
        source_manager.release()
        logger.error("display unavailable: {}", exc)
        return 2

    effective_show_box = show_box or do_not_track
    postprocessor = PostProcessor(
        config.display,
        selection_classes=OVERLAY_CLASSES,
        selection_interval_seconds=config.tracking.reselect_interval_seconds,
        missing_linger_seconds=config.tracking.missing_linger_seconds,
        stabilize_box_pixels=stabilize_box if stabilize_box is not None else config.tracking.stabilize_box_pixels,
        show_boxes=effective_show_box,
        margin=margin,
    )

    selected_source: Source | None = source_manager.sources[0] if mock and source_manager.sources else None
    displayed_source_name: str | None = None
    displayed_overlay_state: bool | None = None
    next_selection_at = 0.0
    frame_delay = 1.0 / max(1.0, config.display.max_fps)
    frames_shown = 0

    try:
        while True:
            loop_started = time.monotonic()

            if loop_started >= next_selection_at:
                previous_source = selected_source.name if selected_source is not None else None
                logger.debug("selection cycle start current_source={}", previous_source)
                selection_started = time.monotonic()
                selected_source = _select_source(config, detector, source_manager, selected_source)
                selected_name = selected_source.name if selected_source is not None else None
                if selected_name != previous_source:
                    logger.info("selected source changed from={} to={}", previous_source, selected_name)
                else:
                    logger.debug("selected source unchanged source={}", selected_name)
                logger.debug("selection cycle done elapsed={:.3f}s", time.monotonic() - selection_started)
                postprocessor.note_source_decision(loop_started)
                if selected_source is None:
                    next_selection_at = loop_started + min(0.5, config.selection.interval_seconds)
                else:
                    next_selection_at = loop_started + config.selection.interval_seconds

            read_started = time.monotonic()
            frame, source_name, should_overlay = _read_main_frame(config, source_manager, selected_source)
            logger.debug(
                "main frame read source={} overlay={} elapsed={:.3f}s",
                source_name,
                should_overlay,
                time.monotonic() - read_started,
            )
            if source_name != displayed_source_name or should_overlay != displayed_overlay_state:
                logger.info("display mode source={} overlay={}", source_name, should_overlay)
                displayed_source_name = source_name
                displayed_overlay_state = should_overlay

            detections: list[Detection] = []
            if should_overlay:
                detect_started = time.monotonic()
                detections = detector.detect(frame, class_names=OVERLAY_CLASSES)
                logger.debug("overlay detections source={} count={} elapsed={:.3f}s", source_name, len(detections), time.monotonic() - detect_started)

            process_started = time.monotonic()
            result = postprocessor.process(
                frame,
                detections,
                source_name=source_name,
                should_crop=should_overlay and not do_not_track,
                now=loop_started,
            )
            logger.debug(
                "postprocess done source={} detections={} elapsed={:.3f}s",
                source_name,
                len(result.detections),
                time.monotonic() - process_started,
            )

            display_started = time.monotonic()
            if not display.show(
                result.frame,
                result.detections,
                source_name,
                show_source_label=result.show_source_label,
                show_boxes=effective_show_box,
            ):
                logger.info("display requested shutdown")
                return 0
            display_elapsed = time.monotonic() - display_started
            if display_elapsed >= 0.2:
                logger.warning("slow display show source={} elapsed={:.3f}s", source_name, display_elapsed)

            frames_shown += 1
            if max_frames is not None and frames_shown >= max_frames:
                logger.info("max frames reached frames={}", frames_shown)
                return 0

            elapsed = time.monotonic() - loop_started
            if elapsed >= frame_delay * 2:
                logger.warning("slow app loop source={} elapsed={:.3f}s target_delay={:.3f}s", source_name, elapsed, frame_delay)
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
    except KeyboardInterrupt:
        logger.info("keyboard interrupt")
        return 0
    finally:
        logger.info("app shutdown frames_shown={}", frames_shown)
        source_manager.release()
        display.close()


def _select_source(
    config: AppConfig,
    detector: Detector,
    source_manager: SourceManager,
    current_source: Source | None,
) -> Source | None:
    source_frames = source_manager.frames_for_selection()
    if not source_frames:
        logger.warning("selection skipped no readable source frames current_source={}", current_source.name if current_source else None)
        return current_source

    frames = [frame for _source, frame in source_frames]
    selection_classes = set(OVERLAY_CLASSES)
    detections_by_frame = detector.detect_batch(frames, class_names=selection_classes)
    candidates: list[Source] = []

    for (source, _frame), detections in zip(source_frames, detections_by_frame, strict=True):
        if any(detection.class_name in selection_classes for detection in detections):
            candidates.append(source)

    logger.debug(
        "selection detections sources={} candidates={}",
        [source.name for source, _frame in source_frames],
        [source.name for source in candidates],
    )
    if not candidates:
        return current_source

    return random.choice(candidates)


def _read_main_frame(
    config: AppConfig,
    source_manager: SourceManager,
    selected_source: Source | None,
) -> tuple[Frame, str, bool]:
    if selected_source is None:
        fallback = source_manager.first_readable_frame()
        if fallback is not None:
            source, frame = fallback
            logger.debug("using raw fallback frame source={}", source.name)
            return frame, source.name, False
        return waiting_frame(config.display.width, config.display.height), "waiting", False

    frame = selected_source.read()
    if frame is None:
        logger.warning("selected source unavailable source={}", selected_source.name)
        fallback = source_manager.first_readable_frame()
        if fallback is not None:
            source, frame = fallback
            logger.debug("using raw fallback frame source={} selected_source={}", source.name, selected_source.name)
            return frame, source.name, False
        return waiting_frame(config.display.width, config.display.height, "source unavailable"), selected_source.name, False
    return frame, selected_source.name, True
