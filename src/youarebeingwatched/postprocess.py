from __future__ import annotations

from dataclasses import dataclass
import random
import time

import cv2
import numpy as np

from .config import DisplayConfig
from .types import Box, CropMask, Detection, Frame


@dataclass(frozen=True)
class PostProcessResult:
    frame: Frame
    detections: list[Detection]
    show_source_label: bool = True


@dataclass(frozen=True)
class TrackedObject:
    slot: int | None
    class_name: str
    confidence: float
    box: Box
    target_box: Box
    display_box: Box
    mask: CropMask
    last_seen_at: float


_BOX_SMOOTHING = 0.75


class PostProcessor:
    """Crops output to tracked objects during selected-source display."""

    def __init__(
        self,
        display: DisplayConfig,
        *,
        selection_classes: tuple[str, ...] = ("person", "dog"),
        selection_interval_seconds: float = 1.0,
        missing_linger_seconds: float = 0.5,
        stabilize_box_pixels: float = 16.0,
        show_boxes: bool = False,
        margin: int = 0,
        rng: random.Random | None = None,
    ) -> None:
        self._display_width = display.width
        self._display_height = display.height
        self._selection_classes = set(selection_classes)
        self._selection_interval_seconds = selection_interval_seconds
        self._missing_linger_seconds = missing_linger_seconds
        self._stabilize_box_pixels = stabilize_box_pixels
        self._show_boxes = show_boxes
        self._margin = margin
        self._rng = rng or random.Random()
        self._tracked_objects: list[TrackedObject] = []
        self._next_selection_at = 0.0
        self._force_selection = True
        self._tracked_source_name: str | None = None

    def note_source_decision(self, now: float | None = None) -> None:
        self._tracked_objects = []
        self._tracked_source_name = None
        self._force_selection = True
        self._next_selection_at = now if now is not None else time.monotonic()

    def process(
        self,
        frame: Frame,
        detections: list[Detection],
        *,
        source_name: str,
        should_crop: bool,
        now: float | None = None,
    ) -> PostProcessResult:
        if not should_crop:
            self._tracked_objects = []
            self._tracked_source_name = None
            self._force_selection = True
            return PostProcessResult(frame=frame, detections=detections)

        now = now if now is not None else time.monotonic()
        if self._tracked_source_name != source_name:
            self._tracked_objects = []
            self._tracked_source_name = source_name
            self._force_selection = True

        candidates = [detection for detection in detections if detection.class_name in self._selection_classes]
        selected = self._select_objects(candidates, now)
        if not selected:
            return PostProcessResult(frame=frame, detections=detections)

        cropped, cropped_detections = _compose_crops(
            frame, selected, self._display_width, self._display_height, self._margin
        )
        return PostProcessResult(
            frame=cropped,
            detections=cropped_detections if self._show_boxes else [],
            show_source_label=False,
        )

    def _select_objects(self, candidates: list[Detection], now: float) -> list[TrackedObject]:
        if self._force_selection or not self._tracked_objects:
            if not candidates:
                self._tracked_objects = []
                self._force_selection = True
                return []
            return self._elect_objects(candidates, now)

        matched, missed, remaining = _match_tracked_objects(self._tracked_objects, candidates, now, self._stabilize_box_pixels)
        lingering = self._lingering_objects(missed, now)

        if now >= self._next_selection_at and not lingering:
            if candidates:
                return self._elect_objects(candidates, now)
            self._tracked_objects = []
            self._force_selection = True
            return []

        if len(matched) == len(self._tracked_objects):
            self._tracked_objects = matched
            return matched

        if lingering:
            return self._retain_lingering_objects(matched, lingering, remaining, now)

        if matched:
            return self._stabilized_recompute(matched, remaining, candidates, now)

        if candidates:
            return self._elect_objects(candidates, now)

        self._tracked_objects = []
        self._force_selection = True
        return []

    def _elect_objects(self, candidates: list[Detection], now: float) -> list[TrackedObject]:
        previous = list(self._tracked_objects)
        selected_count = min(3, len(candidates))
        if selected_count == len(candidates):
            selected = list(candidates)
        else:
            selected = self._rng.sample(candidates, selected_count)

        selected.sort(key=lambda detection: _box_center_x(detection.box))
        slots = _slots_for_count(selected_count)
        self._tracked_objects = []
        for slot, detection in zip(slots, selected, strict=True):
            matched = _match_tracked_object(detection.box, previous)
            if matched is not None:
                previous.remove(matched)
                target_box = (
                    detection.box
                    if _box_deviates(matched.target_box, detection.box, self._stabilize_box_pixels)
                    else matched.target_box
                )
                display_box = _smooth_box(matched.display_box, target_box)
            else:
                target_box = detection.box
                display_box = detection.box
            self._tracked_objects.append(
                TrackedObject(
                    slot=slot,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    box=detection.box,
                    target_box=target_box,
                    display_box=display_box,
                    mask=detection.mask,
                    last_seen_at=now,
                )
            )
        self._force_selection = False
        self._next_selection_at = now + self._selection_interval_seconds
        return self._tracked_objects

    def _lingering_objects(self, missed: list[TrackedObject], now: float) -> list[TrackedObject]:
        return [item for item in missed if now - item.last_seen_at <= self._missing_linger_seconds]

    def _retain_lingering_objects(
        self,
        matched: list[TrackedObject],
        lingering: list[TrackedObject],
        remaining: list[Detection],
        now: float,
    ) -> list[TrackedObject]:
        selected = [*matched, *lingering]
        target_count = min(3, len(selected) + len(remaining))
        if len(selected) < target_count:
            needed = target_count - len(selected)
            selected = [*selected, *_new_tracked_objects(remaining, needed, _unused_slots(selected, target_count), self._rng, now)]

        self._tracked_objects = selected
        self._force_selection = False
        return self._tracked_objects

    def _stabilized_recompute(
        self,
        matched: list[TrackedObject],
        remaining: list[Detection],
        candidates: list[Detection],
        now: float,
    ) -> list[TrackedObject]:
        target_count = min(3, len(candidates))
        if len(matched) >= target_count:
            selected = matched[:target_count]
        else:
            needed = target_count - len(matched)
            selected = [*matched, *_new_tracked_objects(remaining, needed, _unused_slots(matched, target_count), self._rng, now)]

        self._tracked_objects = selected
        self._force_selection = False
        self._next_selection_at = now + self._selection_interval_seconds
        return self._tracked_objects


def _compose_crops(
    frame: Frame,
    objects: list[TrackedObject],
    display_width: int,
    display_height: int,
    margin: int,
) -> tuple[Frame, list[Detection]]:
    if len(objects) == 1 and objects[0].slot is None:
        return _crop_to_canvas(frame, objects[0], display_width, display_height, margin)

    canvas = _black_canvas(frame, display_width, display_height)
    cropped_detections: list[Detection] = []
    edges = _third_edges(display_width)
    for item in objects:
        slot = item.slot if item.slot is not None else 0
        slot_x1 = edges[slot]
        slot_x2 = edges[slot + 1]
        mapped_box = _paste_crop(frame, item, canvas, slot_x1, slot_x2, display_height, margin)
        if mapped_box is None:
            return frame, []
        cropped_detections.append(Detection(item.class_name, item.confidence, mapped_box))
    return canvas, cropped_detections


def _crop_to_canvas(
    frame: Frame, item: TrackedObject, display_width: int, display_height: int, margin: int
) -> tuple[Frame, list[Detection]]:
    canvas = _black_canvas(frame, display_width, display_height)
    mapped_box = _paste_crop(frame, item, canvas, 0, display_width, display_height, margin)
    if mapped_box is None:
        return frame, []
    return canvas, [Detection(item.class_name, item.confidence, mapped_box)]


def _paste_crop(
    frame: Frame,
    item: TrackedObject,
    canvas: Frame,
    slot_x1: int,
    slot_x2: int,
    display_height: int,
    margin: int,
) -> Box | None:
    frame_height, frame_width = frame.shape[:2]
    expanded = _expand_box(item.display_box, margin) if margin > 0 else item.display_box
    clipped = _clip_box(expanded, frame_width, frame_height)
    if clipped is None:
        return None

    crop = frame[clipped.y1 : clipped.y2, clipped.x1 : clipped.x2]
    crop = item.mask.apply(crop, clipped, frame.shape)
    if margin > 0 and clipped != expanded:
        crop = cv2.copyMakeBorder(
            crop,
            clipped.y1 - expanded.y1,
            expanded.y2 - clipped.y2,
            clipped.x1 - expanded.x1,
            expanded.x2 - clipped.x2,
            cv2.BORDER_CONSTANT,
            value=0,
        )
    crop_height, crop_width = crop.shape[:2]
    if crop_height <= 0 or crop_width <= 0:
        return None

    slot_width = slot_x2 - slot_x1
    if slot_width <= 0:
        return None
    scaled_width = max(1, int(round(crop_width * (display_height / crop_height))))
    resized = cv2.resize(crop, (scaled_width, display_height), interpolation=cv2.INTER_LINEAR)

    if scaled_width <= slot_width:
        x = slot_x1 + (slot_width - scaled_width) // 2
        canvas[:, x : x + scaled_width] = resized
        offset_x = x
    else:
        start_x = (scaled_width - slot_width) // 2
        canvas[:, slot_x1:slot_x2] = resized[:, start_x : start_x + slot_width]
        offset_x = slot_x1 - start_x

    target = _clip_box(item.display_box, frame_width, frame_height)
    if target is None:
        return None
    crop_box = expanded if margin > 0 else clipped
    scale = display_height / (crop_box.y2 - crop_box.y1)
    mapped = Box(
        x1=round(offset_x + (target.x1 - crop_box.x1) * scale),
        y1=round((target.y1 - crop_box.y1) * scale),
        x2=round(offset_x + (target.x2 - crop_box.x1) * scale),
        y2=round((target.y2 - crop_box.y1) * scale),
    )
    return _clip_output_box(mapped, slot_x1, slot_x2, display_height)


def _expand_box(box: Box, margin: int) -> Box:
    return Box(box.x1 - margin, box.y1 - margin, box.x2 + margin, box.y2 + margin)


def _clip_output_box(box: Box, x1: int, x2: int, height: int) -> Box | None:
    clipped = _clip_box(box, x2, height)
    if clipped is None:
        return None
    return _clip_box(Box(max(x1, clipped.x1), clipped.y1, min(x2, clipped.x2), clipped.y2), x2, height)


def _black_canvas(frame: Frame, display_width: int, display_height: int) -> Frame:
    if frame.ndim == 2:
        return np.zeros((display_height, display_width), dtype=frame.dtype)
    return np.zeros((display_height, display_width, frame.shape[2]), dtype=frame.dtype)


def _clip_box(box: Box, frame_width: int, frame_height: int) -> Box | None:
    x1 = max(0, min(frame_width, box.x1))
    y1 = max(0, min(frame_height, box.y1))
    x2 = max(0, min(frame_width, box.x2))
    y2 = max(0, min(frame_height, box.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def _match_tracked_objects(
    tracked_objects: list[TrackedObject],
    candidates: list[Detection],
    now: float,
    stabilize_box_pixels: float,
) -> tuple[list[TrackedObject], list[TrackedObject], list[Detection]]:
    remaining = list(candidates)
    matched: list[TrackedObject] = []
    missed: list[TrackedObject] = []
    for tracked in tracked_objects:
        detection = _match_detection(tracked.box, remaining)
        if detection is None:
            missed.append(tracked)
            continue
        remaining.remove(detection)
        target_box = (
            detection.box
            if _box_deviates(tracked.target_box, detection.box, stabilize_box_pixels)
            else tracked.target_box
        )
        matched.append(
            TrackedObject(
                slot=tracked.slot,
                class_name=detection.class_name,
                confidence=detection.confidence,
                box=detection.box,
                target_box=target_box,
                display_box=_smooth_box(tracked.display_box, target_box),
                mask=detection.mask,
                last_seen_at=now,
            )
        )
    return matched, missed, remaining


def _match_detection(tracked_box: Box, people: list[Detection]) -> Detection | None:
    if not people:
        return None

    best_iou = max(people, key=lambda detection: _iou(tracked_box, detection.box))
    if _iou(tracked_box, best_iou.box) >= 0.1:
        return best_iou

    best_distance = min(people, key=lambda detection: _center_distance_squared(tracked_box, detection.box))
    if _center_distance_squared(tracked_box, best_distance.box) <= _distance_threshold_squared(tracked_box):
        return best_distance

    return None


def _match_tracked_object(detection_box: Box, tracked_objects: list[TrackedObject]) -> TrackedObject | None:
    if not tracked_objects:
        return None

    best_iou = max(tracked_objects, key=lambda item: _iou(detection_box, item.box))
    if _iou(detection_box, best_iou.box) >= 0.1:
        return best_iou

    best_distance = min(tracked_objects, key=lambda item: _center_distance_squared(detection_box, item.box))
    if _center_distance_squared(detection_box, best_distance.box) <= _distance_threshold_squared(detection_box):
        return best_distance

    return None


def _iou(left: Box, right: Box) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    left_area = max(0, left.x2 - left.x1) * max(0, left.y2 - left.y1)
    right_area = max(0, right.x2 - right.x1) * max(0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _center_distance_squared(left: Box, right: Box) -> float:
    left_x = (left.x1 + left.x2) / 2
    left_y = (left.y1 + left.y2) / 2
    right_x = (right.x1 + right.x2) / 2
    right_y = (right.y1 + right.y2) / 2
    return (left_x - right_x) ** 2 + (left_y - right_y) ** 2


def _distance_threshold_squared(box: Box) -> float:
    width = max(1, box.x2 - box.x1)
    height = max(1, box.y2 - box.y1)
    threshold = ((width**2 + height**2) ** 0.5) * 0.2
    return threshold**2


def _box_center_x(box: Box) -> float:
    return (box.x1 + box.x2) / 2


def _smooth_box(previous: Box, current: Box) -> Box:
    keep = _BOX_SMOOTHING
    update = 1.0 - keep
    return Box(
        x1=int(round(previous.x1 * keep + current.x1 * update)),
        y1=int(round(previous.y1 * keep + current.y1 * update)),
        x2=int(round(previous.x2 * keep + current.x2 * update)),
        y2=int(round(previous.y2 * keep + current.y2 * update)),
    )


def _box_deviates(previous: Box, current: Box, threshold: float) -> bool:
    return max(
        abs(previous.x1 - current.x1),
        abs(previous.y1 - current.y1),
        abs(previous.x2 - current.x2),
        abs(previous.y2 - current.y2),
    ) >= threshold


def _slots_for_count(count: int) -> list[int | None]:
    if count == 1:
        return [None]
    if count == 2:
        return [1, 2]
    return [0, 1, 2]


def _third_edges(display_width: int) -> list[int]:
    return [0, display_width // 3, (display_width * 2) // 3, display_width]


def _new_tracked_objects(
    remaining: list[Detection],
    count: int,
    slots: list[int | None],
    rng: random.Random,
    now: float,
) -> list[TrackedObject]:
    if count <= 0 or not remaining:
        return []

    selected_count = min(count, len(remaining))
    if selected_count == len(remaining):
        selected = list(remaining)
    else:
        selected = rng.sample(remaining, selected_count)

    selected.sort(key=lambda detection: _box_center_x(detection.box))
    return [
        TrackedObject(
            slot=slot,
            class_name=detection.class_name,
            confidence=detection.confidence,
            box=detection.box,
            target_box=detection.box,
            display_box=detection.box,
            mask=detection.mask,
            last_seen_at=now,
        )
        for slot, detection in zip(slots, selected)
    ]


def _unused_slots(matched: list[TrackedObject], target_count: int) -> list[int | None]:
    used = {item.slot for item in matched}
    preferred = [slot for slot in _slots_for_count(target_count) if slot not in used]
    fallback = [slot for slot in [0, 1, 2] if slot not in used]
    return [*preferred, *[slot for slot in fallback if slot not in preferred]]
