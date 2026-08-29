from __future__ import annotations

from dataclasses import dataclass
import random
import time

import cv2
import numpy as np

from .config import DisplayConfig
from .types import Box, Detection, Frame


@dataclass(frozen=True)
class PostProcessResult:
    frame: Frame
    detections: list[Detection]
    show_source_label: bool = True


@dataclass(frozen=True)
class TrackedObject:
    slot: int | None
    class_name: str
    box: Box


class PostProcessor:
    """Crops output to tracked objects during selected-source display."""

    def __init__(
        self,
        display: DisplayConfig,
        *,
        selection_classes: tuple[str, ...] = ("person", "dog"),
        selection_interval_seconds: float = 5.0,
        rng: random.Random | None = None,
    ) -> None:
        self._display_width = display.width
        self._display_height = display.height
        self._selection_classes = set(selection_classes)
        self._selection_interval_seconds = selection_interval_seconds
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
        candidates = [detection for detection in detections if detection.class_name in self._selection_classes]
        if not candidates:
            self._tracked_objects = []
            self._force_selection = True
            return PostProcessResult(frame=frame, detections=detections)

        if self._tracked_source_name != source_name:
            self._tracked_objects = []
            self._tracked_source_name = source_name
            self._force_selection = True

        selected = self._select_objects(candidates, now)
        if not selected:
            return PostProcessResult(frame=frame, detections=detections)

        cropped = _compose_crops(frame, selected, self._display_width, self._display_height)
        return PostProcessResult(frame=cropped, detections=[], show_source_label=False)

    def _select_objects(self, candidates: list[Detection], now: float) -> list[TrackedObject]:
        if self._force_selection or not self._tracked_objects or now >= self._next_selection_at:
            return self._elect_objects(candidates, now)

        matched, remaining = _match_tracked_objects(self._tracked_objects, candidates)
        if len(matched) == len(self._tracked_objects):
            self._tracked_objects = matched
            return matched
        if matched:
            return self._stabilized_recompute(matched, remaining, candidates, now)
        return self._elect_objects(candidates, now)

    def _elect_objects(self, candidates: list[Detection], now: float) -> list[TrackedObject]:
        selected_count = min(3, len(candidates))
        if selected_count == len(candidates):
            selected = list(candidates)
        else:
            selected = self._rng.sample(candidates, selected_count)

        selected.sort(key=lambda detection: _box_center_x(detection.box))
        slots = _slots_for_count(selected_count)
        self._tracked_objects = [
            TrackedObject(slot=slot, class_name=detection.class_name, box=detection.box) for slot, detection in zip(slots, selected, strict=True)
        ]
        self._force_selection = False
        self._next_selection_at = now + self._selection_interval_seconds
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
            selected = [*matched, *_new_tracked_objects(remaining, needed, _unused_slots(matched, target_count), self._rng)]

        self._tracked_objects = selected
        self._force_selection = False
        self._next_selection_at = now + self._selection_interval_seconds
        return self._tracked_objects


def _crop_to_display(frame: Frame, box: Box, display_width: int, display_height: int) -> Frame:
    return _crop_to_canvas(frame, box, display_width, display_height)


def _compose_crops(frame: Frame, objects: list[TrackedObject], display_width: int, display_height: int) -> Frame:
    if len(objects) == 1 and objects[0].slot is None:
        return _crop_to_display(frame, objects[0].box, display_width, display_height)

    canvas = _black_canvas(frame, display_width, display_height)
    edges = _third_edges(display_width)
    for item in objects:
        slot = item.slot if item.slot is not None else 0
        slot_x1 = edges[slot]
        slot_x2 = edges[slot + 1]
        if not _paste_crop(frame, item.box, canvas, slot_x1, slot_x2, display_height):
            return frame
    return canvas


def _crop_to_canvas(frame: Frame, box: Box, display_width: int, display_height: int) -> Frame:
    canvas = _black_canvas(frame, display_width, display_height)
    if not _paste_crop(frame, box, canvas, 0, display_width, display_height):
        return frame
    return canvas


def _paste_crop(frame: Frame, box: Box, canvas: Frame, slot_x1: int, slot_x2: int, display_height: int) -> bool:
    frame_height, frame_width = frame.shape[:2]
    clipped = _clip_box(box, frame_width, frame_height)
    if clipped is None:
        return False

    crop = frame[clipped.y1 : clipped.y2, clipped.x1 : clipped.x2]
    crop_height, crop_width = crop.shape[:2]
    if crop_height <= 0 or crop_width <= 0:
        return False

    slot_width = slot_x2 - slot_x1
    if slot_width <= 0:
        return False
    scaled_width = max(1, int(round(crop_width * (display_height / crop_height))))
    resized = cv2.resize(crop, (scaled_width, display_height), interpolation=cv2.INTER_LINEAR)

    if scaled_width <= slot_width:
        x = slot_x1 + (slot_width - scaled_width) // 2
        canvas[:, x : x + scaled_width] = resized
        return True

    start_x = (scaled_width - slot_width) // 2
    canvas[:, slot_x1:slot_x2] = resized[:, start_x : start_x + slot_width]
    return True


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


def _match_tracked_objects(tracked_objects: list[TrackedObject], candidates: list[Detection]) -> tuple[list[TrackedObject], list[Detection]]:
    remaining = list(candidates)
    matched: list[TrackedObject] = []
    for tracked in tracked_objects:
        detection = _match_detection(tracked.box, remaining)
        if detection is None:
            continue
        remaining.remove(detection)
        matched.append(TrackedObject(slot=tracked.slot, class_name=detection.class_name, box=detection.box))
    return matched, remaining


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
) -> list[TrackedObject]:
    if count <= 0 or not remaining:
        return []

    selected_count = min(count, len(remaining))
    if selected_count == len(remaining):
        selected = list(remaining)
    else:
        selected = rng.sample(remaining, selected_count)

    selected.sort(key=lambda detection: _box_center_x(detection.box))
    return [TrackedObject(slot=slot, class_name=detection.class_name, box=detection.box) for slot, detection in zip(slots, selected)]


def _unused_slots(matched: list[TrackedObject], target_count: int) -> list[int | None]:
    used = {item.slot for item in matched}
    preferred = [slot for slot in _slots_for_count(target_count) if slot not in used]
    fallback = [slot for slot in [0, 1, 2] if slot not in used]
    return [*preferred, *[slot for slot in fallback if slot not in preferred]]
