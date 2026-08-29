from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Iterable, Protocol

from loguru import logger

from .types import Box, Detection, Frame


class Detector(Protocol):
    def detect(self, frame: Frame, class_names: Iterable[str] | None = None) -> list[Detection]:
        ...

    def detect_batch(self, frames: list[Frame], class_names: Iterable[str] | None = None) -> list[list[Detection]]:
        ...


class YoloDetector:
    def __init__(self, weights: str, confidence: float) -> None:
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))

        from ultralytics import YOLO

        logger.info("loading YOLO weights={} confidence={}", weights, confidence)
        started = time.monotonic()
        self._model = YOLO(weights)
        self._confidence = confidence
        self._names = self._normalize_names(self._model.names)
        logger.info("loaded YOLO model in {:.3f}s classes={}", time.monotonic() - started, len(self._names))

    def detect(self, frame: Frame, class_names: Iterable[str] | None = None) -> list[Detection]:
        return self.detect_batch([frame], class_names=class_names)[0]

    def detect_batch(self, frames: list[Frame], class_names: Iterable[str] | None = None) -> list[list[Detection]]:
        if not frames:
            return []

        class_names = tuple(class_names) if class_names is not None else None
        wanted_ids = self._class_ids(class_names)
        started = time.monotonic()
        logger.debug("YOLO predict start frames={} classes={}", len(frames), class_names or "all")
        results = self._model.predict(
            frames,
            conf=self._confidence,
            classes=wanted_ids,
            verbose=False,
        )
        detections = [self._detections_from_result(result) for result in results]
        elapsed = time.monotonic() - started
        detection_counts = [len(items) for items in detections]
        log = logger.warning if elapsed >= 1.0 else logger.debug
        log("YOLO predict done frames={} detections={} elapsed={:.3f}s", len(frames), detection_counts, elapsed)
        return detections

    def _detections_from_result(self, result: object) -> list[Detection]:
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            detections.append(
                Detection(
                    class_name=self._names.get(class_id, str(class_id)),
                    confidence=confidence,
                    box=Box(
                        x1=int(round(xyxy[0])),
                        y1=int(round(xyxy[1])),
                        x2=int(round(xyxy[2])),
                        y2=int(round(xyxy[3])),
                    ),
                )
            )
        return detections

    def _class_ids(self, class_names: Iterable[str] | None) -> list[int] | None:
        if class_names is None:
            return None

        wanted = set(class_names)
        return [class_id for class_id, class_name in self._names.items() if class_name in wanted]

    @staticmethod
    def _normalize_names(names: object) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, list):
            return {index: str(value) for index, value in enumerate(names)}
        raise TypeError(f"Unsupported YOLO names type: {type(names)!r}")


@dataclass
class MockDetector:
    frame_index: int = 0

    def detect(self, frame: Frame, class_names: Iterable[str] | None = None) -> list[Detection]:
        return self.detect_batch([frame], class_names=class_names)[0]

    def detect_batch(self, frames: list[Frame], class_names: Iterable[str] | None = None) -> list[list[Detection]]:
        started = time.monotonic()
        self.frame_index += 1
        wanted = set(class_names) if class_names is not None else {"person", "dog"}
        output: list[list[Detection]] = []

        for frame in frames:
            height, width = frame.shape[:2]
            detections: list[Detection] = []
            offset = (self.frame_index * 13) % max(1, width // 3)
            if "person" in wanted:
                detections.append(
                    Detection(
                        class_name="person",
                        confidence=0.91,
                        box=Box(
                            x1=width // 5 + offset // 3,
                            y1=height // 5,
                            x2=width // 5 + offset // 3 + width // 5,
                            y2=height * 4 // 5,
                        ),
                    )
                )
            if "dog" in wanted:
                detections.append(
                    Detection(
                        class_name="dog",
                        confidence=0.84,
                        box=Box(
                            x1=width * 3 // 5 - offset // 4,
                            y1=height // 2,
                            x2=width * 4 // 5 - offset // 4,
                            y2=height * 4 // 5,
                        ),
                    )
                )
            output.append(detections)

        logger.debug(
            "mock detect frames={} classes={} detections={} elapsed={:.3f}s",
            len(frames),
            sorted(wanted),
            [len(items) for items in output],
            time.monotonic() - started,
        )
        return output
