from __future__ import annotations

import unittest

import numpy as np

from youarebeingwatched.detector import YoloDetector
from youarebeingwatched.types import BinaryCropMask, UnmaskedCrop


class FakeBox:
    def __init__(self, xyxy: tuple[float, float, float, float], class_id: int, confidence: float) -> None:
        self.xyxy = np.array([xyxy])
        self.cls = np.array([class_id])
        self.conf = np.array([confidence])


class FakeMasks:
    def __init__(self, data: list[np.ndarray]) -> None:
        self.data = data


class FakeResult:
    def __init__(self, boxes: list[FakeBox], masks: FakeMasks | None = None) -> None:
        self.boxes = boxes
        self.masks = masks


class YoloDetectorParsingTest(unittest.TestCase):
    def detector(self, *, segmentation: bool) -> YoloDetector:
        detector = YoloDetector.__new__(YoloDetector)
        detector._names = {0: "person"}
        detector._segmentation = segmentation
        return detector

    def test_segmentation_mode_pairs_masks_by_detection_index(self) -> None:
        detector = self.detector(segmentation=True)
        mask = np.zeros((4, 4), dtype=np.float32)
        mask[:, 1:3] = 1.0
        result = FakeResult([FakeBox((0, 0, 4, 4), 0, 0.9)], FakeMasks([mask]))

        detections = detector._detections_from_result(result)

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections[0].mask, BinaryCropMask)
        crop = np.full((4, 4, 3), 100, dtype=np.uint8)
        masked = detections[0].mask.apply(crop, detections[0].box, crop.shape)
        self.assertTrue(np.all(masked[:, 1:3] == 100))
        self.assertTrue(np.all(masked[:, :1] == 0))
        self.assertTrue(np.all(masked[:, 3:] == 0))

    def test_detection_mode_ignores_available_masks(self) -> None:
        detector = self.detector(segmentation=False)
        mask = np.ones((4, 4), dtype=np.float32)
        result = FakeResult([FakeBox((0, 0, 4, 4), 0, 0.9)], FakeMasks([mask]))

        detections = detector._detections_from_result(result)

        self.assertIsInstance(detections[0].mask, UnmaskedCrop)

    def test_segmentation_mode_falls_back_when_masks_are_absent(self) -> None:
        detector = self.detector(segmentation=True)
        result = FakeResult([FakeBox((0, 0, 4, 4), 0, 0.9)])

        detections = detector._detections_from_result(result)

        self.assertIsInstance(detections[0].mask, UnmaskedCrop)


if __name__ == "__main__":
    unittest.main()
