from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from youarebeingwatched.app import _select_source
from youarebeingwatched.config import AppConfig
from youarebeingwatched.sources import SourceManager
from youarebeingwatched.types import Box, Detection


class FakeSource:
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def is_open(self) -> bool:
        return True

    def read(self) -> np.ndarray:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def release(self) -> None:
        return None


class FakeDetector:
    def __init__(self, detections: list[list[Detection]]) -> None:
        self.detections = detections
        self.class_names = None

    def detect(self, frame: np.ndarray, class_names: object = None) -> list[Detection]:
        return self.detections[0]

    def detect_batch(self, frames: list[np.ndarray], class_names: object = None) -> list[list[Detection]]:
        self.class_names = class_names
        return self.detections


class SourceSelectionTest(unittest.TestCase):
    def test_dog_only_source_can_be_selected(self) -> None:
        source = FakeSource("camera:0")
        detector = FakeDetector([[Detection(class_name="dog", confidence=0.8, box=Box(1, 1, 5, 5))]])

        with patch("youarebeingwatched.app.random.choice", return_value=source):
            selected = _select_source(AppConfig(), detector, SourceManager([source]), None)

        self.assertIs(selected, source)
        self.assertEqual(detector.class_names, {"person", "dog"})


if __name__ == "__main__":
    unittest.main()
