from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from youarebeingwatched.config import AppConfig
from youarebeingwatched.image import next_output_path, run_image
from youarebeingwatched.types import Box, Detection


class FakeDetector:
    last: "FakeDetector | None" = None

    def __init__(self, _weights: str, *, confidence: float, segmentation: bool = False) -> None:
        self.confidence = confidence
        self.segmentation = segmentation
        self.class_names = None
        FakeDetector.last = self

    def detect(self, _frame: np.ndarray, class_names: object = None) -> list[Detection]:
        self.class_names = class_names
        return [Detection(class_name="person", confidence=0.9, box=Box(100, 100, 300, 500))]


class ImageCommandTest(unittest.TestCase):
    def test_next_output_path_skips_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "out-0.jpg").touch()
            (root / "out-1.jpg").touch()

            self.assertEqual(next_output_path(root), root / "out-2.jpg")

    def test_unsupported_extension_returns_failure_without_detection(self) -> None:
        with patch("youarebeingwatched.image.YoloDetector") as detector:
            self.assertEqual(run_image(AppConfig(), source="photo.gif"), 1)

        detector.assert_not_called()

    def test_unreadable_image_returns_failure_without_detection(self) -> None:
        with (
            patch("youarebeingwatched.image.cv2.imread", return_value=None),
            patch("youarebeingwatched.image.YoloDetector") as detector,
        ):
            self.assertEqual(run_image(AppConfig(), source="missing.jpg"), 1)

        detector.assert_not_called()

    def test_success_writes_1080p_frame_and_detects_people_and_dogs(self) -> None:
        written: dict[str, object] = {}

        def imwrite(path: str, frame: np.ndarray) -> bool:
            written["path"] = path
            written["shape"] = frame.shape
            return True

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out-0.jpg"
            with (
                patch("youarebeingwatched.image.cv2.imread", return_value=np.zeros((600, 800, 3), dtype=np.uint8)),
                patch("youarebeingwatched.image.cv2.imwrite", side_effect=imwrite),
                patch("youarebeingwatched.image.next_output_path", return_value=output),
                patch("youarebeingwatched.image.YoloDetector", FakeDetector),
                patch("sys.stdout", io.StringIO()),
            ):
                self.assertEqual(run_image(AppConfig(), source="photo.JPG", threshold=0.7), 0)

        self.assertEqual(written["path"], str(output))
        self.assertEqual(written["shape"], (1080, 1920, 3))
        self.assertIsNotNone(FakeDetector.last)
        self.assertEqual(FakeDetector.last.class_names, ("person", "dog"))
        self.assertEqual(FakeDetector.last.confidence, 0.7)

    def test_write_failure_returns_failure(self) -> None:
        with (
            patch("youarebeingwatched.image.cv2.imread", return_value=np.zeros((600, 800, 3), dtype=np.uint8)),
            patch("youarebeingwatched.image.cv2.imwrite", return_value=False),
            patch("youarebeingwatched.image.YoloDetector", FakeDetector),
            patch("sys.stdout", io.StringIO()),
        ):
            self.assertEqual(run_image(AppConfig(), source="photo.jpg"), 1)


if __name__ == "__main__":
    unittest.main()
