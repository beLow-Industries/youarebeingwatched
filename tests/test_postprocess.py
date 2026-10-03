from __future__ import annotations

import random
import unittest

import numpy as np

from youarebeingwatched.config import DisplayConfig
from youarebeingwatched.postprocess import PostProcessor
from youarebeingwatched.types import BinaryCropMask, Box, Detection


def detection(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> Detection:
    return Detection(class_name="person", confidence=confidence, box=Box(x1=x1, y1=y1, x2=x2, y2=y2))


def dog_detection(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> Detection:
    return Detection(class_name="dog", confidence=confidence, box=Box(x1=x1, y1=y1, x2=x2, y2=y2))


def masked_detection(x1: int, y1: int, x2: int, y2: int, mask: np.ndarray, confidence: float = 0.9) -> Detection:
    return Detection(class_name="person", confidence=confidence, box=Box(x1=x1, y1=y1, x2=x2, y2=y2), mask=BinaryCropMask(mask))


def solid_frame(width: int, height: int, color: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


class SampleSequence:
    def __init__(self, indexes: list[list[int]]) -> None:
        self._indexes = indexes
        self._cursor = 0

    def sample(self, population: list[Detection], count: int) -> list[Detection]:
        indexes = self._indexes[self._cursor]
        self._cursor += 1
        return [population[index] for index in indexes[:count]]


class PostProcessorTest(unittest.TestCase):
    def test_margin_adds_source_pixels_around_crop(self) -> None:
        frame = solid_frame(20, 20, (10, 10, 10))
        frame[5:15, 5:15] = (200, 0, 0)
        processor = PostProcessor(DisplayConfig(width=20, height=20), margin=5, rng=random.Random(0))

        result = processor.process(frame, [detection(5, 5, 15, 15)], source_name="camera:0", should_crop=True, now=0.0)

        self.assertTrue(np.all(result.frame[0, 0] == (10, 10, 10)))
        self.assertTrue(np.all(result.frame[10, 10] == (200, 0, 0)))

    def test_show_boxes_returns_box_in_composed_coordinates(self) -> None:
        frame = solid_frame(20, 20, (10, 10, 10))
        processor = PostProcessor(DisplayConfig(width=20, height=20), show_boxes=True, margin=5, rng=random.Random(0))

        result = processor.process(frame, [detection(5, 5, 15, 15)], source_name="camera:0", should_crop=True, now=0.0)

        self.assertEqual(result.detections[0].box, Box(5, 5, 15, 15))

    def test_margin_clips_to_source_edges_without_padding(self) -> None:
        frame = solid_frame(20, 20, (200, 0, 0))
        processor = PostProcessor(DisplayConfig(width=20, height=20), show_boxes=True, margin=5, rng=random.Random(0))

        result = processor.process(frame, [detection(0, 0, 20, 20)], source_name="camera:0", should_crop=True, now=0.0)

        self.assertEqual(result.frame.shape, (20, 20, 3))
        self.assertEqual(result.detections[0].box, Box(0, 0, 20, 20))
        self.assertTrue(np.all(result.frame[0, 0] == (200, 0, 0)))
        self.assertTrue(np.all(result.frame[-1, -1] == (200, 0, 0)))

    def test_crops_height_fit_and_centers_with_black_sides(self) -> None:
        frame = solid_frame(100, 80, (10, 20, 30))
        frame[10:70, 20:50] = (200, 40, 80)
        processor = PostProcessor(DisplayConfig(width=80, height=120), rng=random.Random(0))

        result = processor.process(frame, [detection(20, 10, 50, 70)], source_name="camera:0", should_crop=True, now=0.0)

        self.assertEqual(result.frame.shape, (120, 80, 3))
        self.assertFalse(result.show_source_label)
        self.assertEqual(result.detections, [])
        self.assertTrue(np.all(result.frame[:, :10] == 0))
        self.assertTrue(np.all(result.frame[:, 70:] == 0))
        self.assertTrue(np.all(result.frame[:, 10:70] == (200, 40, 80)))

    def test_segmentation_mask_blacks_out_pixels_outside_mask(self) -> None:
        frame = solid_frame(4, 4, (200, 40, 80))
        mask = np.zeros((4, 4), dtype=bool)
        mask[:, :2] = True
        processor = PostProcessor(DisplayConfig(width=4, height=4), rng=random.Random(0))

        result = processor.process(frame, [masked_detection(0, 0, 4, 4, mask)], source_name="camera:0", should_crop=True, now=0.0)

        self.assertTrue(np.all(result.frame[:, :2] == (200, 40, 80)))
        self.assertTrue(np.all(result.frame[:, 2:] == 0))

    def test_segmentation_mask_is_retained_during_linger_window(self) -> None:
        frame = solid_frame(4, 4, (200, 40, 80))
        mask = np.zeros((4, 4), dtype=bool)
        mask[:, 1:3] = True
        processor = PostProcessor(DisplayConfig(width=4, height=4), rng=random.Random(0))

        processor.process(frame, [masked_detection(0, 0, 4, 4, mask)], source_name="camera:0", should_crop=True, now=0.0)
        result = processor.process(frame, [], source_name="camera:0", should_crop=True, now=0.25)

        self.assertTrue(np.all(result.frame[:, :1] == 0))
        self.assertTrue(np.all(result.frame[:, 1:3] == (200, 40, 80)))
        self.assertTrue(np.all(result.frame[:, 3:] == 0))

    def test_crops_horizontally_when_height_fit_is_wider_than_display(self) -> None:
        frame = np.zeros((20, 100, 3), dtype=np.uint8)
        frame[:, :50] = (20, 20, 20)
        frame[:, 50:] = (220, 220, 220)
        processor = PostProcessor(DisplayConfig(width=40, height=40), rng=random.Random(0))

        result = processor.process(frame, [detection(0, 0, 100, 20)], source_name="camera:0", should_crop=True, now=0.0)

        self.assertEqual(result.frame.shape, (40, 40, 3))
        self.assertLess(np.mean(result.frame[:, :19]), 30)
        self.assertGreater(np.mean(result.frame[:, 21:]), 210)

    def test_clips_detection_box_to_source_frame(self) -> None:
        frame = solid_frame(40, 30, (0, 0, 0))
        frame[0:20, 0:20] = (90, 100, 110)
        processor = PostProcessor(DisplayConfig(width=40, height=40), rng=random.Random(0))

        result = processor.process(frame, [detection(-10, -10, 20, 20)], source_name="camera:0", should_crop=True, now=0.0)

        self.assertEqual(result.frame.shape, (40, 40, 3))
        self.assertTrue(np.all(result.frame == (90, 100, 110)))

    def test_tracking_box_is_smoothed_between_frames(self) -> None:
        frame = np.zeros((40, 120, 3), dtype=np.uint8)
        frame[:, :40] = (255, 0, 0)
        frame[:, 40:80] = (0, 255, 0)
        frame[:, 80:] = (0, 0, 255)
        processor = PostProcessor(DisplayConfig(width=40, height=40), stabilize_box_pixels=0, rng=random.Random(0))

        first = processor.process(frame, [detection(0, 0, 40, 40)], source_name="camera:0", should_crop=True, now=0.0)
        second = processor.process(frame, [detection(8, 0, 48, 40)], source_name="camera:0", should_crop=True, now=1.0)

        self.assertTrue(np.all(first.frame == (255, 0, 0)))
        self.assertGreater(np.mean(second.frame[:, :, 0]), 100)
        self.assertGreater(np.mean(second.frame[:, :, 1]), 5)
        self.assertLess(np.mean(second.frame[:, :, 2]), 5)

    def test_small_box_jitter_does_not_update_crop_target(self) -> None:
        frame = np.zeros((40, 120, 3), dtype=np.uint8)
        frame[:, :40] = (255, 0, 0)
        frame[:, 40:80] = (0, 255, 0)
        processor = PostProcessor(DisplayConfig(width=40, height=40), stabilize_box_pixels=16, rng=random.Random(0))

        first = processor.process(frame, [detection(0, 0, 40, 40)], source_name="camera:0", should_crop=True, now=0.0)
        second = processor.process(frame, [detection(8, 0, 48, 40)], source_name="camera:0", should_crop=True, now=1.0)

        self.assertTrue(np.array_equal(first.frame, second.frame))

    def test_large_box_deviation_updates_crop_target_with_easing(self) -> None:
        frame = np.zeros((40, 120, 3), dtype=np.uint8)
        frame[:, :40] = (255, 0, 0)
        frame[:, 40:80] = (0, 255, 0)
        processor = PostProcessor(DisplayConfig(width=40, height=40), stabilize_box_pixels=16, rng=random.Random(0))

        first = processor.process(frame, [detection(0, 0, 40, 40)], source_name="camera:0", should_crop=True, now=0.0)
        second = processor.process(frame, [detection(16, 0, 56, 40)], source_name="camera:0", should_crop=True, now=0.5)

        self.assertFalse(np.array_equal(first.frame, second.frame))
        self.assertGreater(np.mean(second.frame[:, :, 0]), 100)
        self.assertGreater(np.mean(second.frame[:, :, 1]), 5)

    def test_latest_detection_keeps_matching_while_crop_target_is_held(self) -> None:
        frame = np.zeros((40, 200, 3), dtype=np.uint8)
        processor = PostProcessor(DisplayConfig(width=40, height=40), stabilize_box_pixels=16, rng=random.Random(0))

        processor.process(frame, [detection(0, 0, 40, 40)], source_name="camera:0", should_crop=True, now=0.0)
        processor.process(frame, [detection(8, 0, 48, 40)], source_name="camera:0", should_crop=True, now=0.1)
        processor.process(frame, [detection(16, 0, 56, 40)], source_name="camera:0", should_crop=True, now=0.2)

        tracked = processor._tracked_objects[0]  # noqa: SLF001
        self.assertEqual(tracked.box, Box(16, 0, 56, 40))
        self.assertEqual(tracked.target_box, Box(16, 0, 56, 40))

    def test_keeps_tracking_same_person_before_interval(self) -> None:
        frame = np.zeros((100, 90, 3), dtype=np.uint8)
        frame[:, :30] = (40, 0, 0)
        frame[:, 60:] = (0, 80, 0)
        processor = PostProcessor(DisplayConfig(width=90, height=100), rng=random.Random(1))

        first = processor.process(
            frame,
            [detection(0, 0, 30, 100), detection(60, 0, 90, 100)],
            source_name="camera:0",
            should_crop=True,
            now=0.0,
        )
        second = processor.process(
            frame,
            [detection(3, 0, 33, 100), detection(60, 0, 90, 100)],
            source_name="camera:0",
            should_crop=True,
            now=2.0,
        )

        self.assertTrue(np.all(first.frame[:, :30] == 0))
        self.assertTrue(np.all(first.frame[:, 30:60] == (40, 0, 0)))
        self.assertTrue(np.all(first.frame[:, 60:90] == (0, 80, 0)))
        self.assertLess(np.mean(second.frame[:, :30]), 10)
        self.assertGreater(np.mean(second.frame[:, 30:60, 0]), 35)
        self.assertGreater(np.mean(second.frame[:, 60:90, 1]), 75)

    def test_reselects_after_interval_and_after_source_decision(self) -> None:
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        frame[:, :30] = (40, 0, 0)
        frame[:, 30:60] = (0, 80, 0)
        frame[:, 60:90] = (0, 0, 120)
        frame[:, 90:120] = (160, 160, 0)
        processor = PostProcessor(DisplayConfig(width=90, height=100), rng=SampleSequence([[0, 1, 2], [1, 2, 3], [0, 2, 3]]))  # type: ignore[arg-type]
        people = [
            detection(0, 0, 30, 100),
            detection(30, 0, 60, 100),
            detection(60, 0, 90, 100),
            detection(90, 0, 120, 100),
        ]

        first = processor.process(frame, people, source_name="camera:0", should_crop=True, now=0.0)
        after_interval = processor.process(frame, people, source_name="camera:0", should_crop=True, now=1.0)
        processor.note_source_decision(now=6.0)
        after_source_decision = processor.process(frame, people, source_name="camera:0", should_crop=True, now=6.0)

        self.assertFalse(np.array_equal(first.frame, after_interval.frame))
        self.assertFalse(np.array_equal(after_interval.frame, after_source_decision.frame))
        self.assertEqual(first.frame.shape, (100, 90, 3))
        self.assertEqual(after_interval.frame.shape, (100, 90, 3))
        self.assertEqual(after_source_decision.frame.shape, (100, 90, 3))

    def test_lingers_tracked_person_when_detection_temporarily_disappears(self) -> None:
        frame = np.zeros((100, 90, 3), dtype=np.uint8)
        frame[:, :30] = (40, 0, 0)
        frame[:, 60:] = (0, 80, 0)
        processor = PostProcessor(DisplayConfig(width=90, height=100), rng=random.Random(1))

        processor.process(
            frame,
            [detection(0, 0, 30, 100), detection(60, 0, 90, 100)],
            source_name="camera:0",
            should_crop=True,
            now=0.0,
        )
        result = processor.process(frame, [detection(60, 0, 90, 100)], source_name="camera:0", should_crop=True, now=0.5)

        self.assertTrue(np.all(result.frame[:, :30] == 0))
        self.assertTrue(np.all(result.frame[:, 30:60] == (40, 0, 0)))
        self.assertTrue(np.all(result.frame[:, 60:] == (0, 80, 0)))
        self.assertFalse(result.show_source_label)

    def test_recompute_keeps_survivors_in_previous_thirds(self) -> None:
        frame = np.zeros((100, 90, 3), dtype=np.uint8)
        frame[:, :30] = (40, 0, 0)
        frame[:, 30:60] = (0, 80, 0)
        frame[:, 60:] = (0, 0, 120)
        processor = PostProcessor(DisplayConfig(width=90, height=100), rng=random.Random(0))

        processor.process(
            frame,
            [detection(0, 0, 30, 100), detection(30, 0, 60, 100), detection(60, 0, 90, 100)],
            source_name="camera:0",
            should_crop=True,
            now=0.0,
        )
        result = processor.process(
            frame,
            [detection(0, 0, 30, 100), detection(60, 0, 90, 100)],
            source_name="camera:0",
            should_crop=True,
            now=0.6,
        )

        self.assertTrue(np.all(result.frame[:, :30] == (40, 0, 0)))
        self.assertTrue(np.all(result.frame[:, 30:60] == 0))
        self.assertTrue(np.all(result.frame[:, 60:] == (0, 0, 120)))

    def test_partial_recompute_does_not_require_all_open_slots_to_be_filled(self) -> None:
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        frame[:, :30] = (40, 0, 0)
        frame[:, 90:] = (160, 160, 0)
        processor = PostProcessor(DisplayConfig(width=90, height=100), rng=random.Random(0))

        processor.process(
            frame,
            [detection(0, 0, 30, 100), detection(30, 0, 60, 100), detection(60, 0, 90, 100)],
            source_name="camera:0",
            should_crop=True,
            now=0.0,
        )
        result = processor.process(
            frame,
            [detection(0, 0, 30, 100), detection(90, 0, 120, 100)],
            source_name="camera:0",
            should_crop=True,
            now=0.5,
        )

        self.assertEqual(result.frame.shape, (100, 90, 3))
        self.assertTrue(np.all(result.frame[:, :30] == (40, 0, 0)))
        self.assertTrue(np.any(result.frame[:, 30:] == (160, 160, 0)))

    def test_single_survivor_stays_in_third_until_next_election(self) -> None:
        frame = np.zeros((100, 90, 3), dtype=np.uint8)
        frame[:, :30] = (40, 0, 0)
        frame[:, 60:] = (0, 80, 0)
        processor = PostProcessor(DisplayConfig(width=90, height=100), rng=random.Random(1))

        processor.process(
            frame,
            [detection(0, 0, 30, 100), detection(60, 0, 90, 100)],
            source_name="camera:0",
            should_crop=True,
            now=0.0,
        )
        immediate = processor.process(frame, [detection(60, 0, 90, 100)], source_name="camera:0", should_crop=True, now=0.5)
        after_interval = processor.process(frame, [detection(60, 0, 90, 100)], source_name="camera:0", should_crop=True, now=2.0)

        self.assertTrue(np.all(immediate.frame[:, :30] == 0))
        self.assertTrue(np.all(immediate.frame[:, 30:60] == (40, 0, 0)))
        self.assertTrue(np.all(immediate.frame[:, 60:] == (0, 80, 0)))
        self.assertTrue(np.all(after_interval.frame[:, :30] == 0))
        self.assertTrue(np.all(after_interval.frame[:, 30:60] == (0, 80, 0)))
        self.assertTrue(np.all(after_interval.frame[:, 60:] == 0))

    def test_full_detection_miss_lingers_previous_crop(self) -> None:
        frame = solid_frame(90, 100, (0, 0, 0))
        frame[:, 30:60] = (40, 0, 0)
        processor = PostProcessor(DisplayConfig(width=90, height=100), rng=random.Random(0))

        processor.process(frame, [detection(30, 0, 60, 100)], source_name="camera:0", should_crop=True, now=0.0)
        result = processor.process(frame, [], source_name="camera:0", should_crop=True, now=0.25)

        self.assertFalse(result.show_source_label)
        self.assertEqual(result.detections, [])
        self.assertTrue(np.all(result.frame[:, 30:60] == (40, 0, 0)))

    def test_full_detection_miss_expires_after_linger_window(self) -> None:
        frame = solid_frame(90, 100, (12, 34, 56))
        processor = PostProcessor(DisplayConfig(width=90, height=100), rng=random.Random(0))

        processor.process(frame, [detection(30, 0, 60, 100)], source_name="camera:0", should_crop=True, now=0.0)
        result = processor.process(frame, [], source_name="camera:0", should_crop=True, now=0.6)

        self.assertIs(result.frame, frame)
        self.assertTrue(result.show_source_label)

    def test_returning_detection_updates_lingering_box(self) -> None:
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        frame[:, :30] = (40, 0, 0)
        frame[:, 5:35] = (0, 80, 0)
        processor = PostProcessor(DisplayConfig(width=30, height=100), rng=random.Random(0))

        processor.process(frame, [detection(0, 0, 30, 100)], source_name="camera:0", should_crop=True, now=0.0)
        processor.process(frame, [], source_name="camera:0", should_crop=True, now=0.25)
        result = processor.process(frame, [detection(5, 0, 35, 100)], source_name="camera:0", should_crop=True, now=0.4)

        self.assertGreater(np.mean(result.frame[:, :, 0]), 0)
        self.assertGreater(np.mean(result.frame[:, :, 1]), 0)

    def test_source_change_clears_lingering_tracks(self) -> None:
        frame = solid_frame(90, 100, (12, 34, 56))
        processor = PostProcessor(DisplayConfig(width=90, height=100), rng=random.Random(0))

        processor.process(frame, [detection(30, 0, 60, 100)], source_name="camera:0", should_crop=True, now=0.0)
        result = processor.process(frame, [], source_name="camera:1", should_crop=True, now=0.25)

        self.assertIs(result.frame, frame)
        self.assertTrue(result.show_source_label)

    def test_raw_display_mode_clears_lingering_tracks(self) -> None:
        frame = solid_frame(90, 100, (12, 34, 56))
        processor = PostProcessor(DisplayConfig(width=90, height=100), rng=random.Random(0))

        processor.process(frame, [detection(30, 0, 60, 100)], source_name="camera:0", should_crop=True, now=0.0)
        raw = processor.process(frame, [], source_name="camera:0", should_crop=False, now=0.25)
        cropped = processor.process(frame, [], source_name="camera:0", should_crop=True, now=0.3)

        self.assertIs(raw.frame, frame)
        self.assertIs(cropped.frame, frame)
        self.assertTrue(cropped.show_source_label)

    def test_two_people_use_center_and_right_thirds(self) -> None:
        frame = np.zeros((100, 90, 3), dtype=np.uint8)
        frame[:, :30] = (40, 0, 0)
        frame[:, 60:] = (0, 80, 0)
        processor = PostProcessor(DisplayConfig(width=90, height=100), rng=random.Random(0))

        result = processor.process(
            frame,
            [detection(0, 0, 30, 100), detection(60, 0, 90, 100)],
            source_name="camera:0",
            should_crop=True,
            now=0.0,
        )

        self.assertTrue(np.all(result.frame[:, :30] == 0))
        self.assertTrue(np.all(result.frame[:, 30:60] == (40, 0, 0)))
        self.assertTrue(np.all(result.frame[:, 60:90] == (0, 80, 0)))

    def test_three_people_fill_all_thirds_in_source_order(self) -> None:
        frame = np.zeros((100, 90, 3), dtype=np.uint8)
        frame[:, :30] = (40, 0, 0)
        frame[:, 30:60] = (0, 80, 0)
        frame[:, 60:] = (0, 0, 120)
        processor = PostProcessor(DisplayConfig(width=90, height=100), rng=random.Random(0))

        result = processor.process(
            frame,
            [detection(60, 0, 90, 100), detection(0, 0, 30, 100), detection(30, 0, 60, 100)],
            source_name="camera:0",
            should_crop=True,
            now=0.0,
        )

        self.assertTrue(np.all(result.frame[:, :30] == (40, 0, 0)))
        self.assertTrue(np.all(result.frame[:, 30:60] == (0, 80, 0)))
        self.assertTrue(np.all(result.frame[:, 60:90] == (0, 0, 120)))

    def test_passes_through_original_frame_when_no_person_is_available(self) -> None:
        frame = solid_frame(64, 48, (12, 34, 56))
        processor = PostProcessor(DisplayConfig(width=80, height=120), rng=random.Random(0))

        result = processor.process(frame, [], source_name="camera:0", should_crop=True, now=0.0)

        self.assertIs(result.frame, frame)
        self.assertTrue(result.show_source_label)

    def test_dog_detection_is_selected_and_displayed(self) -> None:
        frame = solid_frame(100, 80, (10, 20, 30))
        frame[10:70, 20:50] = (200, 120, 20)
        processor = PostProcessor(DisplayConfig(width=80, height=120), rng=random.Random(0))

        result = processor.process(frame, [dog_detection(20, 10, 50, 70)], source_name="camera:0", should_crop=True, now=0.0)

        self.assertEqual(result.frame.shape, (120, 80, 3))
        self.assertFalse(result.show_source_label)
        self.assertEqual(result.detections, [])
        self.assertTrue(np.all(result.frame[:, 10:70] == (200, 120, 20)))

    def test_person_and_dog_share_thirds(self) -> None:
        frame = np.zeros((100, 90, 3), dtype=np.uint8)
        frame[:, :30] = (40, 0, 0)
        frame[:, 60:] = (200, 120, 20)
        processor = PostProcessor(DisplayConfig(width=90, height=100), rng=random.Random(0))

        result = processor.process(
            frame,
            [detection(0, 0, 30, 100), dog_detection(60, 0, 90, 100)],
            source_name="camera:0",
            should_crop=True,
            now=0.0,
        )

        self.assertTrue(np.all(result.frame[:, :30] == 0))
        self.assertTrue(np.all(result.frame[:, 30:60] == (40, 0, 0)))
        self.assertTrue(np.all(result.frame[:, 60:90] == (200, 120, 20)))


if __name__ == "__main__":
    unittest.main()
