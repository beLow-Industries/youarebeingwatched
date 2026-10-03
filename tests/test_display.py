from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from youarebeingwatched.display import LABEL_FONT, LABEL_THICKNESS, _cover_frame, _detection_label, _should_continue, draw_detections
from youarebeingwatched.types import Box, Detection


class DisplayKeyboardTest(unittest.TestCase):
    def test_q_exits_display_loop(self) -> None:
        self.assertFalse(_should_continue(ord("q")))
        self.assertFalse(_should_continue(ord("Q")))

    def test_escape_exits_display_loop(self) -> None:
        self.assertFalse(_should_continue(27))

    def test_other_keys_continue_display_loop(self) -> None:
        self.assertTrue(_should_continue(-1 & 0xFF))
        self.assertTrue(_should_continue(ord("x")))


class DisplayCompositingTest(unittest.TestCase):
    def test_cover_frame_matches_target_without_borders(self) -> None:
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)
        rendered = _cover_frame(frame, 1920, 1200)

        self.assertEqual(rendered.shape, (1200, 1920, 3))

    def test_cover_frame_keeps_matching_frame_unchanged(self) -> None:
        frame = np.zeros((1200, 1920, 3), dtype=np.uint8)

        self.assertIs(_cover_frame(frame, 1920, 1200), frame)

    def test_detection_labels_use_requested_names_and_percentages(self) -> None:
        self.assertEqual(_detection_label(Detection("person", 0.9, Box(1, 1, 3, 3))), "human (90%)")
        self.assertEqual(_detection_label(Detection("dog", 0.42, Box(1, 1, 3, 3))), "doggo (42%)")

    def test_detection_boxes_are_white(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        rendered = draw_detections(frame, [Detection("person", 0.9, Box(4, 4, 12, 12))])

        self.assertEqual(tuple(rendered[4, 4]), (255, 255, 255))

    def test_detection_label_has_white_background_and_black_text(self) -> None:
        frame = np.full((60, 120, 3), 80, dtype=np.uint8)
        rendered = draw_detections(frame, [Detection("person", 0.9, Box(10, 30, 40, 50))])

        self.assertTrue(np.any(np.all(rendered[:30, 10:100] == (255, 255, 255), axis=2)))
        self.assertTrue(np.any(np.mean(rendered[:30, 10:100], axis=2) < 80))

    def test_detection_label_uses_requested_font_style(self) -> None:
        frame = np.zeros((60, 120, 3), dtype=np.uint8)

        with patch("youarebeingwatched.display.cv2.putText") as put_text:
            draw_detections(frame, [Detection("person", 0.9, Box(10, 30, 40, 50))], font_size=1.2)

        self.assertEqual(put_text.call_args.args[3], LABEL_FONT)
        self.assertEqual(put_text.call_args.args[4], 1.2)
        self.assertEqual(put_text.call_args.args[6], LABEL_THICKNESS)


if __name__ == "__main__":
    unittest.main()
