from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from youarebeingwatched import cli
from youarebeingwatched.config import (
    AppConfig,
    DisplayConfig,
    ModelConfig,
    SourcesConfig,
    TrackingConfig,
    with_display_overrides,
)
from youarebeingwatched.v4l2 import CameraMode, select_camera_mode


class ConfigTest(unittest.TestCase):
    def test_default_confidence_is_forty_percent(self) -> None:
        self.assertEqual(ModelConfig().confidence, 0.40)

    def test_default_model_weights_keep_detection_and_segmentation_separate(self) -> None:
        self.assertEqual(ModelConfig().weights, "yolo26n.pt")
        self.assertEqual(ModelConfig().segmentation_weights, "yolo26n-seg.pt")

    def test_display_defaults_to_fullscreen(self) -> None:
        self.assertTrue(DisplayConfig().fullscreen)

    def test_sources_scan_the_first_ten_camera_indices_by_default(self) -> None:
        self.assertEqual(SourcesConfig().scan_indices, tuple(range(10)))

    def test_source_mode_strategy_overrides(self) -> None:
        config = with_display_overrides(AppConfig(), fullscreen=None, scan_max=None, source_mode_strategy="max_resolution")

        self.assertEqual(config.sources.mode_strategy, "max_resolution")

    def test_tracking_defaults_smooth_short_detection_drops(self) -> None:
        self.assertEqual(TrackingConfig().missing_linger_seconds, 0.5)
        self.assertEqual(TrackingConfig().reselect_interval_seconds, 1.0)
        self.assertEqual(TrackingConfig().stabilize_box_pixels, 16.0)

class CliTest(unittest.TestCase):
    def test_max_fps_flag_sets_camera_mode_strategy(self) -> None:
        with patch.object(cli, "run_app", return_value=0) as run_app:
            self.assertEqual(cli._run(["--mock", "--headless", "--max-fps"]), 0)

        config = run_app.call_args.args[0]
        self.assertEqual(config.sources.mode_strategy, "max_fps")

    def test_max_resolution_flag_sets_camera_mode_strategy(self) -> None:
        with patch.object(cli, "run_app", return_value=0) as run_app:
            self.assertEqual(cli._run(["--mock", "--headless", "--max-resolution"]), 0)

        config = run_app.call_args.args[0]
        self.assertEqual(config.sources.mode_strategy, "max_resolution")

    def test_segmentation_flag_is_passed_to_app(self) -> None:
        with patch.object(cli, "run_app", return_value=0) as run_app:
            self.assertEqual(cli._run(["--mock", "--headless", "--segmentation"]), 0)

        self.assertTrue(run_app.call_args.kwargs["segmentation"])

    def test_threshold_flag_is_passed_to_app(self) -> None:
        with patch.object(cli, "run_app", return_value=0) as run_app:
            self.assertEqual(cli._run(["--mock", "--headless", "--threshold", "0.65"]), 0)

        self.assertEqual(run_app.call_args.kwargs["threshold"], 0.65)

    def test_threshold_must_be_between_zero_and_one(self) -> None:
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            cli._run(["--threshold", "1.1"])

    def test_box_margin_stabilization_and_no_track_flags_are_passed_to_app(self) -> None:
        with patch.object(cli, "run_app", return_value=0) as run_app:
            self.assertEqual(
                cli._run(["--mock", "--headless", "--show-box", "--margin", "10", "--stabilize-box", "8.5", "--do-not-track"]),
                0,
            )

        self.assertTrue(run_app.call_args.kwargs["show_box"])
        self.assertEqual(run_app.call_args.kwargs["margin"], 10)
        self.assertEqual(run_app.call_args.kwargs["stabilize_box"], 8.5)
        self.assertTrue(run_app.call_args.kwargs["do_not_track"])

    def test_margin_must_be_nonnegative(self) -> None:
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            cli._run(["--margin", "-1"])

    def test_stabilize_box_must_be_nonnegative(self) -> None:
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            cli._run(["--stabilize-box", "-0.1"])

    def test_camera_mode_flags_are_mutually_exclusive(self) -> None:
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            cli._run(["--max-fps", "--max-resolution"])

    def test_doctor_scans_ten_indices_by_default(self) -> None:
        with patch.object(cli, "run_doctor", return_value=0) as run_doctor:
            self.assertEqual(cli.main(["doctor"]), 0)

        run_doctor.assert_called_once_with(10)


class V4l2SelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.modes = [
            CameraMode("MJPG", "Motion-JPEG", 640, 480, (30.0,)),
            CameraMode("MJPG", "Motion-JPEG", 1280, 720, (10.0,)),
            CameraMode("MJPG", "Motion-JPEG", 800, 600, (60.0,)),
            CameraMode("YUYV", "YUYV", 1920, 1080, (30.0,)),
        ]

    def test_configured_strategy_prefers_configured_resolution(self) -> None:
        selected = select_camera_mode(
            self.modes,
            minimum_fps=25.0,
            preferred_format="MJPG",
            preferred_width=640,
            preferred_height=480,
        )

        self.assertEqual((selected.width, selected.height, selected.max_fps), (640, 480, 30.0))

    def test_max_fps_strategy_uses_highest_fps_mjpg_above_minimum(self) -> None:
        selected = select_camera_mode(
            self.modes,
            minimum_fps=25.0,
            preferred_format="MJPG",
            preferred_width=640,
            preferred_height=480,
            mode_strategy="max_fps",
        )

        self.assertEqual((selected.width, selected.height, selected.max_fps), (800, 600, 60.0))

    def test_max_fps_strategy_prefers_smaller_resolution_on_fps_tie(self) -> None:
        selected = select_camera_mode(
            [
                CameraMode("MJPG", "Motion-JPEG", 640, 480, (30.0,)),
                CameraMode("MJPG", "Motion-JPEG", 1920, 1080, (30.0,)),
            ],
            minimum_fps=25.0,
            preferred_format="MJPG",
            preferred_width=640,
            preferred_height=480,
            mode_strategy="max_fps",
        )

        self.assertEqual((selected.width, selected.height, selected.max_fps), (640, 480, 30.0))

    def test_max_resolution_strategy_uses_largest_mjpg_regardless_of_fps(self) -> None:
        selected = select_camera_mode(
            self.modes,
            minimum_fps=0.0,
            preferred_format="MJPG",
            preferred_width=640,
            preferred_height=480,
            mode_strategy="max_resolution",
        )

        self.assertEqual((selected.width, selected.height, selected.max_fps), (1280, 720, 10.0))


if __name__ == "__main__":
    unittest.main()
