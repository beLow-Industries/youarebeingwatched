from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

CameraModeStrategy = Literal["configured", "max_fps", "max_resolution"]
OVERLAY_CLASSES = ("person", "dog")
DEFAULT_SCAN_INDICES = tuple(range(10))


@dataclass(frozen=True)
class ModelConfig:
    weights: str = "yolo26n.pt"
    segmentation_weights: str = "yolo26n-seg.pt"
    confidence: float = 0.40


@dataclass(frozen=True)
class SelectionConfig:
    interval_seconds: float = 10.0


@dataclass(frozen=True)
class TrackingConfig:
    missing_linger_seconds: float = 0.5
    reselect_interval_seconds: float = 1.0
    stabilize_box_pixels: float = 16.0


@dataclass(frozen=True)
class DisplayConfig:
    window_name: str = "you are being watched"
    fullscreen: bool = True
    max_fps: float = 30.0
    width: int = 1280
    height: int = 720
    output_res: tuple[int, int] | None = None


@dataclass(frozen=True)
class SourcesConfig:
    auto_discover: bool = True
    scan_indices: tuple[int, ...] = DEFAULT_SCAN_INDICES
    entries: tuple[str, ...] = ()
    capture_width: int | None = 640
    capture_height: int | None = 480
    capture_fps: float | None = 25.0
    capture_fourcc: str | None = "MJPG"
    mode_strategy: CameraModeStrategy = "configured"


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = ModelConfig()
    selection: SelectionConfig = SelectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    display: DisplayConfig = DisplayConfig()
    sources: SourcesConfig = SourcesConfig()


def with_source_overrides(config: AppConfig, sources: list[str] | None) -> AppConfig:
    if not sources:
        return config

    return replace(config, sources=replace(config.sources, auto_discover=False, entries=tuple(sources)))


def with_display_overrides(
    config: AppConfig,
    *,
    fullscreen: bool | None,
    scan_max: int | None,
    source_mode_strategy: CameraModeStrategy | None = None,
    output_res: tuple[int, int] | None = None,
) -> AppConfig:
    display = config.display
    sources = config.sources

    if fullscreen is not None:
        display = replace(display, fullscreen=fullscreen)

    if output_res is not None:
        display = replace(display, width=output_res[0], height=output_res[1], output_res=output_res)

    if scan_max is not None:
        sources = replace(sources, scan_indices=tuple(range(max(0, scan_max))))

    if source_mode_strategy is not None:
        sources = replace(sources, mode_strategy=source_mode_strategy)

    return replace(config, display=display, sources=sources)
