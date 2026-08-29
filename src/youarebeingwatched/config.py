from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import tomllib

CameraModeStrategy = Literal["configured", "max_fps", "max_resolution"]


@dataclass(frozen=True)
class ModelConfig:
    weights: str = "yolo26n.pt"
    confidence: float = 0.40
    selection_class: str = "person"
    overlay_classes: tuple[str, ...] = ("person", "dog")


@dataclass(frozen=True)
class SelectionConfig:
    interval_seconds: float = 10.0


@dataclass(frozen=True)
class DisplayConfig:
    window_name: str = "you are being watched"
    fullscreen: bool = False
    max_fps: float = 30.0
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class SourcesConfig:
    auto_discover: bool = True
    scan_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
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
    display: DisplayConfig = DisplayConfig()
    sources: SourcesConfig = SourcesConfig()


def load_config(path: Path | None) -> AppConfig:
    if path is None or not path.exists():
        return AppConfig()

    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    return AppConfig(
        model=_load_model(raw.get("model", {})),
        selection=_load_selection(raw.get("selection", {})),
        display=_load_display(raw.get("display", {})),
        sources=_load_sources(raw.get("sources", {})),
    )


def _load_model(raw: dict[str, Any]) -> ModelConfig:
    defaults = ModelConfig()
    return ModelConfig(
        weights=str(raw.get("weights", defaults.weights)),
        confidence=float(raw.get("confidence", defaults.confidence)),
        selection_class=str(raw.get("selection_class", defaults.selection_class)),
        overlay_classes=tuple(str(item) for item in raw.get("overlay_classes", defaults.overlay_classes)),
    )


def _load_selection(raw: dict[str, Any]) -> SelectionConfig:
    defaults = SelectionConfig()
    return SelectionConfig(interval_seconds=float(raw.get("interval_seconds", defaults.interval_seconds)))


def _load_display(raw: dict[str, Any]) -> DisplayConfig:
    defaults = DisplayConfig()
    return DisplayConfig(
        window_name=str(raw.get("window_name", defaults.window_name)),
        fullscreen=bool(raw.get("fullscreen", defaults.fullscreen)),
        max_fps=float(raw.get("max_fps", defaults.max_fps)),
        width=int(raw.get("width", defaults.width)),
        height=int(raw.get("height", defaults.height)),
    )


def _load_sources(raw: dict[str, Any]) -> SourcesConfig:
    defaults = SourcesConfig()
    return SourcesConfig(
        auto_discover=bool(raw.get("auto_discover", defaults.auto_discover)),
        scan_indices=tuple(int(item) for item in raw.get("scan_indices", defaults.scan_indices)),
        entries=tuple(str(item) for item in raw.get("entries", defaults.entries)),
        capture_width=_optional_int(raw.get("capture_width", defaults.capture_width)),
        capture_height=_optional_int(raw.get("capture_height", defaults.capture_height)),
        capture_fps=_optional_float(raw.get("capture_fps", defaults.capture_fps)),
        capture_fourcc=_optional_str(raw.get("capture_fourcc", defaults.capture_fourcc)),
        mode_strategy=_mode_strategy(raw.get("mode_strategy", defaults.mode_strategy)),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def with_source_overrides(config: AppConfig, sources: list[str] | None) -> AppConfig:
    if not sources:
        return config

    return AppConfig(
        model=config.model,
        selection=config.selection,
        display=config.display,
        sources=SourcesConfig(
            auto_discover=False,
            scan_indices=config.sources.scan_indices,
            entries=tuple(sources),
            capture_width=config.sources.capture_width,
            capture_height=config.sources.capture_height,
            capture_fps=config.sources.capture_fps,
            capture_fourcc=config.sources.capture_fourcc,
            mode_strategy=config.sources.mode_strategy,
        ),
    )


def with_display_overrides(
    config: AppConfig,
    *,
    fullscreen: bool | None,
    scan_max: int | None,
    source_mode_strategy: CameraModeStrategy | None = None,
) -> AppConfig:
    display = config.display
    sources = config.sources

    if fullscreen is not None:
        display = DisplayConfig(
            window_name=display.window_name,
            fullscreen=fullscreen,
            max_fps=display.max_fps,
            width=display.width,
            height=display.height,
        )

    if scan_max is not None:
        sources = SourcesConfig(
            auto_discover=sources.auto_discover,
            scan_indices=tuple(range(max(0, scan_max))),
            entries=sources.entries,
            capture_width=sources.capture_width,
            capture_height=sources.capture_height,
            capture_fps=sources.capture_fps,
            capture_fourcc=sources.capture_fourcc,
            mode_strategy=sources.mode_strategy,
        )

    if source_mode_strategy is not None:
        sources = SourcesConfig(
            auto_discover=sources.auto_discover,
            scan_indices=sources.scan_indices,
            entries=sources.entries,
            capture_width=sources.capture_width,
            capture_height=sources.capture_height,
            capture_fps=sources.capture_fps,
            capture_fourcc=sources.capture_fourcc,
            mode_strategy=source_mode_strategy,
        )

    return AppConfig(model=config.model, selection=config.selection, display=display, sources=sources)


def _mode_strategy(value: Any) -> CameraModeStrategy:
    strategy = str(value)
    if strategy not in ("configured", "max_fps", "max_resolution"):
        raise ValueError(f"Unsupported camera mode strategy: {strategy!r}")
    return strategy
