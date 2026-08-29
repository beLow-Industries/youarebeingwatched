from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from typing import Literal

CameraModeStrategy = Literal["configured", "max_fps", "max_resolution"]


@dataclass(frozen=True)
class CameraMode:
    pixel_format: str
    description: str
    width: int
    height: int
    fps_values: tuple[float, ...]

    @property
    def max_fps(self) -> float:
        if not self.fps_values:
            return 0.0
        return max(self.fps_values)

    @property
    def area(self) -> int:
        return self.width * self.height

    def format_line(self) -> str:
        fps_text = ", ".join(f"{fps:g} fps" for fps in self.fps_values)
        suffix = f" @ {fps_text}" if fps_text else ""
        return f"{self.pixel_format} ({self.description}): {self.width}x{self.height}{suffix}"


def v4l2_available() -> bool:
    return shutil.which("v4l2-ctl") is not None


def list_camera_modes(device: Path) -> list[CameraMode]:
    result = run_v4l2(device, "--list-formats-ext", timeout=3.0)
    if result.returncode != 0:
        return []
    return parse_camera_modes(result.stdout)


def select_camera_mode(
    modes: list[CameraMode],
    *,
    minimum_fps: float,
    preferred_format: str | None,
    preferred_width: int | None,
    preferred_height: int | None,
    mode_strategy: CameraModeStrategy = "configured",
) -> CameraMode | None:
    candidates = [mode for mode in modes if mode.max_fps >= minimum_fps]
    if mode_strategy in ("max_fps", "max_resolution") and preferred_format is not None:
        candidates = [mode for mode in candidates if mode.pixel_format == preferred_format]
    if not candidates:
        return None

    if mode_strategy == "max_fps":
        return max(candidates, key=lambda mode: (mode.max_fps, -mode.area))
    if mode_strategy == "max_resolution":
        return max(candidates, key=lambda mode: (mode.area, mode.max_fps))

    def score(mode: CameraMode) -> tuple[int, int, int, float]:
        format_match = int(preferred_format is not None and mode.pixel_format == preferred_format)
        resolution_match = int(
            preferred_width is not None
            and preferred_height is not None
            and mode.width == preferred_width
            and mode.height == preferred_height
        )
        if preferred_width is not None and preferred_height is not None:
            resolution_distance = abs(mode.width - preferred_width) + abs(mode.height - preferred_height)
        else:
            resolution_distance = -mode.area
        return format_match, resolution_match, -resolution_distance, mode.max_fps

    return max(candidates, key=score)


def parse_camera_modes(output: str) -> list[CameraMode]:
    modes: list[CameraMode] = []
    current_format = ""
    current_description = ""
    current_width = 0
    current_height = 0
    fps_values: list[float] = []

    for raw_line in output.splitlines():
        line = raw_line.strip()
        format_match = re.match(r"\[\d+\]:\s+'([^']+)'\s+\((.*)\)", line)
        if format_match:
            _append_mode(modes, current_format, current_description, current_width, current_height, fps_values)
            current_format = format_match.group(1)
            current_description = format_match.group(2)
            current_width = 0
            current_height = 0
            fps_values = []
            continue

        size_match = re.search(r"(\d+)x(\d+)", line) if line.startswith("Size:") else None
        if size_match:
            _append_mode(modes, current_format, current_description, current_width, current_height, fps_values)
            current_width = int(size_match.group(1))
            current_height = int(size_match.group(2))
            fps_values = []
            continue

        fps_match = re.search(r"\(([\d.]+)\s+fps\)", line) if line.startswith("Interval:") else None
        if fps_match:
            fps_values.append(float(fps_match.group(1)))

    _append_mode(modes, current_format, current_description, current_width, current_height, fps_values)
    return modes


def run_v4l2(device: Path, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["v4l2-ctl", f"--device={device}", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=exc.cmd,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )


def _append_mode(
    modes: list[CameraMode],
    pixel_format: str,
    description: str,
    width: int,
    height: int,
    fps_values: list[float],
) -> None:
    if not pixel_format or width <= 0 or height <= 0:
        return
    modes.append(
        CameraMode(
            pixel_format=pixel_format,
            description=description,
            width=width,
            height=height,
            fps_values=tuple(fps_values),
        )
    )
