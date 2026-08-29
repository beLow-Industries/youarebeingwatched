from __future__ import annotations

import grp
import os
from pathlib import Path
import platform
import pwd
import stat
import tempfile

import cv2

from .config import SourcesConfig
from .sources import SourceSpec, configured_specs, probe_source
from .v4l2 import list_camera_modes, run_v4l2, select_camera_mode, v4l2_available


def run_doctor(scan_max: int) -> int:
    print(f"platform: {platform.platform()}")
    print(f"opencv: {cv2.__version__}")
    print(f"video devices: {_video_devices()}")
    _print_permissions()
    config = SourcesConfig(auto_discover=True, scan_indices=tuple(range(max(0, scan_max))))
    _print_v4l2_diagnostics(config)
    _print_probe_results(config)
    specs = configured_specs(config)
    if not specs:
        print("openable sources: none")
        print("wsl note: if devices exist but are not openable, check Linux group membership and USB attachment")
        print(
            "wsl note: if permissions are ok but v4l2 stream probes time out, "
            "the WSL USB/IP camera path is not delivering frame buffers"
        )
        return 1

    print("openable sources:")
    for spec in specs:
        print(f"  - {spec.name} ({spec.raw})")
    return 0


def _video_devices() -> str:
    devices = sorted(Path("/dev").glob("video*"))
    if not devices:
        return "none"
    return ", ".join(str(device) for device in devices)


def _print_permissions() -> None:
    devices = sorted(Path("/dev").glob("video*"))
    if not devices:
        return

    groups = {group.gr_gid: group.gr_name for group in grp.getgrall()}
    user_group_ids = set(os.getgroups())
    user_name = pwd.getpwuid(os.getuid()).pw_name
    missing_groups: set[str] = set()

    print(f"user: {user_name}")
    print("device permissions:")
    for device in devices:
        info = device.stat()
        group_name = groups.get(info.st_gid, str(info.st_gid))
        mode = stat.filemode(info.st_mode)
        can_read = os.access(device, os.R_OK)
        can_write = os.access(device, os.W_OK)
        membership = "in group" if info.st_gid in user_group_ids else "not in group"
        if info.st_gid not in user_group_ids:
            missing_groups.add(group_name)
        print(f"  - {device}: {mode} root:{group_name} read={can_read} write={can_write} user={membership}")

    if missing_groups:
        names = ", ".join(sorted(missing_groups))
        print(f"permission hint: add {user_name} to group(s) {names}, then restart the WSL session")


def _print_v4l2_diagnostics(config: SourcesConfig) -> None:
    devices = sorted(Path("/dev").glob("video*"))
    if not devices or not v4l2_available():
        return

    print("v4l2 diagnostics:")
    for device in devices:
        capabilities = _v4l2_lines(device, "--all", ("Device Caps", "Video Capture", "Metadata Capture", "Pixel Format"))
        resolutions = _v4l2_resolutions(device)
        if _has_video_capture_formats(device):
            stream_ok, stream_note = _v4l2_stream_probe(device, config)
        else:
            stream_ok, stream_note = False, "metadata/non-capture node"

        print(f"  - {device}: stream={stream_ok} {stream_note}")
        for line in capabilities[:8]:
            print(f"      {line}")
        if resolutions:
            print("      available resolutions:")
            for line in resolutions:
                print(f"        - {line}")


def _v4l2_lines(device: Path, command: str, interesting: tuple[str, ...]) -> list[str]:
    result = run_v4l2(device, command, timeout=3.0)
    if result.returncode != 0:
        text = (result.stderr or result.stdout).strip()
        return [text] if text else [f"{command} failed with exit {result.returncode}"]

    lines: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if any(marker in stripped for marker in interesting):
            lines.append(stripped)
    return lines


def _has_video_capture_formats(device: Path) -> bool:
    result = run_v4l2(device, "--list-formats-ext", timeout=3.0)
    return result.returncode == 0 and "Type: Video Capture" in result.stdout and "[" in result.stdout


def _v4l2_resolutions(device: Path) -> list[str]:
    modes = list_camera_modes(device)
    if modes:
        return [mode.format_line() for mode in modes]
    result = run_v4l2(device, "--list-formats-ext", timeout=3.0)
    text = (result.stderr or result.stdout).strip()
    return [text] if text else [f"list formats failed with exit {result.returncode}"]


def _v4l2_stream_probe(device: Path, config: SourcesConfig) -> tuple[bool, str]:
    stream_args: list[str] = []
    if config.capture_width is not None and config.capture_height is not None:
        stream_args.append(
            f"--set-fmt-video=width={config.capture_width},height={config.capture_height},"
            f"pixelformat={config.capture_fourcc or 'MJPG'}"
        )
    if config.capture_fps is not None:
        stream_args.append(f"--set-parm={config.capture_fps}")

    with tempfile.NamedTemporaryFile(prefix="ybwatch-", suffix=".raw") as frame_file:
        result = run_v4l2(
            device,
            *stream_args,
            "--stream-mmap",
            "--stream-count=1",
            f"--stream-to={frame_file.name}",
            timeout=3.0,
        )

        size = Path(frame_file.name).stat().st_size
        if result.returncode == 0 and size > 0:
            return True, f"captured {size} bytes"
        if result.returncode == 124:
            return False, "timeout waiting for frame buffers"
        text = (result.stderr or result.stdout).strip()
        return False, text or f"failed with exit {result.returncode}"

def _print_probe_results(config: SourcesConfig) -> None:
    print("camera probes:")
    for index in config.scan_indices:
        spec = SourceSpec(raw=str(index))
        opened, readable = probe_source(spec)
        selected_mode = None
        if isinstance(spec.capture_target, str) and spec.capture_target.startswith("/dev/video") and v4l2_available():
            modes = list_camera_modes(Path(spec.capture_target))
            if config.capture_fourcc is not None:
                modes = [mode for mode in modes if mode.pixel_format == config.capture_fourcc]
            selected_mode = select_camera_mode(
                modes,
                minimum_fps=config.capture_fps or 25.0,
                preferred_format=config.capture_fourcc,
                preferred_width=config.capture_width,
                preferred_height=config.capture_height,
            )
        mode_text = f" selected_mode={selected_mode.format_line()}" if selected_mode is not None else ""
        print(f"  - {spec.name} ({spec.capture_target}): opened={opened} frame={readable}{mode_text}")
