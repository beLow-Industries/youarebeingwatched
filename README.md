# You Are Being Watched

Prototype software for an art installation that watches camera sources, selects a stream with detected people or dogs every 10 seconds, and crops the output around tracked subjects.

## Setup

This project uses `uv`.

```bash
UV_CACHE_DIR=.uv-cache uv sync
```

Run the no-camera smoke check:

```bash
UV_CACHE_DIR=.uv-cache uv run ybwatch --mock --headless --max-frames 120
```

Run the no-camera windowed prototype:

```bash
UV_CACHE_DIR=.uv-cache uv run ybwatch --mock --windowed
```

Run with the default config:

```bash
UV_CACHE_DIR=.uv-cache uv run ybwatch --config config/default.toml
```

Prefer a camera's fastest MJPG mode, or its largest MJPG mode:

```bash
UV_CACHE_DIR=.uv-cache uv run ybwatch --max-fps
UV_CACHE_DIR=.uv-cache uv run ybwatch --max-resolution
```

Run a camera or video source directly:

```bash
UV_CACHE_DIR=.uv-cache uv run ybwatch --source 0
UV_CACHE_DIR=.uv-cache uv run ybwatch --source 4
UV_CACHE_DIR=.uv-cache uv run ybwatch --source media/example.mp4
```

Info logging is enabled by default. Use `--log-level DEBUG` or `YBWATCH_LOG_LEVEL=DEBUG` for per-frame detail.
Press `q` in the OpenCV window to exit. Display output is fullscreen by default; use `--windowed` to force a window.

Inspect camera availability:

```bash
UV_CACHE_DIR=.uv-cache uv run ybwatch doctor
```

The first non-mock run downloads `yolo26n.pt` through Ultralytics if it is not already present.

## WSL Webcam Notes

Production is assumed to be a normal Linux host where webcams appear as `/dev/video*`.

For development under WSL, attach USB webcams from Windows before running the app. The usual flow is:

```powershell
usbipd list
usbipd bind --busid <busid>
usbipd attach --wsl --busid <busid>
```

Then confirm Linux can see the device:

```bash
ls /dev/video*
UV_CACHE_DIR=.uv-cache uv run ybwatch doctor
```

If `doctor` shows `/dev/video*` devices owned by `root:video` but `read=False` or `write=False`, add your WSL user to the video group and restart WSL:

```bash
sudo usermod -aG video "$USER"
```

Then from Windows:

```powershell
wsl.exe --shutdown
```

Avoid running the app with `sudo`: `uv` is usually installed in the user PATH, and root-owned `.venv` files are annoying to clean up.

The default camera capture settings prefer `640x480` MJPG, then automatically choose a camera mode that can provide at least `25 fps`. Use `--max-fps` to select the fastest MJPG mode above that threshold, preferring smaller frames when FPS ties, or `--max-resolution` to select the largest MJPG mode regardless of FPS. Adjust `capture_fps`, `capture_width`, or `capture_height` in `config/default.toml` if a camera needs a different target.

Live cameras are captured on a background worker. The app keeps only the newest frame, drops stale frames when processing falls behind, and reopens the camera if reads stall or repeatedly fail. High-risk camera modes selected by `--max-fps` or `--max-resolution` can fall back to safer MJPG modes during recovery; `--max-resolution` descends through lower resolutions progressively.

## Behavior

- Source discovery combines configured sources with an auto-scan of OpenCV camera indices.
- Live camera reads use latest-frame backpressure so YOLO/display work does not block the capture device.
- Every 10 seconds, the app runs one YOLO pass over current frames from all active sources.
- If one or more sources contain a person or dog, the next main output source is chosen randomly among them.
- If neither is detected, the current output source remains active; if there is no source yet, the first readable source is displayed without overlays.
- Detections use a default confidence threshold of `0.40`.
- During a person-selected output interval, YOLO runs on the selected source for `person` and `dog`.
- The image processor chooses detected people or dogs just after each 10-second source decision, then re-elects every second.
- With one subject, output is cropped to that detection rectangle, fit to display height, centered on a black display canvas, and shown without overlays.
- With two subjects, the display is split into vertical thirds and the subjects are shown in the center and right thirds; with three or more subjects, three randomly chosen subjects fill all thirds.
- If a tracked subject disappears, the processor immediately recomputes the arrangement from the visible people and dogs while keeping surviving subjects in their current thirds where possible; if none are visible, the original frame is shown.
- `--headless` is only for development smoke checks; installation output is the OpenCV window.
