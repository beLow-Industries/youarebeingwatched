# You Are Being Watched

Prototype software for an art installation that watches camera sources, selects a stream with detected people or dogs every 10 seconds, and crops the output around tracked subjects.

## Setup

This project uses `uv`.

```bash
uv sync
```

Run the no-camera smoke check or the windowed prototype:

```bash
uv run ybwatch --mock --headless --max-frames 120
uv run ybwatch --mock --windowed
```

## Options

Behavioral options:

| Option | Description |
| --- | --- |
| `--do-not-track` | Show the full selected source frame with detection boxes instead of cropping to subjects. |
| `--segmentation` | Use YOLO segmentation masks to black out crop backgrounds. |
| `--show-box` | Draw detection boxes and labels. Labels use `human`/`doggo` and confidence percentages. |
| `--font-size SCALE` | Set detection label font scale (default: `0.65`). |
| `--margin PIXELS` | Add source pixels around each subject crop. |
| `--stabilize-box PIXELS` | Hold crop targets until a box edge moves this many source pixels (default: `16`). |
| `--threshold VALUE` | Set the detection confidence threshold from `0` to `1` (default: `0.4`). |
| `--fullscreen` / `--windowed` | Force fullscreen or windowed output. Fullscreen output fills the detected display size. |
| `--headless` | Run without opening an OpenCV window; useful for smoke checks. |

Capture and runtime options:

| Option | Description |
| --- | --- |
| `--source SOURCE` | Use a camera index or video/stream path. Repeat for multiple sources; disables auto-discovery. |
| `--max-fps` | Select the highest-FPS MJPG camera mode above the configured minimum FPS. |
| `--max-resolution` | Select the largest MJPG camera mode, regardless of FPS. |
| `--scan-max N` | Auto-discover camera indices from `0` through `N - 1`. |
| `--max-frames N` | Exit after showing `N` frames. |
| `--mock` | Use a generated source and synthetic detections. |
| `--log-level LEVEL` | Set logging verbosity (default: `INFO`). `YBWATCH_LOG_LEVEL` also sets the default. |

Examples:

```bash
uv run ybwatch --segmentation
uv run ybwatch --do-not-track --show-box
uv run ybwatch --margin 40 --source 0
uv run ybwatch --max-fps
uv run ybwatch --max-resolution
uv run ybwatch image --source photo.jpg
```

Info logging is enabled by default. Use `--log-level DEBUG` or `YBWATCH_LOG_LEVEL=DEBUG` for per-frame detail.
Press `q` in the OpenCV window to exit. Display output is fullscreen by default; use `--windowed` to force a window.

Process one JPG or PNG still image and save a 1920x1080 result as the first available `out-N.jpg`:

```bash
uv run ybwatch image --source photo.jpg --margin 40
```

Inspect camera availability:

```bash
uv run ybwatch doctor
```

The first non-mock detection run downloads `yolo26n.pt` through Ultralytics if it is not already present. Segmentation mode downloads `yolo26n-seg.pt` by default.

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
uv run ybwatch doctor
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

The default camera capture settings prefer `640x480` MJPG, then automatically choose a camera mode that can provide at least `25 fps`. Use `--max-fps` to select the fastest MJPG mode above that threshold, preferring smaller frames when FPS ties, or `--max-resolution` to select the largest MJPG mode regardless of FPS. Auto-discovery scans camera indices `0` through `9`; use `--scan-max` to change that range.

Live cameras are captured on a background worker. The app keeps only the newest frame, drops stale frames when processing falls behind, and reopens the camera if reads stall or repeatedly fail. High-risk camera modes selected by `--max-fps` or `--max-resolution` can fall back to safer MJPG modes during recovery; `--max-resolution` descends through lower resolutions progressively.

## Behavior

- Source discovery combines explicit `--source` values with an auto-scan of OpenCV camera indices.
- Live camera reads use latest-frame backpressure so YOLO/display work does not block the capture device.
- Every 10 seconds, the app runs one YOLO pass over current frames from all active sources.
- If one or more sources contain a person or dog, the next main output source is chosen randomly among them.
- If neither is detected, the current output source remains active; if there is no source yet, the first readable source is displayed without overlays.
- Detections use a default confidence threshold of `0.40`.
- Detection mode is the default. Pass `--segmentation` to use YOLO instance masks for the displayed crops.
- During a person-selected output interval, YOLO runs on the selected source for `person` and `dog`.
- Crop targets update only when a YOLO box edge deviates from the held crop box by `--stabilize-box` source pixels, then ease toward the new target.
- The image processor chooses detected people or dogs just after each 10-second source decision, then re-elects every second.
- With one subject, output is cropped to that detection rectangle, fit to display height, centered on a black display canvas, and shown without overlays. In segmentation mode, pixels outside the selected subject mask are blacked out inside the crop.
- With two subjects, the display is split into vertical thirds and the subjects are shown in the center and right thirds; with three or more subjects, three randomly chosen subjects fill all thirds.
- If a tracked subject disappears, the processor keeps cropping the current frame at its last known rectangle for 0.5 seconds to smooth brief detector dropouts. After that grace window, it recomputes the arrangement from the visible people and dogs while keeping surviving subjects in their current thirds where possible; if none are visible, the original frame is shown.
- `--headless` is only for development smoke checks; installation output is the OpenCV window.
