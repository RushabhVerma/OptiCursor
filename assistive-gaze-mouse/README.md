# Assistive Gaze Mouse

A Windows assistive-input prototype that combines MediaPipe facial tracking,
Python gesture recognition, and safe Windows mouse control.

## Current controls

| Action | Input |
|---|---|
| Cursor | Accelerated nose/head movement |
| Left click | Sustained left-eyebrow raise |
| Right click | Lip pucker |
| Scroll mode | Hold mouth wide open for 0.5 seconds |
| Scroll | In scroll mode, move head up/down |
| Safety pause | `P` in the preview window |
| Quit | `Q` in the preview window |

Natural blinks are vetoed from eyebrow clicking. Click gestures are also held
until head movement settles, preventing accidental input while repositioning.

## Run

Install the Python requirements, then start in safe preview mode:

```powershell
python src\python\main.py
```

To enable real Windows cursor movement, clicks, and scrolling:

```powershell
python src\python\main.py --enable-mouse
```

Native input always starts paused. Focus the preview window and press `P` to
arm it. Start on an empty desktop area or harmless document.

Optional flags:

```powershell
python src\python\main.py --mobile
python src\python\main.py --no-cal
```

`--mobile` uses a lighter 640×480 preview profile. `--no-cal` skips the
three-phase per-session facial calibration and falls back to default thresholds.

## Persistent settings

Edit [settings.json](settings.json) to adjust the camera index, cursor
precision/fast gains, fallback gesture thresholds, blink veto, and scroll-mode
timing. Changes are loaded on the next application start. Calibrated eyebrow
and pucker thresholds take precedence over the fallback values unless you run
with `--no-cal`.

## Architecture

```text
camera → MediaPipe Face Landmarker → landmarks + blendshapes
                                      ├─ nose → Kalman smoothing → cursor mapper
                                      └─ gestures → debounce/safety → Win32 input
```

- `src/python/main.py`: camera loop, calibration UI, HUD, adaptive cursor,
  and input safety state.
- `src/python/gestures.py`: eyebrow, pucker, mouth-open scroll mode, blink
  veto, head-stability guard, and debounce policy.
- `src/python/filter.py`: 2-D Kalman smoothing for the nose position.
- `src/python/tracker.py`: MediaPipe landmark extraction and diagnostics.
- `src/native/mouse_win32.py`: Windows `user32` adapter implemented with the
  Python standard library; no C++ compiler is required.

The optional `mouse_win32.cpp` / pybind11 extension is retained for future
native optimization but is not required by the current application.

## Setup guidance

Use the [onboarding guide](ONBOARDING.md) for calibration, lighting, glasses,
gesture technique, scroll mode, and troubleshooting.
