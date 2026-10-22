# Assistive Gaze Mouse — Claude Continuation Context

> Last updated: 2026-10-23. This is the sole current handoff; do not rely on
> older README text, previous chat history, or wink-era modules.

## Product decision and current status

This is a Windows 11 / Python 3.13 assistive mouse prototype using OpenCV and
MediaPipe Face Landmarker. The project deliberately **abandoned wink clicks**:
they caused fatigue and false positives. Step 2 is complete and live-tested by
the user. Native Windows input, calibration, safety controls, scroll mode, a
settings system, and a custom visual UI are all implemented.

| Action | Current control |
|---|---|
| Cursor | Nose/head movement with trackpad-style acceleration |
| Left click | Sustained left-eyebrow raise |
| Right click | Lip pucker / kiss shape |
| Toggle scroll mode | Hold mouth wide open for about 0.5 seconds |
| Scroll up/down | In scroll mode, move the head/nose up/down |
| Arm/pause real input | `P` in the camera window |
| Open/close settings | `O` in the camera window |
| Quit | `Q` in the camera window |

## Run commands

```powershell
cd C:\Users\rusha\Documents\ai-project\assistive-gaze-mouse
python src\python\main.py --enable-mouse
```

Other supported modes:

```powershell
python src\python\main.py             # camera/HUD only; no OS input
python src\python\main.py --mobile    # 640×480 lighter rendering profile
python src\python\main.py --no-cal    # fallback threshold settings only
```

Real OS input always starts **paused**. The user must focus the camera window
and press `P` to arm it. Opening settings automatically pauses input; closing
settings leaves it paused, requiring another `P` to arm. Use a harmless desktop
area for live testing.

## Active pipeline and behavior

### Calibration

The default run has a three-phase per-session wizard in `main.py`:

1. Relaxed face
2. Raise left eyebrow
3. Lip pucker

It collects approximately three seconds of samples per phase, discards
blink-contaminated rest/eyebrow samples, and calls `compute_thresholds()`.
Those calibrated brow/pucker thresholds are used by `GestureState` during the
current run. `--no-cal` instead uses persisted fallback settings.

### Gesture safety and logic: `src/python/gestures.py`

- Eyebrow signal: weighted MediaPipe blendshapes `browOuterUpLeft` (0.90) and
  `browInnerUp` (0.10). It is much more reliable than prior landmark geometry.
- Natural blink veto: `eyeBlinkLeft` or `eyeBlinkRight` at/above the configured
  threshold disables eyebrow input and clears brow smoothing history. This
  prevents delayed clicks after a blink.
- Head-motion guard: eyebrow and pucker clicks require the nose to remain
  settled for `HEAD_SETTLE_MS` (250 ms). This prevents click events while the
  user is repositioning the cursor.
- Click debounce: expressions must be held for the configured duration, fire
  once, then release before rearming.
- Scroll mode uses MediaPipe `jawOpen`, **not** `tongueOut`. Tongue tracking was
  not detectable/reliable for the user, so it was intentionally replaced.
  While scroll mode is active, cursor and click gestures are disabled. Moving
  the nose up returns scroll direction `-1`; moving down returns `+1`.

Do not reintroduce wink logic or tongue mode without explicit user direction.

### Cursor: `src/python/main.py` + `filter.py`

- Nose landmark `4` is smoothed through `KalmanFilter2D`.
- `AdaptiveCursorMapper` is a relative, trackpad-like mapper. On arming, it
  centres the cursor and treats the user’s current nose position as neutral.
- Slow movement uses `precision_gain`; fast movement ramps to `fast_gain`.
  Defaults currently are 3.2× and 6.0×.
- Cursor motion is frozen during scroll mode. The mapper rebases after a scroll
  mode change so no cursor jump is produced on exit.

### Native Windows input: `src/native/mouse_win32.py`

This is the active input module. It uses Python standard-library `ctypes` to
call `user32`:

- `SetCursorPos` for absolute cursor movement
- `SendInput` for left/right clicks and scroll wheel events
- `get_screen_size` and `get_cursor_position` helpers

No Visual Studio Build Tools or compiled module is required. The older C++
pybind11 module `src/native/mouse_win32.cpp` and `setup.py` remain optional
future optimization only. Earlier Build Tools attempts failed because `cl.exe`
was unavailable; that is no longer a blocker.

## UI and settings

### Main camera dashboard

`main.py` renders a dark grouped HUD over the mirrored camera preview. It
shows click and pucker meters, head stability, blink block, scroll mode,
cursor gain, open-mouth score, raw/smoothed nose coordinates, and keyboard
shortcut hints.

### Control Studio: `src/python/ui.py`

Press `O` to open a custom OpenCV settings window called **Control Studio**.
It is not a default trackbar UI: it has a dark visual theme, rounded cards,
accent colors, direct drag sliders, and live values. It controls:

- Precision cursor speed
- Fast cursor speed
- Click hold time
- Open-mouth scroll-mode threshold
- Scroll interval (smaller interval = faster scrolling)

The panel uses `SettingsPanel` and does not require any external GUI library.
Slider interactions call back into `main.py`, apply immediately, and save to
disk. The user has not yet live-tested the redesigned Control Studio; only its
off-screen rendering and slider callback behavior have been programmatically
verified.

### Persistent configuration

`settings.json` is the user-editable, persisted source of settings. It is
validated by `src/python/config.py` on startup. `config.py` contains strict
dataclasses and `load_settings()` / `save_settings()`.

Key fields:

```json
{
  "camera_index": 0,
  "cursor": {
    "precision_gain": 3.2,
    "fast_gain": 6.0,
    "speed_for_max_gain": 0.8
  },
  "gestures": {
    "default_eyebrow_threshold": 0.45,
    "default_lip_pucker_threshold": 0.92,
    "click_hold_ms": 400.0,
    "blink_veto_threshold": 0.2
  },
  "scroll": {
    "open_mouth_threshold": 0.65,
    "open_mouth_hold_ms": 500.0,
    "interval_ms": 160.0,
    "dead_zone": 0.25
  }
}
```

`main.py` loads this file and applies values to the live cursor/gesture
constants. Calibrated eyebrow and pucker thresholds take precedence in normal
calibrated runs; the settings defaults take effect with `--no-cal`.

## Important files

| File | Status / purpose |
|---|---|
| `src/python/main.py` | Active camera loop, calibration, modern HUD, input safety, Control Studio integration |
| `src/python/gestures.py` | Active brow/pucker/mouth-open gestures, blink veto, head guard, debounce, scroll mode |
| `src/python/filter.py` | Active four-state Kalman nose smoother |
| `src/python/tracker.py` | Active dense landmark extraction and face overlay |
| `src/python/config.py` | Active typed JSON settings load/save/validation |
| `src/python/ui.py` | Active custom modern settings renderer and slider interaction |
| `src/native/mouse_win32.py` | Active ctypes Win32 input adapter |
| `settings.json` | Active persisted user configuration |
| `ONBOARDING.md` | Current user guide |
| `README.md` | Current project summary |
| `src/python/calibration.py`, `src/python/quality.py` | Legacy wink-era modules; do not reconnect |
| `calibration_profile.json` | Legacy wink profile; not used by active runtime |

## Verification completed

The following passed after the current changes:

```text
python -m py_compile src/python/main.py src/python/ui.py src/python/config.py
python -m py_compile src/python/gestures.py src/python/filter.py src/python/tracker.py
```

Additional deterministic checks passed:

- Native ctypes adapter imports and reads the screen size and cursor location.
- Centred adaptive cursor mapping passes precision/fast movement checks.
- Wide-open-mouth scroll toggle emits one toggle, then signed up/down scroll
  directions in synthetic landmark tests.
- Settings JSON parses and validates default values.
- Control Studio renders a 600×590 image and accepts synthetic slider drags.

User live-tested and confirmed that eyebrows/pucker/click controls, native
cursor input, cursor acceleration, calibration, mouth-open scroll mode, and
two-way scrolling work. The new visual Control Studio has not yet been
user-acceptance-tested.

## Recommended next work for Claude

1. Ask the user to live-test the Control Studio (`O`) and refine visual layout,
   slider ranges, and copy based on their feedback. Verify sliders actually
   feel useful in a live camera run.
2. Add an explicit **Reset to defaults** button to `SettingsPanel`, with a
   confirmation/safety flow appropriate for changing persistent settings.
3. Consider adding a large visible armed/paused badge and optional keyboard
   emergency-stop key beyond `P`/`Q`; preserve the current safe default pause.
4. Add automated unit tests for `gestures.py`, `config.py`, and cursor mapping
   rather than relying only on one-line manual Python assertions.
5. Decide whether to remove/archive old wink files and `calibration_profile.json`.
   Do not delete user files without user authorization.
6. Only if needed after profiling, revisit the optional C++ pybind11 extension.
   Do not make it a prerequisite—the ctypes module is currently sufficient.

## Constraints

- Preserve user files, especially `face_landmarker.task`.
- Keep type hints and docstrings in new Python code.
- Use `apply_patch` for source changes.
- This workspace is not a Git repository; do not rely on `git status`.
