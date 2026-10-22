# OptiCursor (Assistive Gaze Mouse)

A Windows assistive-input application that combines MediaPipe facial tracking, Python gesture recognition, and native Windows mouse control for hands-free computing.

## Current Controls

| Action | Input |
|---|---|
| **Cursor** | Accelerated nose/head movement with trackpad-style dynamics |
| **Left Click** | Sustained left-eyebrow raise |
| **Right Click** | Lip pucker / kiss shape |
| **Scroll Mode** | Hold mouth wide open for ~0.5 seconds |
| **Scroll Up/Down** | In scroll mode, move head/nose up or down |
| **Safety Pause** | `P` in the camera preview window |
| **Settings Panel** | `O` in the camera preview window |
| **Quit** | `Q` in the camera preview window |

> Natural blinks are vetoed from eyebrow clicking. Click gestures are held until head movement settles, preventing accidental input while repositioning the cursor.

---

## Quickstart & Running

1. **Navigate into the application folder**:
   ```powershell
   cd assistive-gaze-mouse
   ```

2. **Install requirements**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Start in preview mode (camera & HUD only, no OS input)**:
   ```powershell
   python src\python\main.py
   ```

4. **Enable real Windows mouse control**:
   ```powershell
   python src\python\main.py --enable-mouse
   ```

> **Safety Notice:** Native input always starts **paused** for safety. Focus the camera preview window and press `P` to arm it.

### Additional Command-Line Flags

```powershell
python src\python\main.py --mobile    # 640×480 lighter rendering profile
python src\python\main.py --no-cal    # Skip 3-step calibration and use fallback settings
```

---

## Configuration & Tuning

Edit [settings.json](assistive-gaze-mouse/settings.json) or press `O` in the camera window to open the interactive **Control Studio** panel to adjust:
- Camera index
- Cursor precision and fast acceleration gains
- Gesture hold duration and debounce times
- Scroll mode threshold and scroll wheel speed

Calibrated eyebrow and pucker thresholds take precedence over fallback values unless running with `--no-cal`.

---

## Architecture

```text
camera → MediaPipe Face Landmarker → landmarks + blendshapes
                                      ├─ nose → Kalman smoothing → cursor mapper
                                      └─ gestures → debounce/safety → Win32 input
```

- **`src/python/main.py`**: camera loop, 3-step calibration wizard, modern HUD, and input safety state.
- **`src/python/gestures.py`**: eyebrow raise, lip pucker, mouth-open scroll mode, blink veto, and head-motion guard.
- **`src/python/filter.py`**: 2-D Kalman smoothing filter for nose tracking.
- **`src/python/tracker.py`**: MediaPipe Face Landmarker feature extraction.
- **`src/python/ui.py`**: Interactive dark-mode Control Studio settings panel.
- **`src/native/mouse_win32.py`**: Native Windows `user32` standard library `ctypes` adapter (no C++ compiler needed).

---

## Setup & Ergonomics Guide

For detailed guidance on lighting, camera positioning, eyeglasses, gesture technique, and troubleshooting, read the [Onboarding Guide](assistive-gaze-mouse/ONBOARDING.md).
