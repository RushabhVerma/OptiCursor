# Assistive Gaze Mouse — User Guide

## What this build does

This Windows prototype lets you control a cursor with head/nose movement and
use comfortable facial gestures for clicks and scrolling:

| Control | Gesture |
|---|---|
| Move cursor | Move your head/nose |
| Left click | Raise your left eyebrow briefly |
| Right click | Pucker your lips briefly |
| Toggle scroll mode | Hold your mouth wide open for about half a second |
| Scroll up/down | In scroll mode, move your head up/down |
| Pause/arm input | Press `P` in the preview window |
| Quit | Press `Q` in the preview window |

## Start safely

Open PowerShell in the project folder and run:

```powershell
python src\python\main.py --enable-mouse
```

The preview opens and first asks for three short calibration phases: relaxed
face, left-eyebrow raise, and lip pucker. Follow the displayed prompt and
press Space or Enter to collect each phase. Press `S` only if you deliberately
want to use the built-in fallback threshold for that phase.

After calibration, the system starts **paused**. Place the cursor somewhere
safe, such as the desktop or an empty document, then focus the preview window
and press `P` to arm it. Press `P` again at any time to pause it immediately.

## Position and lighting

- Sit at your usual display distance, about 45–70 cm from the camera.
- Put the camera near eye level and centre your face in the preview.
- Use steady front-facing light. Avoid a bright window or lamp behind you.
- If you wear glasses, wear them while calibrating and using the system. Clean
  lenses and reduce strong reflections.
- Keep your head still during each calibration collection window. During
  normal operation, pause briefly after moving your head before attempting a
  click; the app intentionally blocks clicks until head motion settles.

## Cursor movement

When controls are armed, the cursor begins at screen centre. The movement is
trackpad-like: small, slow head motion is precise; quicker movement accelerates
to cover more screen area. The on-screen label `CURSOR 3.2–6.0x` describes the
current minimum and maximum gain.

If the cursor starts to feel uncomfortable, press `P`, return to a natural
posture, then press `P` again to recenter and re-arm.

## Click gestures

### Left click: left eyebrow

Raise your **left eyebrow** and hold it naturally for a short moment. Keep
your eyes open normally; blinking does not count as a click. The app includes
a blink veto and requires the head to be settled before sending a click.

### Right click: lip pucker

Make a deliberate kiss/pucker shape and release. A normal relaxed mouth should
not activate the gesture. This also requires the head to be settled.

## Two-way scroll mode

Open your mouth wide and hold it for about 0.5 seconds. The `OPEN MOUTH`
metric turns green when the mode-toggle gesture is being recognized. Once the
HUD shows `SCROLL MODE — MOVE HEAD UP / DOWN`:

- Move your head/nose upward to scroll up.
- Move your head/nose downward to scroll down.
- Keep your head near the mode-entry position to stop scrolling.
- Open your mouth wide again for about 0.5 seconds to exit scroll mode.

The cursor and click gestures are intentionally disabled while scrolling, so
head movement cannot accidentally click or move the pointer.

## HUD and troubleshooting

- `HEAD STABLE` means click gestures are allowed. `MOVE HEAD, THEN PAUSE`
  means wait briefly before clicking.
- `BLINK BLOCK` means a blink was detected and eyebrow clicking is disabled
  for that frame.
- `OPEN MOUTH current / 0.65` shows whether the scroll-mode toggle is being
  recognized. If it never approaches `0.65`, open your mouth wider, face the
  camera, and improve front lighting.
- The terminal prints every real left click, right click, scroll-mode change,
  and scroll event for confirmation.

For camera-only testing with no operating-system control, omit
`--enable-mouse`.

## Adjusting settings

The project-root `settings.json` is loaded every time the app starts. Most
users should begin by adjusting only `cursor.precision_gain` and
`cursor.fast_gain`: lower values require more head movement but are steadier;
higher values cover the display faster. Keep `fast_gain` at least as large as
`precision_gain`.

The `gestures` values are fallback settings used with `--no-cal`; ordinary
calibrated runs use your newly collected eyebrow and pucker thresholds instead.
The `scroll` section adjusts the wide-open-mouth sensitivity and scroll speed.
Make one small change at a time and restart the app to test it.
