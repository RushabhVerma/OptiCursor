"""
main.py — Step 2 integration loop with startup calibration

Startup sequence
────────────────
  1. Open the camera and build the MediaPipe FaceLandmarker once.
  2. Run a 3-phase calibration wizard:
       Phase 1 — Rest your face (baseline)
       Phase 2 — Raise your LEFT eyebrow
       Phase 3 — Pucker your lips (kiss)
  3. Compute per-session thresholds from the collected samples.
  4. Enter the tracking loop using those thresholds.

Press 'q' at any time to quit.  Press 's' during calibration to skip a phase
and fall back to the built-in defaults for that gesture.

Native input
────────────
Pass ``--enable-mouse`` to enable the bundled Windows input adapter. For
safety, native input starts paused; press ``P`` in the preview window to arm
or pause it.
"""

from __future__ import annotations

import os
import sys
import time
from importlib import import_module
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import mediapipe as mp

sys.path.insert(0, os.path.dirname(__file__))

from filter   import KalmanFilter2D
import gestures as gesture_logic
from config import AppSettings, CursorSettings, GestureSettings, ScrollSettings, load_settings, save_settings
from ui import SettingsPanel
from gestures import (
    CalibrationResult,
    GestureState,
    compute_thresholds,
    detect_gestures,
    eyebrow_raise_distance,
    is_blinking,
    lip_pucker_width,
)
from tracker import (
    LandmarkFrame,
    LEFT_EYE, RIGHT_EYE, NOSE_TIP, MICRO_EXPRESSION_INDICES,
    _extract, _draw_overlay, _draw_connections,
)

# ── HUD constants ─────────────────────────────────────────────────────────────

_HUD_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_HUD_SCALE     = 0.55
_HUD_THICKNESS = 1
_COLOR_OK      = (0,   220,   0)
_COLOR_WARN    = (0,   165, 255)
_COLOR_CLICK   = (0,     0, 255)
_COLOR_TEXT    = (255, 255, 255)
_COLOR_SMOOTH  = (255, 255,   0)
_COLOR_RAW     = (100, 100, 100)
_COLOR_CAL_BG  = (30,   30,  30)   # dark overlay for calibration prompt
_COLOR_CAL_HI  = (0,   220,   0)   # green highlight during collection

MIRROR_CAMERA_FEED: bool = True
MOBILE_CAPTURE_WIDTH:  int = 640
MOBILE_CAPTURE_HEIGHT: int = 480
PAUSE_TOGGLE_KEY: int = ord("p")
SETTINGS_TOGGLE_KEY: int = ord("o")
SETTINGS_WINDOW: str = "Assistive Gaze Mouse — Settings"
# Trackpad-style cursor gains. Slow motion remains precise; fast movement
# reaches farther. Values are normalized against camera width/height per second.
CURSOR_PRECISION_GAIN: float = 3.2
CURSOR_FAST_GAIN: float = 6.0
CURSOR_SPEED_FOR_MAX_GAIN: float = 0.80


def _apply_runtime_settings(settings: AppSettings) -> None:
    """Apply user settings to live cursor and gesture policy values."""
    global CURSOR_PRECISION_GAIN, CURSOR_FAST_GAIN, CURSOR_SPEED_FOR_MAX_GAIN
    CURSOR_PRECISION_GAIN = settings.cursor.precision_gain
    CURSOR_FAST_GAIN = settings.cursor.fast_gain
    CURSOR_SPEED_FOR_MAX_GAIN = settings.cursor.speed_for_max_gain
    gesture_logic.EYEBROW_RAISE_THRESHOLD = settings.gestures.default_eyebrow_threshold
    gesture_logic.LIP_PUCKER_THRESHOLD = settings.gestures.default_lip_pucker_threshold
    gesture_logic.GESTURE_HOLD_MS = settings.gestures.click_hold_ms
    gesture_logic.BLINK_VETO_THRESHOLD = settings.gestures.blink_veto_threshold
    gesture_logic.SCROLL_TOGGLE_THRESHOLD = settings.scroll.open_mouth_threshold
    gesture_logic.SCROLL_TOGGLE_HOLD_MS = settings.scroll.open_mouth_hold_ms
    gesture_logic.SCROLL_INTERVAL_MS = settings.scroll.interval_ms
    gesture_logic.SCROLL_DEAD_ZONE = settings.scroll.dead_zone


def _map_to_screen_axis(
    camera_coordinate: float,
    camera_dimension: int,
    screen_dimension: int,
    gain: float,
) -> int:
    """Map one camera axis around centre with gain, clipping at screen edges."""
    if camera_dimension <= 0 or screen_dimension <= 0:
        raise ValueError("Camera and screen dimensions must be positive.")
    if gain <= 0.0:
        raise ValueError("Cursor gain must be positive.")
    centered_ratio = camera_coordinate / camera_dimension - 0.5
    amplified_ratio = 0.5 + centered_ratio * gain
    bounded_ratio = min(max(amplified_ratio, 0.0), 1.0)
    return round(bounded_ratio * (screen_dimension - 1))


@dataclass
class AdaptiveCursorMapper:
    """Relative nose-to-cursor mapper with trackpad-style speed acceleration."""

    camera_width: int
    camera_height: int
    screen_width: int
    screen_height: int
    _last_camera: tuple[float, float] | None = None
    _last_timestamp: float | None = None
    _screen_position: tuple[float, float] | None = None

    def arm(self, camera_x: float, camera_y: float, timestamp: float) -> tuple[int, int]:
        """Centre the cursor and use the current nose position as the neutral point."""
        self._last_camera = (camera_x, camera_y)
        self._last_timestamp = timestamp
        self._screen_position = (
            (self.screen_width - 1) / 2.0,
            (self.screen_height - 1) / 2.0,
        )
        return round(self._screen_position[0]), round(self._screen_position[1])

    def rebase(self, camera_x: float, camera_y: float, timestamp: float) -> None:
        """Discard movement accumulated while cursor control was temporarily frozen."""
        self._last_camera = (camera_x, camera_y)
        self._last_timestamp = timestamp

    def update(self, camera_x: float, camera_y: float, timestamp: float) -> tuple[int, int]:
        """Convert nose movement into an accelerated but bounded screen position."""
        if self._last_camera is None or self._last_timestamp is None or self._screen_position is None:
            return self.arm(camera_x, camera_y, timestamp)
        elapsed = max(timestamp - self._last_timestamp, 1e-3)
        delta_x = camera_x - self._last_camera[0]
        delta_y = camera_y - self._last_camera[1]
        normalized_speed = (
            (delta_x / self.camera_width) ** 2 + (delta_y / self.camera_height) ** 2
        ) ** 0.5 / elapsed
        acceleration = min(normalized_speed / CURSOR_SPEED_FOR_MAX_GAIN, 1.0)
        gain = CURSOR_PRECISION_GAIN + acceleration * (
            CURSOR_FAST_GAIN - CURSOR_PRECISION_GAIN
        )
        next_x = self._screen_position[0] + delta_x / self.camera_width * self.screen_width * gain
        next_y = self._screen_position[1] + delta_y / self.camera_height * self.screen_height * gain
        self._screen_position = (
            min(max(next_x, 0.0), float(self.screen_width - 1)),
            min(max(next_y, 0.0), float(self.screen_height - 1)),
        )
        self._last_camera = (camera_x, camera_y)
        self._last_timestamp = timestamp
        return round(self._screen_position[0]), round(self._screen_position[1])


def map_to_screen_x(
    camera_x: float,
    camera_width: int,
    screen_width: int,
    gain: float = CURSOR_PRECISION_GAIN,
) -> int:
    """Map a mirrored camera x-coordinate to the screen with cursor gain."""
    if camera_width <= 0 or screen_width <= 0:
        raise ValueError("Camera and screen widths must be positive.")
    return _map_to_screen_axis(camera_x, camera_width, screen_width, gain)


def map_to_screen_y(
    camera_y: float,
    camera_height: int,
    screen_height: int,
    gain: float = CURSOR_PRECISION_GAIN,
) -> int:
    """Map a camera y-coordinate to the screen with cursor gain."""
    if camera_height <= 0 or screen_height <= 0:
        raise ValueError("Camera and screen heights must be positive.")
    return _map_to_screen_axis(camera_y, camera_height, screen_height, gain)


def map_relative_to_screen_axis(
    camera_coordinate: float,
    camera_anchor: float,
    screen_anchor: int,
    camera_dimension: int,
    screen_dimension: int,
    gain: float = CURSOR_PRECISION_GAIN,
) -> int:
    """Map displacement from an armed nose anchor to a bounded screen axis."""
    if camera_dimension <= 0 or screen_dimension <= 0:
        raise ValueError("Camera and screen dimensions must be positive.")
    if gain <= 0.0:
        raise ValueError("Cursor gain must be positive.")
    delta = (camera_coordinate - camera_anchor) / camera_dimension
    target = round(screen_anchor + delta * gain * (screen_dimension - 1))
    return min(max(target, 0), screen_dimension - 1)


def _load_mouse_controller(enabled: bool):
    """Load Windows input only when OS mouse control is explicitly requested."""
    if not enabled:
        return None
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    native_dir = os.path.join(repo_root, "src", "native")
    sys.path.insert(0, native_dir)
    sys.path.insert(0, repo_root)
    try:
        return import_module("mouse_win32")
    except ImportError as error:
        raise RuntimeError(
            "Native mouse control was requested but mouse_win32 is unavailable. "
            "This mode is supported only on Windows."
        ) from error


def _blendshape_scores(result) -> dict[str, float]:
    """Convert MediaPipe's first-face blendshape categories into score lookup."""
    if not result.face_blendshapes:
        return {}
    return {
        category.category_name: float(category.score)
        for category in result.face_blendshapes[0]
    }

# ── Calibration config ────────────────────────────────────────────────────────

# Seconds to show the instruction before collection starts.
CAL_LEAD_IN_SEC: float = 2.0
# Seconds of metric collection per phase.
CAL_COLLECT_SEC: float = 3.0
# Minimum samples required to accept a phase (roughly CAL_COLLECT_SEC × fps).
CAL_MIN_SAMPLES: int = 10


# ── Calibration wizard ────────────────────────────────────────────────────────

@dataclass
class _CalPhase:
    title: str          # short header line
    instruction: str    # what the user should do
    key: str            # "rest" | "eyebrow" | "lip"


_CAL_PHASES: list[_CalPhase] = [
    _CalPhase(
        title="Phase 1 / 3 — Resting face",
        instruction="Relax your face completely and look at the camera.",
        key="rest",
    ),
    _CalPhase(
        title="Phase 2 / 3 — Eyebrow raise  (LEFT CLICK gesture)",
        instruction="Raise your LEFT eyebrow as you would to trigger a click.",
        key="eyebrow",
    ),
    _CalPhase(
        title="Phase 3 / 3 — Lip pucker  (RIGHT CLICK gesture)",
        instruction="Pucker your lips (kiss shape) as you would to right-click.",
        key="lip",
    ),
]


def _draw_cal_overlay(
    frame,
    title: str,
    instruction: str,
    waiting: bool,
    collecting: bool,
    eyebrow_val: float | None,
    lip_val: float | None,
    sample_count: int,
    collect_progress: float = 0.0,   # 0.0–1.0 fill for progress bar
) -> None:
    """Render the calibration UI onto the frame in place."""
    h, w = frame.shape[:2]

    # Semi-transparent dark banner at top
    banner_h = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), _COLOR_CAL_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Title
    cv2.putText(frame, title, (14, 26),
                _HUD_FONT, 0.65, _COLOR_CAL_HI if collecting else _COLOR_WARN,
                1, cv2.LINE_AA)

    # Instruction
    cv2.putText(frame, instruction, (14, 52),
                _HUD_FONT, 0.52, _COLOR_TEXT, 1, cv2.LINE_AA)

    if collecting:
        msg = f"Collecting...  {sample_count} samples"
        cv2.putText(frame, msg, (14, 78),
                    _HUD_FONT, 0.52, _COLOR_CAL_HI, 1, cv2.LINE_AA)
        # Progress bar
        bar_x, bar_y, bar_w, bar_h = 14, 92, w - 28, 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        filled = int(collect_progress * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), _COLOR_CAL_HI, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), _COLOR_TEXT, 1)
    elif waiting:
        msg = "Press SPACE or ENTER to start   |   S = skip   |   Q = quit"
        cv2.putText(frame, msg, (14, 78),
                    _HUD_FONT, 0.50, (0, 220, 220), 1, cv2.LINE_AA)
    else:
        msg = "Get ready..."
        cv2.putText(frame, msg, (14, 78),
                    _HUD_FONT, 0.52, _COLOR_WARN, 1, cv2.LINE_AA)

    # Live metrics at bottom
    if eyebrow_val is not None:
        cv2.putText(frame, f"Eyebrow raise: {eyebrow_val:.3f}", (14, h - 36),
                    _HUD_FONT, _HUD_SCALE, _COLOR_TEXT, _HUD_THICKNESS, cv2.LINE_AA)
    if lip_val is not None:
        cv2.putText(frame, f"Lip width:     {lip_val:.3f}", (14, h - 16),
                    _HUD_FONT, _HUD_SCALE, _COLOR_TEXT, _HUD_THICKNESS, cv2.LINE_AA)


def run_calibration(
    landmarker,
    cap: cv2.VideoCapture,
    actual_w: int,
    actual_h: int,
    mobile: bool,
    connections,
) -> CalibrationResult | None:
    """Run the 3-phase calibration wizard and return computed thresholds.

    Returns None if the user presses 'q' to abort, or if calibration cannot
    produce valid thresholds (the tracking loop will use built-in defaults).
    """
    samples: dict[str, list[float]] = {
        "eyebrow_rest": [],
        "eyebrow_active": [],
        "lip_rest": [],
        "lip_active": [],
    }

    skipped: set[str] = set()

    for phase in _CAL_PHASES:
        # ── wait for user to press Space or Enter ─────────────────────────────
        phase_skipped = False

        while True:
            ret, bgr = cap.read()
            if not ret:
                continue
            if MIRROR_CAMERA_FEED:
                bgr = cv2.flip(bgr, 1)

            timestamp_ms = int(time.monotonic() * 1000)
            rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, timestamp_ms)

            eyebrow_val = lip_val = None
            if result.face_landmarks:
                coords = _extract(result.face_landmarks[0], actual_w, actual_h)
                if not mobile:
                    _draw_connections(bgr, result.face_landmarks[0], connections, actual_w, actual_h)
                _draw_overlay(bgr, coords)
                try:
                    eyebrow_val = eyebrow_raise_distance(coords, _blendshape_scores(result))
                    lip_val     = lip_pucker_width(coords)
                except ValueError:
                    pass

            _draw_cal_overlay(
                bgr, phase.title, phase.instruction,
                waiting=True, collecting=False,
                eyebrow_val=eyebrow_val, lip_val=lip_val, sample_count=0,
            )
            cv2.imshow("Assistive Gaze Mouse — Calibration", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("s"):
                phase_skipped = True
                skipped.add(phase.key)
                print(f"[cal] Skipped: {phase.title}")
                break
            if key in (ord(" "), 13):   # Space or Enter
                break

        if phase_skipped:
            continue

        # ── collection window ─────────────────────────────────────────────────
        col_start = time.monotonic()
        phase_samples: list[float] = []

        while True:
            elapsed = time.monotonic() - col_start
            if elapsed >= CAL_COLLECT_SEC:
                break

            ret, bgr = cap.read()
            if not ret:
                continue
            if MIRROR_CAMERA_FEED:
                bgr = cv2.flip(bgr, 1)

            timestamp_ms = int(time.monotonic() * 1000)
            rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, timestamp_ms)

            eyebrow_val = lip_val = None
            if result.face_landmarks:
                coords = _extract(result.face_landmarks[0], actual_w, actual_h)
                if not mobile:
                    _draw_connections(bgr, result.face_landmarks[0], connections, actual_w, actual_h)
                _draw_overlay(bgr, coords)
                try:
                    blendshapes = _blendshape_scores(result)
                    eyebrow_val = eyebrow_raise_distance(coords, blendshapes)
                    lip_val     = lip_pucker_width(coords)
                    if phase.key == "rest" and not is_blinking(blendshapes):
                        phase_samples.append(eyebrow_val)
                    elif phase.key == "eyebrow" and not is_blinking(blendshapes):
                        phase_samples.append(eyebrow_val)
                    elif phase.key == "lip":
                        phase_samples.append(lip_val)
                    if phase.key == "rest" and lip_val is not None and not is_blinking(blendshapes):
                        samples["lip_rest"].append(lip_val)
                except ValueError:
                    pass

            _draw_cal_overlay(
                bgr, phase.title, phase.instruction,
                waiting=False, collecting=True,
                eyebrow_val=eyebrow_val, lip_val=lip_val,
                sample_count=len(phase_samples),
                collect_progress=min(elapsed / CAL_COLLECT_SEC, 1.0),
            )
            cv2.imshow("Assistive Gaze Mouse — Calibration", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("s"):
                skipped.add(phase.key)
                print(f"[cal] Skipped during collection: {phase.title}")
                break

        # Store collected samples into the right bucket
        if phase.key == "rest":
            samples["eyebrow_rest"] = phase_samples
            # lip_rest was appended inline above
        elif phase.key == "eyebrow":
            samples["eyebrow_active"] = phase_samples
        elif phase.key == "lip":
            samples["lip_active"] = phase_samples

        print(f"[cal] {phase.title}: {len(phase_samples)} samples collected.")

    # ── compute thresholds ────────────────────────────────────────────────────
    # Fill in defaults for any skipped phase so compute_thresholds has data.
    # We mark these as synthetic so the user knows they're not calibrated.
    if "rest" in skipped or len(samples["eyebrow_rest"]) < CAL_MIN_SAMPLES:
        print("[cal] Rest phase skipped/insufficient — using defaults for eyebrow.")
        skipped.add("rest")
    if "eyebrow" in skipped or len(samples["eyebrow_active"]) < CAL_MIN_SAMPLES:
        print("[cal] Eyebrow phase skipped/insufficient — eyebrow threshold uses default.")
        skipped.add("eyebrow")
    if "lip" in skipped or len(samples["lip_active"]) < CAL_MIN_SAMPLES:
        print("[cal] Lip phase skipped/insufficient — lip threshold uses default.")
        skipped.add("lip")

    if skipped:
        # At least one phase was skipped; return None so the caller uses defaults.
        # A partial calibration could be misleading, so we prefer clean defaults.
        print("[cal] One or more phases incomplete. Using built-in defaults.")
        return None

    try:
        result = compute_thresholds(
            eyebrow_rest_samples=samples["eyebrow_rest"],
            eyebrow_active_samples=samples["eyebrow_active"],
            lip_rest_samples=samples["lip_rest"],
            lip_active_samples=samples["lip_active"],
        )
        print("[cal] Calibration complete.")
        print(result.describe())
        return result
    except ValueError as exc:
        print(f"[cal] Calibration failed: {exc}")
        print("[cal] Falling back to built-in defaults.")
        return None


# ── HUD helpers ───────────────────────────────────────────────────────────────

def _draw_gesture_hud(
    frame,
    raw_x: int,
    raw_y: int,
    smooth_x: float,
    smooth_y: float,
    eyebrow_raise: float,
    lip_width: float,
    left_click: bool,
    right_click: bool,
    eyebrow_active: bool,
    scroll_direction: int,
    scroll_mode_enabled: bool,
    scroll_toggle_score: float,
    scroll_toggle_active: bool,
    precision_gain: float,
    fast_gain: float,
    eyebrow_threshold: float,
    lip_threshold: float,
    motion_stable: bool,
    controls_paused: bool,
    blink_blocked: bool,
) -> None:
    """Draw eyebrow/pucker metrics, threshold indicators, and cursor markers."""
    h, w = frame.shape[:2]
    panel_right = min(w - 8, 500)
    panel = frame.copy()
    cv2.rectangle(panel, (6, 4), (panel_right, 158), (16, 16, 16), -1)
    cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)

    left_label  = "LEFT CLICK"  if left_click  else "Forehead-brow gap"
    right_label = "RIGHT CLICK" if right_click else "Lip pucker width"

    # Eyebrow bar: a larger MediaPipe brow-expression score means a stronger raise.
    bar_max_w = 120
    bar_h     = 14
    eyebrow_range = 1.0
    e_fraction = min(max(eyebrow_raise / eyebrow_range, 0.0), 1.0)
    e_filled  = int(e_fraction * bar_max_w)
    e_color   = _COLOR_CLICK if left_click else (_COLOR_CAL_HI if eyebrow_active else _COLOR_TEXT)
    cv2.rectangle(frame, (12, 10), (12 + bar_max_w, 10 + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (12, 10), (12 + e_filled,  10 + bar_h), e_color, -1)
    cv2.rectangle(frame, (12, 10), (12 + bar_max_w, 10 + bar_h), _COLOR_TEXT, 1)
    # Threshold tick
    e_tick = 12 + int(min(max(eyebrow_threshold / eyebrow_range, 0.0), 1.0) * bar_max_w)
    cv2.line(frame, (e_tick, 8), (e_tick, 26), _COLOR_WARN, 2)
    cv2.putText(frame, f"{left_label}: {eyebrow_raise:.3f}", (12 + bar_max_w + 6, 10 + bar_h - 2),
                _HUD_FONT, _HUD_SCALE, e_color, _HUD_THICKNESS, cv2.LINE_AA)

    # Lip bar (full = wide open, shrinks on pucker)
    l_filled = int(min(lip_width / max(lip_threshold * 1.2, 0.01), 1.0) * bar_max_w)
    l_color  = _COLOR_CLICK if right_click else (_COLOR_CAL_HI if lip_width <= lip_threshold else _COLOR_TEXT)
    cv2.rectangle(frame, (12, 34), (12 + bar_max_w, 34 + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (12, 34), (12 + l_filled,  34 + bar_h), l_color, -1)
    cv2.rectangle(frame, (12, 34), (12 + bar_max_w, 34 + bar_h), _COLOR_TEXT, 1)
    l_tick = 12 + int(min(lip_threshold / max(lip_threshold * 1.2, 0.01), 1.0) * bar_max_w)
    cv2.line(frame, (l_tick, 32), (l_tick, 50), _COLOR_WARN, 2)
    cv2.putText(frame, f"{right_label}: {lip_width:.3f}", (12 + bar_max_w + 6, 34 + bar_h - 2),
                _HUD_FONT, _HUD_SCALE, l_color, _HUD_THICKNESS, cv2.LINE_AA)
    motion_text = "HEAD STABLE" if motion_stable else "MOVE HEAD, THEN PAUSE"
    motion_color = _COLOR_CAL_HI if motion_stable else _COLOR_WARN
    cv2.putText(frame, motion_text, (12, 68), _HUD_FONT, _HUD_SCALE,
                motion_color, _HUD_THICKNESS, cv2.LINE_AA)
    if controls_paused:
        cv2.putText(frame, "GESTURES PAUSED — press P to arm", (12, 92),
                    _HUD_FONT, _HUD_SCALE, _COLOR_CLICK, _HUD_THICKNESS, cv2.LINE_AA)
    elif blink_blocked:
        cv2.putText(frame, "BLINK BLOCK — eyebrow click disabled", (12, 92),
                    _HUD_FONT, _HUD_SCALE, _COLOR_WARN, _HUD_THICKNESS, cv2.LINE_AA)
    elif scroll_mode_enabled:
        cv2.putText(frame, "SCROLL MODE — MOVE HEAD UP / DOWN", (12, 92),
                    _HUD_FONT, _HUD_SCALE, _COLOR_CAL_HI, _HUD_THICKNESS, cv2.LINE_AA)

    gain_label = f"CURSOR {precision_gain:.1f}–{fast_gain:.1f}x"
    cv2.putText(frame, gain_label, (12, 116),
                _HUD_FONT, _HUD_SCALE, _COLOR_TEXT, _HUD_THICKNESS, cv2.LINE_AA)
    toggle_color = _COLOR_CAL_HI if scroll_toggle_active else _COLOR_TEXT
    cv2.putText(
        frame,
        f"OPEN MOUTH {scroll_toggle_score:.2f} / {gesture_logic.SCROLL_TOGGLE_THRESHOLD:.2f}",
        (12, 140),
        _HUD_FONT,
        _HUD_SCALE,
        toggle_color,
        _HUD_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(frame, "P pause  |  O settings  |  Q quit", (12, 156),
                _HUD_FONT, 0.45, (190, 190, 190), 1, cv2.LINE_AA)

    # Cursor coordinates
    cv2.putText(frame, f"RAW    ({raw_x:4d}, {raw_y:4d})", (12, h - 36),
                _HUD_FONT, _HUD_SCALE, _COLOR_RAW, _HUD_THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, f"SMOOTH ({int(smooth_x):4d}, {int(smooth_y):4d})", (12, h - 16),
                _HUD_FONT, _HUD_SCALE, _COLOR_SMOOTH, _HUD_THICKNESS, cv2.LINE_AA)

    # Cursor dots
    cv2.circle(frame, (max(0, min(raw_x,         w - 1)), max(0, min(raw_y,         h - 1))), 4, _COLOR_RAW,    -1)
    cv2.circle(frame, (max(0, min(int(smooth_x), w - 1)), max(0, min(int(smooth_y), h - 1))), 6, _COLOR_SMOOTH,  2)


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    camera_index: int | None = None,
    mobile: bool = False,
    skip_cal: bool = False,
    enable_mouse: bool = False,
) -> None:
    """
    Full pipeline: calibration wizard → vision loop → gesture detection → HUD.

    Parameters
    ──────────
    camera_index : webcam index (0 = default camera)
    mobile       : use a 640×480 sparse-overlay profile for constrained CPUs
    skip_cal     : bypass calibration and use built-in default thresholds
    enable_mouse : load native input; starts paused until the user presses P
    """
    settings_path = Path(__file__).resolve().parents[2] / "settings.json"
    settings = load_settings(settings_path)
    _apply_runtime_settings(settings)
    active_camera_index = settings.camera_index if camera_index is None else camera_index
    print(f"[main] Loaded settings from {settings_path.name}.")

    Vision      = mp.tasks.vision
    BaseOptions = mp.tasks.BaseOptions

    model_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "face_landmarker.task")
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"FaceLandmarker model not found at: {model_path}\n"
            "Download with:\n"
            "  curl -L -o face_landmarker.task https://storage.googleapis.com/"
            "mediapipe-models/face_landmarker/face_landmarker/float16/latest/"
            "face_landmarker.task"
        )

    options = Vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=Vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(active_camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {active_camera_index}.")

    capture_w = MOBILE_CAPTURE_WIDTH  if mobile else 1280
    capture_h = MOBILE_CAPTURE_HEIGHT if mobile else 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  capture_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_h)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[main] Capture resolution: {actual_w}x{actual_h}")

    connections = Vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

    with Vision.FaceLandmarker.create_from_options(options) as landmarker:

        # ── Calibration phase ─────────────────────────────────────────────────
        calibration: CalibrationResult | None = None
        if not skip_cal:
            print("[main] Starting calibration wizard — press S to skip a phase, Q to abort.")
            calibration = run_calibration(
                landmarker, cap, actual_w, actual_h, mobile, connections
            )
            cv2.destroyAllWindows()

        gesture_state = GestureState(calibration=calibration)

        if calibration is None:
            print("[main] Using built-in default thresholds.")
        else:
            print(f"[main] Calibrated thresholds — "
                  f"eyebrow≥{calibration.eyebrow_threshold:.3f}  "
                  f"lip≤{calibration.lip_threshold:.3f}")

        kf = KalmanFilter2D()
        mouse_controller = _load_mouse_controller(enable_mouse)
        screen_size = mouse_controller.get_screen_size() if mouse_controller else None
        cursor_mapper = (
            AdaptiveCursorMapper(actual_w, actual_h, *screen_size)
            if screen_size is not None else None
        )
        latest_cursor: tuple[float, float] | None = None
        # Native mouse control always begins paused. Preview-only gesture tests
        # remain armed by default because they cannot alter the operating system.
        controls_paused = enable_mouse
        settings_panel_open = False
        settings_panel: SettingsPanel | None = None

        def update_from_panel(values: dict[str, float]) -> None:
            """Persist and apply a change made by the visual settings panel."""
            nonlocal settings
            settings = AppSettings(
                camera_index=settings.camera_index,
                cursor=CursorSettings(
                    values["precision"], values["fast"], settings.cursor.speed_for_max_gain,
                ),
                gestures=GestureSettings(
                    settings.gestures.default_eyebrow_threshold,
                    settings.gestures.default_lip_pucker_threshold,
                    values["click_hold"] * 1000.0,
                    settings.gestures.blink_veto_threshold,
                ),
                scroll=ScrollSettings(
                    values["mouth"],
                    settings.scroll.open_mouth_hold_ms,
                    values["scroll_interval"] * 1000.0,
                    settings.scroll.dead_zone,
                ),
            )
            _apply_runtime_settings(settings)
            save_settings(settings_path, settings)
        print("[main] Entering tracking loop — press 'p' to pause/arm controls, 'q' to quit.")
        if enable_mouse:
            print("[main] Native mouse control starts PAUSED. Press P in the preview to arm it.")

        # ── Main loop ─────────────────────────────────────────────────────────
        while True:
            ret, bgr = cap.read()
            if not ret:
                continue

            if MIRROR_CAMERA_FEED:
                bgr = cv2.flip(bgr, 1)

            timestamp_ms = int(time.monotonic() * 1000)
            rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, timestamp_ms)

            if result.face_landmarks:
                face_lm = result.face_landmarks[0]
                if not mobile:
                    _draw_connections(bgr, face_lm, connections, actual_w, actual_h)
                coords = _extract(face_lm, actual_w, actual_h)
                _draw_overlay(bgr, coords)

                lf = LandmarkFrame(
                    frame_w=actual_w,
                    frame_h=actual_h,
                    nose=coords[NOSE_TIP],
                    left_eye=[coords[i] for i in LEFT_EYE],
                    right_eye=[coords[i] for i in RIGHT_EYE],
                    micro_features={i: coords[i] for i in MICRO_EXPRESSION_INDICES},
                    frame=bgr,
                )

                # ── Kalman filter ─────────────────────────────────────────────
                raw_x, raw_y       = int(lf.nose[0]), int(lf.nose[1])
                smooth_x, smooth_y = kf.update(raw_x, raw_y)
                latest_cursor = (smooth_x, smooth_y)

                # ── Gesture detection ─────────────────────────────────────────
                gesture = detect_gestures(
                    coords,
                    gesture_state,
                    blendshapes=_blendshape_scores(result),
                )
                if controls_paused:
                    # Reset every paused frame so a held facial expression can
                    # never fire immediately after re-arming.
                    gesture_state.reset()
                    gesture = replace(
                        gesture,
                        left_click=False,
                        right_click=False,
                        eyebrow_active=False,
                        pucker_active=False,
                        scroll_direction=0,
                        scroll_mode_enabled=False,
                        scroll_mode_toggled=False,
                        scroll_toggle_score=0.0,
                        scroll_toggle_active=False,
                    )

                if (
                    mouse_controller
                    and not controls_paused
                    and cursor_mapper is not None
                ):
                    if gesture.scroll_mode_toggled:
                        cursor_mapper.rebase(smooth_x, smooth_y, timestamp_ms / 1000.0)
                    if not gesture.scroll_mode_enabled:
                        mouse_controller.mouse_move(
                            *cursor_mapper.update(smooth_x, smooth_y, timestamp_ms / 1000.0)
                        )
                    if gesture.left_click:
                        mouse_controller.mouse_left_click()
                    elif gesture.right_click:
                        mouse_controller.mouse_right_click()
                    if gesture.scroll_direction:
                        mouse_controller.mouse_scroll(gesture.scroll_direction)

                # ── HUD overlay ───────────────────────────────────────────────
                _draw_gesture_hud(
                    bgr,
                    raw_x, raw_y, smooth_x, smooth_y,
                    gesture.eyebrow_raise, gesture.lip_width,
                    gesture.left_click, gesture.right_click,
                    gesture.eyebrow_active,
                    gesture.scroll_direction,
                    gesture.scroll_mode_enabled,
                    gesture.scroll_toggle_score,
                    gesture.scroll_toggle_active,
                    settings.cursor.precision_gain,
                    settings.cursor.fast_gain,
                    gesture_state.eyebrow_threshold,
                    gesture_state.lip_threshold,
                    gesture.motion_stable,
                    controls_paused,
                    gesture.blink_blocked,
                )

                # ── Console log ───────────────────────────────────────────────
                if gesture.left_click:
                    print(f"\n[gesture] LEFT CLICK event — brow={gesture.eyebrow_raise:.3f}")
                elif gesture.right_click:
                    print(f"\n[gesture] RIGHT CLICK event — pucker={gesture.lip_width:.3f}")
                elif gesture.scroll_mode_toggled:
                    mode = "ON" if gesture.scroll_mode_enabled else "OFF"
                    print(f"\n[gesture] SCROLL MODE {mode} — open-mouth gesture")
                elif gesture.scroll_direction:
                    direction = "DOWN" if gesture.scroll_direction > 0 else "UP"
                    print(f"\n[gesture] SCROLL {direction} event")
                else:
                    print(f"BROW={gesture.eyebrow_raise:.3f}  PUCKER={gesture.lip_width:.3f}  "
                          f"SMOOTH=({int(smooth_x)},{int(smooth_y)})", end="\r")

            else:
                kf.reset()
                gesture_state.reset()
                cv2.putText(bgr, "No face detected", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("Assistive Gaze Mouse", bgr)
            if settings_panel_open:
                if settings_panel is not None:
                    cv2.imshow(SETTINGS_WINDOW, settings_panel.render())
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == PAUSE_TOGGLE_KEY and not settings_panel_open:
                controls_paused = not controls_paused
                state = "PAUSED" if controls_paused else "ARMED"
                if controls_paused:
                    cursor_mapper = (
                        AdaptiveCursorMapper(actual_w, actual_h, *screen_size)
                        if screen_size is not None else None
                    )
                elif mouse_controller and cursor_mapper is not None and latest_cursor is not None:
                    centered_x, centered_y = cursor_mapper.arm(
                        *latest_cursor, time.monotonic()
                    )
                    mouse_controller.mouse_move(centered_x, centered_y)
                print(f"\n[main] Gesture controls {state}.")
            if key == SETTINGS_TOGGLE_KEY:
                settings_panel_open = not settings_panel_open
                if settings_panel_open:
                    controls_paused = True
                    panel_values = {
                        "precision": settings.cursor.precision_gain,
                        "fast": settings.cursor.fast_gain,
                        "click_hold": settings.gestures.click_hold_ms / 1000.0,
                        "mouth": settings.scroll.open_mouth_threshold,
                        "scroll_interval": settings.scroll.interval_ms / 1000.0,
                    }
                    settings_panel = SettingsPanel(panel_values, update_from_panel)
                    cv2.namedWindow(SETTINGS_WINDOW, cv2.WINDOW_AUTOSIZE)
                    cv2.setMouseCallback(SETTINGS_WINDOW, settings_panel.mouse_event)
                    print("\n[settings] Open — controls paused; changes save automatically.")
                else:
                    cv2.destroyWindow(SETTINGS_WINDOW)
                    settings_panel = None
                    print("\n[settings] Closed — press P to arm controls.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(
        mobile=("--mobile" in sys.argv),
        skip_cal=("--no-cal" in sys.argv),
        enable_mouse=("--enable-mouse" in sys.argv),
    )

