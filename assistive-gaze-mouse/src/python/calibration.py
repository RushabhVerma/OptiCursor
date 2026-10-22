"""
calibration.py — Guided face-scan and personal baseline calibration (Step 2b)

What this does
──────────────
On first launch, instead of silently accumulating frames in the background,
we run a structured onboarding sequence:

  Phase 0 — Face detection gate
      Wait until a stable face is visible and centred.  The overlay shows a
      target oval the user should align their face inside.

  Phase 1 — Neutral scan  (NEUTRAL_FRAMES frames)
      "Keep your eyes open and look straight at the camera."
      We accumulate EAR readings for both eyes and compute the personal
      open-eye baseline (EAR_open_L, EAR_open_R).

  Phase 2 — Blink check   (BLINK_FRAMES frames)
      "Now blink naturally a few times."
      We accumulate the minimum EAR observed while blinking to verify the
      user's eyes actually close fully.  This gives us the closed-eye floor
      and a robust wink threshold = open_baseline × CLOSE_RATIO.

  Phase 3 — Wink test  (interactive)
      "Wink your LEFT eye … now your RIGHT eye."
      We confirm that both winks fire before saving the profile.  This step
      has a WINK_TEST_TIMEOUT_S; if it times out we skip to saving anyway.

Output
──────
A CalibrationProfile dataclass that main.py passes straight into GestureState.
The profile is also saved to PROFILE_PATH (JSON) so the calibration can be
skipped on subsequent runs.

Overlay design
──────────────
Full-frame dark overlay (alpha-blended), white text, coloured progress bar.
The face-alignment oval turns green when the face is centred.

All tuneable constants are at the top of this file.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import cv2
import numpy as np

from gestures import compute_ear
from quality import CalibrationQuality, assess_calibration_quality
from tracker import draw_eye_precision_points

# ── Constants ─────────────────────────────────────────────────────────────────

# Precision calibration samples.  At 30 fps this keeps the neutral scan open
# for about six seconds and the blink scan for about four seconds, making a
# single reflection from glasses or a brief gaze shift much less influential.
NEUTRAL_FRAMES: int   = 180
BLINK_FRAMES:   int   = 120

# Mirror the raw image before it reaches MediaPipe so overlay coordinates and
# the user-facing preview describe the same left/right directions.
MIRROR_CAMERA_FEED: bool = True

# Fraction of open baseline that counts as "closed"
CLOSE_RATIO:    float = 0.82  # tighter than the runtime default — more deliberate

# EAR open guard: wink fires only if the other eye is above this × its baseline
OPEN_GUARD_RATIO: float = 0.88

# Countdown seconds shown before each collection phase starts
COUNTDOWN_S: int = 3

# How long to wait for each test wink before giving up
WINK_TEST_TIMEOUT_S: float = 8.0

# Minimum EAR below which we consider a blink genuine (sanity check)
MIN_BLINK_EAR: float = 0.15

# Path where the profile is persisted (relative to repo root)
_REPO_ROOT   = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROFILE_PATH = os.path.join(_REPO_ROOT, "calibration_profile.json")

# Overlay colours (BGR)
_COL_BG       = (20,  20,  20)
_COL_WHITE    = (255, 255, 255)
_COL_GREEN    = (60,  220,  60)
_COL_ORANGE   = (0,   165, 255)
_COL_RED      = (0,    60, 230)
_COL_CYAN     = (255, 220,   0)
_COL_PROGRESS = (80,  200,  80)

# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class CalibrationProfile:
    """
    Personal open-eye baselines and derived thresholds for one user.

    Fields
    ──────
    left_baseline   : mean EAR of the left  eye while open and relaxed
    right_baseline  : mean EAR of the right eye while open and relaxed
    left_threshold  : EAR below which the left  eye is "closed"
                      = left_baseline  × CLOSE_RATIO
    right_threshold : EAR below which the right eye is "closed"
                      = right_baseline × CLOSE_RATIO
    left_open_guard : left  eye EAR that confirms the right eye is truly open
    right_open_guard: right eye EAR that confirms the left  eye is truly open
    min_blink_ear   : lowest EAR seen during the blink phase (sanity check)
    """
    left_baseline:    float
    right_baseline:   float
    left_threshold:   float
    right_threshold:  float
    left_open_guard:  float
    right_open_guard: float
    min_blink_ear:    float

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = PROFILE_PATH) -> None:
        """Persist profile to JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        print(f"[calibration] Profile saved → {path}")

    @classmethod
    def load(cls, path: str = PROFILE_PATH) -> "CalibrationProfile":
        """Load a previously saved profile.  Raises FileNotFoundError if absent."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)

    @classmethod
    def exists(cls, path: str = PROFILE_PATH) -> bool:
        return os.path.isfile(path)

    def summary(self) -> str:
        return (
            f"L baseline={self.left_baseline:.3f}  thr={self.left_threshold:.3f} | "
            f"R baseline={self.right_baseline:.3f}  thr={self.right_threshold:.3f}"
        )


# ── Internal overlay helpers ──────────────────────────────────────────────────

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX


def _overlay(frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Return a darkened copy of frame for overlaying text on."""
    dark = np.full_like(frame, _COL_BG)
    return cv2.addWeighted(frame, 1.0 - alpha, dark, alpha, 0)


def _camera_frame(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    """Read one camera frame and mirror it into the displayed coordinate system."""
    ret, bgr = cap.read()
    if not ret:
        return False, None
    if MIRROR_CAMERA_FEED:
        bgr = cv2.flip(bgr, 1)
    return True, bgr


def _robust_open_baseline(samples: list[float]) -> float:
    """Return the median EAR after removing the outer 10% of valid samples."""
    ordered = np.sort(np.asarray(samples, dtype=float))
    trim = max(1, int(len(ordered) * 0.10))
    core = ordered[trim:-trim] if len(ordered) > trim * 2 else ordered
    return float(np.median(core))


def _text_centred(
    frame: np.ndarray,
    text: str,
    y: int,
    scale: float = 0.9,
    color: tuple = _COL_WHITE,
    thickness: int = 2,
    font=_FONT,
) -> None:
    h, w = frame.shape[:2]
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(frame, text, ((w - tw) // 2, y), font, scale, color, thickness, cv2.LINE_AA)


def _progress_bar(
    frame: np.ndarray,
    fraction: float,
    y: int,
    color: tuple = _COL_PROGRESS,
) -> None:
    h, w = frame.shape[:2]
    bar_w = int(w * 0.6)
    x0 = (w - bar_w) // 2
    # Background
    cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + 18), (60, 60, 60), -1)
    # Fill
    filled = int(fraction * bar_w)
    if filled > 0:
        cv2.rectangle(frame, (x0, y), (x0 + filled, y + 18), color, -1)
    # Border
    cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + 18), _COL_WHITE, 1)


def _draw_quality_status(frame: np.ndarray, quality: CalibrationQuality, y: int) -> None:
    """Show whether the current frame is accepted and how to fix a rejection."""
    if quality.ready:
        _text_centred(frame, "QUALITY CHECK: GOOD", y, scale=0.60, color=_COL_GREEN)
        return
    message = " • ".join(quality.reasons[:2])
    _text_centred(frame, f"QUALITY CHECK: {message}", y, scale=0.55, color=_COL_ORANGE)


def _face_oval(
    frame: np.ndarray,
    centred: bool,
    nose_px: tuple[int, int] | None = None,
) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    color = _COL_GREEN if centred else _COL_ORANGE
    cv2.ellipse(frame, (cx, cy), (int(w * 0.18), int(h * 0.36)), 0, 0, 360, color, 2, cv2.LINE_AA)
    if nose_px:
        cv2.circle(frame, nose_px, 5, color, -1)


def _is_face_centred(nose_px: tuple[int, int], frame_w: int, frame_h: int) -> bool:
    """True when the nose tip is within the central 30% of the frame."""
    nx, ny = nose_px
    margin_x = frame_w * 0.15
    margin_y = frame_h * 0.20
    return (
        abs(nx - frame_w / 2) < margin_x
        and abs(ny - frame_h / 2) < margin_y
    )


# ── Countdown helper ──────────────────────────────────────────────────────────

def _countdown(
    cap: cv2.VideoCapture,
    landmarker,
    seconds: int,
    message: str,
) -> None:
    """
    Block for `seconds` seconds while showing a live countdown overlay.
    Keeps reading frames so the preview stays alive.
    """
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        ret, bgr = _camera_frame(cap)
        if not ret or bgr is None:
            continue
        frame = _overlay(bgr)
        _text_centred(frame, message,      frame.shape[0] // 2 - 60, scale=0.85)
        _text_centred(frame, f"Starting in {int(remaining) + 1}…",
                      frame.shape[0] // 2, scale=1.4, color=_COL_CYAN, font=_FONT_BOLD)
        cv2.imshow("Assistive Gaze Mouse — Calibration", frame)
        cv2.waitKey(1)


# ── Phase runners ─────────────────────────────────────────────────────────────

def _phase_wait_face(
    cap: cv2.VideoCapture,
    landmarker,
    Vision,
    left_eye_indices: list[int],
    right_eye_indices: list[int],
    nose_index: int,
) -> None:
    """Block until a well-centred face is detected for 30 consecutive frames."""
    stable = 0
    NEED   = 30

    while stable < NEED:
        ret, bgr = _camera_frame(cap)
        if not ret or bgr is None:
            continue
        h, w = bgr.shape[:2]

        ts_ms  = int(time.monotonic() * 1000)
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = __import__("mediapipe").Image(
            image_format=__import__("mediapipe").ImageFormat.SRGB, data=rgb
        )
        result = landmarker.detect_for_video(mp_img, ts_ms)

        frame  = _overlay(bgr, alpha=0.45)
        centred = False
        quality = None
        nose_px = None

        if result.face_landmarks:
            lm     = result.face_landmarks[0]
            nose_px = (int(lm[nose_index].x * w), int(lm[nose_index].y * h))
            quality = assess_calibration_quality(bgr, lm)
            draw_eye_precision_points(frame, lm, w, h)
            centred = _is_face_centred(nose_px, w, h) and quality.ready
            if centred:
                stable += 1
            else:
                stable = 0

        _face_oval(frame, centred, nose_px)
        _text_centred(frame, "FACE SETUP", 60, scale=1.1, color=_COL_CYAN, font=_FONT_BOLD)
        _text_centred(frame, "Align your face inside the oval", h // 2 - 120, scale=0.8)
        if centred:
            _text_centred(frame, "Hold still…", h // 2 + 120, scale=0.85, color=_COL_GREEN)
        elif quality is not None:
            _draw_quality_status(frame, quality, h // 2 + 120)
        _progress_bar(frame, stable / NEED, h // 2 + 150)

        cv2.imshow("Assistive Gaze Mouse — Calibration", frame)
        cv2.waitKey(1)


def _phase_collect(
    cap: cv2.VideoCapture,
    landmarker,
    Vision,
    left_eye_indices: list[int],
    right_eye_indices: list[int],
    nose_index: int,
    n_frames: int,
    title: str,
    instruction: str,
    sub_instruction: str = "",
) -> tuple[list[float], list[float]]:
    """
    Collect `n_frames` valid EAR readings.
    Returns (left_ears, right_ears).
    """
    import mediapipe as mp

    left_ears:  list[float] = []
    right_ears: list[float] = []

    while len(left_ears) < n_frames:
        ret, bgr = _camera_frame(cap)
        if not ret or bgr is None:
            continue
        h, w = bgr.shape[:2]

        ts_ms  = int(time.monotonic() * 1000)
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        frame = _overlay(bgr, alpha=0.50)

        if result.face_landmarks:
            lm    = result.face_landmarks[0]
            quality = assess_calibration_quality(bgr, lm)
            draw_eye_precision_points(frame, lm, w, h)
            l_pts = [(lm[i].x * w, lm[i].y * h, lm[i].z) for i in left_eye_indices]
            r_pts = [(lm[i].x * w, lm[i].y * h, lm[i].z) for i in right_eye_indices]
            l_ear = compute_ear(l_pts)
            r_ear = compute_ear(r_pts)

            if quality.ready and l_ear > 0.05 and r_ear > 0.05:
                left_ears.append(l_ear)
                right_ears.append(r_ear)

            # Live EAR readout
            _text_centred(frame, f"L-EAR: {l_ear:.3f}   R-EAR: {r_ear:.3f}",
                          h - 50, scale=0.75, color=_COL_CYAN)
            _draw_quality_status(frame, quality, h - 78)

        fraction = len(left_ears) / n_frames
        _text_centred(frame, title,          70,  scale=1.1, color=_COL_CYAN, font=_FONT_BOLD)
        _text_centred(frame, instruction,    h // 2 - 40, scale=0.85)
        if sub_instruction:
            _text_centred(frame, sub_instruction, h // 2 + 10, scale=0.72, color=_COL_ORANGE)
        _progress_bar(frame, fraction, h // 2 + 60)
        _text_centred(frame, f"{len(left_ears)} / {n_frames} frames",
                      h // 2 + 100, scale=0.7, color=_COL_WHITE)

        cv2.imshow("Assistive Gaze Mouse — Calibration", frame)
        cv2.waitKey(1)

    return left_ears, right_ears


def _phase_wink_test(
    cap: cv2.VideoCapture,
    landmarker,
    Vision,
    left_eye_indices: list[int],
    right_eye_indices: list[int],
    nose_index: int,
    profile: "CalibrationProfile",
) -> None:
    """
    Interactive wink test.  User winks left then right; we confirm detection.
    Times out after WINK_TEST_TIMEOUT_S per eye if the user skips.
    """
    import mediapipe as mp

    for side, thr_attr, guard_attr, label in [
        ("LEFT",  "left_threshold",  "right_open_guard", "Wink your LEFT eye  →"),
        ("RIGHT", "right_threshold", "left_open_guard",  "← Wink your RIGHT eye"),
    ]:
        thr   = getattr(profile, thr_attr)
        guard = getattr(profile, guard_attr)
        detected  = False
        deadline  = time.monotonic() + WINK_TEST_TIMEOUT_S

        while not detected and time.monotonic() < deadline:
            ret, bgr = _camera_frame(cap)
            if not ret or bgr is None:
                continue
            h, w = bgr.shape[:2]

            ts_ms  = int(time.monotonic() * 1000)
            rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            frame = _overlay(bgr, alpha=0.50)

            if result.face_landmarks:
                lm    = result.face_landmarks[0]
                quality = assess_calibration_quality(bgr, lm)
                draw_eye_precision_points(frame, lm, w, h)
                l_pts = [(lm[i].x * w, lm[i].y * h, lm[i].z) for i in left_eye_indices]
                r_pts = [(lm[i].x * w, lm[i].y * h, lm[i].z) for i in right_eye_indices]
                l_ear = compute_ear(l_pts)
                r_ear = compute_ear(r_pts)

                if quality.ready and side == "LEFT" and l_ear < thr and r_ear > guard:
                    detected = True
                if quality.ready and side == "RIGHT" and r_ear < thr and l_ear > guard:
                    detected = True

                _text_centred(frame, f"L-EAR: {l_ear:.3f}   R-EAR: {r_ear:.3f}",
                              h - 50, scale=0.75, color=_COL_CYAN)
                _draw_quality_status(frame, quality, h - 78)

            remaining = max(0, deadline - time.monotonic())
            color = _COL_GREEN if detected else _COL_WHITE
            _text_centred(frame, "WINK TEST", 70, scale=1.1, color=_COL_CYAN, font=_FONT_BOLD)
            _text_centred(frame, label, h // 2 - 40, scale=0.9, color=color)
            _text_centred(frame, f"Timeout in {remaining:.1f}s", h // 2 + 30,
                          scale=0.65, color=_COL_ORANGE)

            if detected:
                _text_centred(frame, "✓  DETECTED!", h // 2 + 80,
                              scale=1.1, color=_COL_GREEN, font=_FONT_BOLD)

            cv2.imshow("Assistive Gaze Mouse — Calibration", frame)
            cv2.waitKey(1)

        status = "DETECTED" if detected else "SKIPPED (timeout)"
        print(f"[calibration] {side} wink test: {status}")


# ── Public entry point ────────────────────────────────────────────────────────

def run_calibration(
    cap: cv2.VideoCapture,
    landmarker,
    left_eye_indices: list[int],
    right_eye_indices: list[int],
    nose_index: int,
    force: bool = False,
) -> CalibrationProfile:
    """
    Run the full calibration sequence and return a :class:`CalibrationProfile`.

    If a saved profile already exists and ``force`` is False, the saved
    profile is loaded and calibration is skipped.

    Parameters
    ──────────
    cap               : already-opened cv2.VideoCapture
    landmarker        : MediaPipe FaceLandmarker (VIDEO mode, already created)
    left_eye_indices  : list of 6 landmark indices for the left  eye (EAR order)
    right_eye_indices : list of 6 landmark indices for the right eye (EAR order)
    nose_index        : landmark index for the nose tip (centring check)
    force             : re-run calibration even if a saved profile exists
    """
    Vision = __import__("mediapipe").tasks.vision

    # ── Fast path: reuse saved profile ───────────────────────────────────────
    if not force and CalibrationProfile.exists():
        profile = CalibrationProfile.load()
        print(f"[calibration] Loaded saved profile — {profile.summary()}")
        _show_loaded_screen(cap, profile)
        return profile

    print("[calibration] Starting guided calibration …")

    # ── Phase 0: wait for stable face ────────────────────────────────────────
    _phase_wait_face(cap, landmarker, Vision,
                     left_eye_indices, right_eye_indices, nose_index)

    # ── Phase 1: neutral scan ─────────────────────────────────────────────────
    _countdown(cap, landmarker, COUNTDOWN_S,
               "Keep your eyes OPEN and look straight at the camera")
    l_open, r_open = _phase_collect(
        cap, landmarker, Vision,
        left_eye_indices, right_eye_indices, nose_index,
        NEUTRAL_FRAMES,
        "PHASE 1 — NEUTRAL SCAN",
        "Keep both eyes open, look straight ahead",
        "This establishes your personal open-eye baseline",
    )

    left_baseline  = _robust_open_baseline(l_open)
    right_baseline = _robust_open_baseline(r_open)

    # ── Phase 2: blink scan ───────────────────────────────────────────────────
    _countdown(cap, landmarker, COUNTDOWN_S,
               "Blink naturally a few times")
    l_blink, r_blink = _phase_collect(
        cap, landmarker, Vision,
        left_eye_indices, right_eye_indices, nose_index,
        BLINK_FRAMES,
        "PHASE 2 — BLINK SCAN",
        "Blink naturally several times",
        "This measures how far your eyes close",
    )

    min_blink_ear = float(min(min(l_blink), min(r_blink)))

    # Derive thresholds from this user's geometry
    # Threshold = midpoint between open baseline and observed blink floor,
    # clamped to [CLOSE_RATIO × baseline, 0.95 × baseline].
    def _threshold(baseline: float, blink_floor: float) -> float:
        midpoint = (baseline + blink_floor) / 2.0
        lo = baseline * CLOSE_RATIO
        hi = baseline * 0.95
        return max(lo, min(hi, midpoint))

    l_blink_floor = float(np.percentile(l_blink, 10))   # 10th pct = typical closed
    r_blink_floor = float(np.percentile(r_blink, 10))

    left_threshold  = _threshold(left_baseline,  l_blink_floor)
    right_threshold = _threshold(right_baseline, r_blink_floor)
    left_open_guard  = left_baseline  * OPEN_GUARD_RATIO
    right_open_guard = right_baseline * OPEN_GUARD_RATIO

    profile = CalibrationProfile(
        left_baseline=left_baseline,
        right_baseline=right_baseline,
        left_threshold=left_threshold,
        right_threshold=right_threshold,
        left_open_guard=left_open_guard,
        right_open_guard=right_open_guard,
        min_blink_ear=min_blink_ear,
    )

    print(f"[calibration] {profile.summary()}")

    # ── Phase 3: wink test ────────────────────────────────────────────────────
    _phase_wink_test(cap, landmarker, Vision,
                     left_eye_indices, right_eye_indices, nose_index, profile)

    # ── Save and return ───────────────────────────────────────────────────────
    profile.save()
    _show_complete_screen(cap, profile)
    return profile


def _show_loaded_screen(cap: cv2.VideoCapture, profile: CalibrationProfile) -> None:
    """Show a 2-second 'profile loaded' confirmation screen."""
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        ret, bgr = _camera_frame(cap)
        if not ret or bgr is None:
            continue
        frame = _overlay(bgr, alpha=0.60)
        h = frame.shape[0]
        _text_centred(frame, "CALIBRATION PROFILE LOADED", h // 2 - 60,
                      scale=0.9, color=_COL_GREEN, font=_FONT_BOLD)
        _text_centred(frame, f"L thr={profile.left_threshold:.3f}   R thr={profile.right_threshold:.3f}",
                      h // 2, scale=0.75, color=_COL_CYAN)
        _text_centred(frame, "Starting tracker…", h // 2 + 50, scale=0.75)
        cv2.imshow("Assistive Gaze Mouse — Calibration", frame)
        cv2.waitKey(1)


def _show_complete_screen(cap: cv2.VideoCapture, profile: CalibrationProfile) -> None:
    """Show a 3-second 'calibration complete' screen before handing off."""
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        ret, bgr = _camera_frame(cap)
        if not ret or bgr is None:
            continue
        frame = _overlay(bgr, alpha=0.60)
        h = frame.shape[0]
        _text_centred(frame, "CALIBRATION COMPLETE  ✓", h // 2 - 80,
                      scale=1.0, color=_COL_GREEN, font=_FONT_BOLD)
        _text_centred(frame, f"L baseline={profile.left_baseline:.3f}  thr={profile.left_threshold:.3f}",
                      h // 2 - 20, scale=0.75, color=_COL_CYAN)
        _text_centred(frame, f"R baseline={profile.right_baseline:.3f}  thr={profile.right_threshold:.3f}",
                      h // 2 + 20, scale=0.75, color=_COL_CYAN)
        _text_centred(frame, "Profile saved. Starting tracker…",
                      h // 2 + 80, scale=0.75)
        cv2.imshow("Assistive Gaze Mouse — Calibration", frame)
        cv2.waitKey(1)
