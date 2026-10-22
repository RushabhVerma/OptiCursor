"""Ergonomic facial gestures: eyebrow raise and lip pucker.

The preferred eyebrow signal is MediaPipe's expression classifier output,
which is trained to separate brow movement from eye movement and head pose.
Landmark geometry remains only as a fallback when blendshape output is absent.

Eyebrow raise measurement
─────────────────────────
The classifier score combines left outer-brow raise and inner-brow raise:

    eyebrow_score = max(browOuterUpLeft, browInnerUp)

At rest this score is near zero. A deliberate left eyebrow raise increases it.

The existing nose-motion settle gate blocks all click gestures while the user
is moving their head to position the cursor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import median
from typing import Mapping, TypeAlias

# ── Landmark indices ─────────────────────────────────────────────────────────

LEFT_BROW_OUTER_INDEX: int = 65
LEFT_BROW_INNER_INDEX: int = 105
RIGHT_BROW_OUTER_INDEX: int = 295
RIGHT_BROW_INNER_INDEX: int = 334
FOREHEAD_INDEX: int = 10        # top of face mask
MOUTH_LEFT_CORNER_INDEX: int = 61
MOUTH_RIGHT_CORNER_INDEX: int = 291
LEFT_EYE_OUTER_INDEX: int = 33
LEFT_EYE_INNER_INDEX: int = 133
RIGHT_EYE_OUTER_INDEX: int = 263
RIGHT_EYE_INNER_INDEX: int = 362

Point3D: TypeAlias = tuple[float, float, float]
LandmarkMap: TypeAlias = Mapping[int, Point3D]
BlendshapeMap: TypeAlias = Mapping[str, float]

BROW_OUTER_UP_LEFT: str = "browOuterUpLeft"
BROW_INNER_UP: str = "browInnerUp"
BROW_OUTER_LEFT_WEIGHT: float = 0.90
BROW_INNER_WEIGHT: float = 0.10
EYE_BLINK_LEFT: str = "eyeBlinkLeft"
EYE_BLINK_RIGHT: str = "eyeBlinkRight"
JAW_OPEN: str = "jawOpen"

# ── Configurable trigger policy (fallback defaults) ──────────────────────────
#
# Eyebrow score increases as the left eyebrow rises. Increase this threshold
# to make a left click harder/more deliberate.

EYEBROW_RAISE_THRESHOLD: float = 0.45
LIP_PUCKER_THRESHOLD: float = 0.92
GESTURE_HOLD_MS: float = 400.0
BLINK_VETO_THRESHOLD: float = 0.20
# A deliberate wide-open mouth is a reliable one-time toggle for two-way
# scroll mode. It does not conflict with the lip-pucker right-click gesture.
SCROLL_TOGGLE_THRESHOLD: float = 0.65
SCROLL_TOGGLE_HOLD_MS: float = 500.0
SCROLL_INTERVAL_MS: float = 160.0
# Vertical nose displacement from scroll-mode entry, normalized by eye span.
SCROLL_DEAD_ZONE: float = 0.25

# Gesture events are inhibited while the nose is moving quickly. Both values
# are normalized by inter-eye span, making the rule independent of camera size.
MAX_NOSE_SPEED: float = 0.35
HEAD_SETTLE_MS: float = 250.0
NOSE_TIP_INDEX: int = 4

# Number of frames to average the eyebrow metric over before comparing to the
# threshold. A blink causes a brief spike; averaging over several frames damps
# it below the threshold while a sustained deliberate raise stays elevated.
EYEBROW_SMOOTH_FRAMES: int = 5


# ── Calibration ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CalibrationResult:
    """Per-session thresholds derived from the user's own resting/active values.

    Attributes
    ──────────
    eyebrow_threshold : Midpoint between resting and raised eyebrow score.
                        Eyebrow raise fires when the metric *exceeds* this.
    lip_threshold     : Midpoint between resting and puckered mouth width.
                        Lip pucker fires when the metric *falls below* this.
    hold_ms           : Milliseconds the gesture must be sustained.
    eyebrow_rest      : Median resting eyebrow score (diagnostics).
    eyebrow_active    : Median active eyebrow score (diagnostics).
    lip_rest          : Median resting lip width metric (diagnostics).
    lip_active        : Median active lip width metric (diagnostics).
    """

    eyebrow_threshold: float
    lip_threshold: float
    hold_ms: float
    eyebrow_rest: float
    eyebrow_active: float
    lip_rest: float
    lip_active: float

    def describe(self) -> str:
        """Human-readable summary for the console."""
        return (
            f"  Eyebrow  rest={self.eyebrow_rest:.3f}  active={self.eyebrow_active:.3f}"
            f"  → threshold={self.eyebrow_threshold:.3f}\n"
            f"  Lip      rest={self.lip_rest:.3f}  active={self.lip_active:.3f}"
            f"  → threshold={self.lip_threshold:.3f}"
        )


def compute_thresholds(
    eyebrow_rest_samples: list[float],
    eyebrow_active_samples: list[float],
    lip_rest_samples: list[float],
    lip_active_samples: list[float],
    hold_ms: float = GESTURE_HOLD_MS,
    margin: float = 0.25,
) -> CalibrationResult:
    """Derive per-session gesture thresholds from collected metric samples.

    The threshold is placed between the median resting value and
    the mean active value, with an optional margin factor that shifts it toward
    the active end to reduce accidental triggers.

    Parameters
    ──────────
    eyebrow_rest_samples   : Eyebrow ratio during relaxed face.
    eyebrow_active_samples : Eyebrow ratio during deliberate raise.
    lip_rest_samples       : Lip width metric during relaxed face.
    lip_active_samples     : Lip width metric during deliberate pucker.
    hold_ms                : Milliseconds the gesture must be sustained.
    margin                 : 0.0 = pure midpoint; 0.15 = shift 15% toward active.

    Raises
    ──────
    ValueError : If any sample list is empty or active ≤ rest (eyebrow) /
                 active ≥ rest (lip), meaning no meaningful gesture separation.
    """
    if not (eyebrow_rest_samples and eyebrow_active_samples
            and lip_rest_samples and lip_active_samples):
        raise ValueError("All sample lists must be non-empty.")

    # Median rejects transient misdetections while a phase is being collected.
    e_rest   = float(median(eyebrow_rest_samples))
    e_active = float(median(eyebrow_active_samples))
    l_rest   = float(median(lip_rest_samples))
    l_active = float(median(lip_active_samples))

    # Eyebrow: active score must exceed rest score.
    if e_active <= e_rest:
        raise ValueError(
            f"Eyebrow active mean ({e_active:.3f}) is not greater than resting mean "
            f"({e_rest:.3f}). Make sure you raised your left eyebrow during calibration."
        )

    # Lip: active value must be lower than rest (pucker decreases mouth width)
    if l_active >= l_rest:
        raise ValueError(
            f"Lip active mean ({l_active:.3f}) is not less than resting mean "
            f"({l_rest:.3f}). Make sure you puckered your lips during calibration."
        )

    eyebrow_mid = (e_rest + e_active) / 2.0
    eyebrow_thr = eyebrow_mid + margin * (e_active - e_rest) / 2.0

    lip_mid = (l_rest + l_active) / 2.0
    lip_thr = lip_mid - margin * (l_rest - l_active) / 2.0

    return CalibrationResult(
        eyebrow_threshold=eyebrow_thr,
        lip_threshold=lip_thr,
        hold_ms=hold_ms,
        eyebrow_rest=e_rest,
        eyebrow_active=e_active,
        lip_rest=l_rest,
        lip_active=l_active,
    )


# ── Gesture result and state ─────────────────────────────────────────────────

@dataclass(frozen=True)
class GestureResult:
    """Normalized gesture metrics and debounced click events for one frame."""

    left_click: bool
    right_click: bool
    eyebrow_raise: float
    lip_width: float
    eyebrow_active: bool
    pucker_active: bool
    scroll_direction: int
    scroll_mode_enabled: bool
    scroll_mode_toggled: bool
    scroll_toggle_score: float
    scroll_toggle_active: bool
    motion_stable: bool
    blink_blocked: bool


@dataclass
class GestureState:
    """Debounce state; each gesture must release before it can fire again.

    If a CalibrationResult is supplied the module-level constants are ignored
    and the calibrated thresholds are used for every detect_gestures() call.
    """

    calibration: CalibrationResult | None = field(default=None)

    _eyebrow_started_at: float | None = field(default=None, repr=False)
    _pucker_started_at: float | None = field(default=None, repr=False)
    _eyebrow_armed: bool = field(default=True, repr=False)
    _pucker_armed: bool = field(default=True, repr=False)
    _scroll_toggle_started_at: float | None = field(default=None, repr=False)
    _scroll_toggle_armed: bool = field(default=True, repr=False)
    _scroll_anchor_nose: Point3D | None = field(default=None, repr=False)
    _next_scroll_at: float | None = field(default=None, repr=False)
    scroll_mode_enabled: bool = field(default=False, init=False)

    # Rolling buffer for eyebrow smoothing — damps blink spikes
    _eyebrow_history: list[float] = field(default_factory=list, repr=False)
    _last_nose: Point3D | None = field(default=None, repr=False)
    _last_nose_timestamp: float | None = field(default=None, repr=False)
    _nose_settled_since: float | None = field(default=None, repr=False)

    # ── resolved thresholds (set once at construction or calibration) ─────────
    eyebrow_threshold: float = field(init=False)
    lip_threshold: float = field(init=False)
    hold_ms: float = field(init=False)

    def __post_init__(self) -> None:
        self._apply_calibration()

    def apply_calibration(self, cal: CalibrationResult) -> None:
        """Replace thresholds with calibrated values and reset debounce state."""
        self.calibration = cal
        self._apply_calibration()
        self.reset()

    def _apply_calibration(self) -> None:
        if self.calibration is not None:
            self.eyebrow_threshold = self.calibration.eyebrow_threshold
            self.lip_threshold     = self.calibration.lip_threshold
            self.hold_ms           = self.calibration.hold_ms
        else:
            self.eyebrow_threshold = EYEBROW_RAISE_THRESHOLD
            self.lip_threshold     = LIP_PUCKER_THRESHOLD
            self.hold_ms           = GESTURE_HOLD_MS

    def smooth_eyebrow(self, raw: float) -> float:
        """Push raw value into the rolling buffer and return its median."""
        self._eyebrow_history.append(raw)
        if len(self._eyebrow_history) > EYEBROW_SMOOTH_FRAMES:
            self._eyebrow_history.pop(0)
        ordered = sorted(self._eyebrow_history)
        return ordered[len(ordered) // 2]

    def clear_eyebrow_history(self) -> None:
        """Discard blink-contaminated brow samples before the next gesture."""
        self._eyebrow_history = []

    def motion_is_settled(self, landmarks: LandmarkMap, now: float) -> bool:
        """Return True only after normalized nose movement stays below the limit."""
        nose = landmarks[NOSE_TIP_INDEX]
        if self._last_nose is None or self._last_nose_timestamp is None:
            self._last_nose = nose
            self._last_nose_timestamp = now
            self._nose_settled_since = now
            return False

        elapsed = now - self._last_nose_timestamp
        if elapsed <= 1e-6:
            return False
        span = _inter_eye_span(landmarks)
        dx = nose[0] - self._last_nose[0]
        dy = nose[1] - self._last_nose[1]
        speed = (dx * dx + dy * dy) ** 0.5 / span / elapsed
        self._last_nose = nose
        self._last_nose_timestamp = now

        if speed > MAX_NOSE_SPEED:
            self._nose_settled_since = None
            return False
        if self._nose_settled_since is None:
            self._nose_settled_since = now
            return False
        return (now - self._nose_settled_since) * 1000.0 >= HEAD_SETTLE_MS

    def reset(self) -> None:
        """Re-arm both gestures after face tracking is lost or paused."""
        self._eyebrow_started_at = None
        self._pucker_started_at  = None
        self._eyebrow_armed      = True
        self._pucker_armed       = True
        self._scroll_toggle_started_at = None
        self._scroll_toggle_armed = True
        self._scroll_anchor_nose = None
        self._next_scroll_at     = None
        self.scroll_mode_enabled = False
        self._eyebrow_history    = []
        self._last_nose          = None
        self._last_nose_timestamp = None
        self._nose_settled_since = None


# ── Public gesture API ───────────────────────────────────────────────────────

def eyebrow_raise_distance(
    landmarks: LandmarkMap,
    blendshapes: BlendshapeMap | None = None,
) -> float:
    """Return a left-eyebrow raise score; larger means a stronger raise.

    Measurement
    ───────────
    Preferred score is an 80/20 weighted combination of MediaPipe's
    left-specific ``browOuterUpLeft`` and shared ``browInnerUp`` expressions.
    The left-specific signal dominates, so raising the right eyebrow does not
    normally create a left-click candidate.

    If blendshapes are unavailable, return ``1 - left_gap / right_gap`` as a
    geometry fallback. The fallback is less accurate and should be used only
    for tracker-only diagnostics.
    """
    if blendshapes is not None:
        return (
            BROW_OUTER_LEFT_WEIGHT * blendshapes.get(BROW_OUTER_UP_LEFT, 0.0)
            + BROW_INNER_WEIGHT * blendshapes.get(BROW_INNER_UP, 0.0)
        )
    forehead_y = landmarks[FOREHEAD_INDEX][1]
    left_brow_y = (
        landmarks[LEFT_BROW_OUTER_INDEX][1] + landmarks[LEFT_BROW_INNER_INDEX][1]
    ) / 2.0
    right_brow_y = (
        landmarks[RIGHT_BROW_OUTER_INDEX][1] + landmarks[RIGHT_BROW_INNER_INDEX][1]
    ) / 2.0
    right_gap = right_brow_y - forehead_y
    if right_gap <= 1e-6:
        return 0.0
    return max(0.0, 1.0 - (left_brow_y - forehead_y) / right_gap)


def is_blinking(blendshapes: BlendshapeMap | None) -> bool:
    """Return True when either eye's blink score vetoes eyebrow input."""
    if blendshapes is None:
        return False
    return max(
        blendshapes.get(EYE_BLINK_LEFT, 0.0),
        blendshapes.get(EYE_BLINK_RIGHT, 0.0),
    ) >= BLINK_VETO_THRESHOLD


def lip_pucker_width(landmarks: LandmarkMap) -> float:
    """Return normalized horizontal mouth width; a pucker produces a smaller value."""
    left_x  = landmarks[MOUTH_LEFT_CORNER_INDEX][0]
    right_x = landmarks[MOUTH_RIGHT_CORNER_INDEX][0]
    return abs(right_x - left_x) / _inter_eye_span(landmarks)


def detect_gestures(
    landmarks: LandmarkMap,
    state: GestureState,
    now: float | None = None,
    blendshapes: BlendshapeMap | None = None,
) -> GestureResult:
    """Emit one click per sustained neutral-to-active facial gesture cycle.

    Thresholds are read from *state* so that calibrated sessions automatically
    use per-user values without any changes to call sites.
    """
    timestamp      = time.monotonic() if now is None else now
    motion_stable  = state.motion_is_settled(landmarks, timestamp)
    blink_blocked  = is_blinking(blendshapes)
    raw_eyebrow = eyebrow_raise_distance(landmarks, blendshapes)
    if blink_blocked:
        state.clear_eyebrow_history()
        eyebrow = 0.0
    else:
        eyebrow = state.smooth_eyebrow(raw_eyebrow)
    lip_w          = lip_pucker_width(landmarks)
    eyebrow_active = (
        motion_stable
        and not blink_blocked
        and eyebrow >= state.eyebrow_threshold
    )
    pucker_active  = motion_stable and lip_w   <= state.lip_threshold

    scroll_toggle_score = 0.0 if blendshapes is None else blendshapes.get(JAW_OPEN, 0.0)
    scroll_toggle_active = scroll_toggle_score >= SCROLL_TOGGLE_THRESHOLD
    scroll_mode_toggled = _toggle_scroll_mode(
        # Do not require the head-stability guard here: opening the mouth is a
        # deliberate mode-selection action and can move the head slightly.
        active=scroll_toggle_active,
        now=timestamp,
        nose=landmarks[NOSE_TIP_INDEX],
        state=state,
    )
    scroll_direction = _scroll_direction(landmarks, timestamp, state)
    if state.scroll_mode_enabled:
        # Scroll mode intentionally repurposes head movement, so never also
        # send click gestures while it is active.
        eyebrow_active = False
        pucker_active = False

    return GestureResult(
        left_click=_debounced_event(
            active=eyebrow_active and not state.scroll_mode_enabled,
            now=timestamp,
            state=state,
            started_attr="_eyebrow_started_at",
            armed_attr="_eyebrow_armed",
            hold_ms=state.hold_ms,
        ),
        right_click=_debounced_event(
            active=pucker_active and not state.scroll_mode_enabled,
            now=timestamp,
            state=state,
            started_attr="_pucker_started_at",
            armed_attr="_pucker_armed",
            hold_ms=state.hold_ms,
        ),
        eyebrow_raise=eyebrow,
        lip_width=lip_w,
        eyebrow_active=eyebrow_active,
        pucker_active=pucker_active,
        scroll_direction=scroll_direction,
        scroll_mode_enabled=state.scroll_mode_enabled,
        scroll_mode_toggled=scroll_mode_toggled,
        scroll_toggle_score=scroll_toggle_score,
        scroll_toggle_active=scroll_toggle_active,
        motion_stable=motion_stable,
        blink_blocked=blink_blocked,
    )


# ── Private helpers ──────────────────────────────────────────────────────────

def _inter_eye_span(landmarks: LandmarkMap) -> float:
    """Return horizontal eye-centre separation, rejecting degenerate geometry."""
    left_centre = (
        landmarks[LEFT_EYE_OUTER_INDEX][0] + landmarks[LEFT_EYE_INNER_INDEX][0]
    ) / 2.0
    right_centre = (
        landmarks[RIGHT_EYE_OUTER_INDEX][0] + landmarks[RIGHT_EYE_INNER_INDEX][0]
    ) / 2.0
    span = abs(right_centre - left_centre)
    if span <= 1e-6:
        raise ValueError("Inter-eye span is zero; cannot normalize gesture distances.")
    return span


def _debounced_event(
    active: bool,
    now: float,
    state: GestureState,
    started_attr: str,
    armed_attr: str,
    hold_ms: float,
) -> bool:
    """Return one event after a hold, then require release before re-arming."""
    if not active:
        setattr(state, started_attr, None)
        setattr(state, armed_attr, True)
        return False

    if not getattr(state, armed_attr):
        return False
    started_at = getattr(state, started_attr)
    if started_at is None:
        setattr(state, started_attr, now)
        return False
    if (now - started_at) * 1000.0 < hold_ms:
        return False

    setattr(state, armed_attr, False)
    setattr(state, started_attr, None)
    return True


def _toggle_scroll_mode(
    active: bool,
    now: float,
    nose: Point3D,
    state: GestureState,
) -> bool:
    """Toggle scroll mode after one sustained wide-open-mouth gesture and release."""
    if not active:
        state._scroll_toggle_started_at = None
        state._scroll_toggle_armed = True
        return False

    if not state._scroll_toggle_armed:
        return False
    if state._scroll_toggle_started_at is None:
        state._scroll_toggle_started_at = now
        return False
    if (now - state._scroll_toggle_started_at) * 1000.0 < SCROLL_TOGGLE_HOLD_MS:
        return False
    state._scroll_toggle_armed = False
    state._scroll_toggle_started_at = None
    state.scroll_mode_enabled = not state.scroll_mode_enabled
    state._scroll_anchor_nose = nose if state.scroll_mode_enabled else None
    state._next_scroll_at = None
    return True


def _scroll_direction(
    landmarks: LandmarkMap,
    now: float,
    state: GestureState,
) -> int:
    """Return -1/0/+1 for rate-limited up/down scrolling in scroll mode."""
    if not state.scroll_mode_enabled or state._scroll_anchor_nose is None:
        return 0
    nose_y = landmarks[NOSE_TIP_INDEX][1]
    offset = (nose_y - state._scroll_anchor_nose[1]) / _inter_eye_span(landmarks)
    if abs(offset) < SCROLL_DEAD_ZONE:
        return 0
    if state._next_scroll_at is not None and now < state._next_scroll_at:
        return 0
    state._next_scroll_at = now + SCROLL_INTERVAL_MS / 1000.0
    # Moving the nose down scrolls down; moving it up scrolls up.
    return 1 if offset > 0.0 else -1
