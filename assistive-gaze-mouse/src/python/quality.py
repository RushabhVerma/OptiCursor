"""Face, image, and head-motion quality gates for reliable wink detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

# Image-quality bounds selected to reject common webcam failure modes without
# requiring studio lighting.
MIN_BRIGHTNESS: float = 55.0
MAX_BRIGHTNESS: float = 210.0
MIN_CONTRAST: float = 18.0
MIN_SHARPNESS: float = 35.0
MIN_EYE_SPAN_FRACTION: float = 0.12
MAX_EYE_WIDTH_RATIO: float = 1.35

# Head pitch proxy and re-anchoring behaviour.  The proxy is the vertical
# nose-to-eye distance divided by the inter-eye distance, so it is scale-free.
PITCH_CHANGE_LIMIT: float = 0.10
PITCH_SETTLE_DELTA: float = 0.012
PITCH_REANCHOR_FRAMES: int = 12

_NOSE_INDEX: int = 4
_LEFT_EYE_CORNERS: tuple[int, int] = (33, 133)
_RIGHT_EYE_CORNERS: tuple[int, int] = (263, 362)


@dataclass(frozen=True)
class CalibrationQuality:
    """Measured preconditions for accepting one calibration sample."""

    ready: bool
    brightness: float
    contrast: float
    sharpness: float
    eye_span_fraction: float
    eye_width_ratio: float
    reasons: tuple[str, ...]


@dataclass
class HeadMotionGuard:
    """Suppress gestures while the pitch proxy is changing, then re-anchor."""

    _reference_pitch: float | None = field(default=None, repr=False)
    _previous_pitch: float | None = field(default=None, repr=False)
    _settled_frames: int = field(default=0, repr=False)

    def update(self, landmarks: Sequence[object]) -> bool:
        """Return whether pitch is stable enough to interpret a wink this frame."""
        pitch = _pitch_proxy(landmarks)
        if pitch is None:
            return False
        if self._reference_pitch is None:
            self._reference_pitch = pitch
            self._previous_pitch = pitch
            return True

        deviation = abs(pitch - self._reference_pitch)
        frame_change = abs(pitch - self._previous_pitch) if self._previous_pitch is not None else 0.0
        self._previous_pitch = pitch
        if deviation <= PITCH_CHANGE_LIMIT:
            self._settled_frames = 0
            self._reference_pitch += 0.01 * (pitch - self._reference_pitch)
            return True

        if frame_change <= PITCH_SETTLE_DELTA:
            self._settled_frames += 1
            if self._settled_frames >= PITCH_REANCHOR_FRAMES:
                self._reference_pitch = pitch
                self._settled_frames = 0
                return True
        else:
            self._settled_frames = 0
        return False


def assess_calibration_quality(
    frame: np.ndarray,
    landmarks: Sequence[object],
) -> CalibrationQuality:
    """Assess whether a frame is suitable for accurate eye calibration."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    height, width = gray.shape

    left_width = _distance_x(landmarks, *_LEFT_EYE_CORNERS)
    right_width = _distance_x(landmarks, *_RIGHT_EYE_CORNERS)
    eye_span = abs(landmarks[_LEFT_EYE_CORNERS[0]].x - landmarks[_RIGHT_EYE_CORNERS[0]].x)
    eye_span_fraction = eye_span
    eye_width_ratio = max(left_width, right_width) / max(min(left_width, right_width), 1e-6)
    nose = landmarks[_NOSE_INDEX]

    reasons: list[str] = []
    if brightness < MIN_BRIGHTNESS:
        reasons.append("more front light needed")
    elif brightness > MAX_BRIGHTNESS:
        reasons.append("reduce glare or direct light")
    if contrast < MIN_CONTRAST:
        reasons.append("increase face lighting")
    if sharpness < MIN_SHARPNESS:
        reasons.append("hold still / clean camera lens")
    if eye_span_fraction < MIN_EYE_SPAN_FRACTION:
        reasons.append("move closer to the camera")
    if eye_width_ratio > MAX_EYE_WIDTH_RATIO:
        reasons.append("face the camera more directly")
    if abs(nose.x - 0.5) > 0.18 or abs(nose.y - 0.5) > 0.24:
        reasons.append("centre your face in the frame")

    return CalibrationQuality(
        ready=not reasons,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        eye_span_fraction=eye_span_fraction,
        eye_width_ratio=eye_width_ratio,
        reasons=tuple(reasons),
    )


def _distance_x(landmarks: Sequence[object], first: int, second: int) -> float:
    """Return horizontal normalized distance between two face landmarks."""
    return abs(landmarks[first].x - landmarks[second].x)


def _pitch_proxy(landmarks: Sequence[object]) -> float | None:
    """Compute scale-normalised vertical nose displacement relative to the eyes."""
    left = landmarks[_LEFT_EYE_CORNERS[0]]
    right = landmarks[_RIGHT_EYE_CORNERS[0]]
    eye_distance = abs(left.x - right.x)
    if eye_distance < 1e-6:
        return None
    eye_y = (left.y + right.y) / 2.0
    return (landmarks[_NOSE_INDEX].y - eye_y) / eye_distance
