"""
tracker.py — MediaPipe vision loop (Step 1, updated for MediaPipe 1.x)

MediaPipe 1.x replaced mp.solutions with mp.tasks.  This module uses
FaceLandmarker (VIDEO mode) via the new Tasks API.

The .task model file is expected at:
    <repo_root>/face_landmarker.task

Download once with:
    curl -L -o face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

Landmark indices (MediaPipe canonical 478-point model, same as before):
  Nose tip  : 4
  Left eye  : 33, 133, 159, 145, 158, 153
  Right eye : 362, 263, 386, 374, 385, 380
"""

from __future__ import annotations

import os
import time
import cv2
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Generator

# ── Landmark index constants ──────────────────────────────────────────────────

NOSE_TIP: int = 4

# EAR point order: [outer_corner, inner_corner, top_outer, bottom_outer,
#                   top_inner,    bottom_inner]
# Indices taken from FaceLandmarksConnections in MediaPipe 1.x.
# Note: MediaPipe 1.x labels are flipped vs 0.x — FACE_LANDMARKS_RIGHT_EYE
# is the user's left eye (camera-right) and vice versa.
LEFT_EYE:  list[int] = [33,  133, 159, 145, 158, 153]   # FACE_LANDMARKS_RIGHT_EYE
RIGHT_EYE: list[int] = [263, 362, 386, 374, 385, 380]   # FACE_LANDMARKS_LEFT_EYE

# Dense feature regions retained alongside the six-point EAR contours.  They
# cover brows, eyelids, irises, lips, and mouth corners—the regions most useful
# for later micro-expression classifiers without forcing callers to keep all
# 478 landmarks every frame.
MICRO_EXPRESSION_INDICES: frozenset[int] = frozenset([
    0, 10, 13, 14, 17, 37, 39, 40, 61, 65, 78, 80, 81, 82, 84, 87, 88, 91,
    95, 146, 152, 159, 145, 158, 153, 191, 263, 267, 269, 270, 291, 295, 308,
    310, 311, 312, 314, 317, 318, 321, 324, 334, 336, 374, 380, 385,
    386, 405, 409, 415, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
    105, 334,
])
# 10  = forehead top  — used as face-height anchor for eyebrow raise metric
# 152 = chin bottom   — used as face-height anchor for eyebrow raise metric

# Full eyelid contours plus both iris rings.  These points are rendered during
# calibration so users can immediately see whether glasses glare is obscuring
# an eye, and are retained for higher-resolution expression features.
LEFT_EYE_PRECISION_INDICES: tuple[int, ...] = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161,
    246, 468, 469, 470, 471, 472,
)
RIGHT_EYE_PRECISION_INDICES: tuple[int, ...] = (
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387,
    388, 466, 473, 474, 475, 476, 477,
)

TRACKED_INDICES: frozenset[int] = frozenset(
    [NOSE_TIP] + LEFT_EYE + RIGHT_EYE + list(MICRO_EXPRESSION_INDICES)
    + list(LEFT_EYE_PRECISION_INDICES) + list(RIGHT_EYE_PRECISION_INDICES)
)

# Private alias retained for internal callers written before the public name.
_TRACKED: frozenset[int] = TRACKED_INDICES

MOBILE_CAPTURE_WIDTH: int = 640
MOBILE_CAPTURE_HEIGHT: int = 480

# Keep MediaPipe input and OpenCV preview in one mirrored coordinate system.
MIRROR_CAMERA_FEED: bool = True

# Default model path — one level above src/python/
_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "..", "face_landmarker.task"
)

# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class LandmarkFrame:
    """
    One frame's worth of data passed downstream.

    All coordinates are in *pixel space* relative to the capture frame.
    `z` is left in MediaPipe's normalized form (negative = closer to camera).
    """
    frame_w: int
    frame_h: int

    # (x_px, y_px, z_norm).  Retain fractional pixels so eyelid motion is not
    # distorted by rounding the six points before EAR measurement.
    nose:      tuple[float, float, float]
    left_eye:  list[tuple[float, float, float]]   # 6 points, order = LEFT_EYE
    right_eye: list[tuple[float, float, float]]   # 6 points, order = RIGHT_EYE
    micro_features: dict[int, tuple[float, float, float]]

    # Raw BGR frame — kept for the preview window; downstream may ignore it.
    frame: object = field(repr=False)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract(face_landmarks, w: int, h: int) -> dict[int, tuple[float, float, float]]:
    """Convert normalized NormalizedLandmark list → pixel coords for our subset."""
    out: dict[int, tuple[float, float, float]] = {}
    for idx in _TRACKED:
        lm = face_landmarks[idx]
        out[idx] = (lm.x * w, lm.y * h, lm.z)
    return out


def _draw_overlay(frame, coords: dict[int, tuple[float, float, float]]) -> None:
    """Colored debug overlay: yellow = nose, green = left eye, blue = right eye."""
    if NOSE_TIP in coords:
        cv2.circle(frame, tuple(map(int, coords[NOSE_TIP][:2])), 5, (0, 255, 255), -1)
    for idx in LEFT_EYE:
        if idx in coords:
            cv2.circle(frame, tuple(map(int, coords[idx][:2])), 3, (0, 255, 0), -1)
    for idx in RIGHT_EYE:
        if idx in coords:
            cv2.circle(frame, tuple(map(int, coords[idx][:2])), 3, (255, 0, 0), -1)


def draw_eye_precision_points(frame, face_landmarks, width: int, height: int) -> None:
    """Draw dense eyelid and iris landmarks for calibration diagnostics."""
    for index in LEFT_EYE_PRECISION_INDICES:
        landmark = face_landmarks[index]
        point = (int(landmark.x * width), int(landmark.y * height))
        color = (0, 255, 255) if index >= 468 else (0, 210, 0)
        cv2.circle(frame, point, 2, color, -1)
    for index in RIGHT_EYE_PRECISION_INDICES:
        landmark = face_landmarks[index]
        point = (int(landmark.x * width), int(landmark.y * height))
        color = (0, 255, 255) if index >= 468 else (255, 120, 0)
        cv2.circle(frame, point, 2, color, -1)


def _draw_connections(frame, landmarks, connections, w, h, color=(80, 80, 80)) -> None:
    """Draw mesh edges between landmark pairs."""
    for conn in connections:
        a = landmarks[conn.start]
        b = landmarks[conn.end]
        pt1 = (int(a.x * w), int(a.y * h))
        pt2 = (int(b.x * w), int(b.y * h))
        cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)


# ── Public interface ──────────────────────────────────────────────────────────

def frame_stream(
    camera_index: int = 0,
    capture_w: int = 1280,
    capture_h: int = 720,
    show_preview: bool = True,
    detection_confidence: float = 0.6,
    tracking_confidence: float = 0.6,
    model_path: str = _DEFAULT_MODEL,
    mobile: bool = False,
) -> Generator[LandmarkFrame | None, None, None]:
    """
    Generator that yields one :class:`LandmarkFrame` per webcam frame.

    Yields ``None`` when no face is detected so the caller can decide
    whether to hold the last known state or skip.

    The generator runs until the capture device is lost or the user
    presses **q** in the preview window.
    """
    Vision      = mp.tasks.vision
    BaseOptions = mp.tasks.BaseOptions

    model_path = os.path.normpath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"FaceLandmarker model not found at: {model_path}\n"
            "Download it with:\n"
            "  curl -L -o face_landmarker.task https://storage.googleapis.com/"
            "mediapipe-models/face_landmarker/face_landmarker/float16/latest/"
            "face_landmarker.task"
        )

    options = Vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=Vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=detection_confidence,
        min_face_presence_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {camera_index}. "
            "Check that the device is connected and not in use."
        )

    if mobile:
        capture_w, capture_h = MOBILE_CAPTURE_WIDTH, MOBILE_CAPTURE_HEIGHT

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  capture_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_h)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[tracker] Capture resolution: {actual_w}x{actual_h}")

    # FaceLandmarksConnections for tessellation drawing
    connections = Vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

    with Vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0

        while True:
            ret, bgr = cap.read()
            if not ret:
                print("[tracker] Dropped frame — retrying.")
                continue

            if MIRROR_CAMERA_FEED:
                bgr = cv2.flip(bgr, 1)

            # VIDEO mode requires a monotonically increasing timestamp in ms
            timestamp_ms = int(time.monotonic() * 1000)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_idx += 1

            if result.face_landmarks:
                face_lm = result.face_landmarks[0]   # list of NormalizedLandmark

                if show_preview and not mobile:
                    _draw_connections(bgr, face_lm, connections, actual_w, actual_h)

                coords = _extract(face_lm, actual_w, actual_h)

                if show_preview:
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
                yield lf

            else:
                if show_preview:
                    cv2.putText(
                        bgr, "No face detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                    )
                yield None

            if show_preview:
                cv2.imshow("Assistive Gaze Mouse — tracker", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()


# ── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    for lf in frame_stream():
        if lf is None:
            continue
        print(
            f"NOSE=({lf.nose[0]:4d},{lf.nose[1]:4d})  "
            f"L_EYE={[p[:2] for p in lf.left_eye]}  "
            f"R_EYE={[p[:2] for p in lf.right_eye]}"
        )
