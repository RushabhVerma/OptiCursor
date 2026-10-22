"""
filter.py — Kalman filter for 2-D cursor coordinate smoothing (Step 2)

Why a Kalman filter?
────────────────────
Raw nose-tip pixel coordinates jitter by ±3–8 px per frame even when the
user's head is perfectly still.  A simple moving average introduces lag
proportional to the window size.  A Kalman filter achieves a better
lag/noise trade-off by maintaining an explicit model of the system:

    State vector x = [x, y, vx, vy]ᵀ  (position + velocity)

The filter predicts the next state from physics, then corrects it with the
noisy measurement.  The gain (how much to trust each) is tuned automatically
via the process noise (Q) and measurement noise (R) matrices.

Tuning guide
────────────
PROCESS_NOISE_COV  (Q scalar): how much the cursor *actually* accelerates
    between frames.  Higher → filter trusts new measurements more (more
    responsive but less smooth).  Lower → smoother but laggier.

MEASUREMENT_NOISE_COV (R scalar): how noisy the raw landmark readings are.
    Higher → filter relies more on its own prediction (smoother but laggier).
    Lower → follows raw input more closely.

Recommended starting point: Q=1e-2, R=1e-1.  For users with high-frequency
tremor, increase R.  For users who need fast cursor response, increase Q.
"""

from __future__ import annotations

import numpy as np

# ── Default noise parameters ──────────────────────────────────────────────────

PROCESS_NOISE_COV:     float = 1e-2   # Q: model uncertainty
MEASUREMENT_NOISE_COV: float = 1e-1   # R: sensor noise


class KalmanFilter2D:
    """
    Constant-velocity Kalman filter for a 2-D point.

    State: [x, y, vx, vy]ᵀ
    Observation: [x, y]ᵀ

    Usage::

        kf = KalmanFilter2D()
        for raw_x, raw_y in coordinate_stream:
            smooth_x, smooth_y = kf.update(raw_x, raw_y)
    """

    def __init__(
        self,
        process_noise:     float = PROCESS_NOISE_COV,
        measurement_noise: float = MEASUREMENT_NOISE_COV,
        dt: float = 1.0,                    # time step (frames); 1.0 = frame-relative
    ) -> None:
        # State vector: [x, y, vx, vy]
        self._x = np.zeros((4, 1), dtype=np.float64)

        # State transition matrix (constant-velocity model)
        self._F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0,  1, 0],
            [0, 0,  0, 1],
        ], dtype=np.float64)

        # Observation matrix: we measure x and y directly
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise covariance
        self._Q = np.eye(4, dtype=np.float64) * process_noise

        # Measurement noise covariance
        self._R = np.eye(2, dtype=np.float64) * measurement_noise

        # Estimate error covariance — start with identity (high uncertainty)
        self._P = np.eye(4, dtype=np.float64)

        self._initialised = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        """
        Feed one raw measurement and return the smoothed (x, y).

        On the first call the filter is seeded with the raw measurement so
        there is no start-up transient.
        """
        z = np.array([[raw_x], [raw_y]], dtype=np.float64)

        if not self._initialised:
            self._x[0, 0] = raw_x
            self._x[1, 0] = raw_y
            self._initialised = True
            return raw_x, raw_y

        # ── Predict ───────────────────────────────────────────────────────────
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # ── Update ────────────────────────────────────────────────────────────
        S = self._H @ P_pred @ self._H.T + self._R          # innovation covariance
        K = P_pred @ self._H.T @ np.linalg.inv(S)           # Kalman gain

        self._x = x_pred + K @ (z - self._H @ x_pred)
        self._P = (np.eye(4) - K @ self._H) @ P_pred

        return float(self._x[0, 0]), float(self._x[1, 0])

    def reset(self) -> None:
        """Reset the filter state (e.g. on face re-acquisition)."""
        self._x[:] = 0
        self._P = np.eye(4, dtype=np.float64)
        self._initialised = False

    @property
    def position(self) -> tuple[float, float]:
        """Current smoothed position without consuming a measurement."""
        return float(self._x[0, 0]), float(self._x[1, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated velocity (pixels/frame)."""
        return float(self._x[2, 0]), float(self._x[3, 0])
