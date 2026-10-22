"""Custom OpenCV presentation layer for the assistive mouse application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


Color = tuple[int, int, int]

_BG: Color = (18, 18, 24)
_CARD: Color = (31, 31, 42)
_CARD_ALT: Color = (39, 39, 53)
_TEXT: Color = (242, 244, 255)
_MUTED: Color = (160, 165, 185)
_ACCENT: Color = (236, 120, 72)
_ACCENT_2: Color = (208, 102, 255)


def rounded_rect(image: np.ndarray, left: int, top: int, right: int, bottom: int, color: Color, radius: int = 14) -> None:
    """Draw a filled rounded rectangle onto an OpenCV BGR image."""
    cv2.rectangle(image, (left + radius, top), (right - radius, bottom), color, -1)
    cv2.rectangle(image, (left, top + radius), (right, bottom - radius), color, -1)
    for center in ((left + radius, top + radius), (right - radius, top + radius), (left + radius, bottom - radius), (right - radius, bottom - radius)):
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)


@dataclass
class Slider:
    """A directly draggable normalized settings slider."""

    key: str
    label: str
    subtitle: str
    value: float
    minimum: float
    maximum: float
    decimals: int
    y: int

    def set_from_x(self, x: int) -> None:
        """Set this slider from its panel-relative mouse x-coordinate."""
        left, right = 52, 548
        fraction = min(max((x - left) / (right - left), 0.0), 1.0)
        self.value = self.minimum + fraction * (self.maximum - self.minimum)

    def display_value(self) -> str:
        """Format the current user-visible slider value."""
        return f"{self.value:.{self.decimals}f}"


class SettingsPanel:
    """Modern no-dependency settings panel rendered entirely through OpenCV."""

    width: int = 600
    height: int = 590

    def __init__(self, initial: dict[str, float], on_change: Callable[[dict[str, float]], None]) -> None:
        self._on_change = on_change
        self._dragging: Slider | None = None
        self._sliders = [
            Slider("precision", "Precision speed", "Slow movement control", initial["precision"], 1.0, 6.0, 1, 166),
            Slider("fast", "Fast speed", "Quick movement acceleration", initial["fast"], 1.0, 10.0, 1, 244),
            Slider("click_hold", "Click hold", "Gesture confirmation time", initial["click_hold"], 0.10, 1.0, 2, 322),
            Slider("mouth", "Scroll mode sensitivity", "Wide-open-mouth threshold", initial["mouth"], 0.10, 0.95, 2, 400),
            Slider("scroll_interval", "Scroll speed", "Lower value scrolls faster", initial["scroll_interval"], 0.04, 0.50, 2, 478),
        ]

    def _values(self) -> dict[str, float]:
        return {slider.key: slider.value for slider in self._sliders}

    def mouse_event(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        """Handle slider drag gestures and publish changed values."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = next((slider for slider in self._sliders if slider.y - 18 <= y <= slider.y + 18), None)
        if event == cv2.EVENT_MOUSEMOVE and self._dragging is not None:
            self._dragging.set_from_x(x)
            self._enforce_ranges()
            self._on_change(self._values())
        if event == cv2.EVENT_LBUTTONUP:
            if self._dragging is not None:
                self._dragging.set_from_x(x)
                self._enforce_ranges()
                self._on_change(self._values())
            self._dragging = None

    def _enforce_ranges(self) -> None:
        fast = next(slider for slider in self._sliders if slider.key == "fast")
        precision = next(slider for slider in self._sliders if slider.key == "precision")
        fast.value = max(fast.value, precision.value)

    def render(self) -> np.ndarray:
        """Return the complete current settings panel image."""
        canvas = np.full((self.height, self.width, 3), _BG, dtype=np.uint8)
        rounded_rect(canvas, 18, 16, self.width - 18, 108, _CARD, 18)
        cv2.circle(canvas, (54, 62), 20, _ACCENT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (54, 62), 9, _BG, -1, cv2.LINE_AA)
        cv2.putText(canvas, "Control Studio", (88, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.78, _TEXT, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Live tuning • saved automatically", (88, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Cursor & gesture preferences", (28, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _MUTED, 1, cv2.LINE_AA)
        for slider in self._sliders:
            self._draw_slider(canvas, slider)
        rounded_rect(canvas, 18, 522, self.width - 18, 568, _CARD_ALT, 12)
        cv2.putText(canvas, "Press O in the camera window to close", (38, 551), cv2.FONT_HERSHEY_SIMPLEX, 0.50, _TEXT, 1, cv2.LINE_AA)
        return canvas

    def _draw_slider(self, canvas: np.ndarray, slider: Slider) -> None:
        left, right = 52, 548
        rounded_rect(canvas, 28, slider.y - 39, 572, slider.y + 30, _CARD, 12)
        cv2.putText(canvas, slider.label, (44, slider.y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, _TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, slider.subtitle, (44, slider.y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, _MUTED, 1, cv2.LINE_AA)
        fraction = (slider.value - slider.minimum) / (slider.maximum - slider.minimum)
        knob_x = round(left + fraction * (right - left))
        cv2.line(canvas, (left, slider.y + 22), (right, slider.y + 22), (73, 75, 94), 5, cv2.LINE_AA)
        cv2.line(canvas, (left, slider.y + 22), (knob_x, slider.y + 22), _ACCENT_2, 5, cv2.LINE_AA)
        cv2.circle(canvas, (knob_x, slider.y + 22), 9, _TEXT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (knob_x, slider.y + 22), 4, _ACCENT, -1, cv2.LINE_AA)
        cv2.putText(canvas, slider.display_value(), (500, slider.y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, _ACCENT, 1, cv2.LINE_AA)
