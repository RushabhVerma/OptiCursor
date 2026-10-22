"""Validated persistent settings for Assistive Gaze Mouse."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CursorSettings:
    """Speed-transfer settings for the adaptive cursor mapper."""

    precision_gain: float = 3.2
    fast_gain: float = 6.0
    speed_for_max_gain: float = 0.8


@dataclass(frozen=True)
class GestureSettings:
    """Fallback click and blink settings used when calibration is skipped."""

    default_eyebrow_threshold: float = 0.45
    default_lip_pucker_threshold: float = 0.92
    click_hold_ms: float = 400.0
    blink_veto_threshold: float = 0.2


@dataclass(frozen=True)
class ScrollSettings:
    """Mode-toggle and vertical scrolling settings."""

    open_mouth_threshold: float = 0.65
    open_mouth_hold_ms: float = 500.0
    interval_ms: float = 160.0
    dead_zone: float = 0.25


@dataclass(frozen=True)
class AppSettings:
    """All persisted user-adjustable application settings."""

    camera_index: int = 0
    cursor: CursorSettings = CursorSettings()
    gestures: GestureSettings = GestureSettings()
    scroll: ScrollSettings = ScrollSettings()


def _number(section: dict[str, Any], key: str, default: float) -> float:
    """Read a finite positive JSON number or use the documented default."""
    value = section.get(key, default)
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive number.")
    return float(value)


def load_settings(path: Path) -> AppSettings:
    """Load and validate settings JSON, returning safe defaults if absent."""
    if not path.exists():
        return AppSettings()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid settings JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("Settings root must be a JSON object.")
    cursor = payload.get("cursor", {})
    gestures = payload.get("gestures", {})
    scroll = payload.get("scroll", {})
    if not all(isinstance(section, dict) for section in (cursor, gestures, scroll)):
        raise ValueError("cursor, gestures, and scroll must be JSON objects.")
    camera_index = payload.get("camera_index", 0)
    if not isinstance(camera_index, int) or isinstance(camera_index, bool) or camera_index < 0:
        raise ValueError("camera_index must be a non-negative integer.")
    settings = AppSettings(
        camera_index=camera_index,
        cursor=CursorSettings(
            _number(cursor, "precision_gain", 3.2),
            _number(cursor, "fast_gain", 6.0),
            _number(cursor, "speed_for_max_gain", 0.8),
        ),
        gestures=GestureSettings(
            _number(gestures, "default_eyebrow_threshold", 0.45),
            _number(gestures, "default_lip_pucker_threshold", 0.92),
            _number(gestures, "click_hold_ms", 400.0),
            _number(gestures, "blink_veto_threshold", 0.2),
        ),
        scroll=ScrollSettings(
            _number(scroll, "open_mouth_threshold", 0.65),
            _number(scroll, "open_mouth_hold_ms", 500.0),
            _number(scroll, "interval_ms", 160.0),
            _number(scroll, "dead_zone", 0.25),
        ),
    )
    if settings.cursor.fast_gain < settings.cursor.precision_gain:
        raise ValueError("cursor.fast_gain must be at least cursor.precision_gain.")
    if settings.gestures.blink_veto_threshold > 1.0 or settings.scroll.open_mouth_threshold > 1.0:
        raise ValueError("Blendshape thresholds must be between 0 and 1.")
    return settings


def save_settings(path: Path, settings: AppSettings) -> None:
    """Persist validated settings in readable JSON for the next application run."""
    path.write_text(json.dumps(asdict(settings), indent=2) + "\n", encoding="utf-8")
