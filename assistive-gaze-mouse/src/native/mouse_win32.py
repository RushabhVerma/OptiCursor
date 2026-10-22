"""Windows mouse-input adapter implemented with the standard library.

This module deliberately matches the public API of the optional pybind11
``mouse_win32`` extension.  It keeps real-input mode usable on systems without
Microsoft C++ Build Tools while retaining the extension as an optional future
optimization.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Final

if sys.platform != "win32":
    raise ImportError("mouse_win32 is available only on Windows.")


_INPUT_MOUSE: Final[int] = 0
_MOUSEEVENTF_LEFTDOWN: Final[int] = 0x0002
_MOUSEEVENTF_LEFTUP: Final[int] = 0x0004
_MOUSEEVENTF_RIGHTDOWN: Final[int] = 0x0008
_MOUSEEVENTF_RIGHTUP: Final[int] = 0x0010
_MOUSEEVENTF_WHEEL: Final[int] = 0x0800
_SM_CXSCREEN: Final[int] = 0
_SM_CYSCREEN: Final[int] = 1
_WHEEL_DELTA: Final[int] = 120

_ULONG_PTR = ctypes.c_size_t


class _MouseInput(ctypes.Structure):
    """Layout of the Win32 MOUSEINPUT structure."""

    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _InputUnion(ctypes.Union):
    """Union within Win32 INPUT; this adapter uses only the mouse variant."""

    _fields_ = [("mi", _MouseInput)]


class _Input(ctypes.Structure):
    """Layout of the Win32 INPUT structure."""

    _anonymous_ = ("data",)
    _fields_ = [("type", wintypes.DWORD), ("data", _InputUnion)]


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.GetSystemMetrics.argtypes = [ctypes.c_int]
_user32.GetSystemMetrics.restype = ctypes.c_int
_user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
_user32.GetCursorPos.restype = wintypes.BOOL
_user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
_user32.SetCursorPos.restype = wintypes.BOOL
_user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_Input), ctypes.c_int]
_user32.SendInput.restype = wintypes.UINT


def _send_mouse_inputs(*flags: int, mouse_data: int = 0) -> None:
    """Inject one or more mouse events and raise if Windows rejects them."""
    inputs = (_Input * len(flags))()
    for input_event, flag in zip(inputs, flags, strict=True):
        input_event.type = _INPUT_MOUSE
        input_event.mi.dwFlags = flag
        input_event.mi.mouseData = ctypes.c_ulong(mouse_data).value
    sent = _user32.SendInput(len(inputs), inputs, ctypes.sizeof(_Input))
    if sent != len(inputs):
        error = ctypes.get_last_error()
        raise OSError(error, f"SendInput sent {sent}/{len(inputs)} mouse events")


def get_screen_size() -> tuple[int, int]:
    """Return the primary monitor dimensions in physical pixels."""
    width = _user32.GetSystemMetrics(_SM_CXSCREEN)
    height = _user32.GetSystemMetrics(_SM_CYSCREEN)
    if width <= 0 or height <= 0:
        raise OSError(ctypes.get_last_error(), "GetSystemMetrics failed")
    return width, height


def get_cursor_position() -> tuple[int, int]:
    """Return the current system cursor position in physical screen pixels."""
    point = wintypes.POINT()
    if not _user32.GetCursorPos(ctypes.byref(point)):
        raise OSError(ctypes.get_last_error(), "GetCursorPos failed")
    return int(point.x), int(point.y)


def mouse_move(x: int, y: int) -> None:
    """Move the system cursor to absolute primary-screen pixel coordinates."""
    width, height = get_screen_size()
    bounded_x = max(0, min(int(x), width - 1))
    bounded_y = max(0, min(int(y), height - 1))
    if not _user32.SetCursorPos(bounded_x, bounded_y):
        raise OSError(ctypes.get_last_error(), "SetCursorPos failed")


def mouse_left_click() -> None:
    """Inject one left-button press and release."""
    _send_mouse_inputs(_MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP)


def mouse_right_click() -> None:
    """Inject one right-button press and release."""
    _send_mouse_inputs(_MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP)


def mouse_scroll(delta: int) -> None:
    """Scroll by signed wheel detents; positive values scroll upward."""
    _send_mouse_inputs(_MOUSEEVENTF_WHEEL, mouse_data=int(delta) * _WHEEL_DELTA)
