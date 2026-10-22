/*
 * mouse_win32.cpp — Windows OS input controller (Step 3)
 *
 * Exposes a minimal C API (and, via pybind11 in Step 4, a Python module)
 * that lets the Python brain issue mouse commands without any Python
 * overhead on the hot path.
 *
 * API surface
 * ───────────
 *   mouse_move(x, y)          Warp cursor to absolute screen coordinates.
 *   mouse_left_click()        Synthesise a left button press + release.
 *   mouse_right_click()       Synthesise a right button press + release.
 *   mouse_scroll(delta)       Vertical scroll; positive = up, negative = down.
 *   get_screen_size(w, h)     Query the primary monitor resolution at runtime.
 *
 * Implementation notes
 * ────────────────────
 * SendInput() is preferred over mouse_event() (deprecated) because it
 * injects events into the raw input stream at higher priority and cannot
 * be blocked by UAC input-queue isolation on the same desktop session.
 *
 * MOUSEEVENTF_ABSOLUTE requires coordinates in the [0, 65535] virtual
 * desktop space regardless of actual monitor resolution.  We normalise
 * from pixel space using the runtime resolution returned by GetSystemMetrics.
 *
 * MOUSEEVENTF_VIRTUALDESK is OR-ed in so the coordinates span the full
 * virtual desktop on multi-monitor setups rather than just the primary
 * monitor.
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdexcept>
#include <string>

// ── Internal helpers ──────────────────────────────────────────────────────────

namespace {

// Virtual desktop coordinate space used by MOUSEEVENTF_ABSOLUTE.
constexpr LONG VDESK_MAX = 65535;

/*
 * Normalise a pixel coordinate to the [0, 65535] absolute space.
 * screen_dim  — actual pixel dimension of the virtual desktop (width or height)
 * pixel       — pixel coordinate to normalise
 */
inline LONG to_absolute(int pixel, int screen_dim) noexcept
{
    if (screen_dim <= 0) return 0;
    // +1 to avoid off-by-one at the far edge (Microsoft KB Q274323)
    return static_cast<LONG>(
        (static_cast<double>(pixel) * VDESK_MAX) / (screen_dim - 1)
    );
}

/*
 * Fire one or more INPUT structures through SendInput.
 * Throws std::runtime_error if SendInput reports a partial send.
 */
void send(INPUT* inputs, UINT count)
{
    UINT sent = SendInput(count, inputs, sizeof(INPUT));
    if (sent != count) {
        DWORD err = GetLastError();
        throw std::runtime_error(
            "SendInput failed: sent " + std::to_string(sent) +
            "/" + std::to_string(count) +
            ", GetLastError=" + std::to_string(err)
        );
    }
}

} // anonymous namespace

// ── Public C API ──────────────────────────────────────────────────────────────

extern "C" {

/*
 * Warp the cursor to (x, y) in pixel coordinates on the primary monitor.
 * Coordinates are clamped to [0, screen_w-1] x [0, screen_h-1].
 */
void mouse_move(int x, int y)
{
    int screen_w = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    int screen_h = GetSystemMetrics(SM_CYVIRTUALSCREEN);

    // Clamp to valid range
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= screen_w) x = screen_w - 1;
    if (y >= screen_h) y = screen_h - 1;

    INPUT inp{};
    inp.type           = INPUT_MOUSE;
    inp.mi.dwFlags     = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK;
    inp.mi.dx          = to_absolute(x, screen_w);
    inp.mi.dy          = to_absolute(y, screen_h);
    inp.mi.mouseData   = 0;
    inp.mi.dwExtraInfo = 0;

    send(&inp, 1);
}

/*
 * Synthesise a left mouse button click (down + up).
 */
void mouse_left_click()
{
    INPUT inputs[2]{};

    inputs[0].type       = INPUT_MOUSE;
    inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;

    inputs[1].type       = INPUT_MOUSE;
    inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;

    send(inputs, 2);
}

/*
 * Synthesise a right mouse button click (down + up).
 */
void mouse_right_click()
{
    INPUT inputs[2]{};

    inputs[0].type       = INPUT_MOUSE;
    inputs[0].mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;

    inputs[1].type       = INPUT_MOUSE;
    inputs[1].mi.dwFlags = MOUSEEVENTF_RIGHTUP;

    send(inputs, 2);
}

/*
 * Scroll vertically by `delta` wheel notches.
 * delta > 0 → scroll up (away from user)
 * delta < 0 → scroll down (toward user)
 *
 * WHEEL_DELTA (120) is one standard detent.  Multiplying by the caller's
 * delta keeps the scroll proportional for continuous scroll zones.
 */
void mouse_scroll(int delta)
{
    INPUT inp{};
    inp.type           = INPUT_MOUSE;
    inp.mi.dwFlags     = MOUSEEVENTF_WHEEL;
    inp.mi.mouseData   = static_cast<DWORD>(delta * WHEEL_DELTA);

    send(&inp, 1);
}

/*
 * Query the primary monitor's pixel dimensions at runtime.
 * Results are written into the caller-supplied pointers.
 *
 * Using SM_CXSCREEN / SM_CYSCREEN (not SM_CXVIRTUALSCREEN) so we return
 * the resolution of the monitor where most applications live.
 */
void get_screen_size(int* out_w, int* out_h)
{
    if (out_w) *out_w = GetSystemMetrics(SM_CXSCREEN);
    if (out_h) *out_h = GetSystemMetrics(SM_CYSCREEN);
}

} // extern "C"

// ── pybind11 module (compiled in Step 4) ─────────────────────────────────────
//
// Guarded so this file compiles as a plain .cpp (Step 3, tested via a small
// standalone main) without pybind11 present, and also as a Python extension
// module when PYBIND11_BUILD is defined by setup.py in Step 4.

#ifdef PYBIND11_BUILD
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(mouse_win32, m)
{
    m.doc() = "Windows raw-input mouse controller for assistive-gaze-mouse";

    m.def("mouse_move",        &mouse_move,
          py::arg("x"), py::arg("y"),
          "Warp cursor to absolute pixel coordinates (x, y).");

    m.def("mouse_left_click",  &mouse_left_click,
          "Synthesise a left mouse button click.");

    m.def("mouse_right_click", &mouse_right_click,
          "Synthesise a right mouse button click.");

    m.def("mouse_scroll",      &mouse_scroll,
          py::arg("delta"),
          "Scroll vertically. delta > 0 = up, delta < 0 = down.");

    m.def("get_screen_size",   [](){
          int w = 0, h = 0;
          get_screen_size(&w, &h);
          return std::make_pair(w, h);
          },
          "Return (width, height) of the primary monitor in pixels.");
}
#endif // PYBIND11_BUILD
