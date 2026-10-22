"""
setup.py — pybind11 compilation script (Step 4)

Builds the C++ mouse controller (src/native/mouse_win32.cpp) as a Python
extension module so it can be imported directly from Python.

Usage
─────
    pip install pybind11
    python setup.py build_ext --inplace

The compiled .pyd will appear in src/native/ (Windows) or src/native/*.so
(Linux/macOS).  Add src/native/ to sys.path or install the package to import:

    import mouse_win32
    mouse_win32.mouse_move(960, 540)
"""

from setuptools import setup, Extension
import pybind11
import sys

ext = Extension(
    name="mouse_win32",
    sources=["src/native/mouse_win32.cpp"],
    include_dirs=[pybind11.get_include()],
    define_macros=[("PYBIND11_BUILD", "1")],   # activates the pybind11 block
    language="c++",
    extra_compile_args=(
        ["/std:c++17", "/O2", "/W3"] if sys.platform == "win32"
        else ["-std=c++17", "-O2", "-Wall"]
    ),
    libraries=(["user32"] if sys.platform == "win32" else []),
)

setup(
    name="assistive-gaze-mouse-native",
    version="0.1.0",
    description="Windows raw-input mouse controller (pybind11 extension)",
    ext_modules=[ext],
)
