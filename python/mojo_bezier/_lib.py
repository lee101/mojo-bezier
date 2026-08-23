"""Build and load the single Mojo compilation unit."""
from __future__ import annotations
import ctypes, os, shutil, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "capi.mojo")
LIB = os.environ.get("MOJO_BEZIER_LIB") or os.path.join(ROOT, "dist", "libmojo-bezier.so")
I, F = ctypes.c_int64, ctypes.c_double
SIGNATURES = {"mbz_evaluate_multi": ([I]*6, None), "mbz_subdivide": ([I]*5, None), "mbz_elevate": ([I]*4, None), "mbz_refine_candidates": ([I]*7, None), "mbz_intersect_candidates": ([I, I, I, I, F, I, I, I, I, I], I), "mbz_intersect": ([I]*8, I)}

def build():
    if os.environ.get("MOJO_BEZIER_LIB") and os.path.exists(LIB): return LIB
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC): return LIB
    mojo = shutil.which("mojo")
    if not mojo: raise RuntimeError("mojo is not on PATH; run through pixi")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run([mojo, "build", "--emit", "shared-lib", SRC, "-o", LIB], capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB): raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB

_library = None
def lib():
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (args, restype) in SIGNATURES.items():
            fn = getattr(_library, name); fn.argtypes, fn.restype = args, restype
    return _library
def f64_fortran(value, *, copy=True, name="array"):
    """Return a finite-layout-safe float64 Fortran array without coercion."""
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64; implicit conversion is not supported.")
    if not array.flags.f_contiguous:
        if not copy:
            raise ValueError(f"{name} must be Fortran-contiguous when copy=False.")
        array = np.asfortranarray(array)
    return array.copy(order="F") if copy else array
def addr(array): return int(array.ctypes.data)
