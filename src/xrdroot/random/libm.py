"""``exp``, ``log``, ``sin``, ``pow`` and the rest, to the bit of the C library's, on arrays.

ROOT's distributions and RooFit's densities call ``std::log`` and the rest,
which are the C library's. Python's ``math`` calls the same library, so ``math.log``
is ROOT's ``log`` to the last bit on the same machine. NumPy's usually is
too - on arm64 and on x86 without AVX-512 it simply calls the C library -
but where the processor has AVX-512 NumPy has vectorised implementations of
its own, which are as accurate but not the same function, and can differ in
the last place. A number that is ROOT's only most of the time is not ROOT's.

So each function is tried once, when this module is loaded, against the C
library's on a few thousand arguments of the kind the distributions pass,
and NumPy's is used only if it agrees on every one; otherwise the C
library's is called for each element, which costs tens of nanoseconds
rather than a few, and is exact. Either way the answer is the C library's -
an overflow an infinity and a point outside the domain a NaN, as in C, where
Python's ``math`` would raise.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "atan", "choose", "choose_pow", "cos", "cosh", "exp", "lgamma", "log", "power", "sin", "sinh",
    "tan",
]  # fmt: skip

#: The arguments each function is tried on, spread evenly over the range the
#: distributions pass it, with a few thousand values between.
_PROBES = {
    "exp": (-60.0, 5.0),
    "log": (1e-12, 1.0),
    "sin": (0.0, 2 * math.pi),
    "cos": (0.0, 2 * math.pi),
    "tan": (-math.pi / 2, math.pi),
    "sinh": (-20.0, 20.0),
    "cosh": (-20.0, 20.0),
    "atan": (-50.0, 50.0),
}


def choose(
    fast: Callable[[Any], Any], exact: Callable[[float], float], low: float, high: float
) -> Any:
    """``fast`` if it agrees with ``exact`` wherever it is tried from ``low`` to ``high``.

    Otherwise the answer is ``exact``, called element by element.
    """
    probe = np.linspace(low, high, 4099)[1:-1]
    if np.array_equal(fast(probe), np.array([exact(x) for x in probe.tolist()])):
        return fast
    return lambda values: _each(exact, fast, values)


def _ieee(exact: Callable[..., float], fast: Callable[..., Any], *args: float) -> float:
    """``exact(*args)``, or - where Python raises rather than overflow or leave the domain, as C
    does - ``fast``'s infinity, zero or NaN, which is the C library's too."""
    try:
        return exact(*args)
    except (OverflowError, ValueError):
        with np.errstate(all="ignore"):
            return float(fast(*args))


def _each(exact: Callable[[float], float], fast: Callable[[Any], Any], values: Any) -> Any:
    """``exact`` of every element, as an array of the same shape - or a number, for a number."""
    values = np.asarray(values, dtype=np.float64)
    found = (_ieee(exact, fast, x) for x in values.ravel().tolist())
    flat = np.fromiter(found, dtype=np.float64, count=values.size)
    return flat.reshape(values.shape)[()]


def choose_pow(fast: Callable[[Any, Any], Any], exact: Callable[[float, float], float]) -> Any:
    """``fast`` if it raises as ``exact`` does on a grid of bases and powers, else ``exact``."""
    bases, powers = np.meshgrid(np.linspace(0.0, 20.0, 66)[1:], np.linspace(-12.0, 12.0, 63))
    found = [exact(b, p) for b, p in zip(bases.ravel().tolist(), powers.ravel().tolist())]
    if np.array_equal(fast(bases.ravel(), powers.ravel()), found):
        return fast
    return lambda base, power: _each_pair(exact, fast, base, power)


def _each_pair(
    exact: Callable[[float, float], float], fast: Callable[[Any, Any], Any], base: Any, power: Any
) -> Any:
    """``exact`` of every pair of elements, broadcast as NumPy broadcasts them."""
    base, power = np.broadcast_arrays(np.asarray(base, np.float64), np.asarray(power, np.float64))
    pairs = zip(base.ravel().tolist(), power.ravel().tolist())
    found = (_ieee(exact, fast, b, p) for b, p in pairs)
    flat = np.fromiter(found, dtype=np.float64, count=base.size)
    return flat.reshape(base.shape)[()]


exp = choose(np.exp, math.exp, *_PROBES["exp"])
log = choose(np.log, math.log, *_PROBES["log"])
sin = choose(np.sin, math.sin, *_PROBES["sin"])
cos = choose(np.cos, math.cos, *_PROBES["cos"])
tan = choose(np.tan, math.tan, *_PROBES["tan"])
sinh = choose(np.sinh, math.sinh, *_PROBES["sinh"])
cosh = choose(np.cosh, math.cosh, *_PROBES["cosh"])
atan = choose(np.arctan, math.atan, *_PROBES["atan"])
power = choose_pow(np.power, math.pow)


def _c_lgamma() -> Callable[[float], float]:
    """The C library's ``lgamma`` - which RooFit's kernels call as ``std::lgamma`` - and not
    Python's ``math.lgamma``, which is CPython's own and differs from it in the last place on
    about half of the integers; Python's where the C library cannot be found."""
    import ctypes
    import ctypes.util

    try:
        found = ctypes.CDLL(ctypes.util.find_library("m") or None).lgamma
    except (OSError, AttributeError):  # pragma: no cover - every platform here has one
        return math.lgamma
    found.restype, found.argtypes = ctypes.c_double, [ctypes.c_double]
    return found


_lgamma = _c_lgamma()


def lgamma(values: Any) -> Any:
    """``std::lgamma`` of every element - an infinity at the poles, as C has it."""
    return _each(_lgamma, np.vectorize(math.lgamma, otypes=[np.float64]), values)
