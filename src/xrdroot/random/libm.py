"""``exp``, ``log``, ``sin``, ``cos`` and ``tan`` to the bit of the C library's, an array at a time.

ROOT's distributions call ``std::log`` and the rest, which are the C
library's. Python's ``math`` module calls the same library, so ``math.log``
is ROOT's ``log`` to the last bit on the same machine. NumPy's usually is
too - on arm64 and on x86 without AVX-512 it simply calls the C library -
but where the processor has AVX-512 NumPy has vectorised implementations of
its own, which are as accurate but not the same function, and can differ in
the last place. A number that is ROOT's only most of the time is not ROOT's.

So each function is tried once, when this module is loaded, against the C
library's on a few thousand arguments of the kind the distributions pass,
and NumPy's is used only if it agrees on every one; otherwise the C
library's is called for each element, which costs tens of nanoseconds
rather than a few, and is exact. Either way the answer is the C library's.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["choose", "cos", "exp", "log", "sin", "tan"]

#: The arguments each function is tried on, spread evenly over the range the
#: distributions pass it, with a few thousand values between.
_PROBES = {
    "exp": (-60.0, 5.0),
    "log": (1e-12, 1.0),
    "sin": (0.0, 2 * math.pi),
    "cos": (0.0, 2 * math.pi),
    "tan": (-math.pi / 2, math.pi),
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
    return lambda values: _each(exact, values)


def _each(exact: Callable[[float], float], values: Any) -> Any:
    """``exact`` of every element, as an array of the same shape."""
    values = np.asarray(values, dtype=np.float64)
    flat = np.fromiter(map(exact, values.ravel().tolist()), dtype=np.float64, count=values.size)
    return flat.reshape(values.shape)


exp = choose(np.exp, math.exp, *_PROBES["exp"])
log = choose(np.log, math.log, *_PROBES["log"])
sin = choose(np.sin, math.sin, *_PROBES["sin"])
cos = choose(np.cos, math.cos, *_PROBES["cos"])
tan = choose(np.tan, math.tan, *_PROBES["tan"])
