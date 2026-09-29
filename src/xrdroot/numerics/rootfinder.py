"""MathCore's ``BrentRootFinder``: a grid to bracket a root, then Brent's method on ``|f|``.

``BrentMethods::MinimStep`` walks ``npx`` points from the lower end and stops
at the first step the function changes sign over - or lands on zero;
``MinimBrent`` then minimises ``|f|`` inside that step. ``Solve`` repeats the
pair, narrowing, up to eleven times until Brent's method converges. This is
not GSL's Brent solver (:mod:`.roots`), which RooStats does not use here.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable

from .minimize1d import minim_brent

__all__ = ["brent_root_finder"]


def _bracket(fn: Callable[[float], float], xmin: float, xmax: float,
             npx: int) -> tuple[float, float, float]:  # fmt: skip
    """``MinimStep`` for a root: the middle of the first step the sign changes over, and its
    ends - or, none found, said on the error stream, the empty range ``(1, 0)``."""
    dx = (xmax - xmin) / (npx - 1)
    last, value = xmin, fn(xmin)
    if value == 0:
        return xmin, xmin, xmin
    for i in range(1, npx):
        x = xmin + i * dx
        y = fn(x)
        if y == 0:
            return x, x, x
        if math.copysign(1.0, y) * math.copysign(1.0, value) < 0:
            return 0.5 * (last + x), last, x
        last, value = x, y
    where = "Info in <ROOT::Math::BrentMethods::MinimStep>: "
    sys.stderr.write(f"{where}Grid search failed to find a root in the  interval \n{where}xmin = "
                     f"{xmin:.6g} xmax = {xmax:.6g} npts = {npx}\n")  # fmt: skip
    return 0.0, 1.0, 0.0


def brent_root_finder(fn: Callable[[float], float], low: float, high: float, npx: int = 100,
                      max_iter: int = 100, abs_tol: float = 1e-8,
                      rel_tol: float = 1e-10) -> tuple[bool, float]:  # fmt: skip
    """``BrentRootFinder::Solve``: whether a root was found, and the root."""
    xmin, xmax, x = low, high, 0.0
    for _ in range(11):
        x, xmin, xmax = _bracket(fn, xmin, xmax, npx)
        if xmin > xmax:
            sys.stderr.write("Error in <ROOT::Math::BrentRootFinder>: Interval does not contain a "
                             "root\n")  # fmt: skip
            return False, 0.0
        x, ok, xmin, xmax = minim_brent(lambda t: abs(fn(t)), xmin, xmax, x, abs_tol, rel_tol,
                                        max_iter)  # fmt: skip
        if ok:
            return True, x
    sys.stderr.write("Error in <ROOT::Math::BrentRootFinder::Solve>: Search didn't converge\n")
    return False, x
