"""``RooCurve``: a function on a plot, sampled where it bends, as RooFit samples it.

The curve starts from a point at each end of every frame bin and, between
two neighbours, adds the midpoint when the function there is further than
a thousandth of the curve's height from the straight line - halving again
until it is not, or until the step is a thousandth of the range
(``RooCurve::addPoints`` and ``addRange``). Beyond the ends it has RooFit's
"wings": a point one step out at the end's height, then down to zero. The
points, and so the picture and any chi-square to them, are ROOT's.

It is an :class:`xrdroot.Graph`, a ``TGraph`` of line width 3 in blue.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ...graph import Graph

__all__ = ["RooCurve", "sample"]

#: ``RooCurve::relativeXEpsilon``: steps closer than this fraction of the range are not split.
EPSILON = 1e-9
#: ``kBlue``, the colour a curve is drawn in unless told otherwise.
BLUE = 600


def _refine(func: Callable[[float], float], x1: float, x2: float, y1: float, y2: float,
            limits: tuple[float, float, float], out: list[tuple[float, float]]) -> None:  # fmt: skip
    """``addRange``: split ``[x1, x2]`` while its middle is off the line, then add its end."""
    min_dy, min_dx, epsilon = limits
    stack = [(x1, x2, y1, y2)]
    while stack:
        a, b, ya, yb = stack.pop()
        if abs(b - a) <= epsilon:
            continue
        mid = 0.5 * (a + b)
        ymid = func(mid)
        dy = ymid - 0.5 * (ya + yb)
        if mid - a >= min_dx and abs(dy) > 0 and abs(dy) >= min_dy:
            stack.append((mid, b, ymid, yb))
            stack.append((a, mid, ya, ymid))
        else:
            out.append((b, yb))


def sample(func: Callable[[Any], Any], low: float, high: float, bins: int,
           precision: float = 1e-3, wings: bool = True) -> tuple[np.ndarray[Any, Any], ...]:  # fmt: skip
    """``RooCurve::addPoints`` over ``bins + 1`` starting points; ``func`` takes an array."""
    count = bins + 1
    dx = (high - low) / (count - 1.0)
    xs = low + np.arange(count) * dx
    probe = xs.copy()
    probe[-1] -= 1e-9 * dx
    ys = np.asarray(func(probe), dtype=np.float64).reshape(-1) * np.ones(count)
    span = float(np.max(ys) - np.min(ys))
    scalar = lambda x: float(np.asarray(func(np.array([x]))).reshape(-1)[0])  # noqa: E731
    points: list[tuple[float, float]] = []
    if wings:
        points += [(low - dx * 1.001, 0.0), (low - dx, float(ys[0]))]
    points.append((low, float(ys[0])))
    limits = (precision * span, precision * (high - low), (high - low) * EPSILON)
    for i in range(1, count):
        _refine(scalar, float(xs[i - 1]), float(xs[i]), float(ys[i - 1]), float(ys[i]), limits, points)
    points.append((high, float(ys[-1])))
    if wings:
        points += [(high + dx, float(ys[-1])), (high + dx * 1.001, 0.0)]
    ordered = sorted(points, key=lambda p: p[0])
    return np.array([p[0] for p in ordered]), np.array([p[1] for p in ordered])


class RooCurve(Graph):
    """A sampled function, drawn as a line."""

    def __init__(self, name: str, title: str, x: Any, y: Any) -> None:
        made = Graph.new(name, x, y, title=title)
        super().__init__("TGraph", made.members)
        self._core["TAttLine"]["fLineWidth"] = 3
        self._core["TAttLine"]["fLineColor"] = BLUE
        self.y_label = title

    def GetName(self) -> str:
        return self.name

    def SetName(self, name: str) -> None:
        self._core["TNamed"]["fName"] = str(name)

    def ClassName(self) -> str:
        return "RooCurve"

    def GetN(self) -> int:
        return len(self.x)

    def interpolate(self, x: Any) -> Any:
        """``RooCurve::interpolate``: the curve's height at ``x``, linearly between its points."""
        return np.interp(x, self.x, self.y)

    def average(self, low: float, high: float) -> float:
        """``RooCurve::average``: the mean height over ``[low, high]``, by the trapezoids of its points."""
        inside = (self.x > low) & (self.x < high)
        xs = np.concatenate([[low], self.x[inside], [high]])
        ys = np.concatenate([[self.interpolate(low)], self.y[inside], [self.interpolate(high)]])
        area = float(np.sum(0.5 * (ys[1:] + ys[:-1]) * np.diff(xs)))
        return area / (high - low) if high > low else float(ys[0])
