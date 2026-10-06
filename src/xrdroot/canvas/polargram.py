"""A polar diagram's arithmetic: ``TGraphPolargram``'s grid and ``TGraphPainter``'s points.

A ``TGraphPolar`` is drawn in a pad whose range is ``-1.25..1.25`` both
ways, in a unit circle: a point ``(theta, r)`` goes to ``(r - rmin) /
(rmax - rmin)`` from the middle, at ``theta`` measured from ``tmin`` and
stretched so that ``tmin..tmax`` is a whole turn. A line leaving the circle
is cut where it crosses it, and picks up again where it comes back in, as
``PaintGraphPolar`` cuts it. The grid is circles at the radial divisions
and spokes at the polar ones, each labelled as ``TGraphPolargram`` labels
them - in fractions of pi for radians, ``%5.3g`` otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["polar_range", "circle", "find_align", "find_text_angle", "reduce_fraction",
           "radian_label", "number_label", "to_unit_circle", "inside_runs"]  # fmt: skip

#: ``PaintCircle``'s most points round a circle, and fewest.
MOST_POINTS, FEWEST_POINTS = 200, 8
#: How wide and high the pad is, in its own units, when a polargram has set its range.
PAD_SPAN = 2.5


def polar_range(x: Any, y: Any, ex: Any = None, ey: Any = None) -> tuple[float, ...]:
    """``CreatePolargram``'s ``rmin, rmax, tmin, tmax``: the points' reach, with margins.

    The radii are widened by a tenth either way, and the angles by one
    point's share of them, as though the points were spread round a turn.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ex = np.zeros(len(x)) if ex is None else np.asarray(ex, dtype=float)
    ey = np.zeros(len(y)) if ey is None else np.asarray(ey, dtype=float)
    rmin, rmax = min(float(y[0]), float((y - ey).min())), max(float(y[-1]), float((y + ey).max()))
    tmin, tmax = min(float(x[0]), float((x - ex).min())), max(float(x[-1]), float((x + ex).max()))
    rmax += 1.0 if rmin == rmax else 0.0
    tmax += 1.0 if tmin == tmax else 0.0
    dr, dt = rmax - rmin, tmax - tmin
    return rmin - 0.1 * dr, rmax + 0.1 * dr, tmin, tmax + dt / len(x)


def circle(r: float, phimin: float = 0.0, phimax: float = 360.0) -> tuple[Any, Any]:
    """``PaintCircle``: the polyline round a circle, its points as many as its length asks."""
    length = math.pi * 2 * r * (phimax - phimin) / 36
    n = min(max(int(MOST_POINTS * length / (2 * PAD_SPAN)), FEWEST_POINTS), MOST_POINTS)
    angles = math.radians(phimin) + np.arange(n + 1) * (phimax - phimin) * math.pi / (180 * n)
    return r * np.cos(angles), r * np.sin(angles)


def _turned(angle: float) -> float:
    """An angle brought into ``0..2 pi``, as ``FindAlign`` brings it."""
    return angle % (2 * math.pi) if angle < 0 or angle > 2 * math.pi else angle


def find_align(angle: float, ortho: bool = False) -> int:
    """``FindAlign``: how a polar label is aligned, by the angle it is at."""
    a, pi = _turned(angle), math.pi
    if ortho:
        return 32 if pi / 2 < a <= 3 * pi / 2 else 12
    exact = {0.0: 12, 2 * pi: 12, pi / 2: 21, pi: 32, 3 * pi / 2: 23}
    if a in exact:
        return exact[a]
    return 11 if a < pi / 2 else 31 if a < pi else 33 if a < 3 * pi / 2 else 13


def find_text_angle(angle: float) -> float:
    """``FindTextAngle``: a label written along its spoke, never upside down."""
    a, pi = _turned(angle), math.pi
    return math.degrees(a + pi if pi / 2 < a <= pi else a - pi if pi < a <= 3 * pi / 2 else a)


def reduce_fraction(num: int, den: int) -> tuple[int, int]:
    """``ReduceFraction``: ``num / den`` in its lowest terms."""
    for i in range(max(num, den), 1, -1):
        if den % i == 0 and num % i == 0:
            num, den = num // i, den // i
    return num, den


def radian_label(i: int, divisions: int) -> str:
    """The label of spoke ``i`` of ``divisions`` in radians: ``0``, ``#frac{#pi}{2}``, ``#pi``..."""
    num, den = reduce_fraction(2 * i, divisions)
    if num == 0:
        return "0"
    top = "#pi" if num == 1 else f"{num}#pi"
    return top if den == 1 else f"#frac{{{top}}}{{{den}}}"


def number_label(value: float) -> str:
    """A label as ``%5.3g`` writes it, the blanks in front taken off as ``LabelsLimits`` does."""
    return f"{value:5.3g}".lstrip()


def to_unit_circle(theta: Any, r: Any, ranges: Sequence[float], scale: float = 1.0) -> Any:
    """Points ``(theta, r)`` in the unit circle a polargram of ``rmin, rmax, tmin, tmax`` draws.

    ``scale`` is 1 for radians, ``180 / pi`` for degrees and ``100 / pi`` for grads.
    """
    rmin, rmax, tmin, tmax = ranges
    radius = (np.asarray(r, dtype=float) - rmin) / (rmax - rmin)
    angle = scale * (np.asarray(theta, dtype=float) - tmin) / ((tmax - tmin) / (2 * math.pi))
    return radius * np.cos(angle), radius * np.sin(angle)


def _crossing(inside: tuple[float, float], outside: tuple[float, float]) -> tuple[float, float]:
    """Where the line through two points crosses the unit circle, nearer ``outside``."""
    (x0, y0), (x1, y1) = inside, outside
    a = (y1 - y0) / (x1 - x0)
    b = y0 - a * x0
    root = math.sqrt(4 * (a * a - b * b + 1))
    ends = [((-2 * a * b + s * root) / (2 * (a * a + 1))) for s in (1, -1)]
    found = [(x, a * x + b) for x in ends]
    return min(found, key=lambda p: math.hypot(p[0] - x1, p[1] - y1))


def inside_runs(xs: Any, ys: Any) -> list[tuple[list[float], list[float]]]:
    """The parts of a line inside the unit circle, each ended where it crosses the circle."""
    runs: list[tuple[list[float], list[float]]] = []
    run: list[tuple[float, float]] = []
    previous: tuple[float, float] | None = None
    for point in zip(map(float, xs), map(float, ys), strict=False):
        if math.hypot(*point) <= 1:
            if not run and previous is not None:
                run.append(_crossing(point, previous))
            run.append(point)
        elif run:
            runs.append(tuple(map(list, zip(*run, _crossing(run[-1], point), strict=False))))  # type: ignore[arg-type]
            run = []
        previous = point
    if len(run) > 1:
        runs.append(tuple(map(list, zip(*run, strict=False))))  # type: ignore[arg-type]
    return runs
