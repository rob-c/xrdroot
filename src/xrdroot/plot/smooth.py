"""A smooth line through points, for ROOT's ``C`` and ``E4``, where a backend has none.

ROOT smooths with a spline of its own making; plotly has one built in, and
for matplotlib and bokeh this lays a Catmull-Rom curve through the points -
one that passes through every point, which is the property that matters
when the points are measurements.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["smoothed"]

#: How many pieces each gap between two points is drawn in.
STEPS = 8


def smoothed(x: Any, y: Any) -> tuple[Any, Any]:
    """``(x, y)`` with a Catmull-Rom curve's points laid between each pair.

    >>> xs, ys = smoothed([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    >>> float(xs[0]), float(xs[-1]), len(xs)
    (0.0, 2.0, 17)
    """
    xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(xs) < 3:
        return xs, ys
    padded = [np.concatenate([[values[0]], values, [values[-1]]]) for values in (xs, ys)]
    t = np.linspace(0.0, 1.0, STEPS, endpoint=False)[:, None]
    made = [_between(values, t) for values in padded]
    return made[0], made[1]


def _between(values: Any, t: Any) -> Any:
    """Each gap's curve, from the four points around it, then the last point."""
    p0, p1, p2, p3 = values[:-3], values[1:-2], values[2:-1], values[3:]
    a = 2 * p1
    b = p2 - p0
    c = 2 * p0 - 5 * p1 + 4 * p2 - p3
    d = 3 * p1 - p0 - 3 * p2 + p3
    curve = 0.5 * (a + b * t + c * t**2 + d * t**3)
    return np.append(curve.T.ravel(), values[-2])
