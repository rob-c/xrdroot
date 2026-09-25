"""What ROOT works out from a graph's points: ``Eval``, ``Integral``, ``Sort``, ``GetMean``.

A graph is a run of points in whatever order they were added, and ROOT's
methods take them in that order: ``TGraph::Eval`` looks for the neighbours
of ``x`` by walking every point rather than assuming they are sorted, and
extrapolates past either end along the two it found last; ``Integral`` is the
area of the polygon the points make, closed from the last back to the first;
the mean and spread are of the points themselves, unweighted by any error.
Each is here as ROOT has it, its sums added in turn.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from .filling import running

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .graph import Graph

__all__ = ["evaluate", "integral", "mean", "rms", "sort"]

#: The per-point arrays a graph's class may keep beside ``fX`` and ``fY``.
PER_POINT = ("fEX", "fEY", "fEXlow", "fEXhigh", "fEYlow", "fEYhigh", "fExL", "fExH")

#: The ones a ``TGraphMultiErrors`` keeps a layer of apiece.
LAYERED = ("fEyL", "fEyH")


def _nearest(
    xs: list[float], x: float, side: Callable[[Any, Any], bool], nearer: Callable[[Any, Any], bool]
) -> tuple[int, int]:
    """The nearest point on one side of ``x``, and the one ROOT's walk keeps behind it.

    ROOT's second is the nearest before the last improvement - or, when the
    first point on that side was never bettered, the next point there - which
    is not always the second nearest, and is what it extrapolates along.
    """
    best = second = -1
    for at, value in enumerate(xs):
        if not side(value, x):
            continue
        if best == -1 or nearer(value, xs[best]):
            second, best = best, at
        elif second == -1:
            second = at
    return best, second


def _walked(xs: list[float], ys: list[float], x: float) -> float:
    """``TGraph::Eval`` of one ``x`` on points in any order, as ROOT walks them."""
    for at, value in enumerate(xs):
        if not (value < x or value > x):
            return ys[at]  # on a point, or a NaN on either side
    low, low2 = _nearest(xs, x, operator.lt, operator.gt)
    up, up2 = _nearest(xs, x, operator.gt, operator.lt)
    if up == -1:
        up, low = low, low2
    if low == -1:
        low, up = up, up2
    if xs[low] == xs[up]:
        return ys[low]
    return ys[up] + (x - xs[up]) * (ys[low] - ys[up]) / (xs[low] - xs[up])


def _sorted_line(xs: Any, ys: Any, points: Any) -> Any:
    """The same for points in strictly increasing ``x``, a whole array at once.

    Every neighbour ROOT's walk would find is then the one either side, and
    past either end the two nearest it, so the answer is the same to the bit.
    """
    count = len(xs)
    found = np.searchsorted(xs, points, side="left")
    up = np.clip(found, 1, count - 1)
    low = up - 1
    with np.errstate(all="ignore"):
        line = ys[up] + (points - xs[up]) * (ys[low] - ys[up]) / (xs[low] - xs[up])
    exact = (found < count) & (xs[np.minimum(found, count - 1)] == points)
    line = np.where(exact, ys[np.minimum(found, count - 1)], line)
    return np.where(np.isnan(points), ys[0], line)


def evaluate(graph: Graph, x: Any) -> Any:
    """``TGraph::Eval(x)``: straight lines between the points, and past the ends along them.

    No points gives zero and one point its own ``y``, as ROOT's does; on a
    point it is that point's ``y``, and two points at the same ``x`` give the
    first one's.
    """
    points = np.asarray(x, dtype=np.float64)
    xs, ys = graph.x, graph.y
    if len(xs) < 2:
        found = np.full(points.shape, float(ys[0]) if len(xs) else 0.0)
    elif np.all(np.diff(xs) > 0):
        found = _sorted_line(xs, ys, points)
    else:
        walk = [_walked(xs.tolist(), ys.tolist(), value) for value in points.ravel().tolist()]
        found = np.array(walk).reshape(points.shape)
    return float(found) if found.ndim == 0 else found


def integral(graph: Graph, first: int, last: int) -> float:
    """``TGraph::Integral``: the area of the polygon from point ``first`` to ``last``, closed.

    ROOT's shoelace formula, its sum taken in turn and its sign dropped;
    ``last`` below zero is the last point, and a range of fewer than two
    points has no area.
    """
    count = len(graph.x)
    first = max(first, 0)
    last = count - 1 if last < 0 else min(last, count - 1)
    if first >= last:
        return 0.0
    xs, ys = graph.x[first : last + 1], graph.y[first : last + 1]
    nx, ny = np.roll(xs, -1), np.roll(ys, -1)
    return 0.5 * abs(running(0.0, (ys + ny) * (nx - xs)))


def _coordinates(graph: Graph, axis: int) -> np.ndarray[Any, Any]:
    if axis not in (0, 1):
        raise ValueError(
            f"axis={axis} is not an axis of a graph, which has x, 0, and y, 1: ROOT's 1 and 2"
        )
    return graph.x if axis == 0 else graph.y


def mean(graph: Graph, axis: int) -> float:
    """``TGraph::GetMean``: the mean of the points' ``x`` - or, for ``axis=1``, ``y``."""
    values = _coordinates(graph, axis)
    return running(0.0, values) / len(values) if len(values) else 0.0


def rms(graph: Graph, axis: int) -> float:
    """``TGraph::GetRMS``: the spread of the points about that mean, never below zero."""
    values = _coordinates(graph, axis)
    if not len(values):
        return 0.0
    average = running(0.0, values) / len(values)
    return math.sqrt(abs(running(0.0, values * values) / len(values) - average * average))


def _reordered(values: Any, order: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """One per-point array in the new order, whatever it keeps past the last point."""
    made = np.array(values, dtype=np.float64)
    made[: len(order)] = made[order]
    return made


def sort(graph: Graph) -> None:
    """``TGraph::Sort``: the points put in increasing ``x``, their error bars with them.

    Points at the same ``x`` keep the order they had. ROOT hands its
    comparison to ``std::stable_sort`` as ``!(a > b)``, which calls such
    points each before the other, and what order they come out in is then
    the C++ library's to choose; keeping them as they were is the choice
    that does not depend on which library that was.
    """
    order = np.argsort(graph.x, kind="stable")
    core, members = graph._core, graph.members
    for name in ("fX", "fY"):
        core[name] = _reordered(core[name], order)
    for name in PER_POINT:
        if members.get(name) is not None:
            members[name] = _reordered(members[name], order)
    for name in LAYERED:
        if members.get(name) is not None:
            members[name] = [_reordered(layer, order) for layer in members[name]]
    graph.x, graph.y = graph.x[order], graph.y[order]
