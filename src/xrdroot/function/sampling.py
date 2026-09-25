"""``TF1::GetRandom``: numbers distributed as a function, drawn the way ROOT draws them.

ROOT does not invert a function. ``TF1::ComputeCdfTable`` cuts the range
into ``fNpx`` equal intervals - of ``log10(x)`` when the range starts above
zero and spans more than ``fNpx`` decades' worth of ratio - integrates the
function over each, adds them up in order and divides by the total. Over
each interval it takes the cumulative integral to be a parabola in ``x``,
fixed by the integral over the whole interval and over its first half, and
for every ``Rndm()`` it finds the interval with ``TMath::BinarySearch`` and
solves that parabola. Every number here is that: the same table, the same
parabolas, the same search, one ``Rndm()`` per number - so the same seed
gives ROOT's numbers.

``GetRandom(xmin, xmax)`` draws ``Uniform(pmin, pmax)`` between the table's
entries either side of the range, inverts it the same way, and draws again
whenever the answer falls outside; each try is one ``Rndm()``, and the tries
are followed through a run of draws at once, keeping the ones ROOT's loop
would keep and leaving the rest for whatever is drawn next.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from ..errors import UnsupportedFeatureError

__all__ = ["Table", "table", "draw", "invert"]

#: The size below which a parabola's curvature is taken as none: ROOT's 1e-8.
FLAT = 1e-8


class Table(NamedTuple):
    """``fIntegral``, ``fAlpha``, ``fBeta`` and ``fGamma``, and how the axis was cut."""

    integral: Any
    alpha: Any
    beta: Any
    gamma: Any
    logarithmic: bool
    low: float
    high: float


def _points(low: float, high: float, npx: int, logarithmic: bool) -> tuple[Any, float]:
    """The ``fNpx + 1`` points the table is cut at, in ``x`` or in ``log10(x)``, and the step."""
    start, stop = (np.log10(low), np.log10(high)) if logarithmic else (low, high)
    dx = (stop - start) / npx
    points = start + np.arange(npx + 1) * dx
    points[npx] = stop
    return points, float(dx)


def table(
    integrate: Callable[[float, float, float], float], low: float, high: float, npx: int
) -> Table:
    """``ComputeCdfTable``: the cumulative integral at ``npx`` points, and a parabola per step.

    ``integrate(a, b, epsrel)`` is ``TF1::Integral``. A step whose integral
    is negative is taken at its size, as ROOT takes it; a function whose
    integral over the whole range is zero has no table, and is refused.
    """
    logarithmic = low > 0 and high / low > npx
    points, dx = _points(low, high, npx, logarithmic)
    ends = 10.0**points if logarithmic else points
    steps = np.abs([integrate(float(ends[i]), float(ends[i + 1]), 0.0) for i in range(npx)])
    cumulative = np.add.accumulate(np.concatenate(([0.0], steps)))
    total = float(cumulative[npx])
    if total == 0:
        raise ValueError(
            "The function integrates to zero over its range, so there is no distribution to "
            "draw from: ROOT's GetRandom gives NaN for it."
        )
    cumulative[1:] = cumulative[1:] / total
    halves = points[:npx] + 0.5 * dx
    half_ends = 10.0**halves if logarithmic else halves
    starts = ends[:npx]
    firsts = np.array([integrate(float(a), float(b), 0.0) for a, b in zip(starts, half_ends)])
    r2 = cumulative[1:] - cumulative[:npx]
    r3 = 2 * r2 - 4 * (firsts / total)
    gamma = np.where(np.abs(r3) > FLAT, r3 / (dx * dx), 0.0)
    beta = r2 / dx - gamma * dx
    return Table(cumulative, points[:npx], beta, 2 * gamma, logarithmic, low, high)


def _search(integral: Any, npx: int, r: Any) -> Any:
    """``TMath::BinarySearch(npx, integral, r)``: the first equal entry, else the one below."""
    at = np.searchsorted(integral[:npx], r, side="left")
    exact = (at < npx) & (integral[np.minimum(at, npx - 1)] == r)
    return np.where(exact, at, at - 1)


def invert(found: Table, r: Any) -> Any:
    """The parabola of the step each ``r`` falls in, solved for ``x``."""
    npx = len(found.alpha)
    step = _search(found.integral, npx, r)
    rr = r - found.integral[step]
    beta, gamma = found.beta[step], found.gamma[step]
    curved = gamma != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        root = (-beta + np.sqrt(beta * beta + 2 * gamma * rr)) / np.where(curved, gamma, 1.0)
        yy = np.where(curved, root, rr / beta)
    return found.alpha[step] + yy


def _span(found: Table, low: float, high: float) -> tuple[float, float]:
    """``pmin`` and ``pmax``: the table's entries either side of ``[low, high]``."""
    if found.logarithmic:
        raise UnsupportedFeatureError(
            "ROOT's GetRandom(xmin, xmax) does not undo the logarithm its table was built in, "
            "so for a function over a range this wide it returns log10(x) rather than x; that "
            "is not a number to give back, and a range is refused for such a function."
        )
    npx = len(found.alpha)
    dx = (found.high - found.low) / npx
    first = max(int((low - found.low) / dx), 0)
    last = min(int((high - found.low) / dx) + 2, npx)
    return float(found.integral[first]), float(found.integral[last])


def draw(found: Table, n: int | None, rng: Any, span: Any) -> Any:
    """``n`` numbers from the table - one if ``n`` is ``None`` - with ``rng``, or ``gRandom``."""
    if rng is None:
        from ..random import gRandom as rng
    count = 1 if n is None else int(n)
    if span is None:
        values = invert(found, np.asarray(rng.rndm(count), dtype=np.float64))
        if found.logarithmic:
            values = np.power(10.0, values)
    else:
        values = _inside(found, rng, count, (float(span[0]), float(span[1])))
    return float(values[0]) if n is None else values


def _inside(found: Table, rng: Any, count: int, span: tuple[float, float]) -> Any:
    """``GetRandom(xmin, xmax)``: ``Uniform(pmin, pmax)`` inverted, drawn again until inside."""
    low, high = span
    pmin, pmax = _span(found, low, high)

    def decode(draws: Any, limit: int) -> tuple[list[Any], int]:
        values = invert(found, pmin + (pmax - pmin) * draws)
        kept = np.flatnonzero((values >= low) & (values <= high))[:limit]
        return [values[kept]], int(kept[-1] + 1) if len(kept) else 0

    (values,) = rng._sequential(count, decode, 1, 1.5)
    return values
