"""A binned dataset's weight between its bin centres: ``RooDataHist::weightInterpolated``.

A histogram density of interpolation order ``n`` is not a staircase: at
each point RooFit takes the ``n + 1`` bin centres nearest to it - ``n/2``
either side, one more on the side the point leans to - and evaluates the
polynomial through their weights there, by Neville's algorithm as
``RooMath::interpolate`` has it. Near an end of the range the missing
centres are the bins inside mirrored outside, so that the curve is flat at
the end, or - for a cumulative distribution - zero below and one above. In
two dimensions it interpolates along the first variable in each of the
``n + 1`` nearest rows of the second, then along the second.

Everything here is evaluated for every event at once: arrays of shape
``(n + 1, events)`` in place of RooFit's per-event buffers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from typing import TypeAlias

__all__ = ["Axis", "clone_rows", "neville", "weights_interpolated"]

#: An array of numbers, one per event - or per point, per event.
Array: TypeAlias = "np.ndarray[Any, Any]"
#: Given each event's bin numbers along the interpolated variable, the weights there.
Lookup: TypeAlias = "Callable[[Array], Array]"


def neville(xa: Array, ya: Array, x: Array) -> Array:
    """``RooMath::interpolate(xa, ya, n, x)``: the polynomial through ``(xa, ya)`` at ``x``.

    ``xa`` and ``ya`` have one row per point and one column per event; the
    algorithm is ROOT's to the order of its operations, the correction
    taken from above or below the table as the nearest point dictates.
    """
    n = len(xa)
    ns = np.argmin(np.abs(x[None, :] - xa), axis=0)
    columns = np.arange(len(x))
    c, d = ya.copy(), ya.copy()
    y = ya[ns, columns]
    ns = ns - 1
    for m in range(1, n):
        ho, hp = xa[: n - m] - x, xa[m:] - x
        den = (c[1 : n - m + 1] - d[: n - m]) / (ho - hp)
        d[: n - m], c[: n - m] = hp * den, ho * den
        upper = 2 * (ns + 1) < n - m
        y = y + np.where(
            upper, c[np.clip(ns + 1, 0, n - 1), columns], d[np.clip(ns, 0, n - 1), columns]
        )
        ns = np.where(upper, ns, ns - 1)
    return y


class Axis:
    """A binning as ROOT computes with it: a uniform one by its width, any other by its boundaries.

    ``RooUniformBinning`` finds a value's bin as ``int((x - low) / width)``
    and puts a centre at ``low + (i + 0.5) * width``; a value on a centre or
    a boundary falls on the side that arithmetic says, which is what decides
    which bins an interpolation takes - so it is that arithmetic, not
    boundaries it would round differently.
    """

    def __init__(self, edges: Any, uniform: bool) -> None:
        self.edges = np.asarray(edges, dtype=np.float64)
        self.uniform = bool(uniform)
        self.low, self.high = float(self.edges[0]), float(self.edges[-1])
        self.count = len(self.edges) - 1
        self.width = (self.high - self.low) / self.count

    @classmethod
    def of(cls, binning: Any) -> Axis:
        return cls(binning.array(), binning.isUniform())

    def numbers(self, x: Array) -> Array:
        """``binNumber``: the bin of each value, the first or last bin for one outside the range."""
        if self.uniform:
            found = np.trunc((x - self.low) / self.width)
        else:
            found = np.searchsorted(self.edges, x, side="right") - 1
        return np.clip(np.nan_to_num(found), 0, self.count - 1).astype(np.int64)

    def centres(self, index: Array) -> Array:
        if self.uniform:
            return self.low + (index + 0.5) * self.width
        return 0.5 * (self.edges[index] + self.edges[index + 1])

    def widths(self) -> Array:
        return np.full(self.count, self.width) if self.uniform else np.diff(self.edges)


def clone_rows(var: Any, low: float, high: float) -> Axis:
    """``var``'s own binning moved to ``[low, high]``: what a density's copy of it has.

    A histogram density keeps copies of its variables with their ranges set
    to the histogram's, so a copy's binning is the variable's number of bins
    over the histogram's range - or, for a binning of any spacing, its
    boundaries inside that range.
    """
    binning = var.getBinning()
    if binning.isUniform():
        return Axis(np.linspace(low, high, binning.numBins() + 1), True)
    inside = [e for e in binning.array() if low < e < high]
    return Axis([low, *inside, high], False)


def _points(axis: Axis, x: Array, order: int) -> tuple[Array, Array, Array]:
    """The ``order + 1`` bins each value is interpolated from: which were wanted, which bins
    stand for them - the ones inside, mirrored, for those outside - and where they are."""
    nbins = axis.count
    central = axis.numbers(x)
    first = central - order // 2 - (x < axis.centres(central)).astype(np.int64)
    wanted = first[None, :] + np.arange(order + 1)[:, None]
    over, under = wanted >= nbins, wanted < 0
    index = np.clip(
        np.where(over, 2 * nbins - wanted - 1, np.where(under, -wanted - 1, wanted)), 0, nbins - 1
    )
    where = axis.centres(index)
    where = np.where(over, 2 * axis.high - where, np.where(under, 2 * axis.low - where, where))
    return wanted, index, where


def along(axis: Axis, lookup: Lookup, x: Array, order: int, cdf: bool = False) -> Array:
    """``RooDataHist::interpolateDim``: the weight at ``x`` along one variable.

    With ``cdf`` the points below the range weigh zero and those above one,
    each a hair further out than the last, as ROOT places them.
    """
    wanted, index, where = _points(axis, x, order)
    values = lookup(index)
    if cdf:
        nbins = axis.count
        over, under = wanted >= nbins, wanted < 0
        values = np.where(over, 1.0, np.where(under, 0.0, values))
        where = np.where(over, axis.high + 1e-10 * (wanted - nbins + 1),
                         np.where(under, axis.low - (-wanted - 1) * 1e-10, where))  # fmt: skip
    return neville(where, values, x)


def weights_interpolated(axes: list[Axis], grid: Array, points: list[Array], order: int,
                         cdf: bool = False, rows: Axis | None = None) -> Array:  # fmt: skip
    """The interpolated weight at each event, from the weights ``grid`` - one axis per variable.

    One variable is interpolated along; two along the first in each of the
    nearest rows of the second, then along the second. More are not - as in
    ROOT, which says so and takes the bin's own weight.

    ROOT finds those rows - and where they are - in the second variable's
    binning *as the density's copy of it has it*, ``rows``, which need not
    be the histogram's: it steps that many rows from the histogram's own
    bin of the point. With the two the same, as they usually are, this is
    the plain interpolation; with them different it is ROOT's, which is not.
    """
    if len(axes) == 1:
        return along(axes[0], lambda index: grid[index], points[0], order, cdf)
    x, y = points
    rows = axes[1] if rows is None else rows
    _, index, where = _points(rows, y, order)
    step = index - rows.numbers(y)[None, :] + axes[1].numbers(y)[None, :]
    flat = grid.reshape(-1)
    ny = grid.shape[1]

    def row(iy: Array) -> Lookup:
        return lambda ix: flat[np.clip(ix * ny + iy[None, :], 0, flat.size - 1)]

    found = np.stack([along(axes[0], row(iy), x, order) for iy in step])
    return neville(where, found, y)
