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
from typing import Any

import numpy as np

__all__ = ["bin_numbers", "neville", "weights_interpolated"]

Array = np.ndarray[Any, Any]
#: Given each event's bin numbers along the interpolated variable, the weights there.
Lookup = Callable[[Array], Array]


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
        y = y + np.where(upper, c[np.clip(ns + 1, 0, n - 1), columns], d[np.clip(ns, 0, n - 1), columns])
        ns = np.where(upper, ns, ns - 1)
    return y


def bin_numbers(edges: Array, x: Array) -> Array:
    """``RooAbsBinning::binNumber``: the bin of each value, the first or last bin outside the range."""
    found = np.searchsorted(edges, x, side="right") - 1
    return np.clip(found, 0, len(edges) - 2)


def _points(edges: Array, x: Array, order: int) -> tuple[Array, Array, Array]:
    """The ``order + 1`` bins each value is interpolated from: which were wanted, which bins
    stand for them - the ones inside, mirrored, for those outside - and where they are."""
    centres = 0.5 * (edges[1:] + edges[:-1])
    nbins = len(centres)
    central = bin_numbers(edges, x)
    first = central - order // 2 - (x < centres[central]).astype(np.int64)
    wanted = first[None, :] + np.arange(order + 1)[:, None]
    over, under = wanted >= nbins, wanted < 0
    index = np.clip(np.where(over, 2 * nbins - wanted - 1, np.where(under, -wanted - 1, wanted)), 0, nbins - 1)
    where = centres[index]
    where = np.where(over, 2 * edges[-1] - where, np.where(under, 2 * edges[0] - where, where))
    return wanted, index, where


def along(edges: Array, lookup: Lookup, x: Array, order: int, cdf: bool = False) -> Array:
    """``RooDataHist::interpolateDim``: the weight at ``x`` along one variable.

    With ``cdf`` the points below the range weigh zero and those above one,
    each a hair further out than the last, as ROOT places them.
    """
    wanted, index, where = _points(edges, x, order)
    values = lookup(index)
    if cdf:
        nbins = len(edges) - 1
        over, under = wanted >= nbins, wanted < 0
        values = np.where(over, 1.0, np.where(under, 0.0, values))
        where = np.where(over, edges[-1] + 1e-10 * (wanted - nbins + 1),
                         np.where(under, edges[0] - (-wanted - 1) * 1e-10, where))  # fmt: skip
    return neville(where, values, x)


def weights_interpolated(edges: list[Array], grid: Array, points: list[Array], order: int,
                         cdf: bool = False) -> Array:  # fmt: skip
    """The interpolated weight at each event, from the weights ``grid`` - one axis per variable.

    One variable is interpolated along; two along the first in each of the
    nearest rows of the second, then along the second. More are not - as in
    ROOT, which says so and takes the bin's own weight.
    """
    if len(edges) == 1:
        return along(edges[0], lambda index: grid[index], points[0], order, cdf)
    x, y = points
    _, index, where = _points(edges[1], y, order)
    rows = np.stack([along(edges[0], lambda ix, iy=iy: grid[ix, iy[None, :]], x, order)
                     for iy in index])  # fmt: skip
    return neville(where, rows, y)
