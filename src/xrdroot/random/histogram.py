"""``TH1::GetRandom``: a number drawn from the shape of a histogram, as ROOT draws it.

ROOT sums the bins in turn into a cumulative integral - ``ComputeIntegral``,
bin contents times widths if asked for ``"width"`` - divides it through by
its last entry, and then for each ``Rndm()`` finds the bin with
``TMath::BinarySearch`` and places the number within it on a straight line
between the bin's edges. Each step here is ROOT's: the sum is added up in
order rather than pairwise, the search returns the first of equal entries
as ``std::lower_bound`` does, and a draw landing exactly on an entry gets
the bin's low edge untouched. An evenly binned axis has its edges and widths
computed from its ends as ``TAxis`` computes them, rather than read from a
list, because that is where the last bit of an edge comes from in ROOT.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["bins", "cumulative", "inverse"]


def bins(edges: Any, nbins: int) -> tuple[Any, Any]:
    """The low edge and the width of every bin, from an axis, its edges or its ends.

    ``edges`` is an :class:`~xrdroot.Axis`, the ``nbins + 1`` edges of a
    variably binned axis, or the ``(low, high)`` ends of an evenly binned
    one (for a single bin the two readings agree).
    """
    if hasattr(edges, "even"):
        edges = (edges.low, edges.high) if edges.even else edges.edges()
    edges = np.asarray(edges, dtype=np.float64).ravel()
    if len(edges) == nbins + 1:
        return edges[:-1], np.diff(edges)
    if len(edges) != 2:
        raise ValueError(
            f"A histogram of {nbins} bins needs {nbins + 1} edges, or the two ends of an even "
            f"axis, but {len(edges)} numbers were given."
        )
    width = (edges[1] - edges[0]) / float(nbins)
    return edges[0] + np.arange(nbins) * width, np.full(nbins, width)


def cumulative(contents: Any, widths: Any | None = None) -> Any | None:
    """``ComputeIntegral``: the running sum of the bins over its total, or ``None`` if that is 0.

    ``widths`` weights every bin by its width, which is ROOT's ``"width"``
    option. ROOT marks a histogram with a negative bin by making the integral
    NaN and then draws from whatever that leaves; that is refused here instead,
    by name, and so is a bin that is NaN.
    """
    y = np.asarray(contents, dtype=np.float64)
    if widths is not None:
        y = y * widths
    if not np.all(y >= 0):
        raise ValueError(
            "A distribution to draw from cannot have a negative or NaN bin: ROOT's GetRandom "
            "gives NaN for such a histogram."
        )
    total = np.add.accumulate(np.concatenate([[0.0], y]))
    if total[-1] == 0:
        return None
    total[1:] = total[1:] / total[-1]
    return total


def inverse(draws: Any, integral: Any, lows: Any, widths: Any) -> Any:
    """Where each draw falls, found and interpolated the way ``TH1::GetRandom`` does."""
    nbins = len(lows)
    lo = np.searchsorted(integral[:nbins], draws, side="left")
    exact = (lo < nbins) & (integral[np.minimum(lo, nbins - 1)] == draws)
    ibin = np.where(exact, lo, lo - 1)
    below = integral[ibin]
    with np.errstate(divide="ignore", invalid="ignore"):
        step = widths[ibin] * (draws - below) / (integral[ibin + 1] - below)
    return np.where(draws > below, lows[ibin] + step, lows[ibin])
