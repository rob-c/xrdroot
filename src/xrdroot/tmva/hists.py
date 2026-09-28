"""The histograms TMVA books, as :class:`xrdroot.Histogram` objects ready to be written.

TMVA works with ``TH1F`` and ``TH1D`` throughout - its reference
histograms, its PDFs, the distributions of every classifier's output - and
writes them into the output file. These helpers book them the way TMVA's
constructors do, fill them with its weights, and set their bins directly the
way ``SetBinContent`` does, so that what is written is what TMVA writes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..hist import Histogram

__all__ = ["book", "book_edges", "bins", "centers", "filled", "find_bin", "renamed", "set_bins"]


def book(name: str, title: str, nbins: int, low: float, high: float, kind: str = "F") -> Histogram:
    """``TH1F(name, title, nbins, low, high)``, or its ``TH1D`` for ``kind="D"``."""
    return Histogram.book(name, (int(nbins), float(low), float(high)), title=title, kind=kind)


def book_edges(name: str, title: str, edges: Any, kind: str = "F") -> Histogram:
    """A histogram of variable bins, from every edge."""
    return Histogram.book(name, [float(edge) for edge in edges], title=title, kind=kind)


def filled(
    name: str,
    title: str,
    spec: tuple[int, float, float],
    values: Any,
    weights: Any = None,
    kind: str = "F",
) -> Histogram:
    """A histogram booked and filled, entry by entry as ``Fill(x, w)`` would fill it."""
    made = book(name, title, *spec, kind=kind)
    if len(values):
        made.fill(np.asarray(values, dtype=np.float64), weight=weights)
    return made


def renamed(histogram: Histogram, name: str, title: str | None = None) -> Histogram:
    """The histogram called ``name``, and titled ``title`` - its name, unless said."""
    named = histogram._core["TNamed"]
    named["fName"] = name
    named["fTitle"] = name if title is None else title
    return histogram


def bins(histogram: Histogram) -> Any:
    """Every bin's content, the flow bins at the ends, as doubles."""
    return np.asarray(histogram.values(flow=True), dtype=np.float64)


def centers(histogram: Histogram) -> Any:
    """``GetBinCenter`` of every bin, the flow bins included."""
    return histogram.axes[0].root_centers()


def find_bin(histogram: Histogram, x: Any) -> Any:
    """``FindBin``: ROOT's bin number of each ``x``."""
    return histogram.axes[0].find_bin(x)


def set_bins(
    histogram: Histogram, contents: Any, errors2: Any = None, entries: float | None = None
) -> Histogram:
    """The bins set - flow bins included - as a loop of ``SetBinContent`` sets them.

    The running sums are zeroed, as ``SetBinContent`` zeroes them, so that
    whoever reads the histogram works its statistics out from the bins.
    """
    cells = histogram._cells()
    cells[:] = np.asarray(contents, dtype=np.float64)
    core = histogram._core
    if errors2 is not None:
        core["fSumw2"] = np.asarray(errors2, dtype=np.float64).copy()
    core["fEntries"] = float(len(cells) - 2 if entries is None else entries)
    core["fTsumw"] = 0.0
    return histogram
