"""``ROOT::Experimental::Hist::ConvertToTH1<X>`` and ``Hist::Internal::ConvertAxis``: a ROOT 7
histogram of one dimension as the ``TH1`` of the same bins.

A regular axis becomes a fixed-bin ``TAxis``, a variable one a ``TAxis`` of
its edges, a categorical one ``[0, n)`` with a label per bin. The bins go
across with the ``TH1``'s numbering, its flow bins left empty when the
axis has none; an ``RBinWithError`` bin brings its sum of squares as the
``TH1``'s ``Sumw2``. An ``RHist``'s entries and running sums come along -
but those of a categorical dimension, which has none - unless they are
tainted; an engine has none to bring, as in ROOT.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .axes import RCategoricalAxis, RRegularAxis
from .contents import RBinWithError
from .hist import RHist, RHistEngine
from .variant import RAxisVariant

__all__ = ["ConvertAxis", "converted"]


def _shape(axis: Any) -> tuple[Any, ...]:
    """What ``TAxis::Set`` (and a ``TH1``'s constructor) takes for ``axis``."""
    count = axis.GetNNormalBins()
    if isinstance(axis, RRegularAxis):
        return count, axis.GetLow(), axis.GetHigh()
    if isinstance(axis, RCategoricalAxis):
        return count, 0.0, float(count)
    return count, np.asarray(axis.edges(), dtype=np.float64)


def _label(dst: Any, axis: Any) -> None:
    """A categorical axis's names, as the labels of ``dst``'s bins."""
    if isinstance(axis, RCategoricalAxis):
        for at, name in enumerate(axis.GetCategories()):
            dst.SetBinLabel(at + 1, str(name))


def ConvertAxis(dst: Any, src: Any) -> None:
    """``ConvertAxis(dst, src)``: ``dst`` made the ``TAxis`` of the axis ``src``."""
    axis = RAxisVariant(src).axis
    dst.Set(*_shape(axis))
    _label(dst, axis)


def _bins(engine: RHistEngine, axis: Any) -> list[tuple[int, Any]]:
    """Each bin of a one-dimensional engine, with its ``TH1`` bin number: a ``TH1``'s first
    bin is the underflow, which only a regular or variable axis has."""
    shift = 0 if getattr(axis, "HasFlowBins", lambda: False)() else 1
    return [(at + shift, bin) for at, bin in enumerate(engine.contents())]


def _statistics(made: Any, hist: RHist) -> None:
    """``ConvertGlobalStatistics``: the entries and the running sums, unless tainted."""
    stats = hist.GetStats()
    if stats.IsTainted():
        return
    made.SetEntries(stats.GetNEntries())
    sums = [stats.GetSumW(), stats.GetSumW2(), 0.0, 0.0]
    if stats.IsEnabled(0):
        first = stats.GetDimensionStats(0)
        sums[2:] = [first.fSumWX, first.fSumWX2]
    made.PutStats(sums)


def _filled(made: Any, bins: list[tuple[int, Any]]) -> None:
    """The bins put in, an ``RBinWithError``'s sum of squares as the ``TH1``'s ``Sumw2``."""
    errors = any(isinstance(bin, RBinWithError) for _, bin in bins)
    if errors:
        made.Sumw2()
    for at, bin in bins:
        made.SetBinContent(at, float(bin) if errors else bin)
        if errors:
            made.SetBinError(at, float(np.sqrt(bin.fSum2)))


def converted(kind: str, hist: Any) -> Any:
    """``ConvertToTH1<kind>(hist)``: a histogram, or an engine, as the ``TH1`` of its bins."""
    from ..core import hists

    if not isinstance(hist, (RHist, RHistEngine)):  # a histogram a frame booked: its value
        hist = hist.GetValue()
    engine = hist.GetEngine() if isinstance(hist, RHist) else hist
    if engine.GetNDimensions() != 1:
        raise ValueError("TH1 requires one dimension")
    axis = RAxisVariant(engine.GetAxes()[0]).axis
    made = getattr(hists, f"TH1{kind}")("", "", *_shape(axis))
    made.SetDirectory(None)
    _label(made.GetXaxis(), axis)
    _filled(made, _bins(engine, axis))
    made.SetEntries(0)
    if isinstance(hist, RHist):
        _statistics(made, hist)
    return made
