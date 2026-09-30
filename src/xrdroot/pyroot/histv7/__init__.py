"""``ROOT.Experimental``: ROOT 7's histograms - ``RHist`` and its axes - and their conversion to
``TH1``.

``ROOT.Experimental.RHist['int'](axis)`` is ``ROOT::Experimental::RHist<int>``;
``Experimental.Hist.ConvertToTH1I(hist)`` makes the ``TH1I`` of the same
bins, contents and statistics, as ROOT's ``ConvertToTH1`` makes it. Other
names under ``ROOT::Experimental`` - ROOT 7's graphics, its RNTuple
writers - are refused by name.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from .axes import RBinIndex, RBinIndexRange, RRegularAxis, RVariableBinAxis
from .hist import (
    RBinWithError,
    RHist,
    RHistConcurrentFiller,
    RHistEngine,
    RHistFillContext,
    _Template,
)
from .stats import RHistStats, RWeight

__all__ = ["Experimental"]


def _converted(kind: str, hist: RHistEngine) -> Any:
    """``ConvertToTH1<kind>``: a one-axis histogram as the ``TH1`` of the same bins."""
    from ..core import hists

    if not isinstance(hist, RHistEngine):  # a histogram a frame booked: its value
        hist = hist.GetValue()
    if hist.GetNDimensions() != 1:
        raise ValueError("TH1 requires one dimension")
    axis = hist.GetAxes()[0]
    edges = np.asarray(axis.edges(), dtype=np.float64)
    made = getattr(hists, f"TH1{kind}")("", "", len(edges) - 1, edges)
    made.SetDirectory(None)
    sums, squares = hist.contents()
    shift = 0 if axis.HasFlowBins() else 1
    if squares is not None:
        made.Sumw2()
    for at, value in enumerate(sums.tolist()):
        made.SetBinContent(at + shift, value)
        if squares is not None:
            made.SetBinError(at + shift, float(np.sqrt(squares[at])))
    if isinstance(hist, RHist):
        _statistics(made, hist.GetStats())
    return made


def _statistics(made: Any, stats: RHistStats) -> None:
    """``ConvertGlobalStatistics``: the entries and the running sums, the RHist's own."""
    made.SetEntries(stats.GetNEntries())
    first = stats.sums[0]
    made.PutStats([stats.GetSumW(), stats.GetSumW2(), first[0], first[1]])


class _Hist:
    """``ROOT::Experimental::Hist``: the conversions to ``TH1``."""

    @staticmethod
    def ConvertToTH1I(hist: Any) -> Any:
        return _converted("I", hist)

    @staticmethod
    def ConvertToTH1F(hist: Any) -> Any:
        return _converted("F", hist)

    @staticmethod
    def ConvertToTH1D(hist: Any) -> Any:
        return _converted("D", hist)

    @staticmethod
    def ConvertToTH1S(hist: Any) -> Any:
        return _converted("S", hist)

    @staticmethod
    def ConvertToTH1C(hist: Any) -> Any:
        return _converted("C", hist)


class _Experimental:
    """``ROOT::Experimental``, as far as ROOT 7's histograms go."""

    RHist = _Template(RHist)
    RHistEngine = _Template(RHistEngine)
    RRegularAxis = RRegularAxis
    RVariableBinAxis = RVariableBinAxis
    RBinIndex = RBinIndex
    RBinIndexRange = RBinIndexRange
    RBinWithError = RBinWithError
    RWeight = RWeight
    RHistStats = RHistStats
    RHistConcurrentFiller = RHistConcurrentFiller
    RHistFillContext = RHistFillContext
    Hist = _Hist()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "ML":  # the machine-learning data loader, TMVA's part of the namespace
            from ..tmva import Experimental as learning

            return learning.ML
        raise UnsupportedFeatureError(f"ROOT::Experimental::{name} is not supported: of ROOT 7's "
                                      "classes, xrdroot has the histograms, RHist and its axes.")

    def __repr__(self) -> str:
        return "<namespace ROOT::Experimental>"


#: ``ROOT.Experimental``.
Experimental = _Experimental()
