"""``THStack`` and ``TMultiGraph``: histograms stacked, and graphs drawn on one frame.

Each keeps the wrappers added to it, in order, and its ``._xrd`` is the
:class:`xrdroot.Stack` or :class:`xrdroot.MultiGraph` of what they stand
for, made when asked, so the graphics draw the objects as they are now.
``GetStack`` is ROOT's running sum, the first histogram at the bottom.
"""

from __future__ import annotations

import functools
import math
import operator
import sys
from functools import partial
from typing import Any

import numpy as np

from ...buffer import Listed as _Listed
from ...stacks import MultiGraph, Stack
from .collections import TList, TObjArray
from .objects import TNamed
from .wrapping import adopt, register, unwrap, wrap

__all__ = ["THStack", "TMultiGraph"]

UNSET = -1111.0

#: The share of its span a frame leaves either side of the points when its axis is linear.
WIDEN = 0.05
#: ``TMultiGraph::GetHistogram`` pulls a logarithmic axis's low end down by this multiple of
#: the base-10 logarithm of its top-to-bottom ratio, and pushes the top up by the other.
LOG_DOWN = 0.5
LOG_UP = 0.2
#: What a logarithmic axis's bottom becomes when the points reach zero or below: this much of
#: the top, so that the ratio below can be taken at all.
LOG_FLOOR = 0.001
#: The fewest bins a multigraph's frame is booked with, however few points it spans.
LEAST_BINS = 100


def _log_axes() -> tuple[bool, bool]:
    """Whether the current pad draws its x and y axes logarithmically.

    The pad belongs to the graphics component above this one, so it is reached through the
    namespace rather than imported - as ``TAxis::UnZoom`` reaches it - and ROOT's ``gPad &&
    gPad->GetLogx()`` is false for both when no canvas has been made.
    """
    pad = getattr(sys.modules.get("xrdroot.pyroot"), "gPad", None)
    if not pad:
        return False, False
    return bool(pad.GetLogx()), bool(pad.GetLogy())


def _merged(spans: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    """The least and greatest of every range given, as ``TMultiGraph`` merges its graphs'."""
    if not spans:
        return 0.0, 0.0, 0.0, 0.0
    lows, downs, highs, ups = zip(*spans, strict=True)
    return min(lows), min(downs), max(highs), max(ups)


def _widened(low: float, high: float, log: bool) -> tuple[float, float]:
    """One axis widened as ``TMultiGraph::GetHistogram`` widens it.

    Linear, a twentieth of the span either way; logarithmic, the ends pushed apart by the
    base-10 logarithm of their ratio, which is why a log frame around a thousandfold spread
    is so much wider than the points. ROOT reads the ratio of a bottom at or below zero as
    one from a thousandth of the top.
    """
    if not log:
        span = WIDEN * (high - low)
        return low - span, high + span
    bottom = low if low > 0 else LOG_FLOOR * high
    reach = math.log10(high / bottom)
    return bottom / (1 + LOG_DOWN * reach), high * (1 + LOG_UP * reach)


class _Holder(TNamed):
    """What the two share: a name, a title, a list of what was added and each one's option."""

    HELD = ""
    XRD: Any = None

    def __init__(self, name: Any = "", title: Any = "", *rest: Any) -> None:
        super().__init__(name, title)
        self._held: list[Any] = []
        self._options: list[str] = []
        self._extremes = [UNSET, UNSET]
        self._histogram: Any = None

    def _adopted(self, xrd: Any) -> None:
        _Holder.__init__(self, xrd.name, xrd.title)
        for item in xrd:
            self.Add(wrap(item))

    @property
    def _xrd(self) -> Any:
        members: dict[str, Any] = {"TNamed": {"fName": self.GetName(), "fTitle": self.GetTitle()},
                   self.HELD: _Listed([unwrap(item) for item in self._held], self._options),
                   "fFunctions": [],
                   "fHistogram": None, "fMinimum": self._extremes[0],
                   "fMaximum": self._extremes[1]}  # fmt: skip
        return self.XRD(self.ClassName(), members)

    def Add(self, obj: Any, option: str = "") -> None:
        self._held.append(obj)
        self._options.append(str(option))

    def RecursiveRemove(self, obj: Any) -> None:
        while obj in self._held:
            at = self._held.index(obj)
            del self._held[at], self._options[at]

    def _listed(self) -> TList:
        made = TList()
        for item in self._held:
            made.Add(item)
        return made

    def SetMaximum(self, maximum: float = UNSET) -> None:
        self._extremes[1] = float(maximum)

    def SetMinimum(self, minimum: float = UNSET) -> None:
        self._extremes[0] = float(minimum)

    def Print(self, option: str = "") -> None:
        """``Print``: each thing held, printed its own way."""
        for item in self._held:
            item.Print(option)

    GetHistogram: Any

    def GetXaxis(self) -> Any:
        return self.GetHistogram().GetXaxis()

    def GetYaxis(self) -> Any:
        return self.GetHistogram().GetYaxis()

    def SetHistogram(self, h: Any) -> None:
        self._histogram = h

    def __iter__(self) -> Any:
        return iter(list(self._held))

    def __len__(self) -> int:
        return len(self._held)


class THStack(_Holder):
    """``THStack``: histograms drawn each on top of those before."""

    CLASS_TITLE = "A collection of histograms"
    HELD = "fHists"
    XRD = Stack

    def GetHists(self) -> TList:
        return self._listed()

    def GetNhists(self) -> int:
        return len(self._held)

    def GetStack(self) -> TObjArray:
        """``GetStack``: the running sums, each histogram with all those before it added."""
        made = TObjArray()
        total: Any = None
        for item in self._held:
            total = unwrap(item).copy() if total is None else total + unwrap(item)
            made.Add(wrap(total.copy()))
        return made

    def GetMaximum(self, option: str = "") -> float:
        """``GetMaximum``: the top of the stack - or with ``"nostack"`` the largest single bin;
        with ``"e"`` each bin's error above it too."""
        if self._extremes[1] != UNSET:
            return self._extremes[1]
        return self._extreme(option, True)

    def GetMinimum(self, option: str = "") -> float:
        """``GetMinimum``: the lowest bin of the stack's top - or with ``"nostack"`` of any
        histogram; with ``"e"`` each bin's error below it too."""
        if self._extremes[0] != UNSET:
            return self._extremes[0]
        return self._extreme(option, False)

    def _extreme(self, option: str, highest: bool) -> float:
        """The highest or lowest bin, of the summed stack or of each histogram, as ROOT's
        ``THStack::GetMaximum`` and ``GetMinimum`` find them; errors are the sum's."""
        if not self._held:
            return 0.0
        opt = str(option).lower()
        hists = [unwrap(item) for item in self._held]
        if "nostack" not in opt:
            hists = [functools.reduce(operator.add, hists[1:], hists[0].copy())]
        found = [_reach(hist, highest, "e" in opt) for hist in hists]
        return max(found) if highest else min(found)

    def Modified(self) -> None:
        """``Modified``: the stack is summed afresh each time it is asked of, so nothing is
        kept to throw away."""

    def GetHistogram(self) -> Any:
        """``GetHistogram``: the frame - an empty copy of the first histogram's axes."""
        if self._histogram is None and self._held:
            first = self._held[0]
            made = wrap(unwrap(first).copy(self.GetName()))
            made.Reset()
            made.SetTitle(self.GetTitle())
            self._histogram = made
        return self._histogram


def _reach(hist: Any, highest: bool, errors: bool) -> float:
    """A histogram's highest or lowest bin, reached up or down by its error if asked."""
    values = np.asarray(hist.values(), dtype=np.float64)
    if errors:
        spread = np.sqrt(np.asarray(hist.variances(), dtype=np.float64))
        values = values + spread if highest else values - spread
    return float(values.max() if highest else values.min())


class TMultiGraph(_Holder):
    """``TMultiGraph``: graphs drawn together on one frame."""

    CLASS_TITLE = "A collection of TGraph objects"
    HELD = "fGraphs"
    XRD = MultiGraph

    def GetListOfGraphs(self) -> TList:
        return self._listed()

    def Fit(
        self, f1: Any, option: str = "", goption: str = "", rxmin: float = 0.0, rxmax: float = 0.0
    ) -> Any:
        """``Fit``: one fit to the points of every graph together."""
        from .fits import fit

        self._fitted = self._xrd
        return fit(self, f1, option, goption, (rxmin, rxmax), self._fitted)

    def GetListOfFunctions(self) -> TList:
        made = TList()
        fitted = getattr(self, "_fitted", None)
        for item in fitted.functions if fitted is not None else []:
            made.Add(wrap(item))
        return made

    def GetFunction(self, name: Any) -> Any:
        fitted = getattr(self, "_fitted", None)
        found = [
            item
            for item in (fitted.functions if fitted is not None else [])
            if item.name == str(name)
        ]
        return wrap(found[0]) if found else None

    def GetHistogram(self) -> Any:
        """``GetHistogram``: the frame, spanning every graph's points.

        ROOT merges each graph's own ``ComputeRange`` - not the points of all of them in
        one graph - and pads that, which is why a logarithmic pad's frame reaches so much
        further than the points do.
        """
        if self._histogram is None:
            self._histogram = self._frame()
        return self._histogram

    def _frame(self) -> Any:
        """The frame as ROOT books it: a hundred bins at least, one per point of the graph
        with most, over the widened range of every graph's points."""
        from .graphs import _frame_histogram

        logx, logy = _log_axes()
        spans = [graph._point_range() for graph in self._held if graph.GetN() > 0]
        count = max([LEAST_BINS, *(graph.GetN() for graph in self._held)])
        low, down, high, up = _merged(spans)
        high += 1.0 if low == high else 0.0
        up += 1.0 if down == up else 0.0
        low, high = _widened(low, high, logx)
        down, up = _widened(down, up, logy)
        return _frame_histogram(self.GetName(), self.GetTitle(), count, low, high, down, up)


for _cls in (THStack, TMultiGraph):
    register(_cls.__name__, factory=partial(adopt, _cls))
