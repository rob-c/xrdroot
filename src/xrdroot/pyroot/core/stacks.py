"""``THStack`` and ``TMultiGraph``: histograms stacked, and graphs drawn on one frame.

Each keeps the wrappers added to it, in order, and its ``._xrd`` is the
:class:`xrdroot.Stack` or :class:`xrdroot.MultiGraph` of what they stand
for, made when asked, so the graphics draw the objects as they are now.
``GetStack`` is ROOT's running sum, the first histogram at the bottom.
"""

from __future__ import annotations

import functools
import operator
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
        """``GetHistogram``: the frame, spanning every graph's points."""
        if self._histogram is None:
            from .graphs import TGraph

            points = TGraph()
            for graph in self._held:
                for at in range(graph.GetN()):
                    points.AddPoint(graph.GetPointX(at), graph.GetPointY(at))
            points.SetNameTitle(self.GetName(), self.GetTitle())
            self._histogram = points.GetHistogram()
        return self._histogram


for _cls in (THStack, TMultiGraph):
    register(_cls.__name__, factory=partial(adopt, _cls))
