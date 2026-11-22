"""``TAxis``: one axis of a histogram, over the members it is written as.

A ``TAxis`` here holds the histogram's own axis members - ``fNbins``,
``fXmin``, ``fXbins``, the ``TAttAxis`` - so ``h.GetXaxis().SetTitle("pt")``
changes what the histogram writes. A range set by ``SetRange`` or
``SetRangeUser`` is ``fFirst``, ``fLast`` and the ``kAxisRange`` bit, as ROOT
keeps it, and the statistics and extremes of the histogram honour it.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from ...fillrandom import AXIS_RANGE
from .messages import message
from .objects import TNamed, renamed

__all__ = ["TAxis", "TAttAxis"]

#: ``TAttAxis``'s members and what a fresh axis has - ROOT's defaults.
AXIS_ATTRIBUTES = {
    "fNdivisions": 510, "fAxisColor": 1, "fLabelColor": 1, "fLabelFont": 42,
    "fLabelOffset": 0.005, "fLabelSize": 0.035, "fTickLength": 0.03, "fTitleOffset": 1.0,
    "fTitleSize": 0.035, "fTitleColor": 1, "fTitleFont": 42,
}  # fmt: skip

#: ``TAxis``'s bits, in its ``fBits``.
CENTER_TITLE = 1 << 12
ROTATE_TITLE = 1 << 15
CENTER_LABELS = 1 << 14
NO_EXPONENT = 1 << 17
MORE_LOG_LABELS = 1 << 16
#: ``TAxis``'s bits in its ``fBits2``: of categories, able to grow, never of categories.
ALPHANUMERIC, CAN_EXTEND, NOT_ALPHA = 1, 2, 4


def _axis_setter(member: str) -> Any:
    def setter(self: Any, value: Any = AXIS_ATTRIBUTES[member], *rest: Any) -> None:
        kind = float if isinstance(AXIS_ATTRIBUTES[member], float) else int
        self._att()[member] = kind(value)

    setter.__doc__ = f"``Set{member[1:]}``: ``{member}`` of the axis's ``TAttAxis``."
    return setter


def _axis_getter(member: str) -> Any:
    def getter(self: Any) -> Any:
        return self._att()[member]

    getter.__doc__ = f"``Get{member[1:]}``."
    return getter


class TAttAxis:
    """``TAttAxis``: how an axis is drawn - divisions, labels, ticks and title."""

    def _att(self) -> dict[str, Any]:
        home: dict[str, Any] = self._row.setdefault("TAttAxis", {})  # type: ignore[attr-defined]
        for member, value in AXIS_ATTRIBUTES.items():
            home.setdefault(member, value)
        return home

    def SetNdivisions(self, n: int = 510, optim: bool = True) -> None:
        self._att()["fNdivisions"] = int(n) if optim else -abs(int(n))

    def ResetAttAxis(self, option: str = "") -> None:
        self._att().update(AXIS_ATTRIBUTES)


for _member in AXIS_ATTRIBUTES:
    if _member != "fNdivisions":
        setattr(TAttAxis, f"Set{_member[1:]}", _axis_setter(_member))
    setattr(TAttAxis, f"Get{_member[1:]}", _axis_getter(_member))
#: ``SetTickSize``: ROOT's other name for ``SetTickLength``, as ``TAttAxis`` has it.
TAttAxis.SetTickSize = TAttAxis.SetTickLength  # type: ignore[attr-defined]


def _checked(spec: Any) -> Any:
    """``TAxis::Set``: the axis's limits as ROOT checks them, and the axis ROOT leaves.

    ROOT reports and returns rather than raising - ``Error in <TAxis::Set>: Axis limits need
    to be finite numbers``, ``Error in <TAxis::Set>: bin edges must be in increasing order``
    - and the axis it leaves is the default one bin from 0 to 1. ROOT's own
    ``TAxis.EqualBinEdges`` case expects exactly those two messages.
    """
    from .messages import Error

    if isinstance(spec, tuple):
        if np.isfinite(spec[1]) and np.isfinite(spec[2]):
            return spec
        Error("TAxis::Set", "Axis limits need to be finite numbers")
        return (1, 0.0, 1.0)
    edges = [float(one) for one in spec]
    if not any(not later > earlier for earlier, later in itertools.pairwise(edges)):
        return spec
    Error("TAxis::Set", "bin edges must be in increasing order")
    return (1, 0.0, 1.0)


class TAxis(TNamed, TAttAxis):
    """``TAxis``: the bins of one axis, where they are, and what the axis is called."""

    CLASS_TITLE = "Axis class"

    #: ``TAxis::EStatusBits``: an axis's status word - which it shares with ``TAxis3D`` and
    #: its siblings, so that a macro may ask ``axis.TestBit(TAxis.kAxisRange)``.
    kDecimals = 1 << 7
    kTickPlus = 1 << 9
    kTickMinus = 1 << 10
    kAxisRange = 1 << 11
    kCenterTitle = 1 << 12
    kCenterLabels = 1 << 14
    kRotateTitle = 1 << 15
    kPalette = 1 << 16
    kNoExponent = 1 << 17
    kLabelsHori = 1 << 18
    kLabelsVert = 1 << 19
    kLabelsDown = 1 << 20
    kLabelsUp = 1 << 21
    kIsInteger = 1 << 22
    kMoreLogLabels = 1 << 23

    def __init__(self, *args: Any) -> None:
        """``TAxis(nbins, xlow, xup)`` or ``TAxis(nbins, edges)``, standing alone."""
        from ...booking import axis_members, binning

        super().__init__()
        spec: Any = (int(args[0]), float(args[1]), float(args[2])) if len(args) == 3 else None
        if spec is None and len(args) == 1 and not isinstance(args[0], (int, np.integer)):
            # ``TAxis(edges)``: ROOT's constructor from a vector of bin edges, which a macro
            # writes as ``TAxis _({1, 1})``.
            spec = [float(one) for one in args[0]]
        elif spec is None:
            spec = list(args[1])[: int(args[0]) + 1] if len(args) == 2 else (1, 0.0, 1.0)
        made = binning(_checked(spec))
        self._row = axis_members("xaxis", made.nbins, made.low, made.high, made.stored)
        self._owner: Any = None

    @classmethod
    def _of(cls, row: dict[str, Any], owner: Any) -> TAxis:
        made = cls.__new__(cls)
        TNamed.__init__(made)
        made._row, made._owner = row, owner
        return made

    # -- naming -----------------------------------------------------------------------------

    def GetName(self) -> str:
        return str(self._row["TNamed"]["fName"])

    def SetName(self, name: Any) -> None:
        renamed(self)
        self._row["TNamed"]["fName"] = str(name)

    def GetTitle(self) -> str:
        return str(self._row["TNamed"]["fTitle"])

    def SetTitle(self, title: Any = "") -> None:
        self._row["TNamed"]["fTitle"] = str(title)

    def _bitword(self) -> int:
        return int(self._row["TNamed"].get("fBits", 0) or 0)

    def _store_bits(self, bits: int) -> None:
        self._row["TNamed"]["fBits"] = int(bits)

    def CenterTitle(self, center: bool = True) -> None:
        self.SetBit(CENTER_TITLE, center)

    def RotateTitle(self, rotate: bool = True) -> None:
        self.SetBit(ROTATE_TITLE, rotate)

    def CenterLabels(self, center: bool = True) -> None:
        self.SetBit(CENTER_LABELS, center)

    def SetNoExponent(self, noExponent: bool = True) -> None:
        self.SetBit(NO_EXPONENT, noExponent)

    def SetMoreLogLabels(self, more: bool = True) -> None:
        self.SetBit(MORE_LOG_LABELS, more)

    def GetCenterTitle(self) -> bool:
        return self.TestBit(CENTER_TITLE)

    # -- the bins -------------------------------------------------------------------------

    def GetNbins(self) -> int:
        return int(self._row["fNbins"])

    def GetXmin(self) -> float:
        return float(self._row["fXmin"])

    def GetXmax(self) -> float:
        return float(self._row["fXmax"])

    def IsVariableBinSize(self) -> bool:
        return len(self._row["fXbins"]) == self.GetNbins() + 1

    def _edges(self) -> np.ndarray[Any, Any]:
        """Every edge of an axis binned unevenly, as it keeps them."""
        return np.asarray(self._row["fXbins"], dtype=np.float64)

    def GetXbins(self) -> Any:
        """``GetXbins``: the edges of an axis binned unevenly, a ``TArrayD`` - empty for one
        binned evenly."""
        from ..vectors import TArrayD

        return TArrayD(np.asarray(self._row["fXbins"], dtype=np.float64))

    def GetBinLowEdge(self, bin: int) -> float:
        """``GetBinLowEdge``: as ROOT has it for the flow bins too, a width past each end."""
        nbins = self.GetNbins()
        if not self.IsVariableBinSize() or not 1 <= bin <= nbins + 1:
            width = (self.GetXmax() - self.GetXmin()) / nbins
            return self.GetXmin() + (bin - 1) * width
        return float(self._edges()[bin - 1])

    def GetBinWidth(self, bin: int) -> float:
        nbins = self.GetNbins()
        if not self.IsVariableBinSize():
            return (self.GetXmax() - self.GetXmin()) / nbins
        edges = self._edges()
        at = min(max(int(bin), 1), nbins)
        return float(edges[at] - edges[at - 1])

    def GetBinUpEdge(self, bin: int) -> float:
        if not self.IsVariableBinSize() or not 1 <= bin <= self.GetNbins():
            width = (self.GetXmax() - self.GetXmin()) / self.GetNbins()
            return self.GetXmin() + bin * width
        return float(self._edges()[bin])

    def GetBinCenter(self, bin: int) -> float:
        """``GetBinCenter``: the middle of the bin, computed as ROOT computes it."""
        if not self.IsVariableBinSize() or not 1 <= bin <= self.GetNbins():
            width = (self.GetXmax() - self.GetXmin()) / self.GetNbins()
            return self.GetXmin() + (bin - 0.5) * width  # as ROOT 6.40 multiplies it out
        edges = self._edges()
        return float(edges[bin - 1] + 0.5 * (edges[bin] - edges[bin - 1]))

    def GetBinCenterLog(self, bin: int) -> float:
        low, high = self.GetBinLowEdge(bin), self.GetBinUpEdge(bin)
        return math.sqrt(low * high) if low > 0 and high > 0 else self.GetBinCenter(bin)

    def GetLowEdge(self, edges: Any) -> None:
        for at in range(self.GetNbins()):
            edges[at] = self.GetBinLowEdge(at + 1)

    def GetCenter(self, centers: Any) -> None:
        for at in range(self.GetNbins()):
            centers[at] = self.GetBinCenter(at + 1)

    def FindBin(self, x: Any) -> int:
        """``FindBin``: 0 below, ``nbins + 1`` above - or the bin of a label, by name."""
        if isinstance(x, str):
            return self.FindFixBin(x)
        if x < self.GetXmin():
            return 0
        if not x < self.GetXmax():
            return self.GetNbins() + 1
        if self.IsVariableBinSize():
            return int(np.searchsorted(self._edges(), x, side="right"))
        return 1 + int(self.GetNbins() * (x - self.GetXmin()) / (self.GetXmax() - self.GetXmin()))

    def FindFixBin(self, x: Any) -> int:
        if isinstance(x, str):
            found = [int(bin) for bin, label in self._labels().items() if label == x]
            return found[0] if found else -1
        return self.FindBin(x)

    # -- the range drawn -----------------------------------------------------------------

    def GetFirst(self) -> int:
        return int(self._row.get("fFirst", 0)) if self.TestBit(AXIS_RANGE) else 1

    def GetLast(self) -> int:
        return int(self._row.get("fLast", 0)) if self.TestBit(AXIS_RANGE) else self.GetNbins()

    def SetRange(self, first: int = 0, last: int = 0) -> None:
        """``SetRange``: bins ``first`` to ``last``, flow bins allowed; ``(0, 0)`` is every bin."""
        cells = self.GetNbins() + 1
        reset = last < first or (first < 0 and last < 0) or (first > cells and last > cells)
        if reset or (first == 0 and last == 0):
            self._row["fFirst"], self._row["fLast"] = 1, self.GetNbins()
            self.SetBit(AXIS_RANGE, False)
            return
        self._row["fFirst"], self._row["fLast"] = max(int(first), 0), min(int(last), cells)
        self.SetBit(AXIS_RANGE, True)

    def SetRangeUser(self, ufirst: float, ulast: float) -> None:
        """``SetRangeUser``: the bins holding ``ufirst`` and ``ulast`` - or, on the axis
        a histogram's contents are drawn along, its minimum and maximum."""
        if self._owner is not None and self.GetName() == self._owner._content_axis():
            self._owner.SetMinimum(ufirst)
            self._owner.SetMaximum(ulast)
            return
        first, last = self.FindFixBin(ufirst), self.FindFixBin(ulast)
        # ROOT's guards against rounding: a bin whose edge is the very value is not in.
        first += int(self.GetBinUpEdge(first) <= ufirst)
        last -= int(self.GetBinLowEdge(last) >= ulast)
        self.SetRange(first, last)

    def UnZoom(self) -> None:
        """``UnZoom``: every bin again - once the axis is drawn, as ROOT needs a pad for it."""
        import sys

        if not getattr(sys.modules.get("xrdroot.pyroot"), "__dict__", {}).get("gPad"):
            message(
                "Warning",
                "TAxis::UnZoom",
                "Cannot UnZoom if gPad does not exist. Did you mean to draw the TAxis first?",
            )
            return
        self.SetRange(0, 0)

    def GetRangeActual(self) -> bool:
        return self.TestBit(AXIS_RANGE)

    def SetLimits(self, xmin: float, xmax: float) -> None:
        self._row["fXmin"], self._row["fXmax"] = float(xmin), float(xmax)

    def Set(self, nbins: int, xlow: Any, xup: Any = None) -> None:
        """``Set(n, low, high)`` or ``Set(n, edges)``: the binning, changed."""
        from ...booking import binning

        spec = (int(nbins), float(xlow), float(xup)) if xup is not None else list(xlow)[: nbins + 1]
        made = binning(spec)
        self._row.update(fNbins=made.nbins, fXmin=made.low, fXmax=made.high, fXbins=made.stored)

    # -- labels and time ---------------------------------------------------------------------

    def _labels(self) -> dict[str, str]:
        """The labels by bin, the bin numbers as text: the members are walked by name."""
        labels: dict[str, str] = self._row.setdefault("_labels", {})
        return labels

    def SetBinLabel(self, bin: int, label: Any) -> None:
        self._labels()[str(int(bin))] = str(label)

    def GetBinLabel(self, bin: int) -> str:
        return self._labels().get(str(int(bin)), "")

    def GetLabels(self) -> Any:
        from .collections import THashList, TObjString

        made = THashList()
        for bin, label in sorted(self._labels().items(), key=lambda pair: int(pair[0])):
            item = TObjString(label)
            item.SetUniqueID(int(bin))
            made.Add(item)
        return made if self._labels() else None

    def LabelsOption(self, option: str = "h") -> None:
        """``LabelsOption``: how labels are drawn, which the drawing reads - handed to the
        histogram the axis is of, as ROOT's is, which sorts its bins by ``a``, ``>`` or ``<``."""
        owner = self._owner
        if owner is not None and hasattr(owner, "LabelsOption"):
            owner.LabelsOption(option, self.GetName()[:1].upper() or "X")
            return
        self.store_labels_option(option)

    def store_labels_option(self, option: str) -> None:
        """How labels are drawn, kept for the drawing."""
        self._row["_labels_option"] = str(option)

    def ChangeLabel(self, *args: Any) -> None:
        """``ChangeLabel``: a drawn label restyled, which the drawing reads."""
        self._row.setdefault("_changed_labels", []).append(args)  # a list: not walked into

    def ChangeLabelByValue(self, *args: Any) -> None:
        """``ChangeLabelByValue``: the label at a value restyled, kept beside those by number."""
        self._row.setdefault("_changed_labels_by_value", []).append(args)

    def SetTimeDisplay(self, value: bool) -> None:
        self._row["fTimeDisplay"] = bool(value)

    def GetTimeDisplay(self) -> bool:
        return bool(self._row.get("fTimeDisplay", False))

    def SetTimeFormat(self, fmt: Any = "") -> None:
        self._row["fTimeFormat"] = str(fmt)

    def GetTimeFormat(self) -> str:
        return str(self._row.get("fTimeFormat", ""))

    def SetTimeOffset(self, toffset: float, option: str = "local") -> None:
        self._row["fTimeFormat"] = f"{self.GetTimeFormat().split('%F')[0]}%F{toffset}"

    def _bit2(self, bit: int) -> bool:
        return bool(int(self._row.get("fBits2", 0)) & bit)

    def _set_bit2(self, bit: int, on: bool) -> None:
        held = int(self._row.get("fBits2", 0))
        self._row["fBits2"] = held | bit if on else held & ~bit

    def CanExtend(self) -> bool:
        """``CanExtend``: whether a fill past the axis, or by a new label, doubles it."""
        return self._bit2(CAN_EXTEND)

    def SetCanExtend(self, extend: bool) -> None:
        self._set_bit2(CAN_EXTEND, bool(extend))

    def IsAlphanumeric(self) -> bool:
        """``IsAlphanumeric``: whether the axis is one of categories, filled by label."""
        return self._bit2(ALPHANUMERIC)

    def SetAlphanumeric(self, alphanumeric: bool = True) -> None:
        self._set_bit2(ALPHANUMERIC, bool(alphanumeric))

    def CanBeAlphanumeric(self) -> bool:
        return not self._bit2(NOT_ALPHA) and not self.IsVariableBinSize()

    def SetNoAlphanumeric(self, no: bool = True) -> None:
        self._set_bit2(NOT_ALPHA, bool(no))
        if no:
            self._set_bit2(ALPHANUMERIC, False)
