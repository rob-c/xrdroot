"""``TH1``, ``TH2``, ``TH3`` and each of their storage kinds, ``C``, ``S``, ``I``, ``F`` and ``D``.

Each is a :class:`xrdroot.Histogram` underneath, in ``._xrd``, with ROOT's
methods over it (see :mod:`.histcore`, :mod:`.histbins`, :mod:`.histstats`
and :mod:`.histops`); what is set here is what is written, and a histogram
read from a file comes back as the class it was written as.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...hist import Histogram
from .histbins import Bins
from .histcore import Booked
from .histops import Operations
from .histstats import Stats
from .objects import TAtt3D, TAttFill, TAttLine, TAttMarker, TNamed
from .wrapping import adopt, register

__all__ = [
    "TH1", "TH2", "TH3",
    "TH1C", "TH1S", "TH1I", "TH1F", "TH1D", "TH1K",
    "TH2C", "TH2S", "TH2I", "TH2F", "TH2D",
    "TH3C", "TH3S", "TH3I", "TH3F", "TH3D",
]  # fmt: skip


class TH1(Booked, Bins, Stats, Operations, TNamed, TAttLine, TAttFill, TAttMarker):
    """``TH1``: a histogram of one axis - and the base of every other."""

    CLASS_TITLE = "1-Dim histogram base class"
    kNoStats = 1 << 9
    kUserContour = 1 << 10
    kLogX = 1 << 15
    kIsZoomed = 1 << 16
    kNoTitle = 1 << 17
    kIsAverage = 1 << 18
    kIsNotW = 1 << 19

    def Print(self, option: str = "") -> None:
        """``Print``: ROOT's one-line summary; ``"all"``, ``"range"`` or ``"base"`` more."""
        entries = int(self.GetEntries())
        print(f"TH1.Print Name  = {self.GetName()}, Entries= {entries}, Total sum= {self.GetSumOfWeights():g}")
        chosen = str(option).lower()
        if "base" in chosen and "all" not in chosen and "range" not in chosen:
            self._print_base()
            return
        if "all" in chosen or "range" in chosen:
            self._print_bins("all" in chosen)

    def _print_base(self) -> None:
        print(f"          Title = {self.GetTitle()}")
        parts = []
        for letter, axis in zip("XYZ", self._axes()):
            low = letter.lower()
            parts.append(f"Nbins{letter}= {axis.GetNbins()}, {low}min= {axis.GetXmin():g}, {low}max={axis.GetXmax():g}")
        print("          " + ", ".join(parts))

    def _print_bins(self, everything: bool) -> None:
        axes = self._axes()
        spans = [(0, axis.GetNbins() + 1) if everything else (axis.GetFirst(), axis.GetLast()) for axis in axes]
        keep_errors = self.GetSumw2N() > 0
        grids = np.meshgrid(*[np.arange(a, b + 1) for a, b in spans], indexing="ij")
        for parts in zip(*(grid.ravel(order="F") for grid in grids)):
            bin = self.GetBin(*parts)
            index = "".join(f"[{int(p)}]" for p in parts)
            where = ", ".join(f"{letter}={axis.GetBinCenter(int(p)):g}" for letter, axis, p in zip("xyz", axes, parts))
            line = f" fSumw{index}={self.GetBinContent(bin):g}, {where}"
            print(f"{line}, error={self.GetBinError(bin):g}" if keep_errors else line)

    # -- UHI, as PyROOT's histograms speak it -----------------------------------------------

    def values(self, flow: bool = False) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd.values(flow))

    def variances(self, flow: bool = False) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd.variances(flow))

    def counts(self, flow: bool = False) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd.counts(flow))

    @property
    def axes(self) -> Any:
        return self._xrd.axes

    @property
    def kind(self) -> str:
        return str(self._xrd.kind)

    def __getitem__(self, index: Any) -> Any:
        found = self._xrd[index]
        return self._wrapped(found)

    def _wrapped(self, found: Any) -> Any:
        from .wrapping import wrap

        return wrap(found) if isinstance(found, Histogram) else found

    def __setitem__(self, index: Any, value: Any) -> None:
        self._xrd[index] = value

    def __len__(self) -> int:
        return self.GetNcells()

    def __bool__(self) -> bool:
        return True

    def __iter__(self) -> Any:
        return iter(self.values().ravel())


class TH2(TH1):
    """``TH2``: a histogram of two axes."""

    CLASS_TITLE = "2-Dim histogram base class"
    DIM = 2


class TH3(TH1, TAtt3D):
    """``TH3``: a histogram of three axes."""

    CLASS_TITLE = "3-Dim histogram base class"
    DIM = 3


def _kind(base: type, letter: str, title: str) -> type:
    """The class of one storage kind: ``TH1F`` is a ``TH1`` keeping floats."""
    name = f"{base.__name__}{letter}"
    return type(name, (base,), {"KIND": letter, "CLASS_TITLE": title, "__doc__": f"``{name}``."})


_TITLES = {"C": "one byte per channel", "S": "one short per channel", "I": "one int per channel",
           "F": "one float per channel", "D": "one double per channel"}  # fmt: skip

TH1C: Any = _kind(TH1, "C", f"1-Dim histograms ({_TITLES['C']})")
TH1S: Any = _kind(TH1, "S", f"1-Dim histograms ({_TITLES['S']})")
TH1I: Any = _kind(TH1, "I", f"1-Dim histograms ({_TITLES['I']})")
TH1F: Any = _kind(TH1, "F", f"1-Dim histograms ({_TITLES['F']})")
TH1D: Any = _kind(TH1, "D", f"1-Dim histograms ({_TITLES['D']})")
TH1K: Any = TH1D
TH2C: Any = _kind(TH2, "C", f"2-Dim histograms ({_TITLES['C']})")
TH2S: Any = _kind(TH2, "S", f"2-Dim histograms ({_TITLES['S']})")
TH2I: Any = _kind(TH2, "I", f"2-Dim histograms ({_TITLES['I']})")
TH2F: Any = _kind(TH2, "F", f"2-Dim histograms ({_TITLES['F']})")
TH2D: Any = _kind(TH2, "D", f"2-Dim histograms ({_TITLES['D']})")
TH3C: Any = _kind(TH3, "C", f"3-Dim histograms ({_TITLES['C']})")
TH3S: Any = _kind(TH3, "S", f"3-Dim histograms ({_TITLES['S']})")
TH3I: Any = _kind(TH3, "I", f"3-Dim histograms ({_TITLES['I']})")
TH3F: Any = _kind(TH3, "F", f"3-Dim histograms ({_TITLES['F']})")
TH3D: Any = _kind(TH3, "D", f"3-Dim histograms ({_TITLES['D']})")

for _cls in (TH1C, TH1S, TH1I, TH1F, TH1D, TH2C, TH2S, TH2I, TH2F, TH2D,
             TH3C, TH3S, TH3I, TH3F, TH3D):  # fmt: skip
    register(_cls.__name__, factory=lambda xrd, cls=_cls: adopt(cls, xrd))
