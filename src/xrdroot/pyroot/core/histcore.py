"""What every ``TH1`` is before it holds anything: made, named, kept in a directory.

ROOT's constructors take a name, a title, then per axis either a count and
two ends or a count and an array of edges; that is read here into the axes
:meth:`xrdroot.Histogram.book` takes. A histogram made while
``TH1::AddDirectoryStatus()`` is on goes into the current directory -
replacing, with ROOT's warning, one of its name - and ``SetDirectory`` moves
it or, given ``nullptr``, takes it out.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from .directories import current_directory
from .objects import Indent

__all__: list[str] = []

#: The bits of ``fBits`` a wrapper may change: the upper byte is ROOT's own.
BIT_MASK = 0x00FFFFFF
#: ``TH1::kNoStats``: draw no statistics box.
NO_STATS = 1 << 9
#: ``TH1::kIsZoomed``, ``kNoTitle``, ``kIsAverage``: bits a script sets and drawing reads.
NO_TITLE = 1 << 17


def axis_specs(args: tuple[Any, ...], dimensions: int) -> tuple[list[Any], tuple[Any, ...]]:
    """Each axis ROOT's constructor arguments describe, and the arguments left over.

    A count followed by an array is variable binning, the first ``n + 1``
    edges of the array; a count followed by two numbers is even binning.
    """
    specs: list[Any] = []
    at = 0
    for _ in range(dimensions):
        count = int(args[at])
        if np.ndim(args[at + 1]) > 0:
            specs.append(list(np.asarray(args[at + 1], dtype=np.float64)[: count + 1]))
            at += 2
        else:
            specs.append((count, float(args[at + 1]), float(args[at + 2])))
            at += 3
    return specs, args[at:]


class Booked:
    """The construction, naming and keeping of a histogram, whatever its kind."""

    #: The storage letter and the number of axes, which each concrete class sets.
    KIND = "D"
    DIM = 1
    #: ``TH1::AddDirectory``'s switch, and ``TH1::SetDefaultSumw2``'s.
    _add_directory: ClassVar[list[bool]] = [True]
    _default_sumw2: ClassVar[list[bool]] = [False]
    _xrd: Any

    def __init__(self, *args: Any) -> None:
        from .objects import TObject

        TObject.__init__(self)  # type: ignore[arg-type]
        self._directory: Any = None
        self._axis_cache: dict[str, Any] = {}
        if args and isinstance(args[0], Booked):
            self._xrd = args[0]._xrd.copy()
        else:
            self._xrd = self._booked(args)
        from .wrapping import remember

        remember(self._xrd, self)
        if self._default_sumw2[0]:
            self._xrd.sumw2()
        if args and self._add_directory[0]:
            current_directory().Append(self, True)

    def _booked(self, args: tuple[Any, ...]) -> Any:
        """The xrdroot histogram ROOT's constructor arguments describe."""
        from ...hist import Histogram

        if not args:
            return Histogram.book("", *[(1, 0.0, 1.0)] * self.DIM, kind=self.KIND)
        specs, _rest = axis_specs(args[2:], self.DIM)
        made = Histogram.book(str(args[0]), *specs, title=str(args[1]), kind=self.KIND)
        for letter, label in zip("XYZ", str(args[1]).split(";")[1:]):
            made._core[f"f{letter}axis"]["TNamed"]["fTitle"] = label
        return made

    def _adopted(self, xrd: Any) -> None:
        """Stand for ``xrd``, one read from a file or made by xrdroot, in no directory."""
        from .objects import TObject

        TObject.__init__(self)  # type: ignore[arg-type]
        self._xrd = xrd
        self._directory = None
        self._axis_cache = {}

    def _replace(self, xrd: Any) -> None:
        """Stand for a new xrdroot histogram from now on - what an in-place ``Rebin`` makes."""
        from .wrapping import remember

        self._xrd = xrd
        self._axis_cache = {}
        remember(xrd, self)

    @classmethod
    def AddDirectory(cls, add: bool = True) -> None:
        Booked._add_directory[0] = bool(add)

    @classmethod
    def AddDirectoryStatus(cls) -> bool:
        return Booked._add_directory[0]

    @classmethod
    def SetDefaultSumw2(cls, sumw2: bool = True) -> None:
        Booked._default_sumw2[0] = bool(sumw2)

    @classmethod
    def GetDefaultSumw2(cls) -> bool:
        return Booked._default_sumw2[0]

    # -- ROOT's class and bits -------------------------------------------------------

    def ClassName(self) -> str:
        return str(self._xrd.classname)

    def _core(self) -> dict[str, Any]:
        core: dict[str, Any] = self._xrd._core
        return core

    def _attribute_holder(self) -> dict[str, Any]:
        return self._core()

    def _bitword(self) -> int:
        return int(self._core()["TNamed"].get("fBits", 0) or 0) & BIT_MASK

    def _store_bits(self, bits: int) -> None:
        named = self._core()["TNamed"]
        named["fBits"] = (int(named.get("fBits", 0) or 0) & ~BIT_MASK) | (int(bits) & BIT_MASK)

    # -- naming ------------------------------------------------------------------------

    def GetName(self) -> str:
        return str(self._core()["TNamed"]["fName"])

    def SetName(self, name: Any) -> None:
        self._core()["TNamed"]["fName"] = str(name)

    def GetTitle(self) -> str:
        return str(self._core()["TNamed"]["fTitle"])

    def SetTitle(self, title: Any = "") -> None:
        """``SetTitle("title;x;y")``: the histogram's title, then its axes' titles."""
        parts = str(title).split(";")
        self._core()["TNamed"]["fTitle"] = parts[0]
        for letter, label in zip("xyz", parts[1:]):
            self._core()[f"f{letter.upper()}axis"]["TNamed"]["fTitle"] = label

    def SetNameTitle(self, name: Any, title: Any) -> None:
        self.SetName(name)
        self.SetTitle(title)

    def GetDimension(self) -> int:
        return len(self._xrd.axes)

    # -- where it is kept ----------------------------------------------------------------

    def SetDirectory(self, directory: Any = None) -> None:
        """``SetDirectory``: kept in ``directory`` from now on - or, given ``nullptr``, nowhere."""
        if self._directory is not None:
            self._directory.Remove(self)
        if directory is not None and directory != 0:
            real = directory.__real__() if hasattr(directory, "__real__") else directory
            real.Append(self, True)

    def GetDirectory(self) -> Any:
        return self._directory

    # -- the axes ---------------------------------------------------------------------------

    def _axis(self, letter: str) -> Any:
        from .axes import TAxis

        if letter not in self._axis_cache:
            self._axis_cache[letter] = TAxis._of(self._core()[f"f{letter}axis"], self)
        return self._axis_cache[letter]

    def GetXaxis(self) -> Any:
        return self._axis("X")

    def GetYaxis(self) -> Any:
        return self._axis("Y")

    def GetZaxis(self) -> Any:
        return self._axis("Z")

    def _content_axis(self) -> str:
        """The axis a histogram's contents are drawn along, whose range is its extremes."""
        return {1: "yaxis", 2: "zaxis"}.get(self.GetDimension(), "")

    def GetNbinsX(self) -> int:
        return int(self.GetXaxis().GetNbins())

    def GetNbinsY(self) -> int:
        return int(self.GetYaxis().GetNbins())

    def GetNbinsZ(self) -> int:
        return int(self.GetZaxis().GetNbins())

    def GetNcells(self) -> int:
        return int(self._core()["fNcells"])

    def SetStats(self, stats: bool = True) -> None:
        """``SetStats(false)``: draw no statistics box - ``kNoStats``, as ROOT keeps it."""
        self.SetBit(NO_STATS, not stats)  # type: ignore[attr-defined]

    def GetStats(self, stats: Any = None) -> Any:
        """``GetStats(stats)``: the running sums, into ``stats``; with nothing given, a list."""
        found = [float(value) for value in self._xrd_statistics()]
        if stats is not None:
            for at, value in enumerate(found):
                stats[at] = value
        return found

    def _xrd_statistics(self) -> list[float]:
        from ...moments import statistics

        return statistics(self._xrd)

    def SetOption(self, option: Any = " ") -> None:
        self._core()["fOption"] = str(option)

    def GetOption(self) -> str:
        return str(self._core().get("fOption", ""))

    def ls(self, option: str = "") -> None:
        """``TH1::ls``: the ``OBJ:`` line and the histogram's address, as ``TNamed`` prints it."""
        line = f"{Indent.text()}OBJ: {self.ClassName()}\t{self.GetName()}\t{self.GetTitle()} : 0"
        print(line if "noaddr" in str(option) else f"{line} at: {hex(id(self))}")
