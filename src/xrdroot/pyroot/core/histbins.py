"""A histogram's bins, by ROOT's numbers: filling them, reading them, setting them.

ROOT numbers bins from 1, with 0 the underflow and ``n + 1`` the overflow,
and a histogram of more than one axis has a *global* bin, x fastest:
``bin = binx + (nx + 2) * (biny + (ny + 2) * binz)``. Everything here takes
either, as ROOT's overloads do - ``GetBinContent(bin)`` or
``GetBinContent(binx, biny)`` - and works on the xrdroot histogram's own
arrays, so a bin set here is a bin written.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__: list[str] = []


class Bins:
    """``Fill``, ``GetBinContent``, ``SetBinContent`` and the rest of a ``TH1``'s bins."""

    _xrd: Any
    _core: Any
    DIM: int

    def _widths(self) -> list[int]:
        """The cells along each axis, flow included, as the global bin number counts them."""
        return [len(axis) + 2 for axis in self._xrd.axes]

    def GetBin(self, binx: int, biny: int = 0, binz: int = 0) -> int:
        """``GetBin``: the global bin of ``(binx, biny, binz)``, each clamped to its flow bins."""
        widths = self._widths()
        parts = [min(max(int(b), 0), width - 1) for b, width in zip((binx, biny, binz), widths)]
        total = 0
        for part, width in zip(reversed(parts), reversed(widths)):
            total = total * width + part
        return total

    def GetBinXYZ(self, bin: int, binx: Any, biny: Any = None, binz: Any = None) -> tuple[int, ...]:
        """``GetBinXYZ``: a global bin's bin along each axis, into what is given, and back."""
        from .refs import store

        found, rest = [], int(bin)
        for width in self._widths():
            rest, part = divmod(rest, width)
            found.append(part)
        found += [0] * (3 - len(found))
        for target, value in zip((binx, biny, binz), found):
            store(target, value)
        return tuple(found)

    def _global(self, args: tuple[Any, ...]) -> int:
        """The global bin a ``(bin)`` or ``(binx, biny[, binz])`` call names."""
        return int(args[0]) if len(args) == 1 else self.GetBin(*args)

    def _split(self, args: tuple[Any, ...], values: int) -> tuple[int, tuple[Any, ...]]:
        """A setter's bin, from however many bin numbers came before its ``values`` values."""
        return self._global(args[: len(args) - values]), args[len(args) - values :]

    # -- filling ------------------------------------------------------------------------

    def Fill(self, *args: Any) -> int:
        """``Fill(x[, y[, z]][, w])``: one entry, and the global bin it went to.

        A string coordinate fills the bin of that label, made on the axis if
        it has none yet, as ROOT fills a histogram of categories.
        """
        if args and np.ndim(args[0]) > 0:
            return self._fill_arrays(args)
        coordinates = [self._coordinate(at, value) for at, value in enumerate(args[: self.DIM])]
        weight = float(args[self.DIM]) if len(args) > self.DIM else 1.0
        self._xrd.fill(*coordinates, weight=None if weight == 1.0 else weight)
        return int(self._xrd.find_bin(*coordinates))

    def _fill_arrays(self, args: tuple[Any, ...]) -> int:
        """``Fill(xs[, ys][, ws])`` - PyROOT's - an entry for each element: ``-1``, no one bin."""
        columns = [np.asarray(value, dtype=np.float64) for value in args[: self.DIM]]
        weights = np.asarray(args[self.DIM], dtype=np.float64) if len(args) > self.DIM else None
        self._xrd.fill(*columns, weight=weights)
        return -1

    def _coordinate(self, at: int, value: Any) -> float:
        if not isinstance(value, str):
            return float(value)
        axis = (self.GetXaxis, self.GetYaxis, self.GetZaxis)[at]()  # type: ignore[attr-defined]
        found = axis.FindFixBin(value)
        if found < 0:
            found = len(axis._labels()) + 1
            axis.SetBinLabel(found, value)
        return float(axis.GetBinCenter(found))

    def FillN(self, ntimes: int, x: Any, *rest: Any) -> None:
        """``FillN(n, x, w[, stride])``, or for two axes ``FillN(n, x, y, w)``: many entries."""
        count = int(ntimes)
        columns = [np.asarray(x, dtype=np.float64)[:count]]
        if self.DIM > 1:
            columns.append(np.asarray(rest[0], dtype=np.float64)[:count])
            rest = rest[1:]
        weights = rest[0] if rest else None
        weight = None if weights is None else np.asarray(weights, dtype=np.float64)[:count]
        self._xrd.fill(*columns, weight=weight)

    # -- reading and setting bins ----------------------------------------------------------

    def GetBinContent(self, *args: Any) -> float:
        """``GetBinContent(bin)`` or ``(binx, biny[, binz])``."""
        bin = self._global(args)
        cells = self._contents()
        return float(cells[bin]) if 0 <= bin < len(cells) else 0.0

    def _contents(self) -> np.ndarray[Any, Any]:
        """Every bin's content, flow included, flat in global-bin order."""
        return np.asarray(self._xrd._bins)

    def RetrieveBinContent(self, bin: int) -> float:
        return self.GetBinContent(bin)

    def SetBinContent(self, *args: Any) -> None:
        """``SetBinContent(bin, c)`` or ``(binx, biny[, binz], c)``: and one more entry."""
        bin, (content,) = self._split(args, 1)
        core = self._core()
        core["fEntries"] = float(core["fEntries"]) + 1.0
        core["fTsumw"] = 0.0
        cells = self._xrd._cells()
        if 0 <= bin < len(cells):
            cells[bin] = content

    def AddBinContent(self, *args: Any) -> None:
        """``AddBinContent(bin[, w])``: add to one bin's content alone, as ROOT's does."""
        w = float(args[1]) if len(args) > 1 else 1.0
        self._xrd._cells()[int(args[0])] += w

    def GetBinError(self, *args: Any) -> float:
        """``GetBinError``: the root of the bin's squared weights, or of its content."""
        bin = self._global(args)
        errors = self._errors()
        return float(errors[bin]) if 0 <= bin < len(errors) else 0.0

    def _errors(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd._bin_errors())

    def GetBinErrorLow(self, *args: Any) -> float:
        return self.GetBinError(*args)

    def GetBinErrorUp(self, *args: Any) -> float:
        return self.GetBinError(*args)

    def SetBinError(self, *args: Any) -> None:
        """``SetBinError(bin, e)``: squared weights kept from now on, this bin's ``e²``."""
        bin, (error,) = self._split(args, 1)
        squares = self._xrd._ensure_sumw2()
        if 0 <= bin < len(squares):
            squares[bin] = float(error) ** 2

    def SetBinsLength(self, n: int = -1) -> None:
        """``SetBinsLength``: the bins are as many as the axes say, so nothing to do."""

    def GetArray(self) -> np.ndarray[Any, Any]:
        """``GetArray``: the bins themselves, flow included, to read or change in place."""
        cells: np.ndarray[Any, Any] = self._xrd._cells()
        return cells

    def GetSumw2(self) -> np.ndarray[Any, Any]:
        """``GetSumw2``: the squared weights, empty for a histogram not keeping them."""
        squares = self._xrd._sumw2()
        return np.zeros(0) if squares is None else squares

    def GetSumw2N(self) -> int:
        return len(self.GetSumw2())

    def Sumw2(self, flag: bool = True) -> None:
        self._xrd.sumw2(bool(flag))

    def Reset(self, option: str = "") -> None:
        """``Reset``: every bin and sum emptied; ``"ICES"`` keeps the contents, as ROOT's does."""
        if "ICE" in str(option).upper():
            self.ResetStats()
            return
        self._xrd.reset()

    def ResetStats(self) -> None:
        """``ResetStats``: the running sums made again from the bins."""
        core = self._core()
        core["fTsumw"], core["fEntries"] = 0.0, 1.0
        core["fEntries"] = float(self._xrd.effective_entries)
