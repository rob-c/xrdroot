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

from .messages import message

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
        parts = [
            min(max(int(b), 0), width - 1)
            for b, width in zip((binx, biny, binz), widths, strict=False)
        ]
        total = 0
        for part, width in zip(reversed(parts), reversed(widths), strict=False):
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
        for target, value in zip((binx, biny, binz), found, strict=False):
            store(target, value)
        return tuple(found)

    def _flows(self, bin: int, iaxis: int, low: bool) -> bool:
        """Whether the global bin is in an underflow (``low``) or overflow along ``iaxis`` -
        any axis, for 0 - as ``IsBinUnderflow`` and ``IsBinOverflow`` have it."""
        parts, rest = [], int(bin)
        widths = self._widths()
        for width in widths:
            rest, part = divmod(rest, width)
            parts.append(part)
        chosen = range(len(widths)) if not iaxis else [int(iaxis) - 1]
        return any(_flowing(parts, widths, i, low) for i in chosen)

    def IsBinUnderflow(self, bin: int, iaxis: int = 0) -> bool:
        return self._flows(bin, iaxis, True)

    def IsBinOverflow(self, bin: int, iaxis: int = 0) -> bool:
        return self._flows(bin, iaxis, False)

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
        if any(coordinate is None for coordinate in coordinates):
            return -1
        return self._fill_one(args, coordinates)

    def _fill_one(self, args: tuple[Any, ...], coordinates: list[Any]) -> int:
        """One entry at its coordinates; by label, the sums along an axis that may grow stay
        as they were, as ROOT's do."""
        weight = float(args[self.DIM]) if len(args) > self.DIM else 1.0
        if not any(isinstance(value, str) for value in args):  # no label, no sum frozen
            return int(self._xrd.fill_one(coordinates, weight))
        frozen = self._frozen_moments(args)
        self._xrd.fill(*coordinates, weight=None if weight == 1.0 else weight)
        self._core().update(frozen)
        return int(self._xrd.find_bin(*coordinates))

    def _frozen_moments(self, args: tuple[Any, ...]) -> dict[str, Any]:
        """The sums along each axis filled by label that may grow: ROOT leaves those alone."""
        axes = self._axes()  # type: ignore[attr-defined]
        letters = [
            letter
            for letter, axis, value in zip("xyz", axes, args, strict=False)
            if isinstance(value, str) and axis.CanExtend() and axis.IsAlphanumeric()
        ]
        core = self._core()
        return {key: core[key] for key in core if key.startswith("fTsumw")
                and any(letter in key[6:] for letter in letters)}  # fmt: skip

    def _fill_arrays(self, args: tuple[Any, ...]) -> int:
        """``Fill(xs[, ys][, ws])`` - PyROOT's - an entry for each element: ``-1``, no one bin."""
        columns = [np.asarray(value, dtype=np.float64) for value in args[: self.DIM]]
        weights = np.asarray(args[self.DIM], dtype=np.float64) if len(args) > self.DIM else None
        self._xrd.fill(*columns, weight=weights)
        return -1

    def _coordinate(self, at: int, value: Any) -> float | None:
        """Where a value falls; a new label takes the next free bin, as ``TAxis::FindBin`` does.

        The first label makes an axis that can be one of categories extendable
        and alphanumeric; one past its last bin doubles it if it may grow, and
        is ignored, with ROOT's ``Info``, if it is not of categories.
        """
        if not isinstance(value, str):
            return float(value)
        axis = self._axes()[at]  # type: ignore[attr-defined]
        found = axis.FindFixBin(value)
        if found >= 0:
            return float(axis.GetBinCenter(found))
        if not axis._labels() and axis.CanBeAlphanumeric():
            axis.SetCanExtend(True)
            axis.SetAlphanumeric(True)
        if not axis.IsAlphanumeric():
            message("Info", "FindBin", "Label %s is not in the list and the axis is not "
                    "alphanumeric - ignore it", value)  # fmt: skip
            return None
        return self._labelled(at, value)

    def _labelled(self, at: int, value: str) -> float:
        """A new label's bin: the next free one, the axis doubled first if full and able to grow."""
        from .deflate import inflated

        axis = self._axes()[at]  # type: ignore[attr-defined]
        found = len(axis._labels()) + 1
        if found > axis.GetNbins() and axis.CanExtend():
            self._replace(inflated(self, at))  # type: ignore[attr-defined]
            axis = self._axes()[at]  # type: ignore[attr-defined]
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

    #: ``TH1::EBinErrorOpt``: errors from the squared weights, or a count's Poisson interval
    #: at one sigma, or at 95%.
    kNormal, kPoisson, kPoisson2 = 0, 1, 2

    def SetBinErrorOption(self, type: int) -> None:
        """``SetBinErrorOption``: how ``GetBinErrorLow`` and ``GetBinErrorUp`` work errors out."""
        self._core()["fBinStatErrOpt"] = int(type)

    def GetBinErrorOption(self) -> int:
        return int(self._core().get("fBinStatErrOpt", 0))

    def _poisson_alpha(self) -> float | None:
        """The Poisson interval's two-sided ``alpha``, or ``None`` where errors are normal.

        A histogram filled with weights other than one has normal errors whatever it is told.
        """
        option = self.GetBinErrorOption()
        core = self._core()
        weighted = self.GetSumw2N() > 0 and core.get("fTsumw") != core.get("fTsumw2")
        if option == self.kNormal or weighted:
            return None
        return 0.05 if option == self.kPoisson2 else 1.0 - 0.682689492

    def _count(self, args: tuple[Any, ...], method: str) -> float | None:
        """The bin's content for a Poisson interval, or ``None`` for one below zero."""
        content = self.GetBinContent(*args)
        if int(content) < 0:
            message("Warning", f"TH1::{method}",
                    "Histogram has negative bin content-force usage to normal errors")  # fmt: skip
            self.SetBinErrorOption(self.kNormal)
            return None
        return content

    def GetBinErrorLow(self, *args: Any) -> float:
        """``GetBinErrorLow``: how far below the content its Poisson interval reaches, if asked."""
        from .rmath import gamma_quantile

        alpha = self._poisson_alpha()
        content = None if alpha is None else self._count(args, "GetBinErrorLow")
        if alpha is None or content is None:
            return self.GetBinError(*args)
        n = int(content)
        return 0.0 if n == 0 else content - gamma_quantile(alpha / 2, n, 1.0)

    def GetBinErrorUp(self, *args: Any) -> float:
        """``GetBinErrorUp``: how far above the content its Poisson interval reaches, if asked."""
        from .rmath import gamma_quantile_c

        alpha = self._poisson_alpha()
        content = None if alpha is None else self._count(args, "GetBinErrorUp")
        if alpha is None or content is None:
            return self.GetBinError(*args)
        return gamma_quantile_c(alpha / 2, int(content) + 1, 1.0) - content

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
        """``ResetStats``: the running sums made again from the bins, the entries from all of them.

        See :func:`xrdroot.arithmetic.reset_statistics` for how ROOT 6.40 counts them.
        """
        from ...arithmetic import reset_statistics

        reset_statistics(self._xrd)


def _flowing(parts: list[int], widths: list[int], axis: int, low: bool) -> bool:
    """Whether the bin ``parts`` is in the under- or overflow of ``axis`` - of an axis the
    histogram lacks, its only bin, 0, is an underflow, as ``GetBinXYZ`` leaves it for ROOT."""
    if axis >= len(widths):
        return low
    return parts[axis] == 0 if low else parts[axis] == widths[axis] - 1
