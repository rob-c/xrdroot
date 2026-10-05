"""``RHist<T>``: a ROOT 7 histogram - its axes, a bin content of type ``T`` for every bin, and
its statistics.

``RHist['int']``, ``RHist['double']`` and ``RHist['RBinWithError']`` are the
contents ROOT's tutorials use; an ``int`` bin given a weight keeps the whole
part of its new sum, as C++'s ``+=`` into an ``int`` does. A bin is found
along every axis at once (:func:`.axes.linearized`), and a fill outside
every bin - past an axis without flow bins - changes no bin, though it
counts in the statistics, as ROOT's does.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .axes import RRegularAxis, as_index, linearized
from .stats import RHistStats, RWeight

__all__ = ["RBinWithError", "RHist", "RHistEngine"]

#: The NumPy type each C++ bin content is kept as.
CONTENTS = {"int": np.int64, "long": np.int64, "long long": np.int64, "short": np.int64,
            "char": np.int64, "float": np.float32, "double": np.float64}  # fmt: skip


class RBinWithError:
    """``RBinWithError``: a bin's sum of weights and sum of their squares."""

    def __init__(self, fSum: float = 0.0, fSum2: float = 0.0) -> None:
        self.fSum, self.fSum2 = float(fSum), float(fSum2)

    def __float__(self) -> float:
        return self.fSum

    def __repr__(self) -> str:
        return f"RBinWithError({self.fSum}, {self.fSum2})"


def _axes_of(args: tuple[Any, ...]) -> list[Any]:
    """The axes an ``RHist`` is made with: each given, a list of them, or ``(n, {low, high})``."""
    if len(args) == 2 and isinstance(args[0], (int, np.integer)):
        return [RRegularAxis(int(args[0]), args[1])]
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return list(args[0])
    return list(args)


class RHistEngine:
    """``RHistEngine<T>``: the axes and the bin contents, without statistics."""

    content = "double"

    def __init__(self, *args: Any) -> None:
        self._axes = _axes_of(args)
        shape = tuple(axis.GetTotalNBins() for axis in self._axes)
        self._errors = self.content == "RBinWithError"
        kind = np.float64 if self._errors else CONTENTS.get(self.content, np.float64)
        self._sum = np.zeros(shape, dtype=kind)
        self._sum2 = np.zeros(shape) if self._errors else None

    def GetAxes(self) -> list[Any]:
        return list(self._axes)

    def GetNDimensions(self) -> int:
        return len(self._axes)

    def GetTotalNBins(self) -> int:
        return int(self._sum.size)

    def _fill_bins(self, values: tuple[Any, ...], weight: float | None) -> None:
        where, found = linearized(self._axes, values)
        if not found:
            return
        w = 1.0 if weight is None else weight
        if self._sum.dtype.kind == "i":
            self._sum[where] = int(self._sum[where] + w)
        else:
            self._sum[where] += w
        if self._sum2 is not None:
            self._sum2[where] += w * w

    def _where(self, args: tuple[Any, ...]) -> tuple[int, ...]:
        given = list(args[0]) if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
        where, found = linearized(self._axes, tuple(as_index(each) for each in given))
        if len(given) != len(self._axes) or not found:
            raise ValueError(f"There is no bin {given} in this histogram of "
                             f"{len(self._axes)} axes.")  # fmt: skip
        return where

    def GetBinContent(self, *args: Any) -> Any:
        """``GetBinContent(indices...)``: normal bins by number, or ``RBinIndex`` for any bin."""
        where = self._where(args)
        if self._sum2 is not None:
            return RBinWithError(self._sum[where], self._sum2[where])
        value = self._sum[where]
        return int(value) if self._sum.dtype.kind == "i" else float(value)

    def SetBinContent(self, *args: Any) -> None:
        *indices, value = args
        where = self._where(tuple(indices))
        self._sum[where] = getattr(value, "fSum", value)
        if self._sum2 is not None:
            self._sum2[where] = getattr(value, "fSum2", 0.0)

    def contents(self) -> tuple[np.ndarray[Any, Any], Any]:
        return self._sum, self._sum2

    def Add(self, other: RHistEngine) -> None:
        self._sum += other._sum
        if self._sum2 is not None and other._sum2 is not None:
            self._sum2 += other._sum2

    def Clear(self) -> None:
        self._sum[...] = 0
        if self._sum2 is not None:
            self._sum2[...] = 0

    def Scale(self, factor: float) -> None:
        self._sum *= factor
        if self._sum2 is not None:
            self._sum2 *= factor * factor

    def Fill(self, *args: Any) -> None:
        values, weight = _split(args)
        self._fill_bins(values, weight)


def _split(args: tuple[Any, ...]) -> tuple[tuple[Any, ...], float | None]:
    """A fill's values, and its weight when the last argument is an ``RWeight``."""
    if args and isinstance(args[-1], RWeight):
        return args[:-1], args[-1].fValue
    return args, None


class RHist(RHistEngine):
    """``RHist<T>``: an engine's bins, and the statistics of every fill."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._stats = RHistStats(len(self._axes))

    def Fill(self, *args: Any) -> None:
        values, weight = _split(args)
        if len(values) != len(self._axes):
            raise ValueError(f"This histogram of {len(self._axes)} axes was filled with "
                             f"{len(values)} values.")  # fmt: skip
        self._fill_bins(values, weight)
        self._stats.fill(tuple(float(v) for v in values), weight)

    def GetEngine(self) -> RHist:
        return self

    def GetStats(self) -> RHistStats:
        return self._stats

    def GetNEntries(self) -> int:
        return self._stats.GetNEntries()

    def ComputeNEffectiveEntries(self) -> float:
        return self._stats.ComputeNEffectiveEntries()

    def ComputeMean(self, dim: int = 0) -> float:
        return self._stats.ComputeMean(int(dim))

    def ComputeStdDev(self, dim: int = 0) -> float:
        return self._stats.ComputeStdDev(int(dim))

    def Add(self, other: RHistEngine) -> None:
        super().Add(other)
        if isinstance(other, RHist):
            self._stats.Add(other._stats)

    def Clear(self) -> None:
        super().Clear()
        self._stats.Clear()

    def Scale(self, factor: float) -> None:
        super().Scale(factor)
        self._stats.Scale(float(factor))


class _Template:
    """``RHist<T>``: the class for each bin content type, by ``[...]``."""

    def __init__(self, base: type) -> None:
        self._base = base
        self._made: dict[str, type] = {}

    def __getitem__(self, kind: Any) -> type:
        name = str(getattr(kind, "__name__", kind)).replace("ROOT::Experimental::", "")
        if name not in self._made:
            self._made[name] = type(f"{self._base.__name__}<{name}>", (self._base,),
                                    {"content": name})  # fmt: skip
        return self._made[name]


class RHistFillContext:
    """``RHistFillContext``: one thread's filling of the histogram a filler shares, each fill
    made whole - its bin and its statistics - before another thread's, as ROOT's atomic
    additions make it."""

    def __init__(self, hist: RHist, lock: Any) -> None:
        self._hist, self._lock = hist, lock

    def Fill(self, *args: Any) -> None:
        with self._lock:
            self._hist.Fill(*args)

    def Flush(self) -> None:
        """``Flush``: every fill went to the histogram as it was made."""


class RHistConcurrentFiller:
    """``RHistConcurrentFiller(hist)``: contexts through which threads fill one histogram."""

    def __init__(self, hist: RHist, *args: Any) -> None:
        self._hist = hist
        self._lock = threading.Lock()

    def CreateFillContext(self) -> RHistFillContext:
        return RHistFillContext(self._hist, self._lock)

    def Flush(self) -> None:
        """``Flush``: nothing waits to be added."""
