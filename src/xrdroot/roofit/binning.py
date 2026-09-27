"""RooFit's binnings: how a variable is cut into bins, and a named range's two ends.

A variable has a default binning - 100 uniform bins over its range - and
may have others by name: a range ``"signal"`` is a binning with only its two
ends, and ``setBins(50, "cache")`` a second uniform binning. All are
:class:`RooBinning`\\ s here, a list of boundaries, with
:class:`RooUniformBinning` and :class:`RooRangeBinning` as ROOT's ways of
making one.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["RooAbsBinning", "RooBinning", "RooRangeBinning", "RooUniformBinning"]


class RooAbsBinning:
    """Boundaries, in order: ``numBins()`` bins between ``lowBound()`` and ``highBound()``."""

    def __init__(self, low: float = -math.inf, high: float = math.inf, name: str = "") -> None:
        self._name = str(name)
        self._low = float(low)
        self._high = float(high)
        self._bins = 1
        self._uniform = True
        self._edges: list[float] = []

    def GetName(self) -> str:
        return self._name

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def lowBound(self) -> float:
        return self._low

    def highBound(self) -> float:
        return self._high

    def setRange(self, low: float, high: float) -> None:
        self._low, self._high = float(low), float(high)
        if not self._uniform:
            self._edges = [e for e in self._edges if self._low <= e <= self._high]

    def setMin(self, low: float) -> None:
        self.setRange(low, self._high)

    def setMax(self, high: float) -> None:
        self.setRange(self._low, high)

    def numBins(self) -> int:
        return self._bins if self._uniform else max(len(self.array()) - 1, 0)

    def numBoundaries(self) -> int:
        return self.numBins() + 1

    def isUniform(self) -> bool:
        return self._uniform

    def array(self) -> np.ndarray[Any, Any]:
        """Every boundary, from the lowest to the highest."""
        if self._uniform:
            return np.linspace(self._low, self._high, self._bins + 1)
        inside = sorted({self._low, self._high, *self._edges})
        return np.asarray(inside, dtype=np.float64)

    def binLow(self, index: int) -> float:
        return float(self.array()[index])

    def binHigh(self, index: int) -> float:
        return float(self.array()[index + 1])

    def binCenter(self, index: int) -> float:
        edges = self.array()
        return float(0.5 * (edges[index] + edges[index + 1]))

    def binWidth(self, index: int) -> float:
        edges = self.array()
        return float(edges[index + 1] - edges[index])

    def averageBinWidth(self) -> float:
        return (self._high - self._low) / self.numBins()

    def binNumber(self, x: float) -> int:
        edges = self.array()
        found = int(np.searchsorted(edges, x, side="right")) - 1
        return min(max(found, 0), len(edges) - 2)

    def clone(self, name: Any = None) -> Any:
        import copy

        made = copy.deepcopy(self)
        if name is not None:
            made._name = str(name)
        return made

    def isParameterized(self) -> bool:
        return False

    def servers(self) -> list[Any]:
        """The functions its ends are - none, for fixed ends."""
        return []


class RooUniformBinning(RooAbsBinning):
    """``n`` equal bins from ``low`` to ``high``."""

    def __init__(
        self, low: float = 0.0, high: float = 1.0, nbins: int = 100, name: str = ""
    ) -> None:
        super().__init__(low, high, name)
        self._bins = int(nbins)

    def setBins(self, nbins: int) -> None:
        self._bins = int(nbins)


class RooRangeBinning(RooUniformBinning):
    """A named range: one bin, its two ends."""

    def __init__(self, low: float = -math.inf, high: float = math.inf, name: str = "") -> None:
        super().__init__(low, high, 1, name)


class RooBinning(RooAbsBinning):
    """Boundaries of any spacing, added one at a time or a uniform run at a time."""

    def __init__(self, *args: Any) -> None:
        name = str(args[-1]) if args and isinstance(args[-1], str) else ""
        numbers = [a for a in args if not isinstance(a, str)]
        if len(numbers) == 3:  # RooBinning(nbins, xlo, xhi)
            super().__init__(numbers[1], numbers[2], name)
            self._uniform = False
            self._edges = list(np.linspace(numbers[1], numbers[2], int(numbers[0]) + 1))
            return
        low, high = (numbers + [-math.inf, math.inf][len(numbers) :])[:2]
        super().__init__(low, high, name)
        self._uniform = False

    def addBoundary(self, boundary: float) -> bool:
        self._edges.append(float(boundary))
        return True

    def addBoundaryPair(self, boundary: float, mirrorPoint: float = 0.0) -> None:
        self.addBoundary(boundary)
        self.addBoundary(2 * mirrorPoint - boundary)

    def addUniform(self, nbins: int, low: float, high: float) -> None:
        for edge in np.linspace(low, high, int(nbins) + 1):
            self.addBoundary(float(edge))

    def removeBoundary(self, boundary: float) -> bool:
        before = len(self._edges)
        self._edges = [e for e in self._edges if e != boundary]
        return len(self._edges) != before


#: The values being evaluated - a dataset's columns, say - that a parameterised
#: range reads its ends from, innermost last: ``RooParamBinning`` in an event loop.
EVALUATING: list[Any] = []


class evaluating:
    """``with evaluating(ctx):``: parameterised ranges read their ends from ``ctx`` meanwhile."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx

    def __enter__(self) -> None:
        EVALUATING.append(self.ctx)

    def __exit__(self, *exc: Any) -> None:
        EVALUATING.pop()


def _end(function: Any) -> Any:
    """An end's value: from the values being evaluated if they have it - a column, perhaps."""
    if EVALUATING and function.GetName() in EVALUATING[-1]:
        return function.compute(EVALUATING[-1])
    return function.getVal()


class RooParamBinning(RooUniformBinning):
    """``setRange(tmin, tmax)`` with functions as its ends: a range that moves with them.

    The ends are read whenever asked for, so a range whose low end is a
    variable of the data is a different range for every event.
    """

    def __init__(self, low: Any, high: Any, nbins: int = 100, name: str = "") -> None:
        super().__init__(0.0, 1.0, nbins, name)
        self.xlo, self.xhi = low, high

    def lowBound(self) -> Any:
        return _end(self.xlo)

    def highBound(self) -> Any:
        return _end(self.xhi)

    def setRange(self, low: float, high: float) -> None:
        raise ValueError(
            "RooParamBinning::setRange: a range whose ends are functions cannot be moved: "
            "set the functions instead."
        )

    def array(self) -> np.ndarray[Any, Any]:
        return np.linspace(float(self.lowBound()), float(self.highBound()), self._bins + 1)

    def clone(self, name: Any = None) -> Any:
        return RooParamBinning(
            self.xlo, self.xhi, self._bins, self._name if name is None else str(name)
        )

    def isParameterized(self) -> bool:
        return True

    def servers(self) -> list[Any]:
        return [self.xlo, self.xhi]
