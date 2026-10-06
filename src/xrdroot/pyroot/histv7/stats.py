"""``RHistStats``: a ROOT 7 histogram's entries and moments, summed over every fill.

The sums take every value filled, in the flow bins or out of any bin, as
``ROOT::Experimental::RHistStats`` takes them - the entries, the sums of
the weights and of their squares, and for each dimension the weighted sums
of the value's first four powers - so the mean and the spread are those of
what was filled, not of the bins.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["RHistStats", "RWeight"]


class RWeight:
    """``RWeight(w)``: the weight a fill is given."""

    def __init__(self, value: float) -> None:
        self.fValue = float(value)


class RHistStats:
    """``RHistStats``: the running sums a histogram's statistics are worked out from."""

    def __init__(self, dimensions: int) -> None:
        self.fNEntries = 0
        self.fSumW = 0.0
        self.fSumW2 = 0.0
        self.sums = [[0.0, 0.0, 0.0, 0.0] for _ in range(int(dimensions))]

    def GetNDimensions(self) -> int:
        return len(self.sums)

    def GetNEntries(self) -> int:
        return self.fNEntries

    def GetSumW(self) -> float:
        return self.fSumW

    def GetSumW2(self) -> float:
        return self.fSumW2

    def fill(self, values: tuple[float, ...], weight: float | None) -> None:
        """One fill: its values, and its weight (``None`` for a fill without one)."""
        self.fNEntries += 1
        w = 1.0 if weight is None else weight
        self.fSumW += w
        self.fSumW2 += w * w
        for sums, x in zip(self.sums, values, strict=False):
            powers = (x, x * x, x * x * x, x * x * x * x)
            for k, power in enumerate(powers):
                sums[k] += power if weight is None else w * power

    def Add(self, other: RHistStats) -> None:
        self.fNEntries += other.fNEntries
        self.fSumW += other.fSumW
        self.fSumW2 += other.fSumW2
        for mine, theirs in zip(self.sums, other.sums, strict=False):
            for k in range(4):
                mine[k] += theirs[k]

    def Clear(self) -> None:
        self.__init__(len(self.sums))  # type: ignore[misc]

    def Scale(self, factor: float) -> None:
        self.fSumW *= factor
        self.fSumW2 *= factor * factor
        self.sums = [[s * factor for s in sums] for sums in self.sums]

    def ComputeNEffectiveEntries(self) -> float:
        return self.fSumW * self.fSumW / self.fSumW2 if self.fSumW2 else math.nan

    def ComputeMean(self, dim: int = 0) -> float:
        return self.sums[dim][0] / self.fSumW if self.fSumW else math.nan

    def ComputeVariance(self, dim: int = 0) -> float:
        if not self.fSumW:
            return math.nan
        mean = self.ComputeMean(dim)
        return self.sums[dim][1] / self.fSumW - mean * mean

    def ComputeStdDev(self, dim: int = 0) -> float:
        variance = self.ComputeVariance(dim)
        return math.sqrt(variance) if variance >= 0 else math.nan

    def dimension(self, dim: int = 0) -> Any:
        """``GetDimensionStats(dim)``: ``fSumWX`` ... ``fSumWX4``."""
        names = ("fSumWX", "fSumWX2", "fSumWX3", "fSumWX4")
        return type("RDimensionStats", (), dict(zip(names, self.sums[dim], strict=False)))()

    GetDimensionStats = dimension
