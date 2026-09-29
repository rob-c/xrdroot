"""``SamplingDistribution``: a test statistic's values over toys, and its quantiles.

The values - and a weight for each - are what ``ToyMCSampler`` gathers; the
integral between two values counts the weights between them, and the
quantile ``InverseCDF(p)`` is the ``p * n``-th of the sorted values, with
RooStats' conventions at the edges and its "variation" by ``sqrt(n)``
values that the adaptive Neyman construction steps by.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .intervals import Named

__all__ = ["SamplingDistribution"]


class SamplingDistribution(Named):
    """The sampled values of a test statistic, with their weights."""

    def __init__(self, name: Any = "SamplingDistribution_DefaultName",
                 title: Any = "SamplingDistribution", values: Any = None,
                 *rest: Any) -> None:  # fmt: skip
        super().__init__(name, title)
        self._var_name = ""
        self._values: list[float] = []
        self._weights: list[float] = []
        self._sorted = False
        if values is not None and hasattr(values, "numEntries"):
            self._from_data(values, rest)
            return
        self._values = [float(v) for v in values or ()]
        weights = rest[0] if rest and not isinstance(rest[0], str) else None
        names = [one for one in rest if isinstance(one, str)]
        self._weights = [float(w) for w in weights] if weights is not None else [1.0] * len(
            self._values)  # fmt: skip
        self._var_name = names[0] if names else ""

    def _from_data(self, data: Any, rest: tuple[Any, ...]) -> None:
        """The column ``<name>_TS0`` of a dataset of toys - or its first - and its weights."""
        if not data.numEntries() or not len(data.get()):
            self._var_name = str(rest[1]) if len(rest) > 1 else ""
            return
        column = str(rest[0]) if rest and rest[0] else f"{self._name}_TS0"
        if data.get().find(column) is None:
            column = data.get()[0].GetName()
        found = data.get().find(column)
        self._var_name = str(rest[1]) if len(rest) > 1 and rest[1] else found.GetTitle()
        self._values = [float(v) for v in data.column(column)]
        self._weights = [float(w) for w in data.weights()]

    def GetVarName(self) -> str:
        return self._var_name

    def GetSize(self) -> int:
        return len(self._values)

    def GetSamplingDistribution(self) -> list[float]:
        return list(self._values)

    def GetSampleWeights(self) -> list[float]:
        return list(self._weights)

    def Add(self, other: Any) -> None:
        """The other's values and weights after these."""
        if other is None:
            return
        self._values += other._values
        self._weights += other._weights
        self._sorted = False
        if not self._var_name and other._var_name:
            self._var_name = other._var_name
        if not self._name and other._name:
            self._name = other._name
        if not self._title and other._title:
            self._title = other._title

    def _sort(self) -> None:
        """``SortValues``: the values in order, their weights with them, and the running sums."""
        if self._sorted:
            return
        order = np.argsort(np.asarray(self._values), kind="stable")
        self._values = [self._values[i] for i in order]
        self._weights = [self._weights[i] for i in order]
        self._sumw = np.cumsum(self._weights).tolist()
        self._sumw2 = np.cumsum(np.square(self._weights)).tolist()
        self._sorted = True

    # -- integrals and quantiles --------------------------------------------------

    def IntegralAndError(self, error: Any, low: float, high: float, normalize: bool = True,
                         lowClosed: bool = True, highClosed: bool = False) -> float:  # fmt: skip
        """The weight of the values from ``low`` to ``high``, its error into ``error`` (a cell)."""
        found, err = self._integral(low, high, normalize, lowClosed, highClosed)
        if hasattr(error, "value"):
            error.value = err
        return found

    def Integral(self, low: float, high: float, normalize: bool = True, lowClosed: bool = True,
                 highClosed: bool = False) -> float:  # fmt: skip
        return self._integral(low, high, normalize, lowClosed, highClosed)[0]

    def _integral(self, low: float, high: float, normalize: bool, low_closed: bool,
                  high_closed: bool) -> tuple[float, float]:  # fmt: skip
        import bisect

        if not self._values:
            return 0.0, math.inf
        self._sort()
        values = self._values
        side_low = bisect.bisect_left if low_closed else bisect.bisect_right
        side_high = bisect.bisect_right if high_closed else bisect.bisect_left
        index_low, index_high = side_low(values, low) - 1, side_high(values, high) - 1
        total = sum2 = 0.0
        if index_high >= 0:
            total, sum2 = self._sumw[index_high], self._sumw2[index_high]
            if index_low >= 0:
                total -= self._sumw[index_low]
                sum2 -= self._sumw2[index_low]
        if normalize:
            norm, norm2 = self._sumw[-1], self._sumw2[-1]
            total /= norm
            return total, math.sqrt(sum2 * (1.0 - 2.0 * total) + norm2 * total * total) / norm
        return total, math.sqrt(sum2)

    def CDF(self, x: float) -> float:
        return self.Integral(-math.inf, x, True, True, True)

    def InverseCDF(self, pvalue: float, sigmaVariation: float = 0.0, *out: Any) -> float:
        """The ``p * n``-th value; with a variation, the one ``sigma * sqrt(n)`` values on too,
        into ``out[0]`` - a cell - as RooStats reports it."""
        value, varied = self.quantile(pvalue, sigmaVariation)
        if out and hasattr(out[0], "value"):
            out[0].value = varied
        return value

    def quantile(self, pvalue: float, sigma: float = 0.0) -> tuple[float, float]:
        """``InverseCDF(p, sigma, variation)``: the quantile and its variation."""
        self._sort()
        n = len(self._values)
        nominal = int(pvalue * n)
        if nominal <= 0:
            return -math.inf, -math.inf
        if nominal >= n - 1:
            return math.inf, math.inf
        if pvalue < 0.5:
            varied = nominal + int(sigma * math.sqrt(1.0 * nominal))
            return self._values[nominal], self._edge(varied, 0)
        varied = nominal + int(sigma * math.sqrt(1.0 * n - nominal))
        return self._values[nominal + 1], self._edge(varied, 1)

    def _edge(self, index: int, shift: int) -> float:
        if index >= len(self._values) - 1:
            return math.inf
        if index <= 0:
            return -math.inf
        return self._values[index + shift]

    def InverseCDFInterpolate(self, pvalue: float) -> float:
        self._sort()
        n = len(self._values)
        nominal = int(pvalue * n)
        if nominal <= 0:
            return -math.inf
        if nominal >= n - 1:
            return math.inf
        upper_x, lower_x = self._values[nominal + 1], self._values[nominal]
        upper_y, lower_y = (nominal + 1.0) / n, float(nominal) / n
        return (upper_x - lower_x) / (upper_y - lower_y) * (pvalue - lower_y) + lower_x
