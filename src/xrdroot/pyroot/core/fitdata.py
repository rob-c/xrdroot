"""``ROOT::Fit``'s data and cost functions, for a fit a macro puts together by hand.

``BinData`` filled from a histogram by ``FillData`` - with ``DataOptions``
and a ``DataRange`` - holds the points ``TH1::Fit`` would fit, chosen by
the same code (:func:`xrdroot.fit.data.from_histogram`); ``Chi2Function``
is the chi-square ``TH1::Fit`` minimises over them, summed in ROOT's order
(:func:`xrdroot.fit.cost.chi2`), of a ``WrappedMultiTF1``. A combined fit
adds such chi-squares in a function of its own and hands that to
``Fitter::FitFCN``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...fit import cost
from ...fit.data import VALUE_ERROR, FitData, from_histogram
from ...fit.data import DataOptions as _Options

__all__ = ["BinData", "Chi2Function", "DataOptions", "DataRange", "FillData", "WrappedMultiTF1"]


class WrappedMultiTF1:
    """``ROOT::Math::WrappedMultiTF1``: a ``TF1`` as a function of its coordinates and
    parameters."""

    def __init__(self, function: Any, dim: int = 0) -> None:
        self._function = function
        self._dim = int(dim) or int(function.GetNdim())

    def NDim(self) -> int:
        return self._dim

    def NPar(self) -> int:
        return int(self._function.GetNpar())

    def __call__(self, x: Any, params: Any = None) -> float:
        return float(self._function.EvalPar(x, params))


class DataOptions:
    """``ROOT::Fit::DataOptions``: ``fUseEmpty``, ``fIntegral``, ``fErrors1`` and the rest."""

    def __init__(self) -> None:
        self.fUseEmpty = self.fIntegral = self.fBinVolume = self.fErrors1 = False
        self.fExpErrors = False

    def _options(self) -> _Options:
        return _Options(integral=bool(self.fIntegral), bin_volume=bool(self.fBinVolume),
                        use_empty=bool(self.fUseEmpty), errors1=bool(self.fErrors1),
                        exp_errors=bool(self.fExpErrors))  # fmt: skip


class _Pair(tuple[float, float]):
    """A ``std::pair`` of a range's ends: ``first`` and ``second``."""

    @property
    def first(self) -> float:
        return self[0]

    @property
    def second(self) -> float:
        return self[1]


class DataRange:
    """``ROOT::Fit::DataRange``: the range of each coordinate - none, all of it."""

    def __init__(self, *ends: float) -> None:
        self._spans: list[Any] = [None, None, None]
        for axis in range(len(ends) // 2):
            self._spans[axis] = (float(ends[2 * axis]), float(ends[2 * axis + 1]))

    def SetRange(self, *args: Any) -> None:
        """``SetRange(xmin, xmax)``, or ``SetRange(icoord, xmin, xmax)``."""
        axis, low, high = (0, *args) if len(args) == 2 else args
        self._spans[int(axis)] = (float(low), float(high))

    def Size(self, icoord: int = 0) -> int:
        return 0 if self._spans[int(icoord)] is None else 1

    def __call__(self, icoord: int = 0, irange: int = 0) -> _Pair:
        span = self._spans[int(icoord)]
        return _Pair(span if span is not None else (-np.inf, np.inf))


class BinData:
    """``ROOT::Fit::BinData``: the points of a histogram, as ``FillData`` chooses them - or,
    made of a size and a dimension, those ``Add`` gives it one by one."""

    def __init__(self, options: Any = None, ranges: Any = None, *rest: Any) -> None:
        dim = int(ranges) if isinstance(options, int) and isinstance(ranges, int) else 0
        made = None if dim else options
        self._options = (made or DataOptions())._options()
        self._spans = (ranges if not dim and ranges is not None else DataRange())._spans
        self._data: Any = None
        self._added: tuple[list[Any], list[float], list[float], int] = ([], [], [], dim)

    def Add(self, x: Any, y: float, error: float = 1.0) -> None:
        """``Add(x, value, error)``: one point - its coordinates, its value, its error."""
        points, values, errors, dim = self._added
        coordinates = [float(x)] if np.ndim(x) == 0 else [float(x[i]) for i in range(dim or 1)]
        points.append(coordinates)
        values.append(float(y))
        errors.append(float(error))
        made = np.array(points, dtype=np.float64)
        self._data = FitData(x=made, y=np.array(values), error=np.array(errors),
                             kind=VALUE_ERROR, options=self._options)  # fmt: skip

    def Size(self) -> int:
        return 0 if self._data is None else int(self._data.size)

    def NDim(self) -> int:
        return 0 if self._data is None else int(self._data.ndim)


def FillData(data: BinData, histogram: Any, function: Any = None) -> None:
    """``ROOT::Fit::FillData``: a histogram's bins in the data's range, as points."""
    xrd = histogram._xrd
    data._data = from_histogram(xrd, data._options, data._spans[: len(xrd.axes)])


class Chi2Function:
    """``ROOT::Fit::Chi2Function``: the chi-square of a function over the data, of its
    parameters."""

    def __init__(self, data: BinData, function: WrappedMultiTF1) -> None:
        self._npar = function.NPar()
        self._cost = cost.chi2(data._data, cost.Predictor(function._function._xrd, data._data))

    def NDim(self) -> int:
        return self._npar

    def __call__(self, params: Any) -> float:
        return float(self._cost(np.asarray([float(params[i]) for i in range(self._npar)])))
