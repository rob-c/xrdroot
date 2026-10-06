"""The functions a HistFactory model is built of: ``ParamHistFunc``, ``FlexibleInterpVar``,
``PiecewiseInterpolation`` and ``RooBinWidthFunction``.

A HistFactory channel is a :class:`~.realsum.RooRealSumPdf` of samples, each
a histogram (``RooHistFunc``) times its normalisation - a
``FlexibleInterpVar`` of the overall systematics, interpolated as
``MathFuncs::flexibleInterpSingle`` interpolates - times, bin by bin, a
``ParamHistFunc`` of the statistical gammas, over the bin width. Each is
binned: RooFit integrates it bin by bin (``RooBinIntegrator``) and a
likelihood of it counts events bin by bin, and the classes say where
their bins are (``binBoundaries``) for that.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from ...random import libm
from ..collections import as_list
from ..real import Context, RooAbsReal

__all__ = ["FlexibleInterpVar", "ParamHistFunc", "PiecewiseInterpolation", "RooBinWidthFunction",
           "interpolate"]  # fmt: skip

#: ``TMath::Limits<double>::Min()``: what a total that is not positive is raised to.
TINY = 2.2250738585072014e-308


def _halves(up: Any, down: Any) -> tuple[Any, Any]:
    """``0.5 * (up + down)`` and ``0.5 * (up - down)``: the symmetric and antisymmetric parts."""
    return 0.5 * (up + down), 0.5 * (up - down)


def _odd(x0: float, a0: Any, s1: Any, a2: Any) -> tuple[Any, Any, Any]:
    """Code 5's polynomial, its odd coefficients."""
    sq, tail = x0 * x0, x0 * x0 * a2
    a = 1.0 / (8 * x0) * (15 * a0 - 7 * x0 * s1 + tail)
    c = 1.0 / (4 * sq * x0) * (-5 * a0 + 5 * x0 * s1 - tail)
    return a, c, _fifth(x0, a0, s1, tail)


def _fifth(x0: float, a0: Any, s1: Any, tail: Any) -> Any:
    sq = x0 * x0
    return 1.0 / (8 * sq * sq * x0) * (3 * a0 - 3 * x0 * s1 + tail)


def _even(x0: float, s0: Any, a1: Any, s2: Any) -> tuple[Any, Any, Any]:
    """Code 5's polynomial, its even coefficients."""
    sq, tail = x0 * x0, x0 * x0 * s2
    b = 1.0 / (8 * sq) * (-24 + 24 * s0 - 9 * x0 * a1 + tail)
    d = 1.0 / (4 * sq * sq) * (12 - 12 * s0 + 7 * x0 * a1 - tail)
    return b, d, _sixth(x0, s0, a1, tail)


def _sixth(x0: float, s0: Any, a1: Any, tail: Any) -> Any:
    sq = x0 * x0
    return 1.0 / (8 * sq * sq * sq) * (-8 + 8 * s0 - 5 * x0 * a1 + tail)


def _inside(x: Any, x0: float, terms: tuple[Any, ...]) -> Any:
    """Code 5 within the boundary: ``1 + x (a + x (b + ... x f))``, of the six terms
    ``up, down`` and their first and second logarithmic derivatives."""
    up, down, up_log, down_log, up_log2, down_log2 = terms
    s0, a0 = _halves(up, down)
    s1, a1 = _halves(up_log, down_log)
    s2, a2 = _halves(up_log2, down_log2)
    a, c, e = _odd(x0, a0, s1, a2)
    b, d, f = _even(x0, s0, a1, s2)
    return 1.0 + x * (a + x * (b + x * (c + x * (d + x * (e + x * f)))))


def _log_or_minus_inf(value: float) -> float:
    import math

    return math.log(value) if value > 0 else -math.inf


def _terms_one(high: float, low: float, x0: float) -> tuple[float, ...]:
    """:func:`_inside`'s terms, of plain floats."""
    import math

    log_hi, log_lo = _log_or_minus_inf(high), _log_or_minus_inf(low)
    up, down = math.exp(x0 * log_hi), math.exp(x0 * log_lo)
    up_log = 0.0 if high <= 0 else up * log_hi
    down_log = 0.0 if low <= 0 else -down * log_lo
    up_log2 = 0.0 if high <= 0 else up_log * log_hi
    down_log2 = 0.0 if low <= 0 else -down_log * log_lo
    return up, down, up_log, down_log, up_log2, down_log2


def _code5_one(low: float, high: float, x0: float, nominal: float, x: float,
               res: float) -> float:  # fmt: skip
    """Code 5 of one parameter value: the same operations as the arrays', in plain floats."""
    import math

    high, low = high / nominal, low / nominal
    if x >= x0:
        return res * (math.pow(high, x) - 1.0)
    if x <= -x0:
        return res * (math.pow(low, -x) - 1.0)
    return float(res * (_inside(x, x0, _terms_one(high, low, x0)) - 1.0))


def _terms(high: Any, low: Any, x0: float) -> tuple[Any, ...]:
    """:func:`_inside`'s terms, of arrays."""
    log_hi, log_lo = libm.log(high), libm.log(low)
    up, down = libm.exp(x0 * log_hi), libm.exp(x0 * log_lo)
    up_log = np.where(high <= 0, 0.0, up * log_hi)
    down_log = np.where(low <= 0, 0.0, -down * log_lo)
    up_log2 = np.where(high <= 0, 0.0, up_log * log_hi)
    down_log2 = np.where(low <= 0, 0.0, -down_log * log_lo)
    return up, down, up_log, down_log, up_log2, down_log2


def _code5(low: Any, high: Any, boundary: float, nominal: Any, x: Any, res: Any) -> Any:
    """Code 5: exponential outside the boundary, a sixth-degree polynomial inside."""
    if all(isinstance(v, float) for v in (low, high, nominal, x, res)):
        return _code5_one(low, high, boundary, nominal, x, res)
    x = np.asarray(x, dtype=np.float64)
    high, low = np.asarray(high / nominal, dtype=np.float64), np.asarray(low / nominal)
    with np.errstate(divide="ignore", invalid="ignore"):
        outside = np.where(x >= boundary, libm.power(high, x), libm.power(low, -x))
        terms = _terms(high, low, boundary)
    inside = _inside(x, boundary, terms)
    mod = np.where((x >= boundary) | (x <= -boundary), outside, inside)
    return res * (mod - 1.0)


def _code4(low: Any, high: Any, boundary: float, nominal: Any, x: Any, res: Any, code: int) -> Any:
    """Codes 4 and 6: linear outside the boundary, sixth-degree inside - 6 multiplicatively."""
    if code == 6:
        high, low, nominal = high / nominal, low / nominal, 1.0
    t = x / boundary
    plus, minus = high - nominal, nominal - low
    half, a = 0.5 * (plus + minus), 0.0625 * (plus - minus)
    inside = x * (half + t * a * (15 + t * t * (-10 + t * t * 3)))  # ROOT's order of products
    mod = np.where(x >= boundary, x * plus, np.where(x <= -boundary, x * minus, inside))
    return mod * res if code == 6 else mod


def interpolate(code: int, low: Any, high: Any, boundary: float, nominal: Any, x: Any,
                res: Any) -> Any:  # fmt: skip
    """``MathFuncs::flexibleInterpSingle``: what one parameter adds to ``res`` at ``x``."""
    if code == 5:
        return _code5(low, high, boundary, nominal, x, res)
    if code in (4, 6):
        return _code4(low, high, boundary, nominal, np.asarray(x, dtype=np.float64), res, code)
    return _simple(code, low, high, nominal, np.asarray(x, dtype=np.float64), res)


def _simple(code: int, low: Any, high: Any, nominal: Any, x: Any, res: Any) -> Any:
    """Codes 0, 1 and 2: linear, exponential, quadratic within one and linear outside."""
    if code == 0:
        return np.where(x > 0, x * (high - nominal), x * (nominal - low))
    if code == 1:
        return np.where(x >= 0, res * (libm.power(high / nominal, x) - 1),
                        res * (libm.power(low / nominal, -x) - 1))  # fmt: skip
    if code == 2:
        return _code2(low, high, nominal, x)
    return 0.0 * x


def _code2(low: Any, high: Any, nominal: Any, x: Any) -> Any:
    """Code 2: quadratic within one, linear outside."""
    a, b = 0.5 * (high + low) - nominal, 0.5 * (high - low)
    return np.where(
        x > 1,
        (2 * a + b) * (x - 1) + high - nominal,
        np.where(x < -1, -(2 * a - b) * (x + 1) + low - nominal, a * x * x + b * x),
    )


class FlexibleInterpVar(RooAbsReal):
    """``RooStats::HistFactory::FlexibleInterpVar``: the nominal normalisation, moved by each
    parameter between its low and high values as its interpolation code says."""

    def __init__(self, name: Any, title: Any, paramList: Any = (), nominal: float = 1.0,
                 low: Any = (), high: Any = (), codes: Any = None) -> None:  # fmt: skip
        super().__init__(name, title)
        self.params = self._list_proxy("paramList", as_list(paramList))
        self._nominal = float(nominal)
        self._low, self._high = [float(v) for v in low], [float(v) for v in high]
        self._codes = [int(c) for c in codes] if codes is not None else [0] * len(self._low)
        self._boundary = 1.0

    def ClassName(self) -> str:
        return "RooStats::HistFactory::FlexibleInterpVar"

    def setInterpCode(self, param: Any, code: int) -> None:
        self._codes[self.params.index(param)] = int(code)

    def setAllInterpCodes(self, code: int) -> None:
        self._codes = [int(code)] * len(self._codes)

    def setNominal(self, nominal: float) -> None:
        self._nominal = float(nominal)

    def interpolationCodes(self) -> list[int]:
        return list(self._codes)

    def low(self) -> list[float]:
        return list(self._low)

    def high(self) -> list[float]:
        return list(self._high)

    def nominal(self) -> float:
        return self._nominal

    def compute(self, ctx: Context) -> Any:
        total: Any = self._nominal
        for param, low, high, code in zip(
            self.params, self._low, self._high, self._codes, strict=False
        ):
            code = 5 if code == 4 else code
            total = total + interpolate(code, low, high, self._boundary, self._nominal,
                                        param.compute(ctx), total)  # fmt: skip
        return np.where(np.asarray(total) <= 0, TINY, total) if np.ndim(total) else (
            TINY if total <= 0 else float(total))  # fmt: skip


class PiecewiseInterpolation(RooAbsReal):
    """``PiecewiseInterpolation``: a nominal function moved towards its low or high variation by
    each parameter - a HistFactory histogram systematic, bin by bin."""

    def __init__(self, name: Any, title: Any, nominal: Any = None, lowSet: Any = (),
                 highSet: Any = (), paramSet: Any = (), codes: Any = None) -> None:  # fmt: skip
        super().__init__(name, title)
        self.nominal = self._proxy("!nominal", nominal)
        self.lows = self._list_proxy("!lowSet", as_list(lowSet))
        self.highs = self._list_proxy("!highSet", as_list(highSet))
        self.params = self._list_proxy("!paramSet", as_list(paramSet))
        self._codes = [int(c) for c in codes] if codes is not None else [0] * len(self.params)
        self._positive = False

    def setPositiveDefinite(self, flag: bool = True) -> None:
        self._positive = bool(flag)

    def positiveDefinite(self) -> bool:
        return self._positive

    def setInterpCode(self, param: Any, code: int, silent: bool = False) -> None:
        self._codes[self.params.index(param)] = int(code)

    def setAllInterpCodes(self, code: int) -> None:
        self._codes = [int(code)] * len(self._codes)

    def interpolationCodes(self) -> list[int]:
        return list(self._codes)

    def compute(self, ctx: Context) -> Any:
        nominal = self.nominal.compute(ctx)
        total: Any = nominal
        for low, high, param, code in zip(
            self.lows, self.highs, self.params, self._codes, strict=False
        ):
            total = total + interpolate(code, low.compute(ctx), high.compute(ctx), 1.0, nominal,
                                        param.compute(ctx), total)  # fmt: skip
        if self._positive:
            total = np.where(np.asarray(total) < 0, 0.0, total)
        return total if np.ndim(total) else float(total)

    def bin_boundaries(self, name: str) -> Any:
        return _boundaries_of([self.nominal], name)

    def isBinnedDistribution(self, obs: Any) -> bool:
        return _binned([self.nominal], obs)


class ParamHistFunc(RooAbsReal):
    """``ParamHistFunc``: a parameter per bin of its variables - the statistical gammas - its
    value the parameter of the bin the variables are in."""

    def __init__(self, name: Any, title: Any, vars: Any = (), paramSet: Any = (),
                 hist: Any = None) -> None:  # fmt: skip
        super().__init__(name, title)
        self.vars = self._list_proxy("!dataVars", as_list(vars))
        self.gammas = self._list_proxy("!paramSet", as_list(paramSet))
        self._edges = [one.getBinning().array() for one in self.vars]
        count = int(np.prod([len(e) - 1 for e in self._edges])) if self._edges else 0
        if count != len(self.gammas):
            raise ValueError(
                f"ParamHistFunc::addParamSet - ERROR - Supplied list of parameters has "
                f"{len(self.gammas)} elements but the ParamHistFunc{self._name} has {count} bins."
            )

    def numBins(self) -> int:
        return len(self.gammas)

    def paramList(self) -> Any:
        return self.gammas

    def getParameter(self, index: int | None = None) -> Any:
        return self.gammas[self._index({}) if index is None else int(index)]

    def setConstant(self, constant: bool = True) -> None:
        for gamma in self.gammas:
            gamma.setAttribute("Constant", bool(constant))

    def _index(self, ctx: Context) -> Any:
        """The parameter of the bin, the first variable's bin fastest, as ``getParameter`` has
        it."""
        found: Any = 0
        stride = 1
        for var, edges in zip(self.vars, self._edges, strict=False):
            values = np.asarray(var.compute(ctx), dtype=np.float64)
            bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
            found = found + bins * stride
            stride *= len(edges) - 1
        return found

    def compute(self, ctx: Context) -> Any:
        index = self._index(ctx)
        values = np.array([np.broadcast_to(g.compute(ctx), np.shape(index)) for g in self.gammas])
        if not np.ndim(index):
            return float(values[int(index)])
        return np.take_along_axis(values, np.asarray(index)[None, ...], axis=0)[0]

    def _volumes(self) -> np.ndarray[Any, Any]:
        """The bins' volumes in the dataset's order, the first variable slowest - which ROOT
        pairs with the parameters in theirs, the first variable fastest, index by index."""
        found = np.ones(1)
        for edges in self._edges:
            found = np.kron(found, np.diff(edges))
        return found

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        mine = frozenset(one.GetName() for one in self.vars)
        return names & mine if not getattr(self, "_force_num_int", False) else frozenset()

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        """``analyticalIntegralWN``: each bin's parameter times its volume, summed."""
        total: Any = 0.0
        for gamma, volume in zip(self.gammas, self._volumes(), strict=False):
            total = total + gamma.compute(ctx) * volume
        return total

    def bin_boundaries(self, name: str) -> Any:
        for var, edges in zip(self.vars, self._edges, strict=False):
            if var.GetName() == name:
                return [float(e) for e in edges]
        return None

    def isBinnedDistribution(self, obs: Any = None) -> bool:
        return True


#: Whether a binned likelihood is being evaluated - in which RooFit replaces every
#: ``RooBinWidthFunction`` by one, the densities then read as yields - and whether the class
#: is switched on at all (``enableClass``/``disableClass``).
_MODE = {"binned": False, "enabled": True}


@contextmanager
def binned_likelihood() -> Iterator[None]:
    """Evaluate as a binned likelihood does: every bin-width function one."""
    before = _MODE["binned"]
    _MODE["binned"] = True
    try:
        yield
    finally:
        _MODE["binned"] = before


class RooBinWidthFunction(RooAbsReal):
    """``RooBinWidthFunction``: the width - or its inverse - of the bin a ``RooHistFunc`` is in."""

    def __init__(self, name: Any, title: Any, histFunc: Any = None,
                 divideByBinWidth: bool = False) -> None:  # fmt: skip
        super().__init__(name, title)
        self.hist = self._proxy("HistFuncForBinWidth", histFunc)
        self._divide = bool(divideByBinWidth)

    @staticmethod
    def enableClass() -> None:
        _MODE["enabled"] = True

    @staticmethod
    def disableClass() -> None:
        _MODE["enabled"] = False

    @staticmethod
    def isClassEnabled() -> bool:
        return bool(_MODE["enabled"])

    def divideByBinWidth(self) -> bool:
        return self._divide

    def histFunc(self) -> Any:
        return self.hist

    def compute(self, ctx: Context) -> Any:
        if _MODE["binned"] or not _MODE["enabled"]:
            return 1.0
        data = self.hist.dataHist()
        bins = self.hist.bin_index(ctx)
        volumes = data.binVolumes()
        found = np.where(np.asarray(bins) >= 0, volumes[np.clip(bins, 0, None)], 1.0)
        found = 1.0 / found if self._divide else found
        return found if np.ndim(found) else float(found)

    def bin_boundaries(self, name: str) -> Any:
        return self.hist.bin_boundaries(name)

    def isBinnedDistribution(self, obs: Any = None) -> bool:
        return bool(self.hist.isBinnedDistribution(obs))


def _boundaries_of(funcs: Any, name: str) -> Any:
    """The first bin boundaries any of ``funcs`` has in ``name`` - ``RooProduct``'s rule."""
    for func in funcs:
        found = getattr(func, "bin_boundaries", None)
        if found is not None and name in func.dependents():
            edges = found(name)
            if edges is not None:
                return edges
    return None


def _binned(funcs: Any, obs: Any) -> bool:
    """Whether every one of ``funcs`` that depends on ``obs`` is binned in it."""
    names = _names(obs)
    for func in funcs:
        if names & func.dependents():
            test = getattr(func, "isBinnedDistribution", None)
            if test is None or not test(obs):
                return False
    return True


def _names(obs: Any) -> frozenset[str]:
    if isinstance(obs, (set, frozenset)) and all(isinstance(o, str) for o in obs):
        return frozenset(obs)
    return frozenset(one.GetName() for one in as_list(obs))
