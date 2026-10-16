"""A native fit can retain the fixed interpolation polynomial while reading each parameter live."""
from __future__ import annotations

import inspect
import math
import types
from collections.abc import Callable
from typing import Any, cast

from ..pdfs.histfactory import TINY, _even, _halves, _odd, _terms_one
from ..variables import RooConstVar, RooRealVar


def _coefficients(high: float, low: float, boundary: float) -> tuple[float, ...]:
    up, down, up_log, down_log, up_log2, down_log2 = _terms_one(high, low, boundary)
    s0, a0 = _halves(up, down)
    s1, a1 = _halves(up_log, down_log)
    s2, a2 = _halves(up_log2, down_log2)
    a, c, e = _odd(boundary, a0, s1, a2)
    b, d, f = _even(boundary, s0, a1, s2)
    return a, b, c, d, e, f


class Variation:
    def __init__(self, parameter: Any, low: float, high: float,
                 boundary: float, nominal: float) -> None:
        self.read = cast(Callable[[dict[str, Any]], float], parameter.compute)
        self.high, self.low = high / nominal, low / nominal
        self.boundary = boundary
        self.coefficients = _coefficients(self.high, self.low, boundary)

    def delta(self, total: float) -> float:
        x = self.read({})
        if x >= self.boundary:
            return total * (math.pow(self.high, x) - 1.0)
        if x <= -self.boundary:
            return total * (math.pow(self.low, -x) - 1.0)
        a, b, c, d, e, f = self.coefficients
        polynomial = 1.0 + x * (a + x * (b + x * (c + x * (d + x * (e + x * f)))))
        return float(total * (polynomial - 1.0))


class Interpolation:
    def __init__(self, node: Any) -> None:
        self.nominal = float(node._nominal)
        self.terms = [Variation(p, low, high, node._boundary, self.nominal)
                      for p, low, high, _ in zip(
                          node.params, node._low, node._high, node._codes, strict=False)]

    def value(self) -> float:
        total = self.nominal
        for term in self.terms:
            total = total + term.delta(total)
        return TINY if total <= 0 else float(total)


def _native_compute(node: Any, names: tuple[tuple[str, str], ...]) -> bool:
    method = inspect.getattr_static(node, "compute", None)
    if type(method) is not types.FunctionType:
        return False
    return (type(method.__module__) is str and type(method.__qualname__) is str
            and (method.__module__, method.__qualname__) in names)


def _scalar_parameters(parameters: Any) -> bool:
    names = (("xrdroot.roofit.variables", "RooAbsRealLValue.compute"),
             ("xrdroot.roofit.variables", "RooConstVar.compute"))
    return all(type(p) in (RooRealVar, RooConstVar) and type(p._val) is float
               and _native_compute(p, names) for p in parameters)


def prepare(node: Any) -> Callable[[], float] | None:
    definition = (("xrdroot.roofit.pdfs.histfactory", "FlexibleInterpVar.compute"),)
    if not _native_compute(node, definition):
        return None
    values = [node._nominal, node._boundary, *node._low, *node._high]
    if not all(type(value) is float for value in values):
        return None
    if not all(code in (4, 5) for code in node._codes):
        return None
    if not _scalar_parameters(node.params):
        return None
    return Interpolation(node).value
