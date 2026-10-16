"""Native fit scopes can bind constraint inputs once without changing their likelihood kernels."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

from .. import kernels
from ..functions import RooProduct
from ..pdfs.basic import RooGaussian
from ..pdfs.shapes import RooPoisson, _poisson_single
from ..variables import RooConstVar, RooRealVar


class NotPrepared(Exception):
    """The general constraint evaluator owns this normalization or input expression."""


class Inputs:
    def __init__(self, context: dict[str, float]) -> None:
        self.context = context
        self.active: set[int] = set()

    def bind(self, node: Any) -> Callable[[], float]:
        if id(node) in self.active:
            raise NotPrepared
        self.active.add(id(node))
        try:
            return self._bind(node)
        finally:
            self.active.remove(id(node))

    def _bind(self, node: Any) -> Callable[[], float]:
        if type(node) is RooConstVar:
            return lambda: float(node.getVal())
        if type(node) is RooRealVar:
            if node.GetName() in self.context:
                value = self.context[node.GetName()]
                return lambda: value
            # A speculative NumPy operation could warn before the fallback repeats it.
            if type(node._val) not in (float, int):
                raise NotPrepared
            return cast(Callable[[], float], node.getVal)
        if type(node) is RooProduct:
            inputs = tuple(self.bind(term) for term in node.terms)
            return lambda: _product(inputs)
        raise NotPrepared


def _product(inputs: tuple[Callable[[], float], ...]) -> float:
    value = 1.0
    for one in inputs:
        value = value * one()
    return value


class Gaussian:
    def __init__(self, node: Any, inputs: Inputs) -> None:
        self.x, self.mean, self.sigma = (inputs.bind(one) for one in (node.x,node.mean,node.sigma))

    def value(self) -> float:
        x, mean, sigma = self.x(), self.mean(), self.sigma()
        if not (sigma > 0. and math.isfinite(sigma)):
            return math.nan
        squared = sigma * sigma
        if not squared > 0.:
            return math.nan
        arg = x - mean
        return float(kernels.fast_exp(arg * arg * (-.5 / squared)))


class Poisson:
    def __init__(self, node: Any, inputs: Inputs) -> None:
        if type(node._no_rounding) is not bool:
            raise NotPrepared
        self.x, self.mean = inputs.bind(node.x), inputs.bind(node.mean)
        self.no_rounding = node._no_rounding

    def value(self) -> float:
        x, mean = self.x(), self.mean()
        if not math.isfinite(x):
            return math.nan
        count = x if self.no_rounding else float(math.floor(x))
        value = _poisson_single(count, mean)
        return math.nan if value is None else value


class Constraints:
    def __init__(self, nll: Any) -> None:
        inputs = Inputs({name: float(value) for name,value in nll._global_values.items()})
        factories: dict[type[Any], Callable[[Any, Inputs], Gaussian | Poisson]] = {
            RooGaussian: Gaussian, RooPoisson: Poisson}
        self.terms: list[Gaussian | Poisson] = []
        for constraint in nll.constraints:
            if nll._constrained(constraint) & constraint.dependents():
                raise NotPrepared
            factory = factories.get(type(constraint))
            if factory is None:
                raise NotPrepared
            self.terms.append(factory(constraint,inputs))

    def total(self) -> float | None:
        value = 0.0
        for term in self.terms:
            raw = term.value()
            if not (raw > 0. and math.isfinite(raw)):
                return None
            value -= math.log(raw)
        return value


def prepare(nll: Any) -> Constraints | None:
    try:
        return Constraints(nll)
    except NotPrepared:
        return None
