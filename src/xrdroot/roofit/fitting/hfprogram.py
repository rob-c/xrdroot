"""Small native HistFactory expressions keep NumPy's operation order without tiny arrays."""
from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from types import CodeType
from typing import Any

import numpy as np

from ..functions import RooProduct
from ..pdfs.histfactory import FlexibleInterpVar, ParamHistFunc, RooBinWidthFunction
from ..pdfs.histpdf import RooHistFunc
from ..pdfs.realsum import RooRealSumPdf
from ..variables import RooConstVar, RooRealVar
from . import hfdata

#: Leave ample headroom below binary64 overflow, including intermediate products.
MAX_BOUND = 1e200
#: Short straight-line blocks keep generated functions below the Halstead limit too.
BLOCK_SIZE = 4


class NotCompiled(Exception):
    """The ordinary evaluator owns this model or its numeric domain."""


class Program:
    """Emit scalar arithmetic for each bin, sharing only values independent of that bin."""

    def __init__(self, columns: dict[str, Any], count: int) -> None:
        self.columns, self.count = columns, count
        self.refs: list[Any] = []
        self.lines: list[str] = []
        self.bounds: list[float] = []
        self.memo: dict[tuple[int, int | None], int] = {}
        self.active: set[tuple[int, int | None]] = set()
        self.static: dict[int, Any] = {}

    def reference(self, node: Any) -> str:
        self.refs.append(node)
        return f"_refs[{len(self.refs) - 1}]"

    def value(self, slot: int) -> str:
        return f"_v[{slot}]"

    def add(self, expression: str, bound: float) -> int:
        if not math.isfinite(bound) or bound > MAX_BOUND:
            raise NotCompiled
        slot = len(self.lines)
        self.lines.append(f" _v[{slot}] = {expression}")
        self.bounds.append(bound)
        return slot

    def scalar(self, node: Any) -> bool:
        return type(node) in (RooConstVar, FlexibleInterpVar, RooBinWidthFunction) or (
            type(node) is RooRealVar and node._name not in self.columns)

    def emit(self, node: Any, index: int) -> int:
        key = (id(node), None if self.scalar(node) else index)
        if key in self.memo:
            return self.memo[key]
        if key in self.active:
            raise NotCompiled
        self.active.add(key)
        try:
            expression, bound = self.expression(node, index)
            slot = self.add(expression, bound)
        finally:
            self.active.remove(key)
        self.memo[key] = slot
        return slot

    def expression(self, node: Any, index: int) -> tuple[str, float]:
        builders = {RooRealVar: self.variable, RooConstVar: self.variable,
                    RooProduct: self.product, RooRealSumPdf: self.summation,
                    RooHistFunc: self.histogram, ParamHistFunc: self.parameter_histogram,
                    FlexibleInterpVar: self.interpolation, RooBinWidthFunction: self.width}
        build = builders.get(type(node))
        if build is None:
            raise NotCompiled
        return build(node, index)

    def variable(self, node: Any, index: int) -> tuple[str, float]:
        if type(node) is RooRealVar and node._name in self.columns:
            if self.columns[node._name].dtype != np.float64:
                raise NotCompiled
            value = float(self.columns[node._name][index])
            return repr(value), abs(value)
        bound = abs(float(node.getVal()))
        if type(node) is RooRealVar and not node.isConstant():
            bound = max(bound, abs(node.getMin()), abs(node.getMax()))
        return f"float({self.reference(node)}.getVal())", bound

    def product(self, node: Any, index: int) -> tuple[str, float]:
        expression, bound = "1.0", 1.0
        for term in hfdata.members(node.terms):
            slot = self.emit(term, index)
            bound *= self.bounds[slot]
            result = self.add(f"({expression} * {self.value(slot)})", bound)
            expression = self.value(result)
        return expression, bound

    def summation(self, node: Any, index: int) -> tuple[str, float]:
        funcs, coefs = hfdata.members(node.funcs), hfdata.members(node.coefs)
        if len(funcs) != len(coefs):
            raise NotCompiled
        coefficients = [self.emit(coef, index) for coef in coefs]
        expression, bound = "0.0", 0.0
        for coefficient, func in zip(coefficients, funcs, strict=False):
            slot = self.emit(func, index)
            bound += self.bounds[coefficient] * self.bounds[slot]
            result = self.add(f"({expression} + ({self.value(coefficient)} * {self.value(slot)}))",
                              bound)
            expression = self.value(result)
        return expression, bound

    def histogram(self, node: Any, index: int) -> tuple[str, float]:
        mode = type(node.order), node.order, type(node.density), node.density
        if mode != (int, 0, bool, False):
            raise NotCompiled
        obs = hfdata.members(node.observables)
        if not obs or len(obs) != len(node._hist_vars):
            raise NotCompiled
        self.observed(obs)
        if not all(_finite_axis(var) for var in hfdata.members(node.data._vars)):
            raise NotCompiled
        if id(node) not in self.static:
            self.static[id(node)] = np.broadcast_to(node.compute(dict(self.columns)),
                                                    (self.count,)).copy()
        value = float(self.static[id(node)][index])
        return repr(value), abs(value)

    def observed(self, variables: list[Any]) -> None:
        if any(type(var) is not RooRealVar or var._name not in self.columns for var in variables):
            raise NotCompiled
        if any(self.columns[var._name].dtype != np.float64 for var in variables):
            raise NotCompiled

    def parameter_histogram(self, node: Any, index: int) -> tuple[str, float]:
        variables = hfdata.members(node.vars)
        self.observed(variables)
        if id(node) not in self.static:
            self.static[id(node)] = np.broadcast_to(node._index(self.columns), (self.count,))
        gammas = hfdata.members(node.gammas)
        slot = self.emit(gammas[int(self.static[id(node)][index])], index)
        return self.value(slot), self.bounds[slot]

    def interpolation(self, node: Any, index: int) -> tuple[str, float]:
        parameters = hfdata.members(node.params)
        if len(parameters) > 4 or any(p._name in self.columns for p in parameters):
            raise NotCompiled
        if not _interpolation_domain(node):
            raise NotCompiled
        for parameter in parameters:
            slot = self.emit(parameter, index)
            if self.bounds[slot] > 8.:
                raise NotCompiled
        return f"float({self.reference(node)}.compute({{}}))", 1e140

    def width(self, node: Any, index: int) -> tuple[str, float]:
        # These functions return one inside binned_likelihood, independent of their divide flag.
        return "1.0", 1.0

    def build(self, pdf: Any) -> tuple[Callable[[], tuple[float, ...]], str]:
        slots = [self.emit(pdf, index) for index in range(self.count)]
        blocks = [self.lines[start:start + BLOCK_SIZE]
                  for start in range(0, len(self.lines), BLOCK_SIZE)]
        source = "\n\n".join(f"def _step{index}(_v):\n" + "\n".join(lines)
                              for index, lines in enumerate(blocks))
        calls = "\n".join(f" _step{index}(_v)" for index in range(len(blocks)))
        result = ",".join(self.value(slot) for slot in slots)
        source += (f"\n\ndef predict():\n _v = [0.0] * {len(self.lines)}\n" + calls
                   + f"\n return ({result},)\n")
        scope: dict[str, Any] = {"_refs": self.refs}
        exec(_compiled(source), scope)
        return scope["predict"], source


def _finite_axis(var: Any) -> bool:
    return bool(0. < var.getMax() - var.getMin() < math.inf)


def _interpolation_domain(node: Any) -> bool:
    return (.01 <= node._boundary <= 4.
            and all(.01 <= value <= 100. for value in [node._nominal, *node._low, *node._high])
            and all(code in (0, 1, 4, 5) for code in node._codes))


@lru_cache(maxsize=128)
def _compiled(source: str) -> CodeType:
    """Code can be shared across fits; each execution still owns its live model references."""
    return compile(source, "<xrdroot native histogram>", "exec")
