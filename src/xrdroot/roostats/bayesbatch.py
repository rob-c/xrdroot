"""Bayesian bindings can evaluate pure parameter points without repeated object assignment.

A scalar point seeds the ordinary evaluator's normalization state. Valid
batches retain its sums and counters; custom models, diagnostics and large
likelihood warnings keep the scalar point order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..random import libm
from ..roofit.fitting import batchmodel, parameterbatch
from ..roofit.pdfs.basic import RooUniform
from ..roofit.pdfs.histfactory import ParamHistFunc
from ..roofit.pdfs.realsum import RooRealSumPdf
from ..roofit.variables import RooRealVar

#: Bound intermediate parameter/event/choice arrays, without changing integration limits.
_MAX_CELLS = 65536


def integrand(like: Any, dimensions: int | None = None) -> Any:
    """Return a vectorized binding only when the original calls are independent."""
    if not _supported(like):
        return None
    if dimensions is not None and dimensions != len(like.nll.params):
        return None
    size = _batch_size(like.nll.func)
    return _LikelihoodBatch(like, like.nll.params, size) if size > 1 else None


def _supported(like: Any) -> bool:
    from .bayesfuncs import Functor, Likelihood

    if type(like) is not Likelihood or type(like.nll) is not Functor:
        return False
    if not batchmodel.eligible(like.nll.func, like.nll.params):
        return False
    return _prior_compatible(like)


def _prior_compatible(like: Any) -> bool:
    from .bayesfuncs import Functor

    if like.prior is None:
        return True
    if type(like.prior) is not Functor:
        return False
    if not _same_parameters(like.nll.params, like.prior.params):
        return False
    return _uniform_supported(like.prior.func, like.prior.params)


def _uniform_supported(pdf: Any, parameters: Any) -> bool:
    if type(pdf) is not RooUniform or not batchmodel._native_instance(pdf):
        return False
    if pdf.getForceNumInt() or pdf._norm_range:
        return False
    names = frozenset(p.GetName() for p in parameters)
    return _fixed_uniform(pdf) and bool(pdf.norm({}, names) > 0.0)


def _same_parameters(first: Any, second: Any) -> bool:
    return len(first) == len(second) and all(a is b for a, b in zip(first, second, strict=False))


def _fixed_uniform(pdf: Any) -> bool:
    return all(
        type(node) is RooRealVar
        and batchmodel.node_supported(node)
        and node.getMax() > node.getMin()
        for node in pdf.servers()
    )


def _batch_size(nll: Any) -> int:
    choices = 1
    for node in nll.pdf._walk():
        if type(node) is ParamHistFunc:
            choices = max(choices, len(node.gammas))
        elif type(node) is RooRealSumPdf:
            choices = max(choices, len(node.funcs))
    return max(1, _MAX_CELLS // (len(nll.w) * choices))


class _LikelihoodBatch:
    """Each chunk has one point per row, with scalar fallback before any counters advance."""

    def __init__(self, like: Any, parameters: Any, size: int) -> None:
        self.like = like
        self.parameters = list(parameters)
        self.base = len(like.nll.params)
        self.size = size
        self.prior_names = (
            frozenset(p.GetName() for p in like.prior.params)
            if like.prior is not None
            else frozenset()
        )

    def __call__(self, points: Any) -> Any:
        if not len(points):
            return np.empty(0)
        values = [self.scalars(points[:1])]
        for start in range(1, len(points), self.size):
            block = points[start : start + self.size]
            found = self.vector(block)
            values.append(self.scalars(block) if found is None else found)
        return np.concatenate(values)

    def scalars(self, points: Any) -> Any:
        found = []
        for point in points:
            for parameter, value in zip(
                self.parameters[self.base :], point[self.base :], strict=False
            ):
                parameter.setVal(float(value))
            found.append(self.like(point[: self.base]))
        return np.asarray(found, dtype=np.float64)

    def restore(self, point: Any) -> None:
        for parameter, value in zip(self.parameters, point, strict=False):
            parameter.setVal(float(value))

    def vector(self, points: Any) -> Any:
        with np.errstate(all="ignore"):
            nll = parameterbatch.evaluate(self.like.nll.func, self.parameters, points)
            if nll is None:
                return None
            found = self.probabilities(nll, points)
        peak = float(np.nanmax(found, initial=0.0))
        if peak >= 1e10:
            return None
        self.like.nll.calls += len(points)
        if self.like.prior is not None:
            self.like.prior.calls += len(points)
        if peak > self.like.max:
            self.like.max = peak
        self.restore(points[-1])
        return found

    def probabilities(self, nll: Any, points: Any) -> Any:
        found = libm.exp(-(nll - self.like.offset))
        if self.like.prior is not None:
            ctx = batchmodel.clipped(self.parameters, points)
            found = found * self.like.prior.func.value(ctx, self.prior_names)
        return np.asarray(found, dtype=np.float64)
