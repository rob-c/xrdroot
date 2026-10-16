"""Counting studies fit one event many times, without needing column allocation.

These scalar formulas follow the same likelihood kernels and integration
rules as the general evaluator. Unsupported models and invalid points use
that evaluator, including its diagnostic messages and NaN payloads.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...random import libm
from .. import kernels
from ..functions import RooAddition, RooProduct
from ..pdfs.basic import RooGaussian
from ..pdfs.prodpdf import RooProdPdf
from ..pdfs.shapes import RooPoisson, _poisson_single
from ..variables import RooConstVar, RooRealVar


class _NotScalar(Exception):
    """The general evaluator owns this model or this evaluation point."""


def _real(node: Any, ctx: dict[str, float]) -> float:
    if type(node) is RooConstVar:
        return float(node.getVal())
    if type(node) is RooRealVar:
        return ctx.get(node.GetName(), node.getVal())
    if type(node) is RooProduct:
        found = 1.0
        for term in node.terms:
            found = found * _real(term, ctx)
        return found
    if type(node) is RooAddition:
        # Keep NumPy's reduction order, including pairwise summation for longer lists.
        return float(np.sum([_real(term, ctx) for term in node.terms]))
    raise _NotScalar


def _positive(value: float) -> float:
    if not (value > 0.0 and math.isfinite(value)):
        raise _NotScalar
    return value


def _bounds(node: Any) -> tuple[float, float]:
    if type(node) is not RooRealVar or node.getBinning().isParameterized():
        raise _NotScalar
    return node.getMin(), node.getMax()


def _gaussian_integral(low: float, high: float, mean: float, sigma: float) -> float:
    """The same upper-tail formula as ``mathfuncs.gaussian_integral``, term for term."""
    scale = 0.5 * math.sqrt(2 * math.pi) * sigma
    xscale = math.sqrt(2.0) * sigma
    smin, smax = (low - mean) / xscale, (high - mean) / xscale
    ecmin, ecmax = math.erfc(abs(smin)), math.erfc(abs(smax))
    if smin * smax < 0.0:
        cond = 2.0 - (ecmin + ecmax)
    else:
        cond = ecmax - ecmin if smax <= 0.0 else ecmin - ecmax
    return scale * cond


def _gaussian(node: Any, ctx: dict[str, float], names: frozenset[str]) -> float:
    x, mean, sigma = (_real(one, ctx) for one in (node.x, node.mean, node.sigma))
    _positive(sigma)
    arg = x - mean
    raw = float(kernels.fast_exp(arg * arg * (-0.5 / (sigma * sigma))))
    if not names:
        return raw
    if names == frozenset([node.x.GetName()]):
        over, other = node.x, mean
    elif names == frozenset([node.mean.GetName()]):
        over, other = node.mean, x
    else:
        raise _NotScalar
    norm = _gaussian_integral(*_bounds(over), other, sigma)
    return raw / _positive(norm)


def _poisson(node: Any, ctx: dict[str, float], names: frozenset[str]) -> float:
    x, mean = _real(node.x, ctx), _real(node.mean, ctx)
    count = x if node._no_rounding else float(math.floor(x))
    raw = _poisson_single(count, mean)
    if raw is None:
        raise _NotScalar
    if not names:
        return raw
    if names != frozenset([node.x.GetName()]):
        raise _NotScalar
    low, high = _bounds(node.x)
    norm = node._over_count(mean, max(0.0, low), high)
    return raw / _positive(norm)


def _product(node: Any, ctx: dict[str, float], names: frozenset[str]) -> float:
    if names and not node._factorizes(names):
        raise _NotScalar
    found = 1.0
    for part in node.pdfs:
        own = node.factor_nset(part, names)
        if own or not names:
            found = found * _density(part, ctx, own)
    return found


#: Subclasses retain their own compute, normalization and diagnostic behavior.
_DENSITIES = {RooGaussian: _gaussian, RooPoisson: _poisson, RooProdPdf: _product}


def _density(node: Any, ctx: dict[str, float], names: frozenset[str]) -> float:
    evaluate = _DENSITIES.get(type(node))
    if evaluate is None:
        raise _NotScalar
    if type(node) is RooProdPdf:
        return evaluate(node, ctx, names)
    names = names & node.dependents()
    if names and (node._norm_range or node.getForceNumInt()):
        raise _NotScalar
    return evaluate(node, ctx, names)


def _global_context(given: dict[str, Any]) -> dict[str, float]:
    # Other scalar types can have their own arithmetic or reject conversion entirely.
    # In particular, NumPy's float32 operations must not silently become doubles.
    if not all(type(value) in (float, int, np.float64) for value in given.values()):
        raise _NotScalar
    return {name: float(value) for name, value in given.items()}


def _evaluated(nll: Any) -> float:
    columns = {name: float(values[0]) for name, values in nll.columns.items()}
    global_values = _global_context(nll._global_values)
    probability = _positive(_density(nll.pdf, columns, nll.nset))
    weight = float(nll.w[0])
    weight = weight * weight if nll._weight_squared else weight
    # The first Kahan addition starts at zero; constraints have their own ordered sum.
    total = 0.0 + (-weight * float(libm.log(probability)))
    return total + _constraints(nll, global_values) if nll.constraints else total


def _constraints(nll: Any, global_values: dict[str, float]) -> float:
    constraints = 0.0
    for constraint in nll.constraints:
        value = _positive(_density(constraint, global_values, nll._constrained(constraint)))
        constraints -= math.log(value)
    return constraints


def constraint_sum(nll: Any) -> float | None:
    """Scalar constraints stay scalar even when the likelihood has many data bins."""
    try:
        return _constraints(nll, _global_context(nll._global_values))
    except (_NotScalar, ValueError, TypeError, OverflowError, ZeroDivisionError):
        return None


def one_entry(nll: Any) -> float | None:
    """A complete one-event likelihood, or ``None`` to keep the general path."""
    if nll.extended or nll.binned or len(nll.w) != 1 or nll.rng:
        return None
    try:
        return _evaluated(nll)
    except (_NotScalar, ValueError, TypeError, OverflowError, ZeroDivisionError):
        return None
