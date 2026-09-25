"""What a fit minimises: ``FitUtil``'s chi-squares and Poisson likelihood, term for term.

* :func:`chi2` is ``EvaluateChi2``: ``((y - f) / e)^2`` summed over the
  points, with ``e`` the point's error - one with ``W`` - or, for Pearson's
  chi-square (``P``), the root of what the function expects, over the
  bin's effective weight when that is ``PW``;
* :func:`effective_chi2` is ``EvaluateChi2Effective``, for a graph with
  errors in x: the error in y widened by the x error times the function's
  slope there - Richardson's derivative, with ROOT's step - and, for
  asymmetric errors, the lower error where the function is below the point
  and the upper where it is above;
* :func:`poisson` is ``EvaluatePoissonLogL``: ``f - y + y log(y / f)`` per
  bin, which with the saturated model's constant is half Baker and
  Cousins's chi-square, and ``WL``'s weighted version of it; without the
  ``f - y`` it is the multinomial likelihood of ``MULTI``.

A term that comes out infinite or NaN is capped at ``DBL_MAX / n``, as ROOT
caps it, and the terms are added in order, as ROOT's loop adds them. The
function is evaluated at the bin centres, or integrated over each bin with
``I`` - a 21-point Gauss-Kronrod rule on the bin in one dimension, the
10-point Gauss rule on each axis of it in more, where ROOT's integrator is
adaptive and would agree to its tolerance of 1e-9 - and multiplied by the
bin's size with ``WIDTH``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from ..filling import running
from ..function import numeric
from .data import ASYM_ERROR, FitData

__all__ = ["Predictor", "chi2", "effective_chi2", "poisson", "eval_log"]

Array = Any
Cost = Callable[[Array], float]

#: ``2 * DBL_MIN``: below this ``Util::EvalLog`` continues the logarithm by a straight line.
TINY = 2.0 * sys.float_info.min
#: ``EvaluateChi2Effective``'s step: 1% of the x error, never below ``8e-8 (|x| + 1e-8)``.
STEP_FRACTION, PRECISION = 0.01, 1e-8
#: The 10-point Gauss rule on [-1, 1], nodes and weights, for integrating over a cell.
GAUSS_NODES, GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(10)


def eval_log(x: Array) -> Array:
    """``ROOT::Math::Util::EvalLog``: the logarithm, continued linearly below ``2 DBL_MIN``."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        line = x / TINY + np.log(TINY) - 1.0
        return np.where(x <= TINY, line, np.log(np.maximum(x, TINY)))


def _cell_rule(data: FitData) -> tuple[Array, Array]:
    """The points to evaluate over every cell, and each one's weight, summing to one per cell."""
    low, high = data.x, data.upper
    half, centre = 0.5 * (high - low), 0.5 * (high + low)
    if data.ndim == 1:
        nodes, weights = numeric._NODES, 0.5 * numeric._KRONROD
        points = centre[:, :1] + half[:, :1] * nodes
        return points.reshape(-1), np.tile(weights, len(low))
    grids = np.meshgrid(*([GAUSS_NODES] * data.ndim), indexing="ij")
    nodes = np.stack([grid.ravel() for grid in grids], axis=1)
    weight = np.prod(np.meshgrid(*([GAUSS_WEIGHTS] * data.ndim), indexing="ij"), axis=0).ravel()
    points = centre[:, None, :] + half[:, None, :] * nodes[None, :, :]
    return points.reshape(-1, data.ndim), np.tile(weight / 2.0**data.ndim, len(low))


class Predictor:
    """What the function says each point should be, for a set of parameters."""

    __slots__ = ("function", "data", "_points", "_weights", "_scale")

    def __init__(self, function: Any, data: FitData) -> None:
        self.function = function
        self.data = data
        options = data.options
        self._weights: Array = None
        self._scale: Array = 1.0
        if options.integral:
            self._points, self._weights = _cell_rule(data)
        elif options.bin_volume:
            centres = 0.5 * (data.upper + data.x)
            self._points = centres[:, 0] if data.ndim == 1 else centres
        else:
            self._points = data.coordinates()
        if options.bin_volume:
            volume = np.prod(np.abs(data.upper - data.x), axis=1)
            self._scale = volume / data.ref_volume if options.norm_bin_volume else volume

    def __call__(self, params: Array) -> Array:
        values = np.asarray(self.function.evaluate(self._points, params), dtype=np.float64)
        if self._weights is not None:
            values = (values * self._weights).reshape(self.data.size, -1).sum(axis=1)
        return values * self._scale


def _capped(terms: Array, n: int) -> float:
    """The sum in order, each term capped at ``DBL_MAX / n`` - which also catches a NaN."""
    ceiling = sys.float_info.max / max(n, 1)
    return running(0.0, np.where(terms < ceiling, terms, ceiling))


def _pearson_inverse(data: FitData, f: Array) -> Array:
    """Pearson's ``1/e``: ``sqrt(w / f)``, ``w`` the bin's inverse weight for ``PW``."""
    weight: Array = 1.0
    if data.options.exp_errors and not data.options.errors1 and data.weighted:
        inverse = 1.0 / data.error
        global_weight = data.sum_content / data.sum_error2
        weight = np.where(data.y != 0, data.y * inverse * inverse, global_weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.where(f > 0, weight / np.where(f > 0, f, 1.0), 0.0))


def chi2(data: FitData, predict: Predictor) -> Cost:
    """``EvaluateChi2``: Neyman's chi-square, or Pearson's with option ``P``."""
    inverse = np.ones(data.size) if data.error is None else 1.0 / data.error

    def cost(params: Array) -> float:
        f = predict(params)
        weight = _pearson_inverse(data, f) if data.options.exp_errors else inverse
        with np.errstate(over="ignore", invalid="ignore"):
            terms = ((data.y - f) * weight) ** 2
        return _capped(np.where(weight > 0, terms, 0.0), data.size)

    return cost


def _slope(function: Any, x: Array, ex: Array, params: Array) -> Array:
    """``RichardsonDerivator::Derivative1`` at each point, stepped as ROOT steps it."""
    h = np.maximum(STEP_FRACTION * np.abs(ex), 8.0 * PRECISION * (np.abs(x) + PRECISION))

    def at(step: Array) -> Array:
        return np.asarray(function.evaluate(x + step, params), dtype=np.float64)

    d0 = at(h) - at(-h)
    d2 = at(h / 2) - at(-h / 2)
    return (1 / (2.0 * h)) * (8 * d2 - d0) / 3.0


def effective_chi2(data: FitData, function: Any) -> Cost:
    """``EvaluateChi2Effective``: the y error widened by the x error times the slope."""
    x, ex = data.coordinates(), data.xerr
    sloped = ex != 0

    def cost(params: Array) -> float:
        f = np.asarray(function.evaluate(x, params), dtype=np.float64)
        residual = data.y - f
        ey = data.error
        if data.kind == ASYM_ERROR:
            ey = np.where(residual < 0, data.yhigh, data.ylow)
        e2 = ey * ey
        if sloped.any():
            spread = ex[sloped] * _slope(function, x[sloped], ex[sloped], params)
            e2 = e2.copy()
            e2[sloped] += spread * spread
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            terms = np.where(e2 > 0, 1.0 / np.where(e2 > 0, e2, 1.0), 0.0) * residual * residual
        return _capped(terms, data.size)

    return cost


def poisson(
    data: FitData, predict: Predictor, extended: bool = True, squares: bool = False
) -> Cost:
    """``EvaluatePoissonLogL``: the Poisson likelihood with the saturated model's constant.

    ``squares`` is the weighted likelihood's second pass, the one its
    errors come from: each bin's likelihood weighted by its effective
    weight, ``e^2 / y``, and an empty bin by the histogram's average one.
    """
    y = data.y
    filled = y > 0
    log_y = np.where(filled, eval_log(np.where(filled, y, 1.0)), 0.0)
    weights: Array = None
    if squares:
        error = data.error if data.error is not None else np.ones(data.size)
        average = data.sum_error2 / data.sum_content
        weights = np.where(y != 0, error * error / np.where(y != 0, y, 1.0), average)

    def cost(params: Array) -> float:
        f = np.maximum(predict(params), 0.0)
        if weights is None:
            terms = np.where(filled, y * (log_y - eval_log(f)), 0.0)
            if extended:
                terms = (f - y) + terms
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = eval_log(f / np.where(y != 0, y, 1.0))
            terms = np.where(y != 0, -weights * y * ratio, 0.0)
            if extended:
                terms = terms + weights * (f - y)
        return running(0.0, terms)

    return cost
