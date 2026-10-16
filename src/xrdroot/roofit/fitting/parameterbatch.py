"""Evaluate independent HistFactory parameter points with the original event sums.

The parameter axis broadcasts against the event axis. Each point retains
its own Kahan carry, and constraints retain their separate ordered sum.
Unknown models and invalid regions return to the ordinary evaluator.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...random import libm
from ...stats import log_gamma
from .. import kernels
from ..pdfs.histfactory import binned_likelihood
from . import batchmodel
from .binned import binned_part
from .kahan import Kahan


class _NotBatch(Exception):
    """The scalar evaluator owns this model or parameter region."""


def evaluate(nll: Any, parameters: Any, points: Any) -> np.ndarray[Any, Any] | None:
    """Return one likelihood per point without changing physical parameter values."""
    if not batchmodel.eligible(nll, parameters):
        return None
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != len(parameters) or not np.isfinite(points).all():
        return None
    if not len(points):
        return np.empty(0, dtype=np.float64)
    try:
        given = batchmodel.clipped(parameters, points)
        total = _channels(nll, given, len(points))
        if nll.constraints:
            total = total + _constraints(nll, given, len(points))
        return np.asarray(total - nll._offset_value, dtype=np.float64)
    except (_NotBatch, ValueError, TypeError, OverflowError, ZeroDivisionError, IndexError):
        return None


def _channels(nll: Any, given: dict[str, Any], count: int) -> Any:
    channels = getattr(nll.pdf, "channels", None)
    if channels is None:
        return _channel(nll, nll.pdf, None, given, count, 0)
    total = np.zeros(count)
    states = nll.columns[nll.pdf.index.GetName()]
    for label, pdf in channels.items():
        keep = states == nll.pdf.index.lookupIndex(label)
        if keep.any():
            total = total + _channel(nll, pdf, keep, given, count, len(channels))
    return total


def _channel(
    nll: Any, pdf: Any, keep: Any, given: dict[str, Any], count: int, simulated: int
) -> Any:
    columns, weights = _channel_data(nll, keep)
    mu = _expectations(pdf, columns, given, count, len(weights))
    total, carry = _event_sum(mu, weights, count)
    if simulated > 1:
        total, carry = _add(
            total, carry, Kahan().extend(weights.tolist()).total * math.log(simulated)
        )
    return total


def _channel_data(nll: Any, keep: Any) -> tuple[Any, Any]:
    columns = nll.columns if keep is None else {k: v[keep] for k, v in nll.columns.items()}
    weights = nll.w if keep is None else nll.w[keep]
    return columns, weights * weights if nll._weight_squared else weights


def _expectations(pdf: Any, columns: Any, given: Any, count: int, events: int) -> Any:
    ctx = {k: v[None, ...] for k, v in columns.items()}
    ctx.update({k: v[:, None] for k, v in given.items() if k not in columns})
    with binned_likelihood():
        mu = np.broadcast_to(
            np.asarray(binned_part(pdf).compute(ctx), dtype=np.float64), (count, events)
        )
    if not np.isfinite(mu).all() or not (mu > 0).all():
        raise _NotBatch
    return mu


def _event_sum(mu: Any, weights: Any, count: int) -> tuple[Any, Any]:
    total, carry = np.zeros(count), np.zeros(count)
    for expectation, observed in zip(mu.T, weights.tolist(), strict=False):
        term = expectation - observed * libm.log(expectation) + log_gamma(observed + 1.0)
        if abs(observed) < 1e-10:
            term = np.where(np.abs(expectation) < 1e-10, 0.0, term)
        total, carry = _add(total, carry, term)
    return total, carry


def _add(total: Any, carry: Any, term: Any) -> tuple[Any, Any]:
    corrected = term - carry
    updated = total + corrected
    return updated, (updated - total) - corrected


def _constraints(nll: Any, given: dict[str, Any], count: int) -> Any:
    ctx = {**given, **nll._global_values}
    total = np.zeros(count)
    with kernels.likelihood(), np.errstate(divide="ignore", invalid="ignore"):
        for constraint in nll.constraints:
            value = np.asarray(
                constraint.value(ctx, nll._constrained(constraint)), dtype=np.float64
            )
            if not np.isfinite(value).all() or not (value >= 0).all():
                raise _NotBatch
            total = total - libm.log(value)
    return total
