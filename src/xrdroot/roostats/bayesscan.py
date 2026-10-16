"""Posterior scan points keep independent adaptive heaps while sharing model evaluations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..roofit import cubature
from ..roofit.cubature_batch import adaptive_integrals
from ..roofit.fitting import batchmodel
from . import bayesbatch

#: Limit simultaneous work arrays; each integral still receives ROOT's full work space.
_WORK_SLOTS = 8000000
#: Public evaluation overrides require their ordinary call sequence.
_OVERRIDES = frozenset(("getVal", "value", "compute", "finish"))


def sample(exact: Any, poi: Any, low: float, high: float, bins: int) -> list[float] | None:
    """Sample only bounded pure posteriors; other functions retain scalar evaluation order."""
    post = _posterior(exact, poi)
    if post is None or not bayesbatch._supported(post.like):
        return None
    if not _domain(post, low, high, bins):
        return None
    parameters = [*post.like.nll.params, poi]
    if not _supported(post, parameters):
        return None
    dx = (high - low) / bins
    xs = np.asarray([low + dx * i for i in range(bins + 1)])
    return _scan(post, parameters, xs)


def _posterior(exact: Any, poi: Any) -> Any:
    from .bayesfuncs import Posterior, PosteriorFunction

    if type(exact) is not Posterior or not _OVERRIDES.isdisjoint(exact.__dict__):
        return None
    post = exact.function
    if type(post) is not PosteriorFunction or not _OVERRIDES.isdisjoint(post.__dict__):
        return None
    return post if exact.poi is poi and post.poi is poi else None


def _domain(post: Any, low: float, high: float, bins: int) -> bool:
    return (
        bins > 0
        and math.isfinite(low)
        and math.isfinite(high)
        and 2 <= len(post.lows) <= 15
        and len(post.lows) == len(post.highs)
        and len(post.lows) == len(post.like.nll.params)
    )


def _supported(post: Any, parameters: Any) -> bool:
    return (
        batchmodel.bounded_posterior(post.like, parameters)
        and bayesbatch._batch_size(post.like.nll.func) > 1
    )


def _options(post: Any) -> tuple[float, float, int, int, int]:
    from .bayesfuncs import MULTI_DIM

    return (
        MULTI_DIM["abs"],
        MULTI_DIM["rel"],
        post.calls or MULTI_DIM["calls"],
        0,
        MULTI_DIM["size"],
    )


def _scan(post: Any, parameters: Any, xs: Any) -> list[float]:
    options = _options(post)
    limits, _, _ = cubature._start(len(post.lows), options)
    width = max(1, _WORK_SLOTS // (limits.iwk + 10))
    values = []
    for start in range(0, len(xs), width):
        packet = _ScanValues(post, parameters, xs[start : start + width])
        results = adaptive_integrals(packet, post.lows, post.highs, len(packet.xs), *options)
        values.extend(packet.finish(results))
    return values


class _ScanValues:
    """Retain each group's final bindings and replay diagnostics in scan order."""

    def __init__(self, post: Any, parameters: Any, xs: Any) -> None:
        self.post, self.xs = post, xs
        self.adapter = bayesbatch._LikelihoodBatch(
            post.like, parameters, bayesbatch._batch_size(post.like.nll.func)
        )
        self.last: dict[int, Any] = {}
        self.peaks: dict[int, float] = {}
        self.calls = post.like.nll.calls
        self.prior_calls = post.like.prior.calls if post.like.prior is not None else 0
        self.maximum = post.like.max

    def __call__(self, groups: Any, points: Any) -> Any:
        combined = np.column_stack([points, self.xs[groups]])
        values = self.adapter(combined)
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            self.last[int(group)] = combined[indices[-1]].copy()
            peak = float(np.nanmax(values[indices], initial=0.0))
            if peak > self.peaks.get(int(group), 0.0):
                self.peaks[int(group)] = peak
        return values

    def finish(self, results: Any) -> list[float]:
        values = []
        for index, result in enumerate(results):
            self.restore(index, result.neval)
            error = result.relerr * abs(result.value)
            values.append(self.post.finish(float(self.xs[index]), result.value, error))
        return values

    def restore(self, index: int, evaluations: int) -> None:
        self.adapter.restore(self.last[index])
        self.calls += evaluations
        self.post.like.nll.calls = self.calls
        if self.post.like.prior is not None:
            self.prior_calls += evaluations
            self.post.like.prior.calls = self.prior_calls
        peak = self.peaks.get(index, 0.0)
        if peak > self.maximum:
            self.maximum = peak
        self.post.like.max = self.maximum
