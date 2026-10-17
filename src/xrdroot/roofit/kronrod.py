"""Embedded Gauss-Kronrod rules and adaptive subdivision in ROOT's arithmetic order.

The Gauss and Kronrod estimates share their function evaluations. Their
difference is rescaled by the integral's absolute deviation, and the region
with the largest error is bisected until the requested accuracy is reached.
Region storage order is retained for the final sum, rather than using a
pairwise or compensated sum that would change ROOT's last bits.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from ..random.libm import power
from .kronrodrules import RULES

Integrand = Callable[[Any], Any]
#: IEEE double precision constants used by the error estimator.
EPSILON, MINIMUM = float(np.finfo(float).eps), float(np.finfo(float).tiny)


class Estimate(NamedTuple):
    """A rule's value, absolute error and absolute integrals used to estimate that error."""

    value: float
    error: float
    absolute: float
    deviation: float


class Result(NamedTuple):
    """The best estimate remains available when roundoff or a subdivision limit stops work."""

    value: float
    error: float
    status: str
    evaluations: int


def _values(func: Integrand, nodes: tuple[float, ...], a: float, b: float) -> tuple[Any, ...]:
    center, half = 0.5 * (a + b), 0.5 * (b - a)
    order = [*range(1, len(nodes) - 1, 2), *range(0, len(nodes) - 1, 2)]
    points = [center]
    for index in order:
        offset = half * nodes[index]
        points.extend((center - offset, center + offset))
    values = np.broadcast_to(np.asarray(func(np.array(points)), dtype=float), (len(points),))
    left, right = np.empty(len(order)), np.empty(len(order))
    for at, index in enumerate(order):
        left[index], right[index] = values[2 * at + 1 : 2 * at + 3]
    return float(values[0]), left, right, half


def _sums(center: float, left: Any, right: Any, gauss: Any, kronrod: Any) -> tuple[float, ...]:
    n = len(kronrod)
    small = center * gauss[-1] if n % 2 == 0 else 0.0
    large = center * kronrod[-1]
    absolute = abs(large)
    for j in range((n - 1) // 2):
        index = 2 * j + 1
        both = float(left[index]) + float(right[index])
        small += gauss[j] * both
        large += kronrod[index] * both
        absolute += kronrod[index] * (abs(float(left[index])) + abs(float(right[index])))
    for index in range(0, n - 1, 2):
        large += kronrod[index] * (float(left[index]) + float(right[index]))
        absolute += kronrod[index] * (abs(float(left[index])) + abs(float(right[index])))
    return small, large, absolute


def _deviation(center: float, left: Any, right: Any, weights: Any, mean: float) -> float:
    found = weights[-1] * abs(center - mean)
    for index in range(len(weights) - 1):
        found += weights[index] * (abs(float(left[index]) - mean) + abs(float(right[index]) - mean))
    return float(found)


def _error(error: float, absolute: float, deviation: float) -> float:
    error = abs(error)
    if deviation != 0 and error != 0:
        scale = float(power(200 * error / deviation, 1.5))
        error = deviation * scale if scale < 1 else deviation
    if absolute > MINIMUM / (50 * EPSILON):
        error = max(error, 50 * EPSILON * absolute)
    return error


def rule(func: Integrand, a: float, b: float, points: int = 21) -> Estimate:
    """One embedded rule, with every addition in the order used by ROOT's qk routines."""
    nodes, gauss, kronrod = RULES[points]
    center, left, right, half = _values(func, nodes, a, b)
    small, large, absolute = _sums(center, left, right, gauss, kronrod)
    deviation = _deviation(center, left, right, kronrod, large * 0.5) * abs(half)
    error = (large - small) * half
    absolute *= abs(half)
    return Estimate(large * half, _error(error, absolute, deviation), absolute, deviation)


class _Regions:
    """Keep the worse child in its parent's slot and append the other, as GSL does."""

    def __init__(self, a: float, b: float, estimate: Estimate, limit: int) -> None:
        self.bounds = [(a, b)]
        self.estimates = [estimate]
        self.order = [0] * max(limit, 2)
        self.at = 0
        self.limit = limit

    def worst(self) -> int:
        return self.at

    def replace(self, at: int, left: Estimate, right: Estimate) -> None:
        a, b = self.bounds[at]
        middle = 0.5 * (a + b)
        first, second = ((a, middle), left), ((middle, b), right)
        if right.error > left.error:
            first, second = second, first
        self.bounds[at], self.estimates[at] = first
        self.bounds.append(second[0])
        self.estimates.append(second[1])
        self._ordered()

    def _ordered(self) -> None:
        """GSL's partial error ordering, including its two directions for ties."""
        last = len(self.estimates) - 1
        maximum = self.order[0]
        if last < 2:
            self.order[:2] = [0, 1]
            self.at = maximum
            return
        top = last if last < self.limit // 2 + 2 else self.limit - last + 1
        index = 1
        while (
            index < top and self.estimates[maximum].error < self.estimates[self.order[index]].error
        ):
            self.order[index - 1] = self.order[index]
            index += 1
        self.order[index - 1] = maximum
        self._insert_last(last, top - 1, index - 2)
        self.at = self.order[0]

    def _insert_last(self, last: int, at: int, stop: int) -> None:
        while at > stop and self.estimates[last].error >= self.estimates[self.order[at]].error:
            self.order[at + 1] = self.order[at]
            at -= 1
        self.order[at + 1] = last

    def total(self) -> float:
        total = 0.0
        for estimate in self.estimates:
            total += estimate.value
        return total


def _first(estimate: Estimate, tolerance: float, limit: int) -> str:
    if estimate.error <= 50 * EPSILON * estimate.absolute and estimate.error > tolerance:
        return "roundoff"
    converged = estimate.error <= tolerance and estimate.error != estimate.deviation
    if converged or estimate.error == 0:
        return "success"
    return "limit" if limit == 1 else "continue"


def _small(a: float, middle: float, b: float) -> bool:
    bound = (1 + 100 * EPSILON) * (abs(middle) + 1000 * MINIMUM)
    return abs(a) <= bound and abs(b) <= bound


def integrate(
    func: Integrand,
    a: float,
    b: float,
    eps_abs: float = 1e-7,
    eps_rel: float = 1e-7,
    limit: int = 100,
    points: int = 21,
) -> Result:
    """Adaptive qag subdivision, retaining the best result when accuracy cannot be reached."""
    if limit < 1:
        raise ValueError("A Gauss-Kronrod workspace needs at least one integration segment.")
    if eps_abs <= 0 and (eps_rel < 50 * EPSILON or eps_rel < 0.5e-28):
        return Result(0.0, 0.0, "tolerance", 0)
    initial = rule(func, a, b, points)
    tolerance = max(eps_abs, eps_rel * abs(initial.value))
    status = _first(initial, tolerance, limit)
    if status != "continue":
        return Result(initial.value, initial.error, status, points)
    run = _Adaptive(func, _Regions(a, b, initial, limit), eps_abs, eps_rel, limit, points)
    return run.finish(initial.value, initial.error)


class _Adaptive:
    """The error and roundoff counters belonging to one independent integral."""

    def __init__(
        self,
        func: Integrand,
        regions: _Regions,
        absolute: float,
        relative: float,
        limit: int,
        points: int,
    ) -> None:
        self.func, self.regions, self.absolute, self.relative = func, regions, absolute, relative
        self.limit, self.points = limit, points
        self.round1 = self.round2 = 0

    def _roundoff(self, old: Estimate, left: Estimate, right: Estimate, iteration: int) -> None:
        if left.deviation == left.error or right.deviation == right.error:
            return
        value, error = left.value + right.value, left.error + right.error
        if abs(old.value - value) <= 1e-5 * abs(value) and error >= 0.99 * old.error:
            self.round1 += 1
        if iteration >= 10 and error > old.error:
            self.round2 += 1

    def finish(self, area: float, error: float) -> Result:
        iteration = 1
        while True:
            at = self.regions.worst()
            a, b = self.regions.bounds[at]
            old = self.regions.estimates[at]
            middle = 0.5 * (a + b)
            left = rule(self.func, a, middle, self.points)
            right = rule(self.func, middle, b, self.points)
            error += left.error + right.error - old.error
            area += left.value + right.value - old.value
            self._roundoff(old, left, right, iteration)
            self.regions.replace(at, left, right)
            status = self._status(a, middle, b, area, error)
            if status != "continue":
                break
            iteration += 1
        evaluations = self.points * (2 * len(self.regions.estimates) - 1)
        return Result(self.regions.total(), error, status, evaluations)

    def _status(self, a: float, middle: float, b: float, area: float, error: float) -> str:
        if error <= max(self.absolute, self.relative * abs(area)):
            return "success"
        if self.round1 >= 6 or self.round2 >= 20:
            return "roundoff"
        if _small(a, middle, b):
            return "bad-integrand"
        return "continue" if len(self.regions.estimates) < self.limit else "limit"
