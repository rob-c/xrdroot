"""RooAcceptReject retains its trial points and refills according to measured efficiency.

Discarding the initial trials changes both the returned sample and the random
stream. A larger maximum found during refill asks the parent generator to trim
its existing production before adding the next accepted event.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: ROOT's trial-cache eviction threshold.
MAX_CACHE = 1_000_000


class EmpiricalSampler:
    """An empirical maximum and cached proposals for one numerical generator."""

    def __init__(self, density: Any, variables: list[Any], rng: Any, trials: int) -> None:
        self.density, self.variables, self.rng = density, variables, rng
        self.trials = trials
        self.maximum = self.weight_sum = 0.0
        self.total = self.used = 0
        self.points: list[Any] = []
        self.weights: list[float] = []
        self.resample_ratio = 1.0
        self.categories = [i for i, var in enumerate(variables) if hasattr(var, "lookupIndex")]
        self.reals = [i for i, var in enumerate(variables) if not hasattr(var, "lookupIndex")]

    def _proposal(self) -> Any:
        point = np.empty(len(self.variables))
        for i in self.categories:
            states = list(self.variables[i].states().values())
            point[i] = states[self.rng.Integer(len(states))]
        for i in self.reals:
            low, high = self.variables[i].getMin(), self.variables[i].getMax()
            point[i] = low + self.rng.Rndm() * (high - low)
        return point

    def _append(self) -> None:
        point = self._proposal()
        value = float(np.asarray(self.density(point[None, :]))[0])
        if value > self.maximum:
            self.maximum = 1.05 * value
        self.weight_sum += value
        self.total += 1
        self.points.append(point)
        self.weights.append(value)

    def _clear(self) -> None:
        self.points.clear()
        self.weights.clear()
        self.used = 0

    def _bootstrap(self) -> None:
        while self.total < self.trials:
            self._append()
            if len(self.points) > MAX_CACHE:
                self._clear()

    def _accepted(self) -> Any:
        while self.used < len(self.points):
            at = self.used
            self.used += 1
            if not self.rng.Rndm() * self.maximum > self.weights[at]:
                return self.points[at]
        return None

    def _refill(self, remaining: int) -> None:
        self._clear()
        if self.total * self.maximum <= 0:
            raise ValueError(
                "Accept/reject sampling cannot estimate an efficiency from a nonpositive density."
            )
        efficiency = self.weight_sum / (self.total * self.maximum)
        count = 1 + int(1.05 * remaining / efficiency)
        for _ in range(count):
            self._append()

    def generate(self, remaining: int = 1) -> Any:
        self._bootstrap()
        before = self.maximum
        self.resample_ratio = 1.0
        while True:
            if self.maximum > before:
                self.resample_ratio = before / self.maximum
            point = self._accepted()
            if point is not None:
                if self.used > MAX_CACHE:
                    self._clear()
                return point.copy()
            self._refill(remaining)
