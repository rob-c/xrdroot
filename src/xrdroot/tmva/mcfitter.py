"""``TMVA::MCFitter``: the best of ``SampleSize`` parameter sets drawn at random.

Each set is drawn from ``TRandom3(Seed)`` exactly as TMVA draws it -
uniformly in each parameter's interval, or with ``Sigma > 0`` by a Gaussian
of ``Sigma`` times the interval's length around the best set so far - and
the first set of least estimator is the answer. Uniform draws do not depend
on what was found, so when the target can evaluate many sets at once
(``batch_estimator``) they are evaluated a batch at a time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..random.mersenne import TRandom3
from .cutsfit import BATCH, draw_mc
from .genetic import FactorVector, GeneticRange, _fill
from .log import Logger
from .options import Options

__all__ = ["MCFitter"]


class MCFitter:
    """``TMVA::MCFitter(target, name, ranges, options)``."""

    def __init__(self, target: Any, name: Any, ranges: Any, options: Any = "") -> None:
        self.target, self.name, self.ranges = target, str(name), list(ranges)
        parsed = Options(options)
        self.samples = parsed.integer("SampleSize", 100000)
        self.sigma = parsed.number("Sigma", -1.0)
        self.seed = parsed.integer("Seed", 100)
        self.log = Logger("FitterBase")

    def Run(self, parameters: Any = None) -> float:
        """``Run(pars)``: the best parameters put in ``pars``; their estimator returned."""
        self.log.header("<MCFitter> Sampling, please be patient ...")
        if parameters is None:
            parameters = FactorVector(interval.GetMean() for interval in self.ranges)
        if self.sigma > 0 or not hasattr(self.target, "batch_estimator"):
            best, fitness = self._sequential()
        else:
            best, fitness = self._batched()
        self.log.info("Elapsed time: 0 sec                           ")
        _fill(parameters, best)
        return fitness

    def _batched(self) -> tuple[list[float], float]:
        drawn = draw_mc(self.ranges, self.samples, self.seed)
        best, fitness = list(drawn[0]), float("inf")
        for start in range(0, len(drawn), BATCH):
            chunk = drawn[start : start + BATCH]
            found = np.asarray(self.target.batch_estimator(chunk))
            index = int(np.argmin(found))
            if found[index] < fitness:
                best, fitness = list(chunk[index]), float(found[index])
        return best, fitness

    def _sequential(self) -> tuple[list[float], float]:
        random = TRandom3(self.seed)
        random.uniform(0.0, 1.0)
        ranges = [GeneticRange(random, interval) for interval in self.ranges]
        best = [r.random_value() for r in ranges]
        fitness = 0.0
        for sample in range(self.samples):
            if self.sigma > 0:
                trial = [r.random_value(True, b, self.sigma) for r, b in zip(ranges, best)]
            else:
                trial = [r.random_value() for r in ranges]
            estimator = float(self.target.EstimatorFunction(FactorVector(trial)))
            if estimator < fitness or sample == 0:
                fitness, best = estimator, trial
        return best, fitness
