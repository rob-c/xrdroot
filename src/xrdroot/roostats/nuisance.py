"""``NuisanceParametersSampler``: the nuisance parameters' values for each toy, from their prior.

A hybrid calculation draws the nuisance parameters from their prior - the
product of their constraints - for every toy: a dataset of ``ntoys`` draws
at a time, handed out one by one, a fresh one drawn when it runs out. With
``expected``, the draws are the prior's expected - binned - values instead,
each with its weight.
"""

from __future__ import annotations

from typing import Any

from ..roofit.cmdargs import RooCmdArg
from ..roofit.collections import RooArgSet, as_list
from ..roofit.messages import INFO, log

__all__ = ["NuisanceParametersSampler"]


class NuisanceParametersSampler:
    """Draws of the nuisance parameters, ``ntoys`` at a time."""

    def __init__(self, prior: Any = None, params: Any = None, ntoys: int = 1000,
                 asimov: bool = False) -> None:  # fmt: skip
        self._prior = prior
        self._params = RooArgSet(as_list(params)) if params is not None else None
        self._ntoys = int(ntoys)
        self._expected = bool(asimov)
        self._points: Any = None
        self._index = 0

    def next_point(self, target: Any) -> float:
        """``NextPoint``: the next draw into ``target``, and its weight - skipping any of none."""
        while True:
            if self._points is None or self._index >= self._ntoys:
                self.Refresh()
                self._index = 0
            self._points.get(self._index)
            RooArgSet(as_list(target)).assign(self._points.get(self._index))
            weight = float(self._points.weight())
            self._index += 1
            if weight != 0.0:
                return weight
            log(None, INFO, "Generation", "Weight 0 encountered. Skipping.")

    def NextPoint(self, nuisPoint: Any, weight: Any = None) -> float:
        found = self.next_point(nuisPoint)
        if weight is not None and hasattr(weight, "value"):
            weight.value = found
        return found

    def Refresh(self) -> None:
        """``ntoys`` new draws from the prior - or its expected values, binned."""
        if self._prior is None or self._params is None:
            return
        if self._expected:
            log(None, INFO, "InputArguments", "Using expected nuisance parameters.")
            for one in self._params:
                one.setBins(self._ntoys)
            self._points = self._prior.generate(
                self._params, RooCmdArg("AllBinned"), RooCmdArg("ExpectedData"),
                RooCmdArg("NumEvents", 1))  # fmt: skip
            if self._points.numEntries() != self._ntoys:
                self._ntoys = self._points.numEntries()
                log(None, INFO, "InputArguments", "Adjusted number of toys to number of bins of "
                    f"nuisance parameters: {self._ntoys}")  # fmt: skip
            return
        log(None, INFO, "InputArguments", "Using randomized nuisance parameters.")
        self._points = self._prior.generate(self._params, self._ntoys)
