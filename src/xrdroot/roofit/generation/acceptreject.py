"""``RooAcceptReject``: events of a conditional density, drawn uniformly and kept by its value.

When a density is generated with prototype data - its conditional
observables taken event by event from a dataset - RooFit cannot sample it
once with TFoam, for it is a different density for every event. It finds
its largest value first, from a thousand (a hundred thousand in two
dimensions) uniform trial points over the generated and the prototype
observables it depends on, times 1.05, and then, for each event, draws
uniform points until one is kept: a point with value ``f`` is kept when a
uniform draw ``r`` has ``r max <= f``. A trial of larger value than the
maximum raises it - to 1.05 times that value - for the event being drawn.
"""

from __future__ import annotations

import math
from typing import Any

from ..messages import WARNING, log
from ..rng import generator
from .contexts import Context

__all__ = ["AcceptRejectContext"]

#: ``nTrial0D`` ... ``nTrial3D``: how many trial points find a density's largest value.
TRIALS = (100, 1000, 100000, 10000000)
#: More trials than this - RooAcceptReject's threshold - and it says they may take a while.
MANY_TRIALS = 10000


def randomize(var: Any) -> None:
    """``randomize``: a value drawn uniformly over the variable's range - or, of a category,
    a state by its place among them."""
    if hasattr(var, "lookupIndex"):
        labels = list(var.states())
        var.load_value(var.states()[labels[generator().Integer(len(labels))]])
        return
    low, high = var.getMin(), var.getMax()
    var.load_value(low + generator().Rndm() * (high - low))


class AcceptRejectContext(Context):
    """``RooGenContext`` for a density with prototype observables: accept-reject sampling."""

    def __init__(self, pdf: Any, names: frozenset[str], conditional: frozenset[str],
                 label: str | None = None) -> None:  # fmt: skip
        super().__init__(pdf, names)
        self.order = [one for one in pdf.leaves() if one.GetName() in names]
        found = [one for one in pdf.leaves() if one.GetName() in conditional]
        self.label = label or f"{pdf.GetName()}_AccRej"
        self.maximum = self._largest(self.order + found)

    def value(self) -> float:
        return float(self.pdf.value({}, self.names))

    def _largest(self, sampled: list[Any]) -> float:
        """``RooAcceptReject::getFuncMax``: 1.05 times the largest value at the trial points."""
        trials = self._trials(sampled)
        saved = [(one, one.getVal()) for one in sampled]
        largest = 0.0
        for _ in range(trials):
            for one in sampled:
                randomize(one)
            found = self.value()
            if found > largest:
                largest = 1.05 * found
        for one, before in saved:
            one.load_value(before)
        return largest

    def _trials(self, sampled: list[Any]) -> int:
        """How many trial points: ``nTrialND`` for the ``N`` real dimensions, times the states of
        each category - and RooAcceptReject's warnings of many dimensions, and of many trials."""
        reals = len([one for one in sampled if not hasattr(one, "lookupIndex")])
        trials = TRIALS[min(reals, 3)] * math.prod(
            len(one.states()) for one in sampled if hasattr(one, "lookupIndex"))  # fmt: skip
        if reals > 1:
            log(None, WARNING, "Generation", f"RooAcceptReject::ctor({self.label}) WARNING: "
                f"performing accept/reject sampling on a p.d.f in {reals} dimensions without "
                "prior knowledge on maximum value of p.d.f. Determining maximum value by "
                f"taking {trials} trial samples. If p.d.f contains sharp peaks smaller than "
                "average distance between trial sampling points these may be missed and p.d.f. "
                "may be sampled incorrectly.")  # fmt: skip
        if trials > MANY_TRIALS:
            log(None, WARNING, "Generation", f"RooAcceptReject::ctor({self.label}): WARNING: "
                f"{trials} trial samples requested by p.d.f for {reals}-dimensional "
                "accept/reject sampling, this may take some time")  # fmt: skip
        return trials

    def event(self, remaining: int) -> dict[str, float]:
        """``generateEvent`` with the maximum known: trial points until one is kept."""
        largest = self.maximum
        rng = generator()
        while True:
            for one in self.order:
                randomize(one)
            found = self.value()
            if found > largest:
                largest = 1.05 * found
            if rng.Rndm() * largest <= found:
                return {one.GetName(): one.getVal() for one in self.order}
