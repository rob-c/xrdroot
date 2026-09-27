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

from typing import Any

from ..messages import WARNING, log
from ..rng import generator
from .contexts import Context

__all__ = ["AcceptRejectContext"]

#: ``nTrial0D`` ... ``nTrial3D``: how many trial points find a density's largest value.
TRIALS = (100, 1000, 100000, 10000000)


def randomize(var: Any) -> None:
    """``RooAbsRealLValue::randomize``: a value drawn uniformly over the variable's range."""
    low, high = var.getMin(), var.getMax()
    var.load_value(low + generator().Rndm() * (high - low))


class AcceptRejectContext(Context):
    """``RooGenContext`` for a density with prototype observables: accept-reject sampling."""

    def __init__(self, pdf: Any, names: frozenset[str], conditional: frozenset[str]) -> None:
        super().__init__(pdf, names)
        self.order = [one for one in pdf.leaves() if one.GetName() in names]
        found = [one for one in pdf.leaves() if one.GetName() in conditional]
        self.maximum = self._largest(self.order + found)

    def value(self) -> float:
        return float(self.pdf.value({}, self.names))

    def _largest(self, sampled: list[Any]) -> float:
        """``RooAcceptReject::getFuncMax``: 1.05 times the largest value at the trial points."""
        trials = TRIALS[min(len(sampled), 3)]
        if len(sampled) > 1:
            log(None, WARNING, "Generation", f"RooAcceptReject::ctor({self.pdf.GetName()}_AccRej) WARNING: "
                f"performing accept/reject sampling on a p.d.f in {len(sampled)} dimensions without prior "
                f"knowledge on maximum value of p.d.f. Determining maximum value by taking {trials} trial "
                "samples. If p.d.f contains sharp peaks smaller than average distance between trial sampling "
                "points these may be missed and p.d.f. may be sampled incorrectly.")  # fmt: skip
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
