"""How a convolved decay draws its events: RooFit's contexts for ``RooAbsAnaConvPdf``.

``RooAbsAnaConvPdf::genContext`` chooses, and so does
:func:`context_for_convolution`:

* with the truth model, the decay draws its own events (``RooGenContext``
  forced to let the density generate the time) - and the categories it
  cannot draw itself are drawn first by accept/reject (:class:`AcceptReject`);
* with a resolution that can draw its own smearing, the decay is drawn with
  the truth model instead and the smearing added to its time
  (``RooConvGenContext``): the smearing first, then the decay, both with no
  bounds on the time, until the sum is in range;
* otherwise the whole density is sampled numerically.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..messages import INFO, log
from ..real import value_of
from ..rng import generator
from .contexts import Context, NumericContext

__all__ = [
    "AcceptReject",
    "ConvolutionContext",
    "DecayContext",
    "context_for_convolution",
    "widened",
]

#: ``nTrial0D``: the trial draws per category state before an accept/reject sampler of categories starts.
TRIALS_PER_STATE = 100
#: No bounds: what ``removeMin``/``removeMax`` leave of a variable's range.
UNBOUNDED = (-math.inf, math.inf)


@contextmanager
def widened(var: Any) -> Iterator[None]:
    """``removeMin(); removeMax()`` on ``var`` for a while, as a generator's copy of the time has it."""
    saved = var._binning
    var._binning = saved.clone()
    var.removeMin()
    var.removeMax()
    try:
        yield
    finally:
        var._binning = saved


class AcceptReject:
    """``RooAcceptReject`` over categories only: trial draws find the maximum, then events are accepted."""

    def __init__(
        self, pdf: Any, categories: list[Any], over: frozenset[str], names: frozenset[str]
    ) -> None:
        self.categories = categories
        norm = value_of(pdf.integrate(names, {}))

        def density(states: tuple[float, ...]) -> float:
            ctx = {one.GetName(): state for one, state in zip(categories, states)}
            return (
                value_of(pdf.integrate(over, ctx)) / norm
                if over
                else value_of(pdf.compute(ctx)) / norm
            )

        self.density = density
        self.min_trials = TRIALS_PER_STATE * math.prod(len(one.states()) for one in categories)
        self.cache: list[tuple[tuple[float, ...], float]] = []
        self.used, self.total, self.sum, self.max = 0, 0, 0.0, 0.0

    def _add(self) -> None:
        rng = generator()
        states = []
        for one in self.categories:
            labels = list(one.states())
            states.append(float(one.states()[labels[rng.Integer(len(labels))]]))
        found = self.density(tuple(states))
        if found > self.max:
            self.max = 1.05 * found
        self.sum += found
        self.cache.append((tuple(states), found))
        self.total += 1

    def _next(self) -> Any:
        while self.used < len(self.cache):
            states, found = self.cache[self.used]
            self.used += 1
            if not generator().Rndm() * self.max > found:
                return states
        return None

    def event(self, remaining: int) -> dict[str, float]:
        while self.total < self.min_trials:
            self._add()
        while True:
            states = self._next()
            if states is not None:
                return {one.GetName(): state for one, state in zip(self.categories, states)}
            self.cache, self.used = [], 0
            extra = 1 + int(1.05 * remaining / (self.sum / (self.total * self.max)))
            for _ in range(extra):
                self._add()


class DecayContext(Context):
    """``RooGenContext`` for a decay that draws its own time: its generator, after accept/reject if needed."""

    def __init__(self, pdf: Any, names: frozenset[str], proto: frozenset[str], forced: frozenset[str],
                 bounds: Any = None) -> None:  # fmt: skip
        super().__init__(pdf, names)
        self.bounds = bounds
        direct = frozenset(one for one in names if one in forced or pdf.is_direct_gen_safe(one))
        static_ok = not (pdf.dependents() & proto)
        self.code, generated = pdf.gen_code(direct, static_ok) if direct else (0, frozenset())
        others = [one for one in pdf.leaves() if one.GetName() in names - generated]
        self.sampler = AcceptReject(pdf, others, frozenset(generated), names) if others else None
        if self.code:
            pdf.init_generator(self.code)

    def event(self, remaining: int) -> dict[str, float]:
        found: dict[str, float] = {}
        if self.sampler is not None:
            found.update(self.sampler.event(remaining))
            for name, value in found.items():
                self.pdf.variable(name).load_value(value)
        if self.code:
            found.update(self.pdf.generate_event(self.code, generator(), self.bounds))
        return found


class ConvolutionContext(Context):
    """``RooConvGenContext``: the decay drawn with the truth model, the resolution's smearing added."""

    def __init__(self, pdf: Any, names: frozenset[str], proto: frozenset[str]) -> None:
        from ..pdfs.resolution import RooTruthModel

        super().__init__(pdf, names)
        self.time = pdf.conv_var()
        twin = pdf.with_model(RooTruthModel("truthModel", "Truth resolution model", self.time))
        with widened(self.time):
            self.decay = DecayContext(twin, names & twin.dependents(), proto, frozenset([self.time.GetName()]),
                                      UNBOUNDED)  # fmt: skip
        self.smearing = pdf.model
        self.code = self.smearing.generator_code(frozenset([self.time.GetName()]))

    def event(self, remaining: int) -> dict[str, float]:
        name = self.time.GetName()
        while True:
            smeared = self.smearing.generate_event(self.code, generator(), UNBOUNDED)[name]
            found = self.decay.event(remaining)
            total = found[name] + smeared
            if self.time.inRange(total):
                found[name] = total
                return found


def context_for_convolution(pdf: Any, names: frozenset[str], proto: Any = None) -> Context:
    """``RooAbsAnaConvPdf::genContext``: the context RooFit makes for a convolved decay."""
    proto = frozenset(proto or ())
    time, model = pdf.conv_var().GetName(), pdf.model
    reasons = ""
    if (model.dependents() & names) - {time}:
        reasons += "Resolution model has more observables than the convolution variable. "
    if not pdf.gen_code(frozenset([time]), True)[0]:
        reasons += "PDF does not support internal generation of convolution observable. "
    if not (model.generator_code(frozenset([time])) and model.is_direct_gen_safe(time)):
        reasons += (
            "Resolution model does not support internal generation of convolution observable. "
        )
    if reasons:
        log(pdf, INFO, "Generation", f"RooAbsAnaConvPdf::genContext({pdf.GetName()}) Using regular accept/reject "
            f"generator for convolution p.d.f because: {reasons}")  # fmt: skip
        return NumericContext(pdf, names)
    if model.is_truth():
        return DecayContext(pdf, names, proto, frozenset([time]))
    return ConvolutionContext(pdf, names, proto)


def _states(categories: list[Any]) -> Iterator[tuple[int, ...]]:
    return itertools.product(*(list(one.states().values()) for one in categories))
