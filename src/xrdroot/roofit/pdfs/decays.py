"""``RooDecay`` and ``RooBMixDecay``: an exponential decay, and B0 mixing, with a resolution.

``RooDecay`` is one basis, ``exp(-t/tau)`` for positive times, ``exp(t/tau)``
flipped or ``exp(-|t|/tau)`` on both sides. ``RooBMixDecay`` adds the
mixing oscillation ``cos(dm t)``, with coefficients of the tagged flavour
and the mixing state: ``(1 - tag dw)`` and ``mix (1 - 2w)``. Both draw
their own events when the resolution is the truth model - the time by
inverting the exponential, the mixing by accept/reject on the oscillation,
the categories from fractions worked out once from the integrals - with
RooFit's random draws in RooFit's order.
"""

from __future__ import annotations

import math
from typing import Any

from ..real import Context, value_of
from .anaconv import RooAbsAnaConvPdf, decay_type
from .basic import ref

__all__ = ["RooBMixDecay", "RooDecay", "draw_time", "value"]

#: The three bases of each decay type: the exponential, and its products with cos and sin.
EXPONENTIAL = {0: "exp(-@0/@1)", 2: "exp(@0/@1)", 1: "exp(-abs(@0)/@1)"}


def value(arg: Any) -> float:
    return value_of(arg.compute({}))


def draw_time(kind: int, tau: float, rng: Any) -> float:
    """A time drawn from the decay: ``-tau log(u)``, flipped, or either side with a half each."""
    rand = rng.Rndm()
    if kind == 0:
        return -tau * math.log(rand)
    if kind == 2:
        return +tau * math.log(rand)
    return -tau * math.log(2 * rand) if rand <= 0.5 else +tau * math.log(2 * (rand - 0.5))


def inside(t: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] < t < bounds[1]


class RooDecay(RooAbsAnaConvPdf):
    """``RooDecay(name, title, t, tau, model, type)``: ``exp(-t/tau)`` convolved with ``model``."""

    def __init__(self, name: Any, title: Any, t: Any, tau: Any, model: Any, type: Any = 1) -> None:
        super().__init__(name, title, model, t)
        self.t = self._proxy("t", t)
        self.tau = self._proxy("tau", ref(tau))
        self._type = decay_type(type)
        self._basis_exp = self.declareBasis(EXPONENTIAL[self._type], [self.tau])

    def coef(self, index: int, ctx: Context) -> Any:
        return 1.0

    def gen_code(self, direct: frozenset[str], static_ok: bool) -> tuple[int, frozenset[str]]:
        mine = frozenset([self.t.GetName()])
        return (1, mine) if mine <= direct else (0, frozenset())

    def init_generator(self, code: int) -> None:
        """Nothing to prepare: the time is drawn by inverting the exponential."""

    def generate_event(self, code: int, rng: Any, bounds: Any = None) -> dict[str, float]:
        bounds = bounds or (self.t.getMin(), self.t.getMax())
        while True:
            t = draw_time(self._type, value(self.tau), rng)
            if inside(t, bounds):
                return {self.t.GetName(): t}


class RooBMixDecay(RooAbsAnaConvPdf):
    """``RooBMixDecay``: B0 mixing - ``(1 - tag dw) + mix (1 - 2w) cos(dm t)`` times the decay."""

    def __init__(
        self,
        name: Any,
        title: Any,
        t: Any,
        mixState: Any,
        tagFlav: Any,
        tau: Any,
        dm: Any,
        mistag: Any,
        delMistag: Any,
        model: Any,
        type: Any = 1,
    ) -> None:
        super().__init__(name, title, model, t)
        self._type = decay_type(type)
        self.mistag = self._proxy("mistag", ref(mistag))
        self.delMistag = self._proxy("delMistag", ref(delMistag))
        self.mixState = self._proxy("mixState", mixState)
        self.tagFlav = self._proxy("tagFlav", tagFlav)
        self.tau = self._proxy("tau", ref(tau))
        self.dm = self._proxy("dm", ref(dm))
        self.t = self._proxy("_t", t)
        exp = EXPONENTIAL[self._type]
        self._basis_exp = self.declareBasis(exp, [self.tau])
        self._basis_cos = self.declareBasis(f"{exp}*cos(@0*@2)", [self.tau, self.dm])
        self._fractions = (0.0, 0.0, 0.0, 0.0)

    def coef(self, index: int, ctx: Context) -> Any:
        if index == self._basis_exp:
            return 1 - self.tagFlav.compute(ctx) * self.delMistag.compute(ctx)
        return self.mixState.compute(ctx) * (1 - 2 * self.mistag.compute(ctx))

    def coef_analytic_names(self, index: int, names: frozenset[str], rng: Any) -> frozenset[str]:
        if rng:
            return frozenset()
        mix, tag = self.mixState.GetName(), self.tagFlav.GetName()
        return names & {mix, tag} if mix in names else names & {tag}

    def coef_analytic(self, index: int, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        mix, tag = self.mixState.GetName() in names, self.tagFlav.GetName() in names
        if not mix and not tag:
            return self.coef(index, ctx)
        if index == self._basis_exp:
            return 4.0 if mix and tag else 2.0 * self.coef(index, ctx) if mix else 2.0
        return 2.0 * self.coef(index, ctx) if tag and not mix else 0.0

    def gen_code(self, direct: frozenset[str], static_ok: bool) -> tuple[int, frozenset[str]]:
        """``getGenerator``: 4 for time, mixing and flavour, 3 without flavour, 2 without mixing."""
        t, mix, tag = self.t.GetName(), self.mixState.GetName(), self.tagFlav.GetName()
        if t not in direct:
            return 0, frozenset()
        for code, wanted in ((4, {mix, tag}), (3, {mix}), (2, {tag})):
            if static_ok and wanted <= direct:
                return code, frozenset(wanted | {t})
        return 1, frozenset([t])

    def _int(self, names: set[str], **states: float) -> float:
        return value_of(self.integrate(frozenset(names), {k: float(v) for k, v in states.items()}))

    def init_generator(self, code: int) -> None:
        """``initGenerator``: the fraction of mixed events, and of B0 in each, from integrals."""
        t, mix, tag = self.t.GetName(), self.mixState.GetName(), self.tagFlav.GetName()
        if code == 2:
            self._fractions = (0.0, self._int({t}, **{tag: 1}) / self._int({t, tag}), 0.0, 0.0)
        elif code == 3:
            self._fractions = (self._int({t}, **{mix: -1}) / self._int({t, mix}), 0.0, 0.0, 0.0)
        elif code == 4:
            total, mixed = self._int({t, mix, tag}), self._int({t, tag}, **{mix: -1})
            unmixed_b0 = self._int({t}, **{mix: 1, tag: 1})
            self._fractions = (
                mixed / total,
                0.0,
                self._int({t}, **{mix: -1, tag: 1}) / mixed,
                unmixed_b0 / (total - mixed),
            )

    def _draw_states(self, code: int, rng: Any) -> dict[str, float]:
        mix_frac, flav_frac, flav_mixed, flav_unmixed = self._fractions
        found: dict[str, float] = {}
        if code in (3, 4):
            found[self.mixState.GetName()] = -1.0 if rng.Rndm() <= mix_frac else 1.0
        if code == 2:
            found[self.tagFlav.GetName()] = 1.0 if rng.Rndm() <= flav_frac else -1.0
        if code == 4:
            chosen = flav_mixed if found[self.mixState.GetName()] == -1 else flav_unmixed
            found[self.tagFlav.GetName()] = 1.0 if rng.Rndm() <= chosen else -1.0
        return found

    def generate_event(self, code: int, rng: Any, bounds: Any = None) -> dict[str, float]:
        found = self._draw_states(code, rng)
        bounds = bounds or (self.t.getMin(), self.t.getMax())
        tag = found.get(self.tagFlav.GetName(), value(self.tagFlav))
        mix = found.get(self.mixState.GetName(), value(self.mixState))
        tau, dm, dw = value(self.tau), value(self.dm), value(self.delMistag)
        dil = 1 - 2.0 * value(self.mistag)
        most = 1 + abs(dw) + abs(dil)
        while True:
            t = draw_time(self._type, tau, rng)
            accept = most * rng.Rndm() < (1 - tag * dw) + mix * dil * math.cos(dm * t)
            if inside(t, bounds) and accept:
                found[self.t.GetName()] = t
                return found
