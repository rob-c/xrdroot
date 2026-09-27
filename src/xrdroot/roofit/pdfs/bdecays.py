"""The CP-violating B decays: ``RooBCPEffDecay``, ``RooBCPGenDecay`` and the general ``RooBDecay``.

The first two are a tagged decay to a CP eigenstate - an exponential and
its products with ``sin(dm t)`` and ``cos(dm t)``, with coefficients of the
tag, the mistag rates and the CP parameters (``|lambda|`` and ``Im lambda``,
or ``C`` and ``S``); ``RooBDecay`` is any combination of ``cosh``, ``sinh``,
``cos`` and ``sin`` of the time, with a lifetime difference, whose four
coefficients are functions the caller gives. Each draws its own events
with the truth model, as RooFit's ``generateEvent`` does.
"""

from __future__ import annotations

import math
from typing import Any

from ..real import Context, value_of
from .anaconv import RooAbsAnaConvPdf, decay_type
from .basic import ref
from .decays import draw_time, inside, value

__all__ = ["RooBCPEffDecay", "RooBCPGenDecay", "RooBDecay"]

#: The exponential basis of each decay type - ``Flipped``'s with RooFit's own typo, which it refuses.
CP_EXPONENTIAL = {0: "exp(-@0/@1)", 2: "exp(@0)/@1)", 1: "exp(-abs(@0)/@1)"}
#: The time dependence of each decay type, which the oscillating bases multiply.
SIDES = {0: "exp(-@0/@1)", 2: "exp(@0/@1)", 1: "exp(-abs(@0)/@1)"}


class _TaggedCP(RooAbsAnaConvPdf):
    """What the two CP decays share: three bases, a tag, and the tag drawn from its B0 fraction."""

    def _declare(self) -> None:
        params = [self.tau, self.dm]
        self._basis_exp = self.declareBasis(CP_EXPONENTIAL[self._type], params)
        self._basis_sin = self.declareBasis(f"{SIDES[self._type]}*sin(@0*@2)", params)
        self._basis_cos = self.declareBasis(f"{SIDES[self._type]}*cos(@0*@2)", params)
        self._b0_fraction = 0.0

    def coef_analytic_names(self, index: int, names: frozenset[str], rng: Any) -> frozenset[str]:
        return frozenset() if rng else names & {self.tag.GetName()}

    def gen_code(self, direct: frozenset[str], static_ok: bool) -> tuple[int, frozenset[str]]:
        t, tag = self.t.GetName(), self.tag.GetName()
        if t not in direct:
            return 0, frozenset()
        return (2, frozenset([t, tag])) if static_ok and tag in direct else (1, frozenset([t]))

    def init_generator(self, code: int) -> None:
        """``initGenerator``: the fraction of B0 tags, from the integrals over the time."""
        if code == 2:
            t, tag = self.t.GetName(), self.tag.GetName()
            total = value_of(self.integrate(frozenset([t, tag]), {}))
            self._b0_fraction = value_of(self.integrate(frozenset([t]), {tag: 1.0})) / total

    def acceptance(self, tag: float, t: float) -> tuple[float, float]:
        """The most the accept/reject weight can be, and its value at ``t`` for ``tag``."""
        raise NotImplementedError

    def generate_event(self, code: int, rng: Any, bounds: Any = None) -> dict[str, float]:
        found: dict[str, float] = {}
        if code == 2:
            found[self.tag.GetName()] = 1.0 if rng.Rndm() <= self._b0_fraction else -1.0
        tag = found.get(self.tag.GetName(), value(self.tag))
        bounds = bounds or (self.t.getMin(), self.t.getMax())
        tau = value(self.tau)
        while True:
            t = draw_time(self._type, tau, rng)
            most, weight = self.acceptance(tag, t)
            if inside(t, bounds) and most * rng.Rndm() < weight:
                found[self.t.GetName()] = t
                return found


class RooBCPEffDecay(_TaggedCP):
    """``RooBCPEffDecay``: CP violation with ``|lambda|`` and ``Im lambda``, B0/B0bar efficiencies."""

    def __init__(self, name: Any, title: Any, t: Any, tag: Any, tau: Any, dm: Any, avgMistag: Any,
                 CPeigenval: Any, absLambda: Any, argLambda: Any, effRatio: Any, delMistag: Any,
                 model: Any, type: Any = 1) -> None:  # fmt: skip
        super().__init__(name, title, model, t)
        self.absLambda = self._proxy("absLambda", ref(absLambda))
        self.argLambda = self._proxy("argLambda", ref(argLambda))
        self.effRatio = self._proxy("effRatio", ref(effRatio))
        self.CPeigenval = self._proxy("CPeigenval", ref(CPeigenval))
        self.avgMistag = self._proxy("avgMistag", ref(avgMistag))
        self.delMistag = self._proxy("delMistag", ref(delMistag))
        self.t = self._proxy("t", t)
        self.tau = self._proxy("tau", ref(tau))
        self.dm = self._proxy("dm", ref(dm))
        self.tag = self._proxy("tag", tag)
        self._type = decay_type(type)
        self._declare()

    def coef(self, index: int, ctx: Context) -> Any:
        tag, al = self.tag.compute(ctx), self.absLambda.compute(ctx)
        if index == self._basis_exp:
            return (1 - tag * self.delMistag.compute(ctx)) * (1 + al * al) / 2
        dilution = -1 * tag * (1 - 2 * self.avgMistag.compute(ctx))
        if index == self._basis_sin:
            return dilution * self.CPeigenval.compute(ctx) * al * self.argLambda.compute(ctx)
        return dilution * (1 - al * al) / 2

    def coef_analytic(self, index: int, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        if not names:
            return self.coef(index, ctx)
        al = self.absLambda.compute(ctx)
        return (1 + al * al) if index == self._basis_exp else 0.0 * al

    def acceptance(self, tag: float, t: float) -> tuple[float, float]:
        al, arg, cp = value(self.absLambda), value(self.argLambda), value(self.CPeigenval)
        dw, w, dm = value(self.delMistag), value(self.avgMistag), value(self.dm)
        max_dil = 1.0
        al2 = al * al
        most = (1 + al2) + abs(max_dil * cp * al * arg) + abs(max_dil * (1 - al2) / 2)
        weight = ((1 + al2) / 2 * (1 - tag * dw) - (tag * (1 - 2 * w)) * (cp * al * arg) * math.sin(dm * t)
                  - (tag * (1 - 2 * w)) * (1 - al2) / 2 * math.cos(dm * t))  # fmt: skip
        return most, weight


class RooBCPGenDecay(_TaggedCP):
    """``RooBCPGenDecay``: CP violation with the ``C`` and ``S`` coefficients and a tag asymmetry ``mu``."""

    def __init__(self, name: Any, title: Any, t: Any, tag: Any, tau: Any, dm: Any, avgMistag: Any,
                 avgC: Any, avgS: Any, delMistag: Any, mu: Any, model: Any, type: Any = 1) -> None:  # fmt: skip
        super().__init__(name, title, model, t)
        self.C = self._proxy("C", ref(avgC))
        self.S = self._proxy("S", ref(avgS))
        self.avgMistag = self._proxy("avgMistag", ref(avgMistag))
        self.delMistag = self._proxy("delMistag", ref(delMistag))
        self.mu = self._proxy("mu", ref(mu))
        self.t = self._proxy("t", t)
        self.tau = self._proxy("tau", ref(tau))
        self.dm = self._proxy("dm", ref(dm))
        self.tag = self._proxy("tag", tag)
        self._type = decay_type(type)
        self._declare()

    def _parts(self, tag: Any, ctx: Context) -> tuple[Any, Any]:
        dw, w, mu = self.delMistag.compute(ctx), self.avgMistag.compute(ctx), self.mu.compute(ctx)
        return (1 - tag * dw + mu * tag * (1.0 - 2.0 * w)), (tag * (1 - 2 * w) + mu * (1.0 - tag * dw))

    def coef(self, index: int, ctx: Context) -> Any:
        flat, oscillating = self._parts(self.tag.compute(ctx), ctx)
        if index == self._basis_exp:
            return flat
        if index == self._basis_sin:
            return oscillating * self.S.compute(ctx)
        return -1.0 * oscillating * self.C.compute(ctx)

    def coef_analytic(self, index: int, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        if not names:
            return self.coef(index, ctx)
        mu = self.mu.compute(ctx)
        if index == self._basis_exp:
            return 2.0 + 0.0 * mu
        return 2 * mu * self.S.compute(ctx) if index == self._basis_sin else -2 * mu * self.C.compute(ctx)

    def acceptance(self, tag: float, t: float) -> tuple[float, float]:
        s, c, dm = value(self.S), value(self.C), value(self.dm)
        flat, oscillating = self._parts(tag, {})
        max_dil = 1.0
        most = 2 + abs(max_dil * s) + abs(max_dil * c)
        return most, flat + oscillating * s * math.sin(dm * t) - oscillating * c * math.cos(dm * t)


class RooBDecay(RooAbsAnaConvPdf):
    """``RooBDecay``: ``f0 cosh(dG t/2) + f1 sinh(dG t/2) + f2 cos(dm t) + f3 sin(dm t)`` times the decay."""

    def __init__(self, name: Any, title: Any, t: Any, tau: Any, dgamma: Any, f0: Any, f1: Any, f2: Any,
                 f3: Any, dm: Any, model: Any, type: Any = 1) -> None:  # fmt: skip
        super().__init__(name, title, model, t)
        self.t = self._proxy("t", t)
        self.tau = self._proxy("tau", ref(tau))
        self.dgamma = self._proxy("dgamma", ref(dgamma))
        self.fs = [self._proxy(f"f{i}", ref(one)) for i, one in enumerate((f0, f1, f2, f3))]
        self.dm = self._proxy("dm", ref(dm))
        self._type = decay_type(type)
        side = SIDES[self._type]
        for form, params in (("cosh(@0*@2/2)", [self.tau, self.dgamma]), ("sinh(@0*@2/2)", [self.tau, self.dgamma]),
                             ("cos(@0*@2)", [self.tau, self.dm]), ("sin(@0*@2)", [self.tau, self.dm])):  # fmt: skip
            self.declareBasis(f"{side}*{form}", params)

    def coef(self, index: int, ctx: Context) -> Any:
        return self.fs[index].compute(ctx)

    def gen_code(self, direct: frozenset[str], static_ok: bool) -> tuple[int, frozenset[str]]:
        mine = frozenset([self.t.GetName()])
        return (1, mine) if mine <= direct else (0, frozenset())

    def init_generator(self, code: int) -> None:
        """Nothing to prepare: the time is drawn under an exponential envelope."""

    def generate_event(self, code: int, rng: Any, bounds: Any = None) -> dict[str, float]:
        """``generateEvent``: a time under the envelope ``exp(-gamma_min |t|)``, accepted by the density."""
        low, high = bounds or (self.t.getMin(), self.t.getMax())
        tau, dgamma, dm = value(self.tau), value(self.dgamma), value(self.dm)
        f0, f1, f2, f3 = (value(one) for one in self.fs)
        gammamin = 1 / tau - abs(dgamma) / 2
        while True:
            t = -math.log(rng.Rndm()) / gammamin
            if self._type == 2 or (self._type == 1 and rng.Rndm() < 0.5):
                t *= -1
            if t < low or t > high:
                continue
            ft = abs(t)
            f = math.exp(-ft / tau) * (f0 * math.cosh(dgamma * t / 2) + f1 * math.sinh(dgamma * t / 2)
                                       + f2 * math.cos(dm * t) + f3 * math.sin(dm * t))  # fmt: skip
            envelope = 1.001 * math.exp(-ft * gammamin) * (abs(f0) + abs(f1) + math.sqrt(f2 * f2 + f3 * f3))
            if f < 0 or envelope < f:
                raise RuntimeError(f"RooBDecay::generateEvent({self.GetName()}): the density is below zero or "
                                   "above its envelope, so no event can be drawn.")  # fmt: skip
            if envelope * rng.Rndm() > f:
                continue
            return {self.t.GetName(): t}
