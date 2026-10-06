"""RooFit's generator contexts: how each kind of density draws one event.

``pdf.generate(x, 1000)`` asks the density for a *context* and the context
for an event a thousand times. The context is RooFit's choice, made the way
RooFit makes it, because it decides which random numbers are drawn in what
order:

* a density that samples its own observables - a Gaussian - draws them
  itself (``RooGenContext`` with the density's ``generateEvent``);
* a sum draws a uniform number to pick a component, then an event of that
  component (``RooAddGenContext``);
* a product draws each factor's observables from that factor
  (``RooProdGenContext``);
* anything else is sampled numerically, by TFoam as ``RooFoamGenerator``
  drives it (:mod:`.foam`).

Every context is made - and every numerical sampler initialised, which
draws numbers too - before the first event is drawn, as in RooFit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..rng import generator

__all__ = ["Context", "DirectContext", "context_for"]


class Context:
    """How one density draws its observables ``names``, an event at a time."""

    def __init__(self, pdf: Any, names: frozenset[str]) -> None:
        self.pdf = pdf
        self.names = names

    def event(self, remaining: int) -> dict[str, float]:
        raise NotImplementedError


class DirectContext(Context):
    """``RooGenContext`` with the density's own generator for every observable."""

    def __init__(self, pdf: Any, names: frozenset[str], code: int) -> None:
        super().__init__(pdf, names)
        self.code = code

    def event(self, remaining: int) -> dict[str, float]:
        return dict(self.pdf.generate_event(self.code, generator()))


class NumericContext(Context):
    """``RooGenContext`` with the default sampler: TFoam over the observables' ranges."""

    def __init__(self, pdf: Any, names: frozenset[str]) -> None:
        from .foam import FoamGenerator

        super().__init__(pdf, names)
        self.order = [one for one in pdf.leaves() if one.GetName() in names]
        ranges = [(one.getMin(), one.getMax()) for one in self.order]
        keys = [one.GetName() for one in self.order]

        norm = pdf.norm({}, names)  # constant while the parameters are: RooFit caches it too

        def density(points: Any) -> Any:
            ctx = {key: points[:, i] for i, key in enumerate(keys)}
            return np.broadcast_to(pdf.compute(ctx) / norm, (len(points),))

        self.sampler = FoamGenerator(density, ranges, generator(), vectorized=True)

    def event(self, remaining: int) -> dict[str, float]:
        point = self.sampler.generate()
        return {one.GetName(): float(v) for one, v in zip(self.order, point, strict=False)}


def context_for(
    pdf: Any, names: frozenset[str], conditional: frozenset[str] | None = None
) -> Context:
    """The context RooFit would make for ``pdf`` to generate ``names``, given ``conditional``
    ones."""
    make = getattr(pdf, "gen_context", None)
    if make is not None:
        return make(names)  # type: ignore[no-any-return]
    code = pdf.generator_code(names) if hasattr(pdf, "generator_code") else 0
    if code:
        return DirectContext(pdf, names, code)
    if conditional is not None:
        from .acceptreject import AcceptRejectContext

        return AcceptRejectContext(pdf, names, conditional)
    direct = _direct_part(pdf, names)
    if direct is not None:
        return MixedContext(pdf, names, *direct)
    return NumericContext(pdf, names)


def _direct_part(pdf: Any, names: frozenset[str]) -> tuple[frozenset[str], int] | None:
    """``getGenerator`` over the observables safe to draw directly: the first the density can
    draw itself - ``x`` of a Gaussian whose mean is a function of ``y`` - and its code."""
    if len(names) < 2 or not hasattr(pdf, "generator_code"):
        return None
    for one in pdf.leaves():
        own = frozenset([one.GetName()])
        if one.GetName() in names and _direct_safe(pdf, one.GetName()):
            code = pdf.generator_code(own)
            if code:
                return own, code
    return None


def _direct_safe(pdf: Any, name: str) -> bool:
    """``isDirectGenSafe``: a server of the density itself, and of nothing else it serves."""
    servers = pdf.servers()
    mine = [one for one in servers if one.GetName() == name]
    return bool(mine) and not any(
        name in one.dependents() for one in servers if one.GetName() != name
    )


class MixedContext(Context):
    """``RooGenContext`` drawing some observables itself and the rest numerically: TFoam over
    the rest, by the density integrated over the direct ones, then the density's own draw."""

    def __init__(self, pdf: Any, names: frozenset[str], direct: frozenset[str], code: int) -> None:
        from ..integration import announce
        from .foam import FoamGenerator

        super().__init__(pdf, names)
        self.code, self.direct = code, direct
        self.order = [one for one in pdf.leaves() if one.GetName() in names - direct]
        keys = [one.GetName() for one in self.order]
        announce(pdf, names, normalising=True)  # the accept-reject function's own integral

        def density(points: Any) -> Any:
            ctx = {key: points[:, i] for i, key in enumerate(keys)}
            return np.broadcast_to(pdf.fraction(direct, ctx, names, None), (len(points),))

        ranges = [(one.getMin(), one.getMax()) for one in self.order]
        self.sampler = FoamGenerator(density, ranges, generator(), vectorized=True)

    def event(self, remaining: int) -> dict[str, float]:
        found = {
            one.GetName(): float(v)
            for one, v in zip(self.order, self.sampler.generate(), strict=False)
        }
        for one in self.order:
            one.load_value(found[one.GetName()])
        found.update(self.pdf.generate_event(self.code, generator()))
        return found


def _factor_context(factor: Any, own: frozenset[str]) -> Context:
    """A factor's own ``RooGenContext``, which normalises its copy of the factor as the top
    context does its density: twice."""
    from ..integration import announce

    for _ in range(2):
        announce(factor, own, normalising=True)
    return context_for(factor, own)


class ProductContext(Context):
    """``RooProdGenContext``: each factor draws its own observables, those it is conditional on
    first."""

    def __init__(self, pdf: Any, names: frozenset[str]) -> None:
        super().__init__(pdf, names)
        self.parts: list[Context] = []
        done: set[str] = set()
        waiting = list(pdf.pdfs)
        while waiting:
            ready = [f for f in waiting if not ((self._imports(f) - done) & names)] or waiting[:]
            for factor in ready:
                own = pdf.factor_nset(factor, names)
                if own:
                    self.parts.append(_factor_context(factor, own))
                done |= own
                waiting.remove(factor)

    def _imports(self, factor: Any) -> set[str]:
        return set(factor.dependents() & self.names) - set(self.pdf.factor_nset(factor, self.names))

    def event(self, remaining: int) -> dict[str, float]:
        found: dict[str, float] = {}
        for part in self.parts:
            drawn = part.event(remaining)
            for name, value in drawn.items():
                self.pdf.variable(name).load_value(value)
            found.update(drawn)
        return found


class SumContext(Context):
    """``RooAddGenContext``: a uniform draw picks the component, which draws the event."""

    def __init__(self, pdf: Any, names: frozenset[str]) -> None:
        super().__init__(pdf, names)
        self.parts = [context_for(component, names) for component in pdf.pdfs]

    def event(self, remaining: int) -> dict[str, float]:
        shares = [
            float(np.asarray(c).reshape(-1)[0]) for c in self.pdf.coefficients({}, self.names)
        ]
        draw = generator().Rndm()
        low = 0.0
        for share, part in zip(shares, self.parts, strict=False):
            if low < draw < low + share:
                return part.event(remaining)
            low += share
        return self.event(remaining)  # pragma: no cover - a draw exactly on a threshold: redrawn
