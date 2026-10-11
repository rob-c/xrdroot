"""``RooEffProd``: a density times an efficiency, normalised again.

``RooEffProd("m", "m", pdf, eff)`` is ``eff(x) * pdf(x)``, normalised
numerically, as RooFit has no closed form for it. Its events are the
density's own, each kept with the efficiency's probability, as
``RooEffGenContext`` keeps them: the density draws an event, a uniform
number below the efficiency keeps it.
"""

from __future__ import annotations

from typing import Any

from ..generation.contexts import Context, context_for
from ..pdf import RooAbsPdf
from ..real import Context as Values
from ..rng import generator

__all__ = ["RooEffProd"]


class RooEffProd(RooAbsPdf):
    """A density times an efficiency function."""

    #: ``RooEffGenContext`` draws the density's own events: no copy of the product is
    #: normalised to generate it, nor announced.
    generated_unnormalised = True

    def __init__(self, name: Any, title: Any, inPdf: Any, inEff: Any) -> None:
        super().__init__(name, title)
        self.pdf = self._proxy("pdf", inPdf)
        self.eff = self._proxy("eff", inEff)

    def compute(self, ctx: Values) -> Any:
        """The efficiency times the density - unnormalised, where RooFit's proxy hands it over
        normalised: by a constant, which the product's own normalisation takes out again."""
        return self.eff.compute(ctx) * self.pdf.compute(ctx)

    def gen_context(self, names: frozenset[str]) -> Any:
        return EfficiencyContext(self, names)


class EfficiencyContext(Context):
    """``RooEffGenContext``: the density's events, each kept with the efficiency's chance."""

    def __init__(self, pdf: RooEffProd, names: frozenset[str]) -> None:
        super().__init__(pdf, names)
        self.inner = context_for(pdf.pdf, names)

    def event(self, remaining: int) -> dict[str, float]:
        while True:
            found = self.inner.event(remaining)
            for name, value in found.items():
                self.pdf.variable(name).load_value(value)
            if float(self.pdf.eff.getVal()) > generator().Rndm():
                return found
