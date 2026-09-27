"""``RooExtendPdf``: a density that also says how many events it expects.

``RooExtendPdf("e", "e", pdf, n)`` is ``pdf`` in shape and ``n`` in number -
or, given a range name, ``n`` events in that range, so the whole expects
``n`` divided by the fraction of ``pdf`` there.
"""

from __future__ import annotations

from typing import Any

from ..pdf import CAN_BE_EXTENDED, RooAbsPdf
from ..real import Context
from .basic import ref

__all__ = ["RooExtendPdf"]


class RooExtendPdf(RooAbsPdf):
    """A density and its expected number of events."""

    def __init__(self, name: Any, title: Any, pdf: Any, norm: Any, rangeName: Any = None) -> None:
        super().__init__(name, title)
        self.pdf = self._proxy("pdf", pdf)
        self.n = self._proxy("n", ref(norm))
        self._range = str(rangeName) if rangeName else None

    def compute(self, ctx: Context) -> Any:
        return self.pdf.compute(ctx)

    def value(self, ctx: Context, nset: Any = None, rng: Any = None) -> Any:
        return self.pdf.value(ctx, nset, rng)

    def selfNormalized(self) -> bool:
        return True

    def fraction(self, names: frozenset[str], ctx: Context, nset: Any, rng: Any,
                 norm_rng: Any = None) -> Any:  # fmt: skip
        return self.pdf.fraction(names, ctx, nset, rng, norm_rng)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return self.pdf.analytic_names(names, rng)

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        return self.pdf.analytic(names, ctx, rng)

    def extendMode(self) -> int:
        return CAN_BE_EXTENDED

    def expected(self, nset: Any, rng: Any = None) -> float:
        """``n`` - scaled from its range to all of it - and to the fit's range, if there is one."""
        found = float(self.n.getVal())
        names = frozenset(nset or ())
        if self._range and names:
            found /= float(self.pdf.fraction(names, {}, names, self._range))
        if rng and names:
            found *= float(self.pdf.fraction(names, {}, names, rng))
        if self.pdf.canBeExtended():
            found *= self.pdf.expected(nset, rng)
        return found

    def gen_context(self, names: frozenset[str]) -> Any:
        from ..generation.contexts import context_for

        return context_for(self.pdf, names)

    def normalized_name(self, observables: Any, rng: Any = None) -> str:
        return self._name
