"""``RooProjectedPdf``: a density integrated over some of its observables, a density of the rest.

``pdf.createProjection(y)`` is ``RooProjectedPdf("pdf_Proj[y]", ..., pdf,
y)``: at each point, the integral of ``pdf`` over ``y`` - normalised, as
RooFit normalises it, over ``y`` and the observables it is asked of, each
factor of a product its own way - so the projection of ``f(x, y)`` onto
``x`` is ``Int f dy / Int f dx dy``, a density of ``x`` already.
"""

from __future__ import annotations

from typing import Any

from ..collections import as_list
from ..pdf import RooAbsPdf
from ..real import Context

__all__ = ["RooProjectedPdf", "projection"]


class RooProjectedPdf(RooAbsPdf):
    """``pdf`` integrated over ``intobs``."""

    def __init__(self, name: Any, title: Any, pdf: Any, intobs: Any) -> None:
        super().__init__(name, title)
        self.intpdf = self._proxy("!IntegratedPdf", pdf)
        self._integrated = frozenset(one.GetName() for one in as_list(intobs))

    def dependents(self) -> frozenset[str]:
        """What the projection is a function of: the density's variables but those integrated."""
        return frozenset(self.intpdf.dependents()) - self._integrated

    def compute(self, ctx: Context) -> Any:
        return self.intpdf.integrate(self._integrated, ctx)

    def selfNormalized(self) -> bool:
        return True

    def value(self, ctx: Context, nset: Any = None, rng: Any = None) -> Any:
        """The density integrated over the variables projected out, normalised over them and
        ``nset`` - each factor of a product its own way, a conditional one over its own."""
        whole = (frozenset(nset or ()) & self.intpdf.dependents()) | self._integrated
        return self.intpdf.fraction(self._integrated, ctx, whole, None)

    def fraction(
        self, names: frozenset[str], ctx: Context, nset: Any, rng: Any, norm_rng: Any = None
    ) -> Any:
        """The projection's integral over ``names`` in ``rng``: the density's over them and
        the variables projected out together."""
        whole = frozenset(nset or ()) | self._integrated
        return self.intpdf.fraction(frozenset(names) | self._integrated, ctx, whole, rng, norm_rng)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return frozenset(names) & self.dependents()

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        return self.intpdf.integrate(frozenset(names) | self._integrated, ctx, rng)

    def printMetaArgs(self) -> str:
        return ""


def projection(pdf: Any, iset: Any) -> RooProjectedPdf:
    """``createProjection(iset)``: named ``pdf_Proj[y,z]``, as RooFit names it."""
    names = ",".join(one.GetName() for one in as_list(iset))
    label = f"{pdf.GetName()}_Proj[{names}]"
    return RooProjectedPdf(label, label, pdf, iset)
