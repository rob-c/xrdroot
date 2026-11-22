"""``RooExtendPdf``: a density that also says how many events it expects.

``RooExtendPdf("e", "e", pdf, n)`` is ``pdf`` in shape and ``n`` in number -
or, given a range name, ``n`` events in that range, so the whole expects
``n`` divided by the fraction of ``pdf`` there.

``RooExtendedBinding`` is the same expected number without a density around
it: a plain function of the model, so anything that can multiply a number by
a function - an ``RooFormulaVar``, a fit's parameterisation - can use the
number of events a density expects as one of its terms.
"""

from __future__ import annotations

from typing import Any

from ..collections import as_list
from ..pdf import CAN_BE_EXTENDED, RooAbsPdf
from ..real import Context, RooAbsReal
from .basic import ref

__all__ = ["RooExtendPdf", "RooExtendedBinding"]


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

    def fraction(
        self, names: frozenset[str], ctx: Context, nset: Any, rng: Any, norm_rng: Any = None
    ) -> Any:
        return self.pdf.fraction(names, ctx, nset, rng, norm_rng)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return frozenset(self.pdf.analytic_names(names, rng))

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        return self.pdf.analytic(names, ctx, rng)

    def extendMode(self) -> int:
        return CAN_BE_EXTENDED

    def expected(self, nset: Any, rng: Any = None, fit: bool = False) -> float:
        """``n``, scaled from its own range to the one the density is normalised over.

        ROOT divides ``n`` by the integral over the extension's range of the
        density normalised where the fit normalises it - the fit's range
        ``rng``, else everywhere; a likelihood (``fit``) rounds that as ``n``
        times the reciprocal of ``I(range) / I(rng)``. With no range of its
        own ``n`` is what the fit's range expects, unscaled: ROOT does not
        shrink it to the part of the data fitted.
        """
        found = float(self.n.getVal())
        names = frozenset(nset or ())
        if self._range and names:
            share = float(self.pdf.fraction(names, {}, names, self._range, rng))
            found = found * (1.0 / share) if fit else found / share
        if self.pdf.canBeExtended():
            found *= self.pdf.expected(nset, rng, fit)
        return found

    def normalized_name(self, observables: Any, rng: Any = None) -> str:
        return self._name


class RooExtendedBinding(RooAbsReal):
    """``pdf.expectedEvents()`` as a function, over the observables it is given.

    A density's expected events depend on the normalisation set the caller
    asks it about: with none, a density reports the ``1.0`` its own
    normalisation is worth, and with one, the integral over those
    observables. The binding holds the set it was made with and passes it on
    every time, so two bindings of the same density differ by the range they
    were made over rather than by who evaluates them.
    """

    def __init__(self, name: Any, title: Any, pdf: Any, normSet: Any = None) -> None:
        super().__init__(name, title)
        self.pdf = self._proxy("pdf", pdf)
        self._obsList = None if normSet is None else self._list_proxy("obsList", as_list(normSet))

    def compute(self, ctx: Context) -> Any:
        """The density's expected events over the held set - ``ctx`` never changes them."""
        return self.pdf.expectedEvents(self._obsList)
