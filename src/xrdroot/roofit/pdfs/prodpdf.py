"""``RooProdPdf``: a product of densities - of independent observables, or conditional ones.

``RooProdPdf("p", "p", [gx, gy])`` is ``gx(x) gy(y)``, each factor
normalised over its own observables, so the product needs no normalisation
of its own. ``Conditional(model, y)`` makes a factor a density of ``y``
only, given the others: normalised over ``y`` alone. Factors that share an
observable are normalised together, numerically, as RooFit must.

Generating from a product draws each factor's observables from that
factor, the factors a conditional one depends on first
(``RooProdGenContext``).
"""

from __future__ import annotations

from typing import Any

from ..cmdargs import RooCmdArg
from ..collections import as_list
from ..pdf import CAN_BE_EXTENDED, CAN_NOT_BE_EXTENDED, RooAbsPdf
from ..real import Context

__all__ = ["RooProdPdf"]


def _factors(args: tuple[Any, ...]) -> tuple[list[Any], dict[str, tuple[frozenset[str], bool]]]:
    """The factors, and for each conditional one its normalised observables and which way round."""
    pdfs: list[Any] = []
    conditional: dict[str, tuple[frozenset[str], bool]] = {}
    for arg in args:
        if isinstance(arg, RooCmdArg):
            if arg.name == "Conditional":
                chosen = as_list(arg.value(0))
                observables = frozenset(one.GetName() for one in as_list(arg.value(1)))
                for pdf in chosen:
                    conditional[pdf.GetName()] = (observables, bool(arg.value(2, False)))
                pdfs.extend(p for p in chosen if all(p is not q for q in pdfs))
            continue
        pdfs.extend(p for p in as_list(arg) if all(p is not q for q in pdfs))
    return pdfs, conditional


class RooProdPdf(RooAbsPdf):
    """A product of densities."""

    def __init__(self, name: Any, title: Any = "", *args: Any, **kwargs: Any) -> None:
        super().__init__(name, title)
        from ..cmdargs import make

        args = args + tuple(make(key, value) for key, value in kwargs.items())
        numbers = [a for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
        pdfs, self._conditional = _factors(tuple(a for a in args if a not in numbers or isinstance(a, bool)))
        self._cutoff = float(numbers[0]) if numbers else 0.0
        self.pdfs = self._list_proxy("!pdfs", pdfs)

    def factor_nset(self, pdf: Any, nset: frozenset[str]) -> frozenset[str]:
        """What ``pdf`` is normalised over within ``nset``: all of it, or its conditional part."""
        mine = nset & pdf.dependents()
        found = self._conditional.get(pdf.GetName())
        if found is None:
            return mine
        observables, reverse = found
        return mine - observables if reverse else mine & observables

    def _factorizes(self, nset: frozenset[str]) -> bool:
        seen: set[str] = set()
        for pdf in self.pdfs:
            mine = self.factor_nset(pdf, nset)
            if seen & mine:
                return False
            seen |= mine
        return True

    def compute(self, ctx: Context) -> Any:
        found: Any = 1.0
        for pdf in self.pdfs:
            found = found * pdf.compute(ctx)
        return found

    def value(self, ctx: Context, nset: Any = None, rng: Any = None) -> Any:
        if not nset:
            return self.compute(ctx)
        nset = frozenset(nset)
        if not self._factorizes(nset):
            return super().value(ctx, nset, rng)
        found: Any = 1.0
        for pdf in self.pdfs:
            found = found * pdf.value(ctx, self.factor_nset(pdf, nset), rng)
        return found

    def fraction(self, names: frozenset[str], ctx: Context, nset: Any, rng: Any,
                 norm_rng: Any = None) -> Any:  # fmt: skip
        nset = frozenset(nset)
        if rng and "," in str(rng):  # a product of sums is not a sum of products: part by part
            return sum(self.fraction(names, ctx, nset, part, norm_rng)
                       for part in str(rng).split(",") if part)  # fmt: skip
        if not self._factorizes(nset) or self._conditional:
            return super().fraction(names, ctx, nset, rng, norm_rng)
        found: Any = 1.0
        for pdf in self.pdfs:
            mine = self.factor_nset(pdf, nset)
            part = names & mine
            found = found * (pdf.fraction(part, ctx, mine, rng, norm_rng) if part
                             else pdf.value(ctx, mine, norm_rng))  # fmt: skip
        return found

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        """A product of factors of separate observables integrates factor by factor."""
        if self._conditional or not self._factorizes(frozenset(names)):
            return frozenset()
        return frozenset(names)

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        found: Any = 1.0
        for pdf in self.pdfs:
            part = names & pdf.dependents()
            found = found * (pdf.integrate(part, ctx, rng) if part else pdf.compute(ctx))
        return found

    # -- extended and generation --------------------------------------------------

    def _extended(self) -> Any:
        return next((pdf for pdf in self.pdfs if pdf.canBeExtended()), None)

    def extendMode(self) -> int:
        return CAN_BE_EXTENDED if self._extended() is not None else CAN_NOT_BE_EXTENDED

    def expected(self, nset: Any, rng: Any = None) -> float:
        found = self._extended()
        return 0.0 if found is None else float(found.expected(nset, rng))

    def gen_context(self, names: frozenset[str]) -> Any:
        from ..generation.contexts import ProductContext

        return ProductContext(self, names)

    def pdfList(self) -> Any:
        return self.pdfs

    def state_word(self) -> str:
        return "Dirty"
