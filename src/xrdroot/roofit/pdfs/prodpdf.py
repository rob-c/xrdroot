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
from . import prodcond

__all__ = ["RooProdPdf"]


def _factors(args: tuple[Any, ...]) -> tuple[list[Any], dict[str, tuple[frozenset[str], bool]]]:
    """The factors, and for each conditional one its normalised observables and which way round."""
    pdfs: list[Any] = []
    conditional: dict[str, tuple[frozenset[str], bool]] = {}
    for arg in args:
        if not isinstance(arg, RooCmdArg):
            _add_new(pdfs, as_list(arg))
        elif arg.name == "Conditional":
            chosen = as_list(arg.value(0))
            observables = frozenset(one.GetName() for one in as_list(arg.value(1)))
            conditional.update(
                {p.GetName(): (observables, bool(arg.value(2, False))) for p in chosen}
            )
            _add_new(pdfs, chosen)
    return pdfs, conditional


def _add_new(pdfs: list[Any], more: list[Any]) -> None:
    """``more`` added to ``pdfs``, but those already in it."""
    pdfs.extend(p for p in more if all(p is not q for q in pdfs))


class RooProdPdf(RooAbsPdf):
    """A product of densities."""

    def __init__(self, name: Any, title: Any = "", *args: Any, **kwargs: Any) -> None:
        from ..cmdargs import make

        super().__init__(name, title)
        args = args + tuple(make(key, value) for key, value in kwargs.items())  # Conditional=(...)
        numbers = [a for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
        pdfs, self._conditional = _factors(
            tuple(a for a in args if a not in numbers or isinstance(a, bool))
        )
        self._cutoff = float(numbers[0]) if numbers else 0.0
        self.pdfs = self._list_proxy("!pdfs", pdfs)

    def factor_nset(self, pdf: Any, nset: frozenset[str]) -> frozenset[str]:
        """What ``pdf`` is normalised over within ``nset``: all of it, or its conditional part."""
        mine = frozenset(nset & pdf.dependents())
        found = self._conditional.get(pdf.GetName())
        if found is None:
            return mine
        observables, reverse = found
        return frozenset(mine - observables if reverse else mine & observables)

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
            mine = self.factor_nset(pdf, nset)
            if mine:  # a factor of none of the observables - a constraint - is left out
                found = found * pdf.value(ctx, mine, rng)
        return found

    def constraint_terms(
        self, observables: frozenset[str], params: list[Any], strip: bool
    ) -> list[Any]:
        """``getConstraints``: the factors of none of the observables, but of the parameters -
        of those the other factors have too, unless the parameters were named."""
        names = {one.GetName() for one in params}
        found = []
        for pdf in self.pdfs:
            mine = pdf.dependents()
            if not mine & observables and mine & names:
                others = frozenset().union(*(q.dependents() for q in self.pdfs if q is not pdf))
                if not strip or mine & names & others:
                    found.append(pdf)
        return found

    def fraction(
        self, names: frozenset[str], ctx: Context, nset: Any, rng: Any, norm_rng: Any = None
    ) -> Any:
        nset = frozenset(nset)
        if rng and "," in str(rng):  # a product of sums is not a sum of products: part by part
            return sum(
                self.fraction(names, ctx, nset, part, norm_rng)
                for part in str(rng).split(",")
                if part
            )
        if not self._factorizes(nset):
            return super().fraction(names, ctx, nset, rng, norm_rng)
        if self._conditional:
            return prodcond.fraction(self, frozenset(names), ctx, nset, rng, norm_rng)
        found: Any = 1.0
        for pdf in self.pdfs:
            mine = self.factor_nset(pdf, nset)
            part = names & mine
            found = found * (
                pdf.fraction(part, ctx, mine, rng, norm_rng)
                if part
                else pdf.value(ctx, mine, norm_rng)
            )
        return found

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        """A product of factors of separate observables integrates factor by factor - with
        conditional
        factors, all but the variables some factor is conditional on and must be integrated with."""
        names = frozenset(names)
        if self._conditional and self._factorizes(names):
            return names - prodcond.joint_names(self, names, names)
        if self._conditional or not self._factorizes(names):
            return frozenset()
        return names

    def announce_projection(self, names: frozenset[str], nset: frozenset[str]) -> bool:
        """Say the ``SPECINT`` a plot projection over ``names`` makes, if it makes one."""
        return bool(self._conditional) and prodcond.announce(self, names, nset)

    def normalized_name(self, observables: Any, rng: Any = None) -> str:
        """A product normalises itself, factor by factor - ``selfNormalized``: its own name."""
        return self._name

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        if self._conditional:
            return prodcond.fraction(self, names, ctx, names, rng)
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

    def expected(self, nset: Any, rng: Any = None, fit: bool = False) -> float:
        found = self._extended()
        return 0.0 if found is None else float(found.expected(nset, rng, fit))

    def gen_context(self, names: frozenset[str]) -> Any:
        from ..generation.contexts import ProductContext

        return ProductContext(self, names)

    def pdfList(self) -> Any:
        return self.pdfs

    def state_word(self) -> str:
        return "Dirty"
