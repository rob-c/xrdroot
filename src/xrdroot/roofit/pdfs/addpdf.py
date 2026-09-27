"""``RooAddPdf``: a sum of densities, each normalised, weighted by fractions or by yields.

``RooAddPdf("m", "m", [sig, bkg], [f])`` is ``f sig + (1-f) bkg``: one
fraction fewer than densities, the last taking what is left. As many
coefficients as densities are yields - numbers of events - and make the
sum extended, its fractions the yields over their total; densities that
are themselves extended need no coefficients at all. Recursive fractions
(``RooAddPdf(..., True)``) are ``f1 A + (1-f1)(f2 B + (1-f2) C)``, made -
as RooFit makes them - of ``RooRecursiveFraction`` terms.

Each component is normalised over the question's observables on its own,
so the sum of normalised densities is itself one, and needs no
normalisation of its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..collections import RooArgList, as_list
from ..messages import WARNING, log
from ..pdf import CAN_NOT_BE_EXTENDED, MUST_BE_EXTENDED, RooAbsPdf
from ..printing import g
from ..real import Context, RooAbsReal
from ..variables import RooConstVar

__all__ = ["RooAddPdf", "RooRecursiveFraction"]


class RooRecursiveFraction(RooAbsReal):
    """``a0 (1-a1)(1-a2)...``: one recursive fraction, from its list in reverse."""

    def __init__(self, name: Any, title: Any, fractions: Any) -> None:
        super().__init__(name, title)
        self.fractions = self._list_proxy("list", list(reversed(as_list(fractions))))

    def compute(self, ctx: Context) -> Any:
        values = [one.compute(ctx) for one in self.fractions]
        product = values[0]
        for one in values[1:]:
            product = product * (1.0 - one)
        return product

    def state_word(self) -> str:
        return "Clean"


class RooAddPdf(RooAbsPdf):
    """A weighted sum of densities."""

    def __init__(self, name: Any, title: Any = "", *args: Any) -> None:
        super().__init__(name, title)
        pdfs, coefs, recursive = _arguments(args)
        self._recursive = recursive
        self._all_extendable = not coefs
        self._have_last = False
        if len(pdfs) > len(coefs) + 1 or len(pdfs) < len(coefs):
            raise ValueError(f"RooAddPdf::RooAddPdf({self._name}) number of pdfs and coefficients "
                             "inconsistent, must have Npdf=Ncoef or Npdf=Ncoef+1.")  # fmt: skip
        if recursive and len(pdfs) != len(coefs) + 1:
            raise ValueError(f"RooAddPdf::RooAddPdf({self._name}): Recursive fractions option can "
                             "only be used if Npdf=Ncoef+1.")  # fmt: skip
        if self._all_extendable:
            for pdf in pdfs:
                if not pdf.canBeExtended():
                    raise ValueError(f"RooAddPdf::RooAddPdf({self._name}) pdf {pdf.GetName()} is not "
                                     "extendable, RooAddPdf constructor call is invalid!")  # fmt: skip
        made = self._recursive_coefs(pdfs, coefs) if recursive else list(coefs)
        self._have_last = recursive or (bool(coefs) and len(coefs) == len(pdfs))
        self.pdfs = self._list_proxy("!pdfs", pdfs)
        self.coefs = self._list_proxy("!coefficients", made)

    def _recursive_coefs(self, pdfs: list[Any], coefs: list[Any]) -> list[Any]:
        made: list[Any] = []
        given: list[Any] = []
        for pdf, coef in zip(pdfs, [*coefs, RooConstVar("1", "1", 1.0)]):
            given.append(coef)
            if len(given) == 1:
                made.append(coef)
                continue
            name = f"{self._name}_recursive_fraction_{pdf.GetName()}_{len(given)}"
            made.append(RooRecursiveFraction(name, "Recursive Fraction", list(given)))
        return made

    def servers(self) -> list[Any]:
        """The components and their coefficients in turn: RooFit adds them to the list that way."""
        found: list[Any] = []
        for i, pdf in enumerate(self.pdfs):
            found.append(pdf)
            if i < len(self.coefs):
                found.append(self.coefs[i])
        return found

    # -- coefficients -------------------------------------------------------------

    def coefficients(self, ctx: Context, nset: Any = None) -> list[Any]:
        """``updateCoefficients``: each component's share of the sum."""
        if self._all_extendable:
            yields = [pdf.expected(nset or frozenset()) for pdf in self.pdfs]
            total = sum(yields)
            return [one / total for one in yields]
        values = [c.compute(ctx) for c in self.coefs]
        if self._have_last:
            total = sum(values)
            return [v / total for v in values]
        last = 1.0 - sum(values)
        if np.any(np.asarray(last) < 0) or np.any(np.asarray(last) > 1):
            log(self, WARNING, "Eval", f"RooAddPdf::updateCoefCache({self._name}) WARNING: sum of "
                f"PDF coefficients not in range [0-1], value={g(1 - np.asarray(last).reshape(-1)[0])}")
        return [*values, last]

    def value(self, ctx: Context, nset: Any = None, rng: Any = None) -> Any:
        total: Any = 0.0
        for coef, pdf in zip(self.coefficients(ctx, nset), self.pdfs):
            total = total + coef * pdf.value(ctx, nset, rng)
        return total

    def compute(self, ctx: Context) -> Any:
        return self.value(ctx, None)

    def selfNormalized(self) -> bool:
        return True

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        return names

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        total: Any = 0.0
        for coef, pdf in zip(self.coefficients(ctx), self.pdfs):
            total = total + coef * pdf.integrate(names, ctx, rng)
        return total

    def fraction(self, names: frozenset[str], ctx: Context, nset: Any, rng: Any,
                 norm_rng: Any = None) -> Any:  # fmt: skip
        total: Any = 0.0
        for coef, pdf in zip(self.coefficients(ctx, nset), self.pdfs):
            total = total + coef * pdf.fraction(names, ctx, nset, rng, norm_rng)
        return total

    def normalized_name(self, observables: Any, rng: Any = None) -> str:
        return self._name if not rng else super().normalized_name(observables, rng)

    # -- extended -----------------------------------------------------------------

    def extendMode(self) -> int:
        extendable = (self._have_last and not self._recursive) or self._all_extendable
        return MUST_BE_EXTENDED if extendable else CAN_NOT_BE_EXTENDED

    def expected(self, nset: Any, rng: Any = None) -> float:
        """The total yield - of the events inside ``rng`` if one is given."""
        if self._all_extendable:
            yields = [pdf.expected(nset, rng) for pdf in self.pdfs]
            return float(sum(yields))
        yields = [float(c.getVal()) for c in self.coefs]
        if rng and nset:
            yields = [y * float(pdf.fraction(nset, {}, nset, rng)) for y, pdf in zip(yields, self.pdfs)]
        return float(sum(yields))

    def pdfList(self) -> RooArgList:
        return self.pdfs

    def coefList(self) -> RooArgList:
        return self.coefs

    # -- printing -----------------------------------------------------------------

    def printMetaArgs(self) -> str:
        """``RooRealSumPdf::printMetaArgs``: ``f * sig + [%] * bkg``."""
        parts = [f"{c.GetName()} * {p.GetName()}" for c, p in zip(self.coefs, self.pdfs)]
        if len(self.pdfs) > len(self.coefs):
            parts.append(f"[%] * {self.pdfs[len(self.coefs)].GetName()}")
        if not self.coefs:
            parts = [p.GetName() for p in self.pdfs]
        return " + ".join(parts) + " "

    def printValue(self) -> str:
        return f"{g(np.asarray(self.compute({})).reshape(-1)[0])}/1"

    def state_word(self) -> str:
        return "Clean"


def _arguments(args: tuple[Any, ...]) -> tuple[list[Any], list[Any], bool]:
    """The components, the coefficients and the recursive flag, from either constructor."""
    flags = [a for a in args if isinstance(a, bool)]
    rest = [a for a in args if not isinstance(a, bool)]
    if rest and all(not isinstance(a, (list, tuple, set)) and not hasattr(a, "_list") for a in rest):
        if len(rest) == 3:  # RooAddPdf(name, title, pdf1, pdf2, coef1)
            return [rest[0], rest[1]], [rest[2]], bool(flags and flags[0])
    lists = [as_list(a) for a in rest]
    return lists[0], (lists[1] if len(lists) > 1 else []), bool(flags and flags[0])
