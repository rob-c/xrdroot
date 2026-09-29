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
from ..selection import active
from ..variables import RooConstVar


def g6(value: Any) -> str:
    """A number as a fresh ``std::ostringstream`` prints one: six figures, whatever ``cout`` is
    at."""
    return g(value, 6)


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
        #: A coefficient list given, even empty, makes fractions: one density is all of the sum.
        rest = [a for a in args if not isinstance(a, bool)]
        given = len(rest) > 1 and _is_list(rest[-1])
        self._all_extendable = not coefs and not given
        self._check(pdfs, coefs, recursive)
        made = self._recursive_coefs(pdfs, coefs) if recursive else list(coefs)
        self._have_last = recursive or (bool(coefs) and len(coefs) == len(pdfs))
        self.pdfs = self._list_proxy("!pdfs", pdfs)
        self.coefs = self._list_proxy("!coefficients", made)

    def _check(self, pdfs: list[Any], coefs: list[Any], recursive: bool) -> None:
        """RooFit's refusals: counts that do not match, or components that cannot give yields."""
        if coefs and (len(pdfs) > len(coefs) + 1 or len(pdfs) < len(coefs)):
            raise ValueError(
                f"RooAddPdf::RooAddPdf({self._name}) number of pdfs and coefficients "
                "inconsistent, must have Npdf=Ncoef or Npdf=Ncoef+1."
            )
        if recursive and len(pdfs) != len(coefs) + 1:
            raise ValueError(
                f"RooAddPdf::RooAddPdf({self._name}): Recursive fractions option can "
                "only be used if Npdf=Ncoef+1."
            )
        for pdf in pdfs if self._all_extendable else []:
            if not pdf.canBeExtended():
                raise ValueError(
                    f"RooAddPdf::RooAddPdf({self._name}) pdf {pdf.GetName()} is not "
                    "extendable, RooAddPdf constructor call is invalid!"
                )

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
            inverse = 1.0 / sum(values)  # RooFit multiplies by the reciprocal, not divides
            return [v * inverse for v in values]
        last = 1.0 - sum(values)
        self._warn_sum(last)
        return [*values, last]

    def _warn_sum(self, last: Any) -> None:
        """``updateCoefCache``'s warning of coefficients summing to more than one, or below 0."""
        if np.any(np.asarray(last) < 0) or np.any(np.asarray(last) > 1):
            log(
                self,
                WARNING,
                "Eval",
                f"RooAddPdf::updateCoefCache({self._name}) WARNING: sum of "
                "PDF coefficients not in range [0-1], "
                f"value={g(1 - np.asarray(last).reshape(-1)[0])}",
            )

    def value(self, ctx: Context, nset: Any = None, rng: Any = None) -> Any:
        """The sum - its coefficients those of the full range, then normalised within ``rng``.

        In a range RooFit projects the coefficients (``updateCoefficients``):
        each is multiplied by its component's share of ``rng`` (:meth:`_shares`),
        the products divided by their sum, and each component normalised over
        ``rng`` - the same density as normalising the full-range sum, but
        rounded as RooFit rounds it, which a fit's last steps depend on.
        """
        coefs = self.coefficients(ctx, nset)
        if rng and nset:
            coefs = [c * share for c, share in zip(coefs, self._shares(ctx, nset, rng))]
            projected: Any = 0.0
            for coef in coefs:
                projected = projected + coef
            coefs = [coef / projected for coef in coefs]
        else:
            rng = None if nset else rng
        total: Any = 0.0
        for coef, pdf in zip(coefs, self.pdfs):
            if active(pdf):
                total = total + coef * pdf.value(ctx, nset, rng)
        return total

    def _shares(self, ctx: Context, nset: Any, rng: Any) -> list[Any]:
        """Each component's share of the range ``rng``, as RooFit's projection rounds it.

        RooFit divides the integral over ``rng`` - normalised there, so one -
        by the full integral normalised over ``rng``: one over ``I / I(rng)``,
        not ``I(rng) / I``, which is the same fraction a bit away. The two are
        each component's own integrals, ``rng`` taken part by part if it is
        several ranges: a product of densities over two boxes in ``x`` and
        ``y`` is no product of its factors' integrals over the union.
        """
        names = frozenset(nset)
        shares = []
        for pdf in self.pdfs:
            mine = names & pdf.dependents()
            shares.append(1.0 / (pdf.integrate(mine, ctx, None) / pdf.integrate(mine, ctx, rng)))
        return shares

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

    def fraction(
        self, names: frozenset[str], ctx: Context, nset: Any, rng: Any, norm_rng: Any = None
    ) -> Any:
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

    def expected(self, nset: Any, rng: Any = None, fit: bool = False) -> float:
        """The total yield - of the events inside the fit range ``rng`` if one is given.

        Each component's yield - its coefficient, or what it expects over
        everything - is scaled by its share of ``rng`` (:meth:`_shares`), as
        RooFit's ``expectedEvents`` projects it.
        """
        if self._all_extendable:
            yields = [pdf.expected(nset, None, fit) for pdf in self.pdfs]
        else:
            yields = [float(c.getVal()) for c in self.coefs]
        if rng and nset:
            yields = [float(share) * y for share, y in zip(self._shares({}, nset, rng), yields)]
        total = 0.0
        for one in yields:
            total += one
        return total

    def gen_context(self, names: frozenset[str]) -> Any:
        from ..generation.contexts import SumContext

        return SumContext(self, names)

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
        if not len(self.coefs):
            parts = [p.GetName() for p in self.pdfs]
        return " + ".join(parts) + " "

    def compiled_origin(self, nset: frozenset[str], rng: Any = None) -> str:
        """How RooFit describes this sum in a fit: its terms as the normalised densities they
        are."""
        labels = [
            p.normalized_label(nset & p.dependents(), rng)
            if not p.selfNormalized()
            else p.GetName()
            for p in self.pdfs
        ]
        parts = [f"{c.GetName()} * {label}" for c, label in zip(self.coefs, labels)]
        if len(labels) > len(self.coefs):
            parts.append(f"[%] * {labels[len(self.coefs)]}")
        return f"RooAddPdf::{self._name}[ " + " + ".join(parts) + " ]"

    def compiled_servers(self, nset: frozenset[str], rng: Any = None) -> str:
        """Its inputs' values now: the observables, the normalised terms, the coefficients."""
        observables = [one for one in self.leaves() if one.GetName() in nset]
        terms = []
        for pdf in self.pdfs:
            label = (
                pdf.normalized_label(nset & pdf.dependents(), rng)
                if not pdf.selfNormalized()
                else pdf.GetName()
            )
            terms.append(
                f"{label} = {g6(float(np.asarray(pdf.value({}, nset & pdf.dependents(), rng))))}"
            )
        return (
            "!refCoefNorm=("
            + ",".join(f"{o.GetName()} = {g6(o.getVal())}" for o in observables)
            + "), !pdfs=("
            + ",".join(terms)
            + "), !coefficients=("
            + ",".join(f"{c.GetName()} = {g6(c.getVal())}" for c in self.coefs)
            + ")"
        )

    def printValue(self) -> str:
        return f"{g(np.asarray(self.compute({})).reshape(-1)[0])}/1"

    def state_word(self) -> str:
        return "Clean"


def _arguments(args: tuple[Any, ...]) -> tuple[list[Any], list[Any], bool]:
    """The components, the coefficients and the recursive flag, from either constructor."""
    recursive = next((a for a in args if isinstance(a, bool)), False)
    rest = [a for a in args if not isinstance(a, bool)]
    if len(rest) == 3 and not any(_is_list(a) for a in rest):  # (name, title, pdf1, pdf2, coef1)
        return [rest[0], rest[1]], [rest[2]], recursive
    lists = [*(as_list(a) for a in rest), []]
    return lists[0], lists[1], recursive


def _is_list(arg: Any) -> bool:
    return isinstance(arg, (list, tuple, set)) or hasattr(arg, "_list")
