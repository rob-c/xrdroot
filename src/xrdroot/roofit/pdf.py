"""``RooAbsPdf``: a probability density, normalised over whichever variables are its observables.

A density's formula - ``exp(-(x-m)^2/2s^2)`` for a Gaussian - is not
normalised; RooFit divides it by its integral over the *normalisation
set*, the variables it is a density of, at the parameters' current values,
and so does :meth:`RooAbsPdf.value`. The normalisation set is not a
property of the density but of the question: the same Gaussian is a density
of ``x`` in a fit to ``x`` and a function of ``x`` inside a product that
normalises itself.

Fitting, generating and plotting are what a density is for, and each has a
module of its own; the methods here are ROOT's names for them.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .collections import as_list
from .messages import WARNING, log
from .printing import g
from .real import Context, RooAbsReal, names_in, value_of

__all__ = ["RooAbsPdf", "check_range"]

#: ``RooAbsPdf::ExtendMode``.
CAN_NOT_BE_EXTENDED, CAN_BE_EXTENDED, MUST_BE_EXTENDED = 0, 1, 2


class RooAbsPdf(RooAbsReal):
    """A density: its formula, normalised over the observables a question names."""

    #: ``RooAbsPdf::CanNotBeExtended`` and the rest, as ROOT's class members.
    CanNotBeExtended, CanBeExtended, MustBeExtended = (
        CAN_NOT_BE_EXTENDED, CAN_BE_EXTENDED, MUST_BE_EXTENDED
    )  # fmt: skip

    def __init__(self, name: Any = "", title: Any = "") -> None:
        super().__init__(name, title)
        self._norm_range: str | None = None
        #: The normalisation set of the last ``getVal`` asked with one, which ``Print`` shows.
        self._last_norm: frozenset[str] | None = None

    # -- values -------------------------------------------------------------------

    def selfNormalized(self) -> bool:
        return False

    def value(self, ctx: Context, nset: Any = None, rng: Any = None) -> Any:
        """The formula divided by its integral over ``nset``, in ``rng`` or the norm range."""
        raw = self.compute(ctx)
        if not nset or self.selfNormalized():
            return raw
        return raw / self.norm(ctx, nset, rng)

    def norm(self, ctx: Context, nset: Any, rng: Any = None) -> Any:
        """The normalisation integral: over the observables in ``nset`` this depends on."""
        names = frozenset(nset) & self.dependents()
        if not names:
            return self.compute(ctx)
        return self.integrate(names, ctx, rng or self._norm_range)

    def getVal(self, nset: Any = None) -> float:
        names = names_in(nset)
        if names:
            self._last_norm = names
        return value_of(self.value({}, names))

    def getNorm(self, nset: Any = None) -> float:
        names = names_in(nset)
        return 1.0 if not names else value_of(self.norm({}, names))

    def getLogVal(self, nset: Any = None) -> float:
        found = self.getVal(nset)
        return math.log(found) if found > 0 else math.nan

    def setNormRange(self, rng: Any) -> None:
        self._norm_range = str(rng) if rng else None

    def normRange(self) -> Any:
        return self._norm_range

    def fraction(self, names: frozenset[str], ctx: Context, nset: Any, rng: Any,
                 norm_rng: Any = None) -> Any:  # fmt: skip
        """The integral over ``names`` in ``rng`` of the density normalised over ``nset``."""
        top = self.integrate(names, ctx, rng)
        return top / self.norm(ctx, nset, norm_rng)

    def normalized_name(self, observables: Any, rng: Any = None) -> str:
        """``RooNormalizedPdf``'s name: ``g_over_g_Int[x]``, or ``g_over_g_Int[x|left,right]``."""
        from .integration import integral_name

        if self.selfNormalized():
            return self._name
        names = frozenset(one.GetName() for one in as_list(observables)) & self.dependents()
        return f"{self._name}_over_{integral_name(self, names, rng)}"

    # -- extended densities -------------------------------------------------------

    def extendMode(self) -> int:
        return CAN_NOT_BE_EXTENDED

    def canBeExtended(self) -> bool:
        return self.extendMode() != CAN_NOT_BE_EXTENDED

    def mustBeExtended(self) -> bool:
        return self.extendMode() == MUST_BE_EXTENDED

    def expected(self, nset: Any, rng: Any = None) -> float:
        """How many events the density expects, in ``rng`` if it is given."""
        return 0.0

    def expectedEvents(self, nset: Any = None) -> float:
        return self.expected(names_in(nset) or frozenset())

    def extendedTerm(self, observed: float, expected: float, sumw2: float = 0.0) -> float:
        """``RooAbsPdf::extendedTerm``: the Poisson term of an extended likelihood."""
        if abs(expected) < 1e-10 and abs(observed) < 1e-10:
            return 0.0
        extra = expected - observed * math.log(expected) if expected > 0 else math.nan
        if sumw2 != 0.0:
            extra *= sumw2 / observed
        return extra

    # -- what a density is for ----------------------------------------------------

    def fitTo(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        from .fitting.fit import fit_to

        return fit_to(self, data, args, kwargs)

    def createNLL(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        from .fitting.nll import create_nll

        return create_nll(self, data, args, kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        from .generation.generate import generate

        return generate(self, args, kwargs)

    def generateBinned(self, *args: Any, **kwargs: Any) -> Any:
        from .generation.generate import generate_binned

        return generate_binned(self, args, kwargs)

    def plotOn(self, frame: Any, *args: Any, **kwargs: Any) -> Any:
        from .plot.curves import plot_pdf

        return plot_pdf(self, frame, args, kwargs)

    def paramOn(self, frame: Any, *args: Any, **kwargs: Any) -> Any:
        from .plot.params import param_on

        return param_on(self, frame, args, kwargs)

    # -- printing -----------------------------------------------------------------

    def printValue(self) -> str:
        raw = value_of(self.compute({}))
        if self._last_norm is None:
            return g(raw)
        return f"{g(raw)}/{g(self.getNorm(list(self._by_names(self._last_norm))))}"

    def _by_names(self, names: frozenset[str]) -> list[Any]:
        return [one for one in self.leaves() if one.GetName() in names]


def check_range(pdf: Any, params: Any, low: float, high: float = math.inf,
                closed: bool = False, extra: str = "") -> None:  # fmt: skip
    """``RooHelpers::checkRangeOfParameters``: warn of parameters that can leave their safe range."""
    shown = ("[" if closed else "(") + ("-inf" if low <= -1.7976931348623157e308 else g(low))
    shown += ", " + ("inf" if high >= 1.7976931348623157e308 else g(high)) + ("]" if closed else ")")
    for param in as_list(params):
        if not param.InheritsFrom("RooAbsRealLValue"):
            continue
        pmin, pmax = param.getMin(), param.getMax()
        outside = pmin < low or pmax > high
        if outside or (not closed and (pmin == low or pmax == high)):
            log(pdf, WARNING, "InputArguments", f"The parameter '{param.GetName()}' with range "
                f"[{g(pmin)}, {g(pmax)}] of the {pdf.ClassName()} '{pdf.GetName()}' exceeds the safe "
                f"range of {shown}. Advise to limit its range." + (f"\n{extra}" if extra else ""))


def names(items: Any) -> frozenset[str]:
    return frozenset(one.GetName() for one in as_list(items))


def as_array(value: Any) -> np.ndarray[Any, Any]:
    return np.asarray(value, dtype=np.float64)
