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

from . import evalerrors
from .binning import evaluating
from .collections import as_list
from .messages import ERROR, WARNING, log
from .nanpack import pack
from .printing import g, kArgs, kClassName, kInline, kName, kValue
from .real import Context, RooAbsReal, names_in, value_of

__all__ = ["RooAbsPdf", "check_range"]

#: ``RooAbsPdf::ExtendMode``.
CAN_NOT_BE_EXTENDED, CAN_BE_EXTENDED, MUST_BE_EXTENDED = 0, 1, 2


class RooAbsPdf(RooAbsReal):
    """A density: its formula, normalised over the observables a question names."""

    #: ``RooAbsPdf::CanNotBeExtended`` and the rest, as ROOT's class members.
    CanNotBeExtended, CanBeExtended, MustBeExtended = (
        CAN_NOT_BE_EXTENDED,
        CAN_BE_EXTENDED,
        MUST_BE_EXTENDED,
    )

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
        norm = self.norm(ctx, nset, rng)
        if evalerrors.active() and not (
            np.all(np.asarray(norm) > 0) and np.all(np.asarray(raw) >= 0)
        ):
            self._log_failures(raw, norm, frozenset(nset), rng)
        elif not evalerrors.quiet() and _unnormalisable(raw, norm):
            self.logEvalError(f"p.d.f normalization integral is zero or negative: {float(norm):f}")
        return normalized(raw, norm)

    def logEvalError(self, message: str) -> None:
        """``RooAbsReal::logEvalError`` outside a fit: the error printed at once - the object,
        the message, and each input's value as its proxy prints it."""
        servers = ", ".join(_proxy_values(p) for p in self._proxies)
        origin = self.printStream(kClassName | kName | kArgs, kInline)
        text = f"RooAbsReal::logEvalError({self._name}) evaluation error, \n origin       : "
        text += f"{origin}\n message      : {message}\n server values: {servers}"
        log(self, ERROR, "Eval", text)

    def normalized_label(self, nset: frozenset[str], rng: Any = None) -> str:
        """``g_over_g_Int[x]``: what RooFit calls this density normalised over ``nset``."""
        return self.normalized_name(list(self._by_names(nset)), rng)

    def normalized_origin(self, nset: frozenset[str], rng: Any = None) -> str:
        from .integration import integral_name

        norm = integral_name(self, nset & self.dependents(), rng)
        return (
            f"RooFit::Detail::RooNormalizedPdf::{self.normalized_label(nset, rng)}[ numerator="
            f"{self._name} denominator={norm} ]"
        )

    def normalized_servers(self, nset: frozenset[str], rng: Any = None) -> str:
        from .integration import integral_name

        norm = integral_name(self, nset & self.dependents(), rng)
        return (
            f"numerator={self._name}={g(value_of(self.compute({})), 6)}, denominator={norm}="
            f"{g(value_of(self.norm({}, nset, rng)), 6)}"
        )

    def _log_failures(self, raw: Any, norm: Any, nset: frozenset[str], rng: Any) -> None:
        """``RooNormalizedPdf::doEval``'s messages, by kind - with its kernel's thresholds."""
        raw, norm = (
            np.asarray(raw, dtype=np.float64),
            np.broadcast_to(np.asarray(norm), np.shape(raw)),
        )
        bad_norm = (norm < 0) | ((norm == 0) & (raw != 0))
        negative = ~bad_norm & (raw < 0)
        nan = ~bad_norm & ~negative & np.isnan(raw)
        counts = (
            int(np.count_nonzero(bad_norm)),
            int(np.count_nonzero(negative)),
            int(np.count_nonzero(nan)),
        )
        messages = (
            "p.d.f normalization integral is zero or negative",
            "p.d.f value is less than zero, trying to recover",
            "p.d.f value is Not-a-Number",
        )
        for kind, (number, message) in enumerate(zip(counts, messages, strict=False)):
            if number > kind:  # the kernel reports a kind only above that many: RooFit's own quirk
                evalerrors.record(
                    ("norm", id(self)),
                    lambda: self.normalized_origin(nset, rng),
                    message,
                    lambda: self.normalized_servers(nset, rng),
                    number,
                )

    def norm(self, ctx: Context, nset: Any, rng: Any = None) -> Any:
        """The normalisation integral: over the observables in ``nset`` this depends on."""
        names = frozenset(nset) & self.dependents()
        if not names:  # "Unit Normalization": none of the set is the density's
            return 1.0
        rng = rng or self._norm_range
        with evaluating(ctx):  # a range with a column for an end is one per event
            key = self._norm_key(names, ctx, rng)
            cached = self.__dict__.get("_norm_cache")
            if key is not None and cached is not None and cached[0] == key:
                return cached[1]
            found = self.integrate(names, ctx, rng)
        if key is not None:
            self.__dict__["_norm_cache"] = (key, found)
        return found

    def _norm_key(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        """What the normalisation depends on - the other variables' values and the ranges -
        or ``None`` when one of them is a column, which is not worth remembering.

        RooFit caches its normalisation integrals the same way, recomputing one
        only when a parameter it depends on has changed.
        """
        values: list[tuple[Any, ...]] = []
        for leaf in self.leaves():
            name = leaf.GetName()
            if name in names and not hasattr(
                leaf, "getMin"
            ):  # a category: all its states, or rng's
                values.append((name,))
                continue
            if name in names:
                ends = (leaf.getMin(rng), leaf.getMax(rng))
                if np.ndim(ends[0]) or np.ndim(ends[1]):
                    return None
                values.append((name, *ends))
                continue
            value = ctx.get(name, None)
            if value is not None and np.ndim(value):
                return None
            values.append((name, float(leaf.getVal() if value is None else value)))
        return (names, rng, tuple(values))

    def getVal(self, nset: Any = None) -> float:
        names = names_in(nset)
        if names:
            self._last_norm = names
            self._announce_norm(names)
        return value_of(self.value({}, names))

    def _announce_norm(self, names: frozenset[str]) -> None:
        """The normalisation integral over ``names``, said the first time this density makes it -
        once per set, as its normalisation cache keeps one."""
        from .integration import announce

        owner, made = self.__dict__.get("_norms_made", (None, set()))
        if owner != id(self):  # a copy's cache is its own, and starts empty, as a clone's does
            made = set()
            self.__dict__["_norms_made"] = (id(self), made)
        if names not in made:
            made.add(names)
            announce(self, names & self.dependents())

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

    def fraction(
        self, names: frozenset[str], ctx: Context, nset: Any, rng: Any, norm_rng: Any = None
    ) -> Any:
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

    def expected(self, nset: Any, rng: Any = None, fit: bool = False) -> float:
        """How many events the density expects, in the fit range ``rng`` if it is given.

        ``fit`` asks for the number as a likelihood computes it: RooFit's
        likelihood evaluates the expected events through compiled nodes that
        round some divisions as multiplications by a reciprocal, where
        ``expectedEvents`` divides - the same number to within a bit or two,
        and the likelihood's last bit is what Minuit steps by.
        """
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

    def createChi2(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        from .fitting.chi2 import create_chi2

        return create_chi2(self, data, args, kwargs)

    def chi2FitTo(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        from .fitting.chi2 import chi2_fit_to

        return chi2_fit_to(self, data, args, kwargs)

    def createNLL(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        from .fitting.nll import create_nll

        return create_nll(self, data, args, kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        from .generation.generate import generate

        return generate(self, args, kwargs)

    def generateSimGlobal(self, whatVars: Any, nEvents: int) -> Any:
        """Global observables, ``nEvents`` sets of them: for a single density, ``generate``."""
        return self.generate(whatVars, int(nEvents))

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


def normalized(raw: Any, norm: Any) -> Any:
    """``computeNormalizedPdf``: ``raw / norm``, or a NaN saying how bad it is where that is wrong.

    A normalisation below zero - or zero under a value that is not - and a
    value below zero give NaNs whose payload is how far below zero they are,
    as RooFit makes them, so that the likelihood can tell Minuit how far to
    back away; a NaN stays NaN, and nothing over nothing is nothing.
    """
    raw, norm = np.asarray(raw, dtype=np.float64), np.asarray(norm, dtype=np.float64)
    if np.all(norm > 0) and np.all(raw >= 0):  # the usual case, and nothing to pack
        found = raw / norm
        return found if found.ndim else float(found)
    bad_norm = (norm < 0) | ((norm == 0) & (raw != 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        found = np.where((raw == 0) & (norm == 0), 0.0, raw / norm)
    found = np.where(np.isnan(raw), raw, found)
    found = np.where(raw < 0, pack(-raw), found)
    found = np.where(bad_norm, pack(-norm + np.where(raw < 0, -raw, 0.0)), found)
    return found if found.ndim else float(found)


def _shown_range(low: float, high: float, closed: bool) -> str:
    """The safe range as ``RooHelpers`` prints it: ``(0, inf)``, ``[-1, 1]``."""
    lower = "-inf" if low <= -1.7976931348623157e308 else g(low)
    upper = "inf" if high >= 1.7976931348623157e308 else g(high)
    return ("[" if closed else "(") + f"{lower}, {upper}" + ("]" if closed else ")")


def _unsafe(param: Any, low: float, high: float, closed: bool) -> bool:
    """Whether a parameter's range goes past - or, for an open range, reaches - the safe one."""
    pmin, pmax = param.getMin(), param.getMax()
    if pmin < low or pmax > high:
        return True
    return not closed and (pmin == low or pmax == high)


#: Whether the range checks are off: a density read from a file is streamed, not constructed.
UNCHECKED = [False]


def check_range(
    pdf: Any, params: Any, low: float, high: float = math.inf, closed: bool = False, extra: str = ""
) -> None:
    """``RooHelpers::checkRangeOfParameters``: warn of parameters that can leave their safe
    range."""
    if UNCHECKED[0]:
        return
    shown = _shown_range(low, high, closed)
    for param in as_list(params):
        if param.InheritsFrom("RooAbsRealLValue") and _unsafe(param, low, high, closed):
            log(
                pdf,
                WARNING,
                "InputArguments",
                f"The parameter '{param.GetName()}' with range "
                f"[{g(param.getMin())}, {g(param.getMax())}] of the {pdf.ClassName()} "
                f"'{pdf.GetName()}' exceeds the safe range of {shown}. Advise to limit its range."
                + (f"\n{extra}" if extra else ""),
            )


def names(items: Any) -> frozenset[str]:
    return frozenset(one.GetName() for one in as_list(items))


def as_array(value: Any) -> np.ndarray[Any, Any]:
    return np.asarray(value, dtype=np.float64)


def _proxy_values(proxy: Any) -> str:
    """``RooAbsProxy::print(os, addContents=true)``: ``x=x=0``, or ``c=(a = 1 +/- 0,b = 2)``."""
    if proxy.many:
        inline = ",".join(one.printStream(kValue | kName, kInline) for one in proxy.target)
        return f"{proxy.name}=({inline})"
    return f"{proxy.name}={proxy.target.GetName()}={g(proxy.target.getVal())}"


def _unnormalisable(raw: Any, norm: Any) -> bool:
    """``RooAbsPdf::getValV``'s test: one normalisation, negative - or zero under a value."""
    return bool(np.ndim(norm) == 0 and (norm < 0 or (norm == 0 and np.any(np.asarray(raw) != 0))))
