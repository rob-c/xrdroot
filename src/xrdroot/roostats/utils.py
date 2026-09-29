"""``RooStatsUtils.h`` and ``NumberCountingUtils``: significances, p-values, and the helpers
every calculator uses.

A significance is the number of Gaussian standard deviations whose upper
tail holds a given p-value - ``normal_quantile_c`` - and back again; the
number-counting significances are the "Z_Bi" of Cousins, Linnemann and
Tucker: the p-value of a Poisson main measurement with a Poisson sideband,
which is an incomplete beta function.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from ..efficiency import regularized_beta
from ..roofit.collections import RooArgSet, as_list

__all__ = [
    "AsimovSignificance",
    "FactorizePdf",
    "MakeNuisancePdf",
    "MakeUnconstrainedPdf",
    "StripConstraints",
    "NumberCountingUtils",
    "PValueToSignificance",
    "RemoveConstantParameters",
    "SetAllConstant",
    "SetParameters",
    "SignificanceToPValue",
]


def PValueToSignificance(pvalue: float) -> float:
    """``normal_quantile_c(p, 1)``: how many sigma out the upper tail holds ``p``."""
    if pvalue <= 0.0:
        return math.inf
    if pvalue >= 1.0:
        return -math.inf
    return -statistics.NormalDist().inv_cdf(float(pvalue))


def SignificanceToPValue(Z: float) -> float:
    """``normal_cdf_c(Z)``: the upper tail beyond ``Z`` sigma."""
    return 0.5 * math.erfc(float(Z) / math.sqrt(2.0))


def AsimovSignificance(s: float, b: float, sigma_b: float = 0.0) -> float:
    """The median significance of ``s`` over ``b`` (with ``sigma_b`` its uncertainty), by the
    Asimov data set - ``RooStats::AsimovSignificance``."""
    if sigma_b == 0.0:
        return math.sqrt(2.0 * ((s + b) * math.log(1.0 + s / b) - s))
    sb2 = sigma_b * sigma_b
    first = (s + b) * math.log((s + b) * (b + sb2) / (b * b + (s + b) * sb2))
    second = b * b / sb2 * math.log(1.0 + sb2 * s / (b * (b + sb2)))
    return math.sqrt(2.0 * (first - second))


def SetParameters(desiredVals: Any, paramsToChange: Any) -> None:
    """``paramsToChange = *desiredVals``: the values of the one set given to the other."""
    paramsToChange.assign(desiredVals)


def RemoveConstantParameters(params: Any) -> None:
    """Take the constant variables out of ``params``, in place."""
    for one in [p for p in params if p.isConstant()]:
        params.remove(one)


def SetAllConstant(coll: Any, constant: bool = True) -> bool:
    """Make every variable of ``coll`` constant (or free); whether any changed."""
    changed = False
    for one in as_list(coll):
        if hasattr(one, "setConstant") and one.isConstant() != bool(constant):
            one.setConstant(bool(constant))
            changed = True
    return changed


def _binomial_p(main: float, auxiliary: float, tau: float) -> float:
    return regularized_beta(1.0 / (1.0 + tau), main, auxiliary + 1.0)


class NumberCountingUtils:
    """``RooStats::NumberCountingUtils``: the Z_Bi of a number-counting experiment."""

    @staticmethod
    def BinomialExpP(signalExp: float, backgroundExp: float, relativeBkgUncert: float) -> float:
        tau = 1.0 / backgroundExp / (relativeBkgUncert * relativeBkgUncert)
        return _binomial_p(signalExp + backgroundExp, backgroundExp * tau, tau)

    @staticmethod
    def BinomialWithTauExpP(signalExp: float, backgroundExp: float, tau: float) -> float:
        return _binomial_p(signalExp + backgroundExp, backgroundExp * tau, tau)

    @staticmethod
    def BinomialObsP(mainObs: float, backgroundObs: float, relativeBkgUncert: float) -> float:
        tau = 1.0 / backgroundObs / (relativeBkgUncert * relativeBkgUncert)
        return _binomial_p(mainObs, backgroundObs * tau, tau)

    @staticmethod
    def BinomialWithTauObsP(mainObs: float, auxiliaryObs: float, tau: float) -> float:
        return _binomial_p(mainObs, auxiliaryObs, tau)

    @staticmethod
    def BinomialExpZ(signalExp: float, backgroundExp: float, relativeBkgUncert: float) -> float:
        return PValueToSignificance(
            NumberCountingUtils.BinomialExpP(signalExp, backgroundExp, relativeBkgUncert)
        )

    @staticmethod
    def BinomialWithTauExpZ(signalExp: float, backgroundExp: float, tau: float) -> float:
        return PValueToSignificance(
            NumberCountingUtils.BinomialWithTauExpP(signalExp, backgroundExp, tau)
        )

    @staticmethod
    def BinomialObsZ(mainObs: float, backgroundObs: float, relativeBkgUncert: float) -> float:
        return PValueToSignificance(
            NumberCountingUtils.BinomialObsP(mainObs, backgroundObs, relativeBkgUncert)
        )

    @staticmethod
    def BinomialWithTauObsZ(mainObs: float, auxiliaryObs: float, tau: float) -> float:
        return PValueToSignificance(
            NumberCountingUtils.BinomialWithTauObsP(mainObs, auxiliaryObs, tau)
        )


def copy_of(items: Any) -> RooArgSet:
    """A set holding the same variables as ``items`` - not copies of them."""
    return RooArgSet(as_list(items))


# -- constraints ------------------------------------------------------------------


def _states(sim: Any) -> list[Any]:
    """A simultaneous density's densities, in its category's order of states."""
    index = sim.indexCat()
    return [sim.getPdf(label) for label in index.states() if sim.getPdf(label) is not None]


def FactorizePdf(observables: Any, pdf: Any, obsTerms: Any, constraints: Any) -> None:
    """The terms of ``pdf`` that depend on the observables, and those that do not - the
    constraints - through products, extended densities and every state of a simultaneous one."""
    if hasattr(observables, "GetObservables"):  # (ModelConfig, pdf, ...)
        observables = observables.GetObservables()
    if pdf.InheritsFrom("RooProdPdf"):
        for one in pdf.pdfList():
            FactorizePdf(observables, one, obsTerms, constraints)
    elif pdf.InheritsFrom("RooExtendPdf"):
        FactorizePdf(observables, pdf.servers()[0], obsTerms, constraints)
    elif pdf.InheritsFrom("RooSimultaneous"):
        for one in _states(pdf):
            FactorizePdf(observables, one, obsTerms, constraints)
    else:
        target = obsTerms if pdf.dependsOn(observables) else constraints
        if not any(one is pdf for one in target):
            target.add(pdf)


def MakeNuisancePdf(pdf: Any, observables: Any = None, name: str = "") -> Any:
    """The product of every constraint term of ``pdf`` - or of a ModelConfig's density."""
    from ..roofit.messages import ERROR, WARNING, log
    from ..roofit.pdfs.prodpdf import RooProdPdf

    if hasattr(pdf, "GetPdf"):  # (ModelConfig, name)
        model, name = pdf, str(observables)
        if model.GetPdf() is None or model.GetObservables() is None:
            log(None, ERROR, "InputArguments", "RooStatsUtils::MakeNuisancePdf - invalid input "
                "model: missing pdf and/or observables")  # fmt: skip
            return None
        pdf, observables = model.GetPdf(), model.GetObservables()
    terms, constraints = RooArgSet(), RooArgSet()
    FactorizePdf(observables, pdf, terms, constraints)
    if not len(constraints):
        log(None, WARNING, "Eval", "RooStatsUtils::MakeNuisancePdf - no constraints found on "
            "nuisance parameters in the input model")  # fmt: skip
        return None
    return RooProdPdf(name, "", list(constraints))


def StripConstraints(pdf: Any, observables: Any) -> Any:
    """A copy of ``pdf`` without its constraint terms - ``None`` if nothing else is left."""
    from ..roofit.pdfs.prodpdf import RooProdPdf

    if pdf.InheritsFrom("RooProdPdf"):
        kept = [k for k in (StripConstraints(one, observables) for one in pdf.pdfList()) if k]
        if not kept:
            return None
        if len(kept) == 1:
            return kept[0].clone(f"{kept[0].GetName()}_unconstrained")
        return RooProdPdf(f"{pdf.GetName()}_unconstrained", f"{pdf.GetTitle()} without "
                          "constraints", kept)  # fmt: skip
    if pdf.InheritsFrom("RooExtendPdf"):
        from ..roofit.pdfs.extend import RooExtendPdf

        inner, number = pdf.servers()[0], pdf.servers()[1]
        stripped = StripConstraints(inner, observables)
        if stripped is None:
            return None
        return RooExtendPdf(f"{pdf.GetName()}_unconstrained", f"{pdf.GetTitle()} without "
                            "constraints", stripped, number)  # fmt: skip
    if pdf.InheritsFrom("RooSimultaneous"):
        return _stripped_simultaneous(pdf, observables)
    if pdf.dependsOn(observables):
        return pdf.clone(f"{pdf.GetName()}_unconstrained")
    return None


def _stripped_simultaneous(sim: Any, observables: Any) -> Any:
    from ..roofit.pdfs.simultaneous import RooSimultaneous

    index = sim.indexCat()
    made = {}
    for label in index.states():
        one = sim.getPdf(label)
        stripped = StripConstraints(one, observables) if one is not None else None
        if stripped is None:
            return None
        made[label] = stripped
    return RooSimultaneous(f"{sim.GetName()}_unconstrained", f"{sim.GetTitle()} without "
                           "constraints", made, index)  # fmt: skip


def MakeUnconstrainedPdf(pdf: Any, observables: Any = None, name: Any = None) -> Any:
    """``pdf`` - or a ModelConfig's density - without its constraints, named ``name`` if given."""
    from ..roofit.messages import ERROR, log

    if hasattr(pdf, "GetPdf"):  # (ModelConfig, name)
        model, name = pdf, observables
        if model.GetPdf() is None or model.GetObservables() is None:
            log(None, ERROR, "InputArguments", "RooStatsUtils::MakeUnconstrainedPdf - invalid "
                "input model: missing pdf and/or observables")  # fmt: skip
            return None
        pdf, observables = model.GetPdf(), model.GetObservables()
    found = StripConstraints(pdf, observables)
    if found is None:
        log(None, ERROR, "InputArguments", "RooStats::MakeUnconstrainedPdf - invalid observable "
            "list passed (observables not found in original pdf) or invalid pdf passed (without "
            "observables)")  # fmt: skip
        return None
    if name:
        found.SetName(str(name))
    return found
