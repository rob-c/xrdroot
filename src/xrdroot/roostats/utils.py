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
