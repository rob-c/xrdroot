"""RooStats' test statistics: numbers a toy or the data is summarised by, for a parameter point.

The profile likelihood ratio ``-log(L(mu, hat-hat theta) / L(hat mu, hat theta))`` -
two fits, the second with the parameters of interest held at the point -
is ``ProfileLikelihoodTestStat``, one- or two-sided, signed or not; the
likelihood ratio of two fixed points, the ratio of two profiles, the number
of events and a maximum-likelihood estimate are the others. Each keeps its
likelihood and gives it each toy's data in turn, as RooStats reuses it, and
fits with RooFit's messages held back below fatal ones.
"""

from __future__ import annotations

from typing import Any

from ..fit.defaults import default, minimizer_algo
from ..roofit.cmdargs import RooCmdArg
from ..roofit.collections import RooArgSet, as_list
from ..roofit.fitting.minimizer import RooMinimizer
from ..roofit.messages import FATAL, WARNING, log
from .modelconfig import quieted
from .utils import RemoveConstantParameters

__all__ = ["ProfileLikelihoodTestStat", "TestStatistic", "fit_as_args", "fit_nll"]

#: ``fgAlwaysReuseNll``: every statistic gives its likelihood each toy's data, not a new one.
_ALWAYS_REUSE = [True]


class TestStatistic:
    """``TestStatistic``: what the samplers evaluate on the data and on each toy."""

    def Evaluate(self, data: Any, nullPOI: Any) -> float:
        raise NotImplementedError

    def GetVarName(self) -> str:
        return ""

    def PValueIsRightTail(self) -> bool:
        return True

    def GetDetailedOutput(self) -> Any:
        return None

    def SetConditionalObservables(self, items: Any) -> None:
        """What the statistic's likelihood leaves unnormalised - none, unless it says so."""

    def SetGlobalObservables(self, items: Any) -> None:
        """What the statistic's likelihood takes global observables to be."""


def fit_nll(nll: Any, strategy: int, tolerance: float, print_level: int, kind: str = "") -> Any:
    """``GetMinNLL``: MIGRAD's ``Minimize`` - Migrad, then Simplex and Migrad if it fails -
    tried again as RooStats tries it while it fails, after a scan each time; the fit's
    result."""
    minim = RooMinimizer(nll)
    minim.setStrategy(strategy)
    minim.setPrintLevel(-1 if print_level == 0 else print_level - 2)
    minim.setEps(tolerance)
    minim.optimizeConst(2)
    algorithm = minimizer_algo()
    algorithm = "Minimize" if algorithm == "Migrad" else algorithm
    tries = 1
    while True:
        status = minim.minimize(kind, algorithm)
        if status % 1000 == 0 or tries == 4:
            break
        minim.minimize(kind, "Scan")
        if tries == 2 and strategy == 0:
            minim.setStrategy(1)
        elif tries == 2:
            tries += 1
        if tries == 3:
            kind, algorithm = "Minuit", "migradimproved"
        tries += 1
    return minim.save()


def fit_as_args(fit: Any, prefix: str, pulls: bool = False) -> list[Any]:
    """``DetailedOutputAggregator::GetAsArgSet``: a fit's parameters - with their pulls from
    their initial values, if asked - then its minimum, status, covariance quality and number of
    invalid evaluations, each named with ``prefix``."""
    from ..roofit.variables import RooRealVar

    found = []
    initial = fit.floatParsInit()
    for par in fit.floatParsFinal():
        name = f"{prefix}{par.GetName()}"
        made = RooRealVar(name, f"{prefix}{par.GetTitle()}", par.getVal())
        made.setError(par.getError())
        found.append(made)
        if pulls:
            error, truth = par.getError(), initial.find(par.GetName()).getVal()
            pull = (par.getVal() - truth) / error if error > 0 else 0.0
            found.append(RooRealVar(f"{name}_pull", f"{name}_pull", pull))
    for what, value in (("minNLL", fit.minNll()), ("fitStatus", fit.status()),
                        ("covQual", fit.covQual()), ("numInvalidNLLEval", fit.numInvalidNLL())):
        found.append(RooRealVar(f"{prefix}{what}", f"{prefix}{what}", value))
    return found


#: ``ProfileLikelihoodTestStat::LimitType``.
TWO_SIDED, ONE_SIDED, ONE_SIDED_DISCOVERY = 0, 1, 2


class ProfileLikelihoodTestStat(TestStatistic):
    """The profile likelihood ratio at a point of the parameters of interest."""

    def __init__(self, pdf: Any = None) -> None:
        self._pdf = pdf
        self._nll: Any = None
        self._limit = TWO_SIDED
        self._signed = False
        self._reuse = False
        self._minimizer = ""
        self._strategy = int(default("Strategy"))
        self._tolerance = max(1.0, float(default("Tolerance")))
        self._print_level = int(default("PrintLevel"))
        self._offset = "none"
        self._conditional = RooArgSet()
        self._global = RooArgSet()
        self._var_name = "Profile Likelihood Ratio"
        self._detailed = self._pulls = False
        self._output: Any = None

    def SetOneSided(self, flag: bool = True) -> None:
        self._limit = ONE_SIDED if flag else TWO_SIDED

    def SetOneSidedDiscovery(self, flag: bool = True) -> None:
        self._limit = ONE_SIDED_DISCOVERY if flag else TWO_SIDED

    def SetSigned(self, flag: bool = True) -> None:
        self._signed = bool(flag)

    def IsTwoSided(self) -> bool:
        return self._limit == TWO_SIDED

    def IsOneSidedDiscovery(self) -> bool:
        return self._limit == ONE_SIDED_DISCOVERY

    @staticmethod
    def SetAlwaysReuseNLL(flag: bool) -> None:
        _ALWAYS_REUSE[0] = bool(flag)

    def SetReuseNLL(self, flag: bool) -> None:
        self._reuse = bool(flag)

    def SetLOffset(self, flag: Any = True) -> None:
        self._offset = str(flag) if isinstance(flag, str) else ("initial" if flag else "none")

    def SetMinimizer(self, name: str) -> None:
        self._minimizer = str(name)

    def SetStrategy(self, strategy: int) -> None:
        self._strategy = int(strategy)

    def SetTolerance(self, tolerance: float) -> None:
        self._tolerance = float(tolerance)

    def SetPrintLevel(self, level: int) -> None:
        self._print_level = int(level)

    def SetConditionalObservables(self, items: Any) -> None:
        self._conditional = RooArgSet(as_list(items))

    def SetGlobalObservables(self, items: Any) -> None:
        self._global = RooArgSet(as_list(items))

    def SetVarName(self, name: str) -> None:
        self._var_name = str(name)

    def GetVarName(self) -> str:
        return self._var_name

    def GetPdf(self) -> Any:
        return self._pdf

    def EnableDetailedOutput(self, flag: bool = True, withErrorsAndPulls: bool = False) -> None:
        self._detailed, self._pulls = bool(flag), bool(withErrorsAndPulls)
        self._output = None

    def GetDetailedOutput(self) -> Any:
        """Each fit of the last evaluation, as ``fitUncond_`` and ``fitCond_`` variables."""
        return self._output

    def Evaluate(self, data: Any, paramsOfInterest: Any) -> float:
        return self.EvaluateProfileLikelihood(0, data, paramsOfInterest)

    def _likelihood(self, data: Any) -> Any:
        """The likelihood of the data: made the first time, given the data after, as RooStats
        reuses it."""
        if self._nll is not None and (self._reuse or _ALWAYS_REUSE[0]):
            self._nll.setData(data, False)
            return self._nll
        params = RooArgSet(self._pdf.getParameters(data))
        RemoveConstantParameters(params)
        self._nll = self._pdf.createNLL(
            data,
            RooCmdArg("CloneData", False),
            RooCmdArg("Constrain", params),
            RooCmdArg("GlobalObservables", self._global),
            RooCmdArg("ConditionalObservables", self._conditional),
            RooCmdArg("Offset", self._offset == "initial"),
        )
        return self._nll

    def _minimum(self, nll: Any, prefix: str) -> tuple[float, int]:
        found = fit_nll(nll, self._strategy, self._tolerance, self._print_level, self._minimizer)
        if self._detailed:
            self._output.add(fit_as_args(found, prefix, self._pulls))
        return float(found.minNll()), int(found.status())

    def EvaluateProfileLikelihood(self, kind: int, data: Any, paramsOfInterest: Any) -> float:
        """``kind`` 0 the ratio, 1 the unconditional minimum, 2 the conditional one."""
        poi = as_list(paramsOfInterest)
        first = poi[0] if poi else None
        initial = float(first.getVal()) if first is not None else 0.0
        if self._detailed:
            self._output = RooArgSet()
        with quieted(FATAL) if self._print_level < 3 else _nothing():
            nll = self._likelihood(data)
            attached = RooArgSet(list(nll.getParameters()))
            attached.assign(poi)
            before, point = attached.snapshot(), RooArgSet(poi).snapshot()
            uncond, mu_hat, status_d = 0.0, 0.0, 0
            if kind != 2:
                uncond, status_d = self._fitted(nll, attached, "fitUncond_")
                mu_hat = attached.getRealValue(first.GetName()) if first is not None else 0.0
            cond, status_n = uncond, 0
            skip = not self._signed and kind == 0 and (
                (self._limit == ONE_SIDED and mu_hat >= initial)
                or (self._limit == ONE_SIDED_DISCOVERY and mu_hat <= initial))  # fmt: skip
            if kind != 1 and not skip:
                attached.assign(point)
                for par in poi:
                    found = attached.find(par.GetName())
                    if found is not None:
                        found.setConstant(True)
                cond, status_n = self._fitted(nll, attached, "fitCond_")
            pll = self._ratio(kind, uncond, cond, mu_hat, initial)
            attached.assign(before)
        return -1.0 if status_n or status_d else pll

    def _fitted(self, nll: Any, attached: Any, prefix: str) -> tuple[float, int]:
        """The minimum over the free parameters - the value itself, if there are none."""
        if not any(not p.isConstant() for p in attached):
            return float(nll.getVal()), 0
        return self._minimum(nll, prefix)

    def _ratio(self, kind: int, uncond: float, cond: float, mu_hat: float, initial: float) -> float:
        if kind == 1:
            return uncond
        if kind == 2:
            return cond
        pll = cond - uncond
        if self._signed:
            if pll < 0.0:
                if self._print_level > 0:
                    log(None, WARNING, "Eval", "pll is negative - setting it to zero ")
                pll = 0.0
            ahead = mu_hat < initial if self._limit == ONE_SIDED_DISCOVERY else mu_hat > initial
            pll = -pll if ahead else pll
        return pll


class _nothing:
    """A block with nothing held back."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None
