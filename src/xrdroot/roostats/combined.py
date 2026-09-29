"""``CombinedCalculator`` and ``ProfileLikelihoodCalculator``: an interval or a test from the
profile likelihood ratio, by Wilks' theorem.

The calculator fits the model to the data - RooStats' ``DoGlobalFit``, with
every free parameter constrained and the model's global observables - and
hands back a :class:`~.likelihoodinterval.LikelihoodInterval` over the
profile of that likelihood, or a test of the null parameters: half the
chi-square tail of twice the difference between the conditional and the
global minimum.
"""

from __future__ import annotations

from typing import Any

from ..fit.defaults import default
from ..roofit.cmdargs import RooCmdArg
from ..roofit.collections import RooArgSet, as_list
from ..roofit.fitting.minimizer import RooMinimizer
from ..roofit.messages import INFO, PROGRESS, WARNING, log
from .utils import RemoveConstantParameters

__all__ = ["CombinedCalculator", "ProfileLikelihoodCalculator", "minimize_nll"]


class CombinedCalculator:
    """What an interval calculator and a hypothesis test calculator share: data, model, sets."""

    def __init__(self, data: Any = None, model: Any = None, *args: Any) -> None:
        self._data = data
        self._pdf: Any = None
        self._size = 0.05
        self._sets: dict[str, RooArgSet] = {
            k: RooArgSet() for k in ("poi", "null", "alt", "nuis", "cond", "glob")
        }
        if model is not None and hasattr(model, "GetPdf"):  # (data, ModelConfig, [size])
            self.SetModel(model)
            if args:
                self.SetTestSize(float(args[0]))
        elif model is not None:  # (data, pdf, poi, [size], [null], [alt], [nuis])
            self._pdf = model
            self.SetParameters(args[0])
            self.SetTestSize(float(args[1]) if len(args) > 1 else 0.05)
            for key, given in zip(("null", "alt", "nuis"), args[2:]):
                if given is not None:
                    self._sets[key] = RooArgSet(as_list(given))

    def SetTestSize(self, size: float) -> None:
        self._size = float(size)

    def SetConfidenceLevel(self, cl: float) -> None:
        self._size = 1.0 - float(cl)

    def Size(self) -> float:
        return self._size

    def ConfidenceLevel(self) -> float:
        return 1.0 - self._size

    def SetData(self, data: Any) -> None:
        self._data = data

    def SetModel(self, model: Any) -> None:
        self._pdf = model.GetPdf()
        for key, found in (
            ("poi", model.GetParametersOfInterest()),
            ("null", model.GetSnapshot()),
            ("nuis", model.GetNuisanceParameters()),
            ("cond", model.GetConditionalObservables()),
            ("glob", model.GetGlobalObservables()),
        ):
            if found is not None:
                self._sets[key] = RooArgSet(as_list(found))

    def SetPdf(self, pdf: Any) -> None:
        self._pdf = pdf

    def SetParameters(self, items: Any) -> None:
        self._sets["poi"] = RooArgSet(as_list(items))

    def SetNullParameters(self, items: Any) -> None:
        self._sets["null"] = RooArgSet(as_list(items))

    def SetAlternateParameters(self, items: Any) -> None:
        self._sets["alt"] = RooArgSet(as_list(items))

    def SetNuisanceParameters(self, items: Any) -> None:
        self._sets["nuis"] = RooArgSet(as_list(items))

    def SetConditionalObservables(self, items: Any) -> None:
        self._sets["cond"] = RooArgSet(as_list(items))

    def SetGlobalObservables(self, items: Any) -> None:
        self._sets["glob"] = RooArgSet(as_list(items))

    def GetPdf(self) -> Any:
        return self._pdf

    def GetData(self) -> Any:
        return self._data


def minimize_nll(nll: Any, owner: str = "ProfileLikelihoodCalcultor") -> Any:
    """``DoMinimizeNLL``: MIGRAD with the default strategy and tolerance, tried again - after a
    scan, then with strategy 1, then improved - while it fails; the fit as a result."""
    kind, algorithm = "", str(default("Algorithm"))
    strategy = int(default("Strategy"))
    minim = RooMinimizer(nll)
    minim.setStrategy(strategy)
    minim.setEps(float(default("Tolerance")))
    minim.setPrintLevel(int(default("PrintLevel")) - 1)
    minim.optimizeConst(2)
    log(None, PROGRESS, "Minimization", f"{owner}::DoMinimizeNLL - using {minim.minimizer_type} "
        f"/ {algorithm} with strategy {strategy}")  # fmt: skip
    tries = 1
    while tries <= 4:
        status = minim.minimize(kind, algorithm)
        if status % 1000 == 0:
            break
        if tries < 4:
            log(None, WARNING, "Minimization", "    ----> Doing a re-scan first")
            if tries == 2 and strategy == 0:
                log(None, WARNING, "Minimization", "    ----> trying with strategy = 1")
                minim.setStrategy(1)
            elif tries == 2:
                tries += 1  # strategy 1 already: the next try is the improved one
            if tries == 3:
                log(None, WARNING, "Minimization", "    ----> trying with improve")
                kind, algorithm = "Minuit", "migradimproved"
        tries += 1
    return minim.save()


class ProfileLikelihoodCalculator(CombinedCalculator):
    """Intervals and tests from the profile likelihood ratio, as Wilks' theorem calibrates it."""

    def __init__(self, data: Any = None, model: Any = None, *args: Any) -> None:
        super().__init__(data, model, *args)
        self._fit: Any = None
        self._global_fit_done = False

    def GetFitResult(self) -> Any:
        return self._fit

    def _constrained(self) -> RooArgSet:
        found = RooArgSet(self._pdf.getParameters(self._data))
        RemoveConstantParameters(found)
        return found

    def DoGlobalFit(self) -> Any:
        """The likelihood, and the fit of it the first time: its result said as an INFO line."""
        self._fit = None if not self._global_fit_done else self._fit
        nll = self._pdf.createNLL(
            self._data,
            RooCmdArg("CloneData", True),
            RooCmdArg("Constrain", self._constrained()),
            RooCmdArg("ConditionalObservables", self._sets["cond"]),
            RooCmdArg("GlobalObservables", self._sets["glob"]),
            RooCmdArg("Offset", False),
        )
        if self._fit is not None and self._global_fit_done:
            return nll
        log(None, PROGRESS, "Minimization", "ProfileLikelihoodCalcultor::DoGLobalFit - find MLE ")
        self._fit = minimize_nll(nll)
        _said(self._fit)
        if self._fit.status() != 0:
            log(None, WARNING, "Minimization", "ProfileLikelihoodCalcultor::DoGlobalFit -  Global "
                f"fit failed - status = {self._fit.status()}")  # fmt: skip
        else:
            self._global_fit_done = True
        return nll

    def GetInterval(self) -> Any:
        """A :class:`~.likelihoodinterval.LikelihoodInterval` over the profile, at the best fit."""
        from .likelihoodinterval import LikelihoodInterval

        if self._data is None or self._pdf is None or not len(self._sets["poi"]):
            return None
        nll = self.DoGlobalFit()
        profile = nll.createProfile(self._sets["poi"])
        fitted = RooArgSet(list(self._fit.floatParsFinal()))
        for par in self._sets["poi"]:
            found = fitted.find(par.GetName())
            if found is not None:
                par.setVal(found.getVal())
                par.setError(found.getError())
        best = RooArgSet(
            [(fitted.find(p.GetName()) or p).clone(p.GetName()) for p in self._sets["poi"]]
        )
        interval = LikelihoodInterval("LikelihoodInterval_", profile, self._sets["poi"], best)
        interval.SetConfidenceLevel(1.0 - self._size)
        return interval


def _said(fit: Any) -> None:
    """``fit->printStream(oocoutI(nullptr, Minimization), ...)``: the result as an INFO line."""
    text = fit.printStream(fit.defaultPrintContents(None), fit.defaultPrintStyle(None))
    log(None, INFO, "Minimization", text)
