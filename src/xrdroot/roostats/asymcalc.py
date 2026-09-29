"""The ``AsymptoticCalculator`` itself: its set-up fits, and a test at the null snapshot.

Made, it fits the data unconditionally and makes the Asimov data set at the
alternate's parameter of interest - with the nuisance parameters at their
fit to the data, or, asked for nominal Asimov data, as they are - and fits
that too at the alternate. Each test then fits both again at the null's
value, and the formulae of :mod:`.asympformulae` turn the two ratios into
p-values.
"""

from __future__ import annotations

import math
from typing import Any

from ..roofit.collections import RooArgSet
from ..roofit.messages import ERROR, INFO, PROGRESS, WARNING, log
from ..roofit.printing import g
from . import asymptotic as asym
from .calculators import HypoTestCalculatorGeneric
from .utils import RemoveConstantParameters

__all__ = ["AsymptoticCalculator"]


class AsymptoticCalculator(HypoTestCalculatorGeneric):
    """Tests of the null against the alternate by the asymptotic formulae."""

    def __init__(self, data: Any, altModel: Any, nullModel: Any, nominalAsimov: bool = False
                 ) -> None:  # fmt: skip
        super().__init__(data, altModel, nullModel, None)
        self._one_sided = self._discovery = False
        self._nominal = bool(nominalAsimov)
        self._qtilde = -1
        self._nll_obs = self._nll_asimov = 0.0
        self._asimov: Any = None
        self._best_poi, self._best_params = RooArgSet(), RooArgSet()
        self._asimov_globs = RooArgSet()
        self._initialized = False
        if not self.Initialize():
            return
        mu = next(iter(nullModel.GetSnapshot()))
        if mu.getVal() == mu.getMin():
            self._discovery = True
            if asym.PRINT_LEVEL[0] > 0:
                log(None, INFO, "InputArguments", f"AsymptotiCalculator: Minimum of POI is "
                    f"{g(mu.getMin())} corresponds to null  snapshot   - default configuration is  "
                    "one-sided discovery formulae  ")  # fmt: skip

    # -- the switches -------------------------------------------------------------

    def SetOneSided(self, on: bool) -> None:
        self._one_sided = bool(on)

    def SetOneSidedDiscovery(self, on: bool) -> None:
        self._discovery = bool(on)

    def SetTwoSided(self) -> None:
        self._one_sided = self._discovery = False

    def SetQTilde(self, on: bool) -> None:
        self._qtilde = int(bool(on))

    def IsTwoSided(self) -> bool:
        return not (self._one_sided or self._discovery)

    def IsOneSidedDiscovery(self) -> bool:
        return self._discovery

    def GetNLL(self) -> float:
        return self._nll_obs

    def GetExpectedNLL(self) -> float:
        return self._nll_asimov

    def GetAsimovData(self) -> Any:
        return self._asimov

    def GetBestFitPoi(self) -> Any:
        return self._best_poi

    @staticmethod
    def SetPrintLevel(level: int) -> None:
        asym.PRINT_LEVEL[0] = int(level)

    @staticmethod
    def GetExpectedPValues(pnull: float, palt: float, nsigma: float, useCls: bool,
                           oneSided: bool = True) -> float:  # fmt: skip
        from .asympformulae import expected_p_values

        return expected_p_values(pnull, palt, nsigma, useCls, oneSided)

    @staticmethod
    def GenerateAsimovData(pdf: Any, observables: Any) -> Any:
        from .asimov import GenerateAsimovData

        return GenerateAsimovData(pdf, observables)

    @staticmethod
    def MakeAsimovData(*args: Any) -> Any:
        """``(data, model, poiValues, asimovGlobObs[, genPoiValues])`` - the nuisance parameters
        fitted to the data first - or ``(model, allParamValues, asimovGlobObs)``."""
        if hasattr(args[0], "GetPdf"):
            return asym.make_asimov_nominal(*args)
        return asym.make_asimov_data(*args)

    # -- set-up -------------------------------------------------------------------

    def _refusal(self) -> str:
        """Why the calculator cannot start: no density, data, parameter of interest, snapshot."""
        null = self._null
        if null.GetPdf() is None:
            return "AsymptoticCalculator::Initialize - ModelConfig has not a pdf defined"
        if self._data is None:
            return "AsymptoticCalculator::Initialize - data set has not been defined"
        poi = null.GetParametersOfInterest()
        if poi is None or not len(poi):
            return "AsymptoticCalculator::Initialize -  ModelConfig has not POI defined."
        if len(poi) > 1:
            log(None, WARNING, "InputArguments", "AsymptoticCalculator::Initialize - ModelConfig "
                "has more than one POI defined \n\tThe asymptotic calculator works for only one "
                "POI - consider as POI only the first parameter")  # fmt: skip
        snapshot = null.GetSnapshot()
        if snapshot is None or not len(snapshot):
            return ("AsymptoticCalculator::Initialize - Null model needs a snapshot. Set using "
                    "modelconfig->SetSnapshot(poi).")  # fmt: skip
        return ""

    def Initialize(self) -> bool:
        """The unconditional fit to the data, the Asimov data, and its fit at the alternate."""
        if asym.PRINT_LEVEL[0] >= 0:
            log(None, PROGRESS, "Eval", "AsymptoticCalculator::Initialize....")
        refusal = self._refusal()
        if refusal:
            log(None, ERROR, "InputArguments", refusal)
            return False
        params = self._null.GetPdf().getParameters(self._data)
        RemoveConstantParameters(params)
        nominal = params.snapshot() if self._nominal else RooArgSet()
        self._observed_fit(params)
        alt = self._alt.GetSnapshot()
        if alt is None or not len(alt):
            log(None, ERROR, "InputArguments", "Alt (Background)  model needs a snapshot. Set "
                "using modelconfig->SetSnapshot(poi).")  # fmt: skip
            return False
        log(None, PROGRESS, "Eval", "AsymptoticCalculator: Building Asimov data Set")
        with _data_bins(self._null, self._data):
            if not self._make_asimov(RooArgSet(list(alt)), nominal):
                return False
            self._asimov_fit(RooArgSet(list(alt)))
        self._initialized = True
        return True

    def _observed_fit(self, params: Any) -> None:
        if asym.PRINT_LEVEL[0] >= 0:
            log(None, PROGRESS, "Eval", "AsymptoticCalculator::Initialize - Find  best "
                "unconditional NLL on observed data")  # fmt: skip
        self._nll_obs = asym.evaluate_nll(self._null, self._data)
        self._best_poi = self._null.GetParametersOfInterest().snapshot()
        best = next(iter(self._best_poi))
        if asym.PRINT_LEVEL[0] >= 0:
            log(None, PROGRESS, "Eval", f"Best fitted POI value = {g(best.getVal())} +/- "
                f"{g(best.getError())}")  # fmt: skip
        self._best_params = params.snapshot()

    def _make_asimov(self, alt: Any, nominal: Any) -> bool:
        if not self._nominal:
            if asym.PRINT_LEVEL[0] >= 0:
                log(None, INFO, "InputArguments", "AsymptoticCalculator: Asimov data will be "
                    "generated using fitted nuisance parameter values")  # fmt: skip
            self._asimov = asym.make_asimov_data(self._data, self._null, alt, self._asimov_globs,
                                                 alt.snapshot())  # fmt: skip
        else:
            if asym.PRINT_LEVEL[0] >= 0:
                log(None, INFO, "InputArguments", "AsymptoticCalculator: Asimovdata set will be "
                    "generated using nominal (current) nuisance parameter values")  # fmt: skip
            nominal.assign(alt)
            self._asimov = asym.make_asimov_nominal(self._null, nominal, self._asimov_globs)
        if self._asimov is None:
            log(None, ERROR, "InputArguments", "AsymptoticCalculator: Error : Asimov data set "
                "could not be generated ")  # fmt: skip
            return False
        return True

    def _asimov_fit(self, alt: Any) -> None:
        """The fit to the Asimov data at the alternate, its global observables at theirs."""
        mu = next(iter(alt))
        with _globals_at(self._null, self._asimov_globs):
            if asym.PRINT_LEVEL[0] >= 0:
                log(None, PROGRESS, "Eval", "AsymptoticCalculator::Initialize Find  best "
                    f"conditional NLL on ASIMOV data set for given alt POI ( {mu.GetName()} ) = "
                    f"{g(mu.getVal())}")  # fmt: skip
            self._nll_asimov = asym.evaluate_nll(self._null, self._asimov, alt)


    # -- a test -------------------------------------------------------------------

    def GetHypoTest(self) -> Any:
        """The test at the null snapshot: the ratio on the data and on the Asimov data."""
        from .hypotest import HypoTestResult

        if not self._initialized and not self.Initialize():
            log(None, ERROR, "InputArguments", "AsymptoticCalculator::GetHypoTest - Error "
                "initializing Asymptotic calculator - return nullptr result ")  # fmt: skip
            return None
        if self._asimov is None:
            log(None, ERROR, "InputArguments", "AsymptoticCalculator::GetHypoTest - Asimov data "
                "set has not been generated - return nullptr result ")  # fmt: skip
            return None
        test = RooArgSet(list(self._null.GetSnapshot()))
        if len(test) > 1:
            log(None, WARNING, "InputArguments", "AsymptoticCalculator::GetHypoTest: snapshot has "
                "more than one POI - assume as POI first parameter ")  # fmt: skip
        self._null.GetPdf().getParameters(self._data).assign(self._best_params)
        mu = test.find(next(iter(self._best_poi)).GetName())
        qmu = self._observed_q(test, mu)
        qmu_a = None if qmu is None else self._asimov_q(test, mu)
        if qmu_a is None:
            return HypoTestResult()
        return self._result(qmu, qmu_a, mu)


    def _observed_q(self, test: Any, mu: Any) -> Any:
        """``qmu`` on the data - refitting unconditionally if it came out negative - or ``None``
        for a dummy result."""
        verbose = asym.PRINT_LEVEL[0] > 0
        if verbose:
            log(None, INFO, "Eval", "\nAsymptoticCalculator::GetHypoTest: - perform  an "
                f"hypothesis test for  POI ( {mu.GetName()} ) = {g(mu.getVal())}")  # fmt: skip
            log(None, PROGRESS, "Eval", "AsymptoticCalculator::GetHypoTest -  Find  best "
                "conditional NLL on OBSERVED data set ..... ")  # fmt: skip
        cond = asym.evaluate_nll(self._null, self._data, test)
        qmu = 2.0 * (cond - self._nll_obs)
        if verbose:
            log(None, PROGRESS, "Eval", f"\t OBSERVED DATA :  qmu   = {g(qmu)} condNLL = "
                f"{g(cond)} uncond {g(self._nll_obs)}")  # fmt: skip
        if qmu < -_tolerance() or math.isnan(self._nll_obs):
            qmu = self._refit_observed(cond, qmu)
        return self._checked(qmu, mu, "qmu")

    def _refit_observed(self, cond: float, qmu: float) -> float:
        log(None, WARNING, "Minimization", "AsymptoticCalculator:  Found a negative value of the "
            "qmu - retry to do the unconditional fit " if qmu < 0 else "AsymptoticCalculator:  "
            "unconditional fit failed before - retry to do it now ")  # fmt: skip
        nll = asym.evaluate_nll(self._null, self._data)
        if nll < self._nll_obs or (math.isnan(self._nll_obs) and not math.isnan(nll)):
            best = next(iter(self._best_poi))
            log(None, WARNING, "Minimization", "AsymptoticCalculator:  Found a better "
                f"unconditional minimum  old NLL = {g(self._nll_obs)} old muHat "
                f"{g(best.getVal())}")  # fmt: skip
            self._nll_obs = nll
            self._best_poi = self._null.GetParametersOfInterest().snapshot()
            log(None, WARNING, "Minimization", "AsymptoticCalculator:  New minimum  found for"
                f"                           NLL = {g(nll)}    muHat  "
                f"{g(next(iter(self._best_poi)).getVal())}")  # fmt: skip
            qmu = 2.0 * (cond - nll)
            if asym.PRINT_LEVEL[0] > 0:
                log(None, PROGRESS, "Eval", "After unconditional refit,  new qmu value is "
                    f"{g(qmu)}")  # fmt: skip
        return qmu

    def _asimov_q(self, test: Any, mu: Any) -> Any:
        """``qmu_A`` on the Asimov data, its global observables at theirs - refitting as for the
        data - or ``None``. RooStats tests it for NaN by testing ``qmu`` again: a NaN passes."""
        verbose = asym.PRINT_LEVEL[0] > 0
        with _globals_at(self._null, self._asimov_globs):
            if verbose:
                log(None, PROGRESS, "Eval", "AsymptoticCalculator::GetHypoTest -- Find  best "
                    "conditional NLL on ASIMOV data set .... ")  # fmt: skip
            cond = asym.evaluate_nll(self._null, self._asimov, test)
            qmu_a = 2.0 * (cond - self._nll_asimov)
            if verbose:
                log(None, PROGRESS, "Eval", f"\t ASIMOV data qmu_A = {g(qmu_a)} condNLL = "
                    f"{g(cond)} uncond {g(self._nll_asimov)}")  # fmt: skip
            if qmu_a < -_tolerance() or math.isnan(self._nll_asimov):
                qmu_a = self._refit_asimov(cond, qmu_a)
            if qmu_a < -_tolerance():
                return self._checked(qmu_a, mu, "qmu_A")
        return qmu_a

    def _refit_asimov(self, cond: float, qmu_a: float) -> float:
        log(None, WARNING, "Minimization", "AsymptoticCalculator:  Found a negative value of the "
            "qmu Asimov- retry to do the unconditional fit " if qmu_a < 0 else
            "AsymptoticCalculator:  Fit failed for  unconditional the qmu Asimov- retry  "
            "unconditional fit ")  # fmt: skip
        nll = asym.evaluate_nll(self._null, self._asimov)
        if nll < self._nll_asimov or (math.isnan(self._nll_asimov) and not math.isnan(nll)):
            log(None, WARNING, "Minimization", "AsymptoticCalculator:  Found a better "
                f"unconditional minimum for Asimov data set old NLL = {g(self._nll_asimov)}")
            self._nll_asimov = nll
            log(None, WARNING, "Minimization", "AsymptoticCalculator:  New minimum  found for"
                f"                           NLL = {g(nll)}")  # fmt: skip
            qmu_a = 2.0 * (cond - nll)
            if asym.PRINT_LEVEL[0] > 0:
                log(None, PROGRESS, "Eval", "After unconditional Asimov refit,  new qmu_A value "
                    f"is {g(qmu_a)}")  # fmt: skip
        return qmu_a


    def _use_qtilde(self, mu: Any) -> bool:
        """``qtilde`` - never for discovery - when the parameter's minimum is the alternate's
        value, decided once and said."""
        if self._discovery:
            return False
        if self._qtilde == -1:
            alt = next(iter(self._alt.GetSnapshot()))
            self._qtilde = int(mu.getMin() == alt.getVal())
            log(None, INFO, "InputArguments", f"Minimum of POI is {g(mu.getMin())} corresponds to "
                "alt  snapshot   - using qtilde asymptotic formulae  " if self._qtilde else
                f"Minimum of POI is {g(mu.getMin())} is different to alt snapshot "
                f"{g(alt.getVal())} - using standard q asymptotic formulae  ")  # fmt: skip
        return bool(self._qtilde)

    def _result(self, qmu: float, qmu_a: float, mu: Any) -> Any:
        from .asympformulae import p_values
        from .hypotest import HypoTestResult

        qtilde = self._use_qtilde(mu)
        best = next(iter(self._best_poi)).getVal()
        if self._one_sided and best > mu.getVal():
            log(None, INFO, "Eval", "Using one-sided qmu - setting qmu to zero  muHat = "
                f"{g(best)} muTest = {g(mu.getVal())}")  # fmt: skip
            qmu = 0.0
        if self._discovery and best < mu.getVal():
            log(None, INFO, "Eval", "Using one-sided discovery qmu - setting qmu to zero  muHat = "
                f"{g(best)} muTest = {g(mu.getVal())}")  # fmt: skip
            qmu = 0.0
        tol = _tolerance()
        qmu = 0.0 if -tol < qmu < 0 else qmu
        qmu_a = 0.0 if -tol < qmu_a < 0 else qmu_a
        pnull, palt = p_values(qmu, qmu_a, self._one_sided, self._discovery, qtilde, tol)
        result = HypoTestResult("HypoTestAsymptotic_result", pnull, palt)
        if asym.PRINT_LEVEL[0] > 0:
            root_a = math.sqrt(qmu_a) if qmu_a > 0 else 0.0
            sigma = mu.getVal() / root_a if root_a else math.copysign(math.inf, mu.getVal())
            sigma = sigma if mu.getVal() or root_a else math.nan
            log(None, PROGRESS, "Eval", f"poi = {g(mu.getVal())} qmu = {g(qmu)} qmu_A = "
                f"{g(qmu_a)} sigma = {g(sigma)}  CLsplusb = {g(pnull)} CLb = {g(palt)} CLs = "
                f"{g(result.CLs())}")  # fmt: skip
        return result


    @staticmethod
    def _checked(q: float, mu: Any, what: str) -> Any:
        """``q``, or ``None`` - said - if it is still negative, or not a number."""
        if q < -_tolerance():
            log(None, ERROR, "Minimization", f"AsymptoticCalculator:  {what} is still < 0  for "
                f"mu = {g(mu.getVal())} return a dummy result ")  # fmt: skip
            return None
        if math.isnan(q):
            log(None, ERROR, "Minimization", "AsymptoticCalculator:  failure in fitting for qmu or "
                f"qmuA {g(mu.getVal())} return a dummy result ")  # fmt: skip
            return None
        return q


class _Restored:
    """A context that runs ``undo`` on leaving."""

    def __init__(self, undo: Any) -> None:
        self.undo = undo

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        self.undo()


def _tolerance() -> float:
    """How negative a ratio may come out of the fits and still count as zero."""
    from ..fit.defaults import default

    return 2e-3 * max(1.0, float(default("Tolerance")))


def _data_bins(model: Any, data: Any) -> _Restored:
    """A binned dataset's bins, for the one observable, while the Asimov data is made."""
    observables = model.GetObservables()
    if observables is None or len(observables) != 1 or not data.InheritsFrom("RooDataHist"):
        return _Restored(lambda: None)
    xobs = next(iter(observables))
    before = xobs.getBins()
    if data.numEntries() == before:
        return _Restored(lambda: None)
    log(None, WARNING, "InputArguments", f"AsymptoticCalculator: number of bins in "
        f"{xobs.GetName()} are different than data bins  set the same data bins "
        f"{data.numEntries()} in range  [ {g(xobs.getMin())} , {g(xobs.getMax())} ]")  # fmt: skip
    xobs.setBins(data.numEntries())
    return _Restored(lambda: xobs.setBins(before))


def _globals_at(model: Any, values: Any) -> _Restored:
    """The model's global observables at ``values`` - the Asimov ones - and back afterwards."""
    found = model.GetGlobalObservables()
    if found is None:
        return _Restored(lambda: None)
    gobs = RooArgSet(list(found))
    saved = gobs.snapshot()
    gobs.assign(values)
    return _Restored(lambda: gobs.assign(saved))
