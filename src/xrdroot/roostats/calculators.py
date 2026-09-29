"""The toy-based hypothesis test calculators: ``FrequentistCalculator`` and ``HybridCalculator``.

Both evaluate the test statistic on the data at the null snapshot, then
sample its distribution from toys of the null model and of the alternate
- ``HypoTestCalculatorGeneric``'s outline, which each fills in: the
frequentist one fixes the nuisance parameters at their conditional best
fit to the data first, the hybrid one draws them from a prior for each toy.
The p-values are the toys' tails beyond the data's value.
"""

from __future__ import annotations

from typing import Any

from ..roofit.collections import RooArgSet, as_list
from ..roofit.messages import ERROR, INFO, PROGRESS, log
from ..roofit.printing import g
from .hypotest import HypoTestResult
from .sampling import SamplingDistribution
from .toymc import ToyMCSampler

__all__ = ["FrequentistCalculator", "HybridCalculator", "HypoTestCalculatorGeneric"]


class HypoTestCalculatorGeneric:
    """The test statistic on the data, then its distribution under each hypothesis, from toys."""

    def __init__(self, data: Any, altModel: Any, nullModel: Any, sampler: Any = None) -> None:
        self._data, self._alt, self._null = data, altModel, nullModel
        self._sampler = sampler
        self._alt_seed = 0
        if sampler is None:
            from .moretests import RatioOfProfiledLikelihoodsTestStat

            statistic = RatioOfProfiledLikelihoodsTestStat(nullModel.GetPdf(), altModel.GetPdf(),
                                                           altModel.GetSnapshot())  # fmt: skip
            toys = ToyMCSampler(statistic, 1000)
            binned = data.InheritsFrom("RooDataHist")
            if binned:
                toys.SetProtoData(data)
            toys.SetGenerateBinned(binned)
            self._sampler = toys

    def SetNullModel(self, model: Any) -> None:
        self._null = model

    def SetAlternateModel(self, model: Any) -> None:
        self._alt = model

    def SetData(self, data: Any) -> None:
        self._data = data

    def GetData(self) -> Any:
        return self._data

    def GetNullModel(self) -> Any:
        return self._null

    def GetAlternateModel(self) -> Any:
        return self._alt

    def GetTestStatSampler(self) -> Any:
        return self._sampler

    def GetFitInfo(self) -> Any:
        return None

    def UseSameAltToys(self) -> None:
        from ..roofit.rng import generator

        self._alt_seed = generator().Integer(4294967295 - 1) + 1

    # -- hooks, which the calculators fill in -------------------------------------

    def _check(self) -> int:
        return 0

    def _pre(self) -> None:
        """``PreHook``."""

    def _pre_null(self, point: Any, observed: float) -> int:
        return 0

    def _pre_alt(self, point: Any, observed: float) -> int:
        return 0

    def _setup(self, model: Any) -> None:
        """``SetupSampler``: the null's observables and parameters, then ``model``'s density."""
        self._null.LoadSnapshot()
        self._sampler.SetObservables(self._null.GetObservables())
        self._sampler.SetParametersForTestStat(self._null.GetParametersOfInterest())
        model.LoadSnapshot()
        self._sampler.SetSamplingDistName(model.GetName())
        self._sampler.SetPdf(model.GetPdf())
        self._sampler.SetNuisanceParameters(model.GetNuisanceParameters())

    # -- the test -----------------------------------------------------------------

    def GetHypoTest(self) -> Any:
        """The data's test statistic, the null's and the alternate's toys, and the p-values."""
        self._pre()
        self._null.GuessObsAndNuisance(self._data.get())
        self._alt.GuessObsAndNuisance(self._data.get())
        null_snapshot = self._null.GetSnapshot()
        if null_snapshot is None:
            log(None, ERROR, "Generation", "Null model needs a snapshot. Set using "
                "modelconfig->SetSnapshot(poi).")  # fmt: skip
            return None
        if self._check() != 0:
            log(None, ERROR, "Generation", "There was an error in CheckHook(). Stop.")
            return None
        if self._sampler is None or self._sampler.GetTestStatistic() is None:
            log(None, ERROR, "InputArguments", "Test Statistic Sampler or Test Statistics not "
                "defined. Stop.")  # fmt: skip
            return None
        both = RooArgSet(list(self._null.GetPdf().getParameters(self._data)))
        both.add(list(self._alt.GetPdf().getParameters(self._data)), True)
        saved = both.snapshot()
        statistics = self._sampler.EvaluateAllTestStatistics(self._data, RooArgSet(
            list(null_snapshot)))  # fmt: skip
        observed = float(statistics[0].getVal())
        log(None, PROGRESS, "Generation", f"Test Statistic on data: {g(observed)}")
        both.assign(saved)
        null = self._distribution(self._null, self._pre_null, observed)
        both.assign(saved)
        alt = self._alt_distribution(observed)
        result = HypoTestResult("HypoTestCalculator_result")
        result.SetPValueIsRightTail(self._sampler.GetTestStatistic().PValueIsRightTail())
        result.SetTestStatisticData(observed)
        result.SetAltDistribution(alt)
        result.SetNullDistribution(null)
        if len(statistics) > 1:
            result.SetAllTestStatisticsData(statistics)
        both.assign(saved)
        return result

    def _events_if_needed(self, model: Any) -> None:
        """A toy of the data's size where the model cannot say how many events it expects."""
        pdf = model.GetPdf()
        if self._sampler.nEventsPerToy() == 0 and (
            not pdf.canBeExtended() or pdf.expectedEvents(model.GetObservables()) <= 0
        ):
            self._sampler.SetNEventsPerToy(self._data.sumEntries())

    def _distribution(self, model: Any, hook: Any, observed: float) -> Any:
        """One hypothesis' sampling distribution: its hook, then its toys."""
        self._setup(model)
        point = RooArgSet(list(as_list(model.GetParametersOfInterest())))
        if hook(point, observed) != 0:
            which = "PreNullHook" if model is self._null else "PreAltHook"
            log(None, ERROR, "Generation", f"{which} did not return 0.")
        self._events_if_needed(model)
        found = self._sampler.GetSamplingDistributions(point)
        if found is None:
            return None
        return SamplingDistribution(found.GetName(), found.GetTitle(), found)

    def _alt_distribution(self, observed: float) -> Any:
        """The alternate's distribution - from the same seed each time, if asked."""
        from ..roofit.rng import generator

        previous = 0
        if self._alt_seed > 0:
            previous = generator().Integer(4294967295 - 1) + 1
            generator().SetSeed(self._alt_seed)
        found = self._distribution(self._alt, self._pre_alt, observed)
        if previous > 0:
            generator().SetSeed(previous)
        return found


class _Toys(HypoTestCalculatorGeneric):
    """What both calculators add: their toys' numbers, and the adaptive tails."""

    def __init__(self, data: Any, altModel: Any, nullModel: Any, sampler: Any = None) -> None:
        super().__init__(data, altModel, nullModel, sampler)
        self._toys_null = self._toys_alt = -1
        self._tail_null = self._tail_alt = 0

    def SetToys(self, toysNull: int, toysAlt: int) -> None:
        self._toys_null, self._toys_alt = int(toysNull), int(toysAlt)

    def SetNToysInTails(self, toysNull: int, toysAlt: int) -> None:
        self._tail_null, self._tail_alt = int(toysNull), int(toysAlt)

    def _configure(self, which: str, observed: float) -> None:
        """The sampler's toys for the null or the alternate, and where their tail is."""
        toys = self._sampler
        if not isinstance(toys, ToyMCSampler):
            return
        log(None, INFO, "InputArguments", f"Using a ToyMCSampler. Now configuring for {which}.")
        number, tail = ((self._toys_null, self._tail_null) if which == "Null" else
                        (self._toys_alt, self._tail_alt))  # fmt: skip
        if number >= 0:
            toys.SetNToys(number)
        self._global_observables(which)
        if tail:
            log(None, INFO, "InputArguments", "Adaptive Sampling")
            right = toys.GetTestStatistic().PValueIsRightTail()
            if right == (which == "Null"):
                toys.SetToysRightTail(tail, observed)
            else:
                toys.SetToysLeftTail(tail, observed)
        else:
            toys.SetToysBothTails(0, 0, observed)
        if which == "Null":
            self._null.LoadSnapshot()

    def _global_observables(self, which: str) -> None:
        """Only the frequentist calculator tells the sampler the global observables."""


class FrequentistCalculator(_Toys):
    """Toys with the nuisance parameters at their conditional best fits to the data."""

    def __init__(self, data: Any, altModel: Any, nullModel: Any, sampler: Any = None) -> None:
        super().__init__(data, altModel, nullModel, sampler)
        self._mles_null: Any = None
        self._mles_alt: Any = None
        self._store_fit_info = False
        self._fit_info: Any = None

    def SetConditionalMLEsNull(self, items: Any) -> None:
        self._mles_null = RooArgSet(as_list(items)).snapshot() if items is not None else None

    def SetConditionalMLEsAlt(self, items: Any) -> None:
        self._mles_alt = RooArgSet(as_list(items)).snapshot() if items is not None else None

    def StoreFitInfo(self, flag: bool = True) -> None:
        self._store_fit_info = bool(flag)

    def GetFitInfo(self) -> Any:
        return self._fit_info

    def _pre(self) -> None:
        self._fit_info = RooArgSet() if self._store_fit_info else None

    def _pre_null(self, point: Any, observed: float) -> int:
        self._profiled(self._null, self._mles_null, "Null")
        if self._null.GetNuisanceParameters() is not None:
            point.add(list(self._null.GetNuisanceParameters()))
        self._configure("Null", observed)
        return 0

    def _pre_alt(self, point: Any, observed: float) -> int:
        self._profiled(self._alt, self._mles_alt, "Alt")
        if self._alt.GetNuisanceParameters() is not None:
            point.add(list(self._alt.GetNuisanceParameters()))
        self._configure("Alt", observed)
        return 0

    def _global_observables(self, which: str) -> None:
        model = self._null if which == "Null" else self._alt
        self._sampler.SetGlobalObservables(model.GetGlobalObservables())

    def _profile_fit(self, model: Any, params: Any, rest: Any, observables: tuple[Any, Any],
                     which: str) -> None:  # fmt: skip
        """The profile of the likelihood in ``rest``, fitted - its fit kept, if asked."""
        from ..roofit.cmdargs import RooCmdArg
        from ..roofit.messages import FATAL
        from .modelconfig import quieted

        cond, glob = observables
        with quieted(FATAL):
            nll = model.GetPdf().createNLL(self._data, RooCmdArg("CloneData", False),
                                           RooCmdArg("Constrain", params),
                                           RooCmdArg("GlobalObservables", glob),
                                           RooCmdArg("ConditionalObservables", cond),
                                           RooCmdArg("Offset", False))  # fmt: skip
            profile = nll.createProfile(list(rest))
            profile.minimizer().setPrintLevel(_default_print_level() - 1)
            profile.getVal()
            if self._fit_info is not None:
                self._fit_info.add(_fit_as_set(profile.minimizer().save(), f"fit{which}_"))

    def _profiled(self, model: Any, mles: Any, which: str) -> None:
        """The nuisance parameters at their best fit to the data with the parameters of interest
        fixed - unless given, all of them."""
        from .utils import RemoveConstantParameters

        params = RooArgSet(list(model.GetPdf().getParameters(self._data)))
        RemoveConstantParameters(params)
        nuisance = model.GetNuisanceParameters()
        if nuisance is None:
            return
        rest = RooArgSet([p for p in params if nuisance.find(p.GetName()) is None])
        if mles is not None and _given(params, rest, nuisance, mles, which):
            return
        log(None, INFO, "InputArguments", f"Profiling conditional MLEs for {which}.")
        cond = RooArgSet(list(model.GetConditionalObservables() or ()))
        glob = RooArgSet(list(model.GetGlobalObservables() or ()))
        self._profile_fit(model, params, rest, (cond, glob), which)
        self._statistic_observables(cond, glob)

    def _statistic_observables(self, cond: Any, glob: Any) -> None:
        """The test statistic told the conditional and global observables."""
        statistic = self._sampler.GetTestStatistic() if self._sampler is not None else None
        if statistic is not None:
            statistic.SetConditionalObservables(cond)
            statistic.SetGlobalObservables(glob)


def _given(params: Any, rest: Any, nuisance: Any, mles: Any, which: str) -> bool:
    """The conditional MLEs given, taken - and whether they are all the nuisance parameters,
    so nothing is left to fit."""
    log(None, INFO, "InputArguments", f"Using given conditional MLEs for {which}.")
    params.assign(mles)
    rest.add(list(mles), True)
    return not [p for p in nuisance if mles.find(p.GetName()) is None]


def _fit_as_set(fit: Any, prefix: str) -> list[Any]:
    """``DetailedOutputAggregator::GetAsArgSet``: a fit's status, minimum and parameters."""
    from ..roofit.variables import RooRealVar

    found = [RooRealVar(f"{prefix}minNLL", "", fit.minNll()),
             RooRealVar(f"{prefix}fitStatus", "", fit.status())]  # fmt: skip
    for par in fit.floatParsFinal():
        found.append(RooRealVar(f"{prefix}{par.GetName()}", par.GetTitle(), par.getVal()))
    return found


class HybridCalculator(_Toys):
    """Toys with the nuisance parameters drawn from a prior for each one."""

    def __init__(self, data: Any, altModel: Any, nullModel: Any, sampler: Any = None) -> None:
        from .utils import MakeNuisancePdf

        super().__init__(data, altModel, nullModel, sampler)
        self._prior_null = MakeNuisancePdf(nullModel, "PriorNuisanceNull")
        self._prior_alt = MakeNuisancePdf(altModel, "PriorNuisanceAlt")

    def ForcePriorNuisanceNull(self, prior: Any) -> None:
        self._prior_null = prior

    def ForcePriorNuisanceAlt(self, prior: Any) -> None:
        self._prior_alt = prior

    def SetNullModel(self, model: Any) -> None:
        from .utils import MakeNuisancePdf

        self._null = model
        self._prior_null = MakeNuisancePdf(model, "PriorNuisanceNull")

    def SetAlternateModel(self, model: Any) -> None:
        from .utils import MakeNuisancePdf

        self._alt = model
        self._prior_alt = MakeNuisancePdf(model, "PriorNuisanceAlt")

    def _check(self) -> int:
        for prior, model, which in ((self._prior_null, self._null, "Null"),
                                    (self._prior_alt, self._alt, "Alt")):  # fmt: skip
            nuisance = model.GetNuisanceParameters()
            if prior is not None and (nuisance is None or not len(nuisance)):
                log(None, ERROR, "InputArguments", "HybridCalculator - Nuisance PDF has been "
                    "specified, but is unaware of which parameters are the nuisance parameters. "
                    f"Must set nuisance parameters in the {which} ModelConfig"
                    + ("." if which == "Null" else ""))  # fmt: skip
                return -1
        return 0

    def _prior(self, prior: Any, model: Any, which: str) -> None:
        nuisance = model.GetNuisanceParameters()
        if prior is not None:
            self._sampler.SetPriorNuisance(prior)
        elif nuisance is None or not len(nuisance):
            log(None, INFO, "InputArguments", "HybridCalculator - No nuisance parameters "
                f"specified for {which} model and no prior forced. Case is reduced to simple "
                "hypothesis testing with no uncertainty.")  # fmt: skip
        else:
            log(None, INFO, "InputArguments", "HybridCalculator - Using uniform prior on "
                f"nuisance parameters ({which} model).")  # fmt: skip

    def _pre_null(self, point: Any, observed: float) -> int:
        self._prior(self._prior_null, self._null, "Null")
        self._configure("Null", observed)
        return 0

    def _pre_alt(self, point: Any, observed: float) -> int:
        self._prior(self._prior_alt, self._alt, "Alt")
        self._configure("Alt", observed)
        return 0


def _default_print_level() -> int:
    from ..fit.defaults import default

    return int(default("PrintLevel"))
