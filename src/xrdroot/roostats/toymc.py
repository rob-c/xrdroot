"""``ToyMCSampler``: a test statistic's sampling distribution, from toys of a model at a point.

Each toy is RooStats' order of things: the model's parameters set to the
point, the global observables generated if there are any, the nuisance
parameters drawn from their prior if one is given, the observables
generated - as many events as asked, or as many as expected, Poisson-varied
- the parameters put back, and the test statistic evaluated on the toy at
the point it is tested at. The draws are RooRandom's, so a script that
seeds it gets ROOT's toys.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from ..roofit.cmdargs import RooCmdArg
from ..roofit.collections import RooArgSet, as_list
from ..roofit.messages import ERROR, INFO, PROGRESS, WARNING, log, log_plain, service
from .sampling import SamplingDistribution

__all__ = ["ToyMCSampler"]


class ToyMCSampler:
    """Toys of a model at a point, and a test statistic of each."""

    #: ``fgAlwaysUseMultiGen``.
    _multigen: ClassVar[list[bool]] = [False]

    def __init__(self, ts: Any = None, ntoys: int = 1000) -> None:
        self._name = ts.GetVarName() if ts is not None else ""
        self._ntoys = int(ntoys)
        self._max_toys = math.inf
        self._low, self._high = -math.inf, math.inf
        self._statistics: list[Any] = []
        self._pdf: Any = None
        self._point: Any = None
        self._prior: Any = None
        self._nuisance: Any = None
        self._observables: Any = None
        self._global: Any = None
        self._events = 0
        self._size = 0.05
        self._expected_nuisance = False
        self._binned = False
        self._binned_tag = ""
        self._auto_binned = True
        self._tails = 0.0
        self._proto: Any = None
        self._nuisance_sampler: Any = None
        stream = service().getStream(1)
        stream.removeTopic(16384)  # NumericIntegration: RooStats silences it on the INFO stream
        if ts is not None:
            self._statistics.append(ts)

    # -- settings -----------------------------------------------------------------

    def AddTestStatistic(self, ts: Any = None) -> None:
        if ts is None:
            log(None, INFO, "InputArguments", "No test statistic given. Doing nothing.")
            return
        self._statistics.append(ts)

    def SetTestStatistic(self, ts: Any, index: int = 0) -> None:
        if index == len(self._statistics):
            self._statistics.append(ts)
        else:
            self._statistics[index] = ts

    def GetTestStatistic(self, index: int = 0) -> Any:
        return self._statistics[index] if index < len(self._statistics) else None

    def GetNToys(self) -> int:
        return self._ntoys

    def SetNToys(self, ntoys: int) -> None:
        self._ntoys = int(ntoys)

    def SetNEventsPerToy(self, nevents: int) -> None:
        self._events = int(nevents)

    def nEventsPerToy(self) -> int:
        return self._events

    def SetParametersForTestStat(self, poi: Any) -> None:
        self._point = RooArgSet(as_list(poi)).snapshot()

    def SetPdf(self, pdf: Any) -> None:
        self._pdf = pdf

    def SetPriorNuisance(self, pdf: Any) -> None:
        self._prior = pdf
        self._nuisance_sampler = None

    def SetNuisanceParameters(self, items: Any) -> None:
        self._nuisance = RooArgSet(as_list(items))

    def SetObservables(self, items: Any) -> None:
        self._observables = RooArgSet(as_list(items))

    def SetGlobalObservables(self, items: Any) -> None:
        self._global = RooArgSet(as_list(items))

    def SetTestSize(self, size: float) -> None:
        self._size = float(size)

    def SetConfidenceLevel(self, cl: float) -> None:
        self._size = 1.0 - float(cl)

    def ConfidenceLevel(self) -> float:
        return 1.0 - self._size

    def SetExpectedNuisancePar(self, flag: bool = True) -> None:
        self._expected_nuisance = bool(flag)

    SetAsimovNuisancePar = SetExpectedNuisancePar

    def SetGenerateBinned(self, flag: bool = True) -> None:
        self._binned = bool(flag)

    def SetGenerateBinnedTag(self, tag: str = "") -> None:
        self._binned_tag = str(tag)

    def SetGenerateAutoBinned(self, flag: bool = True) -> None:
        self._auto_binned = bool(flag)

    def SetSamplingDistName(self, name: str) -> None:
        if name:
            self._name = str(name)

    def GetSamplingDistName(self) -> str:
        return self._name

    def SetMaxToys(self, toys: float) -> None:
        self._max_toys = float(toys)

    def SetToysLeftTail(self, toys: float, threshold: float) -> None:
        self._tails, self._low, self._high = float(toys), float(threshold), math.inf

    def SetToysRightTail(self, toys: float, threshold: float) -> None:
        self._tails, self._low, self._high = float(toys), -math.inf, float(threshold)

    def SetToysBothTails(self, toys: float, low: float, high: float) -> None:
        self._tails, self._low, self._high = float(toys), float(low), float(high)

    def SetProtoData(self, data: Any) -> None:
        self._proto = data

    def SetUseMultiGen(self, flag: bool = True) -> None:
        """Generation is the same draws either way here: kept for the call's sake."""

    @staticmethod
    def SetAlwaysUseMultiGen(flag: bool) -> None:
        ToyMCSampler._multigen[0] = bool(flag)

    # -- sampling -----------------------------------------------------------------

    def CheckConfig(self) -> bool:
        good = True
        for missing, text in ((not self._statistics, "Test statistic not set."),
                              (self._observables is None, "Observables not set."),
                              (self._point is None, "Parameter values used to evaluate the test "
                               "statistic are not set."),
                              (self._pdf is None, "Pdf not set.")):  # fmt: skip
            if missing:
                log_plain(None, ERROR, "InputArguments", text + "\n")
                good = False
        return good

    def EvaluateTestStatistic(self, data: Any, poi: Any, index: int = 0) -> float:
        return float(self._statistics[index].Evaluate(data, poi))

    def GetSamplingDistribution(self, point: Any) -> Any:
        """The first statistic's distribution over ``GetNToys()`` toys at ``point``."""
        found = self.GetSamplingDistributions(point)
        if found is None or not found.numEntries():
            log(None, WARNING, "Generation", "no sampling distribution generated")
            return None
        return SamplingDistribution(found.GetName(), found.GetTitle(), found)

    def AppendSamplingDistribution(self, point: Any, last: Any, additional: int) -> Any:
        """``additional`` toys more, appended to ``last``."""
        before, self._ntoys = self._ntoys, int(additional)
        fresh = self.GetSamplingDistribution(point)
        self._ntoys = before
        if last is not None:
            last.Add(fresh)
            return last
        return fresh

    def GetSamplingDistributions(self, point: Any) -> Any:
        """``GetSamplingDistributionsSingleWorker``: a dataset of every toy's statistics."""
        if not self.CheckConfig():
            log(None, ERROR, "InputArguments", "Bad COnfiguration in ToyMCSampler ")
            return None
        at = RooArgSet(as_list(point)).snapshot()
        variables = RooArgSet(list(self._pdf.getVariables()))
        saved = variables.snapshot()
        rows: list[list[float]] = []
        weights: list[float] = []
        tails, i = 0.0, 0
        while i < self._max_toys and not (tails >= self._tails and i + 1 > self._ntoys):
            self._progress(i, tails)
            variables.assign(saved)
            toy, weight = self.GenerateToyData(at, None, with_weight=True)
            variables.assign(self._point)
            values = self._all_statistics(toy, variables)
            i += 1
            if values[0] != values[0]:
                log(None, WARNING, "Generation", f"skip: {values[0]:g}, {weight:g}")
                continue
            rows.append(values)
            weights.append(weight)
            tails += self._in_tails(values[0], weight)
        variables.assign(saved)
        return self._as_data(rows, weights)

    def _progress(self, i: int, tails: float) -> None:
        """Every five hundred toys, how many - and in the tails, when toys run until enough are."""
        if i % 500 == 0 and i > 0:
            text = f"generated toys: {i} / {self._ntoys}"
            text += f" (tails: {tails:g} / {self._tails:g})" if self._tails else ""
            log(None, PROGRESS, "Generation", text)

    def _in_tails(self, value: float, weight: float) -> float:
        """A toy's count towards the tails: its weight - one, if negative - when it is in them."""
        if value <= self._low or value >= self._high:
            return weight if weight >= 0.0 else 1.0
        return 0.0

    def _all_statistics(self, data: Any, variables: Any, point: Any = None) -> list[float]:
        """``EvaluateAllTestStatistics``: each statistic at the point, the model's variables put
        back after each."""
        saved = variables.snapshot() if variables is not None else None
        point = self._point if point is None else point
        found = []
        for ts in self._statistics:
            found.append(float(ts.Evaluate(data, RooArgSet(list(as_list(point))).snapshot())))
            if saved is not None:
                variables.assign(saved)
        return found

    def EvaluateAllTestStatistics(self, data: Any, poi: Any) -> Any:
        from ..roofit.variables import RooRealVar

        variables = RooArgSet(list(self._pdf.getVariables())) if self._pdf is not None else None
        values = self._all_statistics(data, variables, poi)
        pairs = zip(self._statistics, values, strict=False)
        return [RooRealVar(f"{self._name}_TS{i}", ts.GetVarName(), v)
                for i, (ts, v) in enumerate(pairs)]  # fmt: skip

    def _as_data(self, rows: list[list[float]], weights: list[float]) -> Any:
        """``DetailedOutputAggregator::GetAsDataSet``: a column per statistic, weighted."""
        import numpy as np

        from ..roofit.data.dataset import RooDataSet
        from ..roofit.variables import RooRealVar

        columns = [RooRealVar(f"{self._name}_TS{i}", ts.GetVarName(), 0.0)
                   for i, ts in enumerate(self._statistics)]  # fmt: skip
        weight = RooRealVar("weight", "weight", 1.0)
        made = RooDataSet(self._name, self._name, [*columns, weight], WeightVar="weight")
        made._columns = {one.GetName(): np.array([r[i] for r in rows], dtype=np.float64)
                         for i, one in enumerate(columns)}  # fmt: skip
        made._weights = np.asarray(weights, dtype=np.float64)
        return made

    # -- toys ---------------------------------------------------------------------

    def GenerateToyData(self, point: Any, pdf: Any = None, with_weight: bool = False) -> Any:
        """A toy at ``point``: global observables, nuisance parameters, observables, as RooStats
        draws them in turn; with ``with_weight``, the toy and its weight."""
        pdf = self._pdf if pdf is None or not hasattr(pdf, "generate") else pdf
        if self._observables is None:
            log_plain(None, ERROR, "InputArguments", "Observables not set.\n")
            return (None, 1.0) if with_weight else None
        variables = RooArgSet(list(self._pdf.getVariables()))
        variables.assign(as_list(point))
        self._make_nuisance_sampler()
        observables = self._generated_observables(pdf)
        saved = variables.snapshot()
        weight = self._nuisance_weight(point, variables)
        data = self.Generate(pdf, observables)
        variables.assign(saved)
        return (data, weight) if with_weight else data

    def _make_nuisance_sampler(self) -> None:
        """The prior's sampler of the nuisance parameters, made the first time one is wanted."""
        if self._nuisance_sampler is None and self._prior is not None and self._nuisance:
            from .nuisance import NuisanceParametersSampler

            self._nuisance_sampler = NuisanceParametersSampler(
                self._prior, self._nuisance, self._ntoys, self._expected_nuisance
            )

    def _generated_observables(self, pdf: Any) -> RooArgSet:
        """The observables to draw - the global ones drawn first, apart."""
        observables = RooArgSet(list(self._observables))
        if self._global is not None and len(self._global):
            for one in list(self._global):
                observables.remove(one, True, True)
            self.GenerateGlobalObservables(pdf)
        return observables

    def _nuisance_weight(self, point: Any, variables: Any) -> float:
        """The nuisance parameters drawn from the prior, and the toy's weight - one if none."""
        if self._nuisance_sampler is None:
            return 1.0
        names = {one.GetName() for one in as_list(point)}
        return float(self._nuisance_sampler.next_point(
            RooArgSet([v for v in variables if v.GetName() not in names])))  # fmt: skip

    def GenerateGlobalObservables(self, pdf: Any) -> None:
        """New values of the global observables, drawn from their constraints."""
        if self._global is None or not len(self._global):
            log_plain(None, ERROR, "InputArguments", "Global Observables not set.\n")
            return
        one = pdf.generateSimGlobal(self._global, 1)
        RooArgSet(list(pdf.getVariables())).assign(one.get(0))

    def Generate(self, pdf: Any, observables: Any, protoData: Any = None,
                 forceEvents: int = 0) -> Any:  # fmt: skip
        """The toy's events: as many as asked for, or as many as the model expects,
        Poisson-varied."""
        if self._proto is not None:
            protoData, forceEvents = self._proto, self._proto.numEntries()
        events = forceEvents or self._events
        options = [RooCmdArg("AutoBinned", self._auto_binned),
                   RooCmdArg("GenBinned", self._binned_tag)]  # fmt: skip
        if self._binned:
            options = [RooCmdArg("AllBinned")]
        if protoData is not None:
            options.append(RooCmdArg("ProtoData", protoData, True, True))
        if events == 0:
            if not pdf.canBeExtended() or pdf.expectedEvents(observables) <= 0:
                text = ("ToyMCSampler: Error : pdf is not extended and number of events per toy "
                        "is zero")  # fmt: skip
                log(None, ERROR, "InputArguments", text)
                raise RuntimeError(text)
            return pdf.generate(observables, RooCmdArg("Extended"), *options)
        return pdf.generate(observables, RooCmdArg("NumEvents", events), *options)
