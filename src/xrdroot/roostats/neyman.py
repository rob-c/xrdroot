"""``NeymanConstruction`` and ``FeldmanCousins``: an interval from toys at every point of a scan.

At each point the test statistic of the data is set against its sampling
distribution there - toys, adaptively more of them while the data's value is
within a statistical wobble of the acceptance region's edge - and the point
is in the interval when the data's value is inside the region. Feldman and
Cousins' construction is the Neyman construction with the profile
likelihood ratio, the region's left tail empty, and the nuisance
parameters at their conditional best fit at each point.
"""

from __future__ import annotations

from typing import Any

from ..errors import UnsupportedFeatureError
from ..roofit.collections import RooArgSet
from ..roofit.messages import ERROR, INFO, PROGRESS, log, log_plain
from ..roofit.printing import g
from .belt import ConfidenceBelt, PointSetInterval

__all__ = ["FeldmanCousins", "NeymanConstruction"]


def _progress(text: str) -> None:
    log_plain(None, PROGRESS, "Eval", text)


class NeymanConstruction:
    """The points of a scan whose data statistic is in their acceptance region."""

    def __init__(self, data: Any, model: Any) -> None:
        self._data, self._model = data, model
        self._size = 0.05
        self._sampler: Any = None
        self._points: Any = None
        self._left = 0.0
        self._belt: Any = None
        self._adaptive = False
        self._factor = 1.0
        self._create_belt = False

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
        self._model = model

    def SetTestStatSampler(self, sampler: Any) -> None:
        self._sampler = sampler

    def GetTestStatSampler(self) -> Any:
        return self._sampler

    def SetParameterPointsToTest(self, points: Any) -> None:
        self._points = points
        self._belt = ConfidenceBelt("ConfBelt", points)

    def SetLeftSideTailFraction(self, fraction: float = 0.0) -> None:
        self._left = float(fraction)

    def UseAdaptiveSampling(self, flag: bool = True) -> None:
        self._adaptive = bool(flag)

    def AdditionalNToysFactor(self, factor: float) -> None:
        self._factor = float(factor)

    def SaveBeltToFile(self, flag: bool = True) -> None:
        if flag:
            raise UnsupportedFeatureError(
                "NeymanConstruction writes its sampling distributions to "
                "SamplingDistributions.root as RooStats objects, which this does not write"
            )

    def CreateConfBelt(self, flag: bool = True) -> None:
        self._create_belt = bool(flag)

    def GetConfidenceBelt(self) -> Any:
        return self._belt

    # -- the construction ---------------------------------------------------------

    def _toys(self) -> int:
        tails = min(self._left, 1.0 - self._left)
        total = int(2.0 / self._size / tails) if tails > 0 else int(2.0 / self._size)
        return int(float(total) * self._factor)

    def _adaptive_distribution(self, point: Any, statistic: float) -> Any:
        """Toys, twice as many again each round, while the data's statistic is within a
        statistical wobble of an edge of the region - and fewer than ``100 / size``."""
        found: Any = None
        total = self._toys()
        upper_p, lower_p = 1.0 - (1.0 - self._left) * self._size, self._left * self._size
        while True:
            found = self._sampler.AppendSamplingDistribution(point, found, 2 * total)
            if found is None:
                return None
            total = found.GetSize()
            upper, up_plus = found.quantile(upper_p, 1)
            _, up_minus = found.quantile(upper_p, -1)
            lower, low_plus = found.quantile(lower_p, 1)
            _, low_minus = found.quantile(lower_p, -1)
            near = (up_minus < statistic <= upper) or (upper <= statistic < up_plus) or (
                low_minus < statistic <= lower) or (lower <= statistic < low_plus)  # fmt: skip
            if not (near and total < 100.0 / self._size):
                return found

    def GetInterval(self) -> Any:
        """``GetInterval``: each point tested, its line of progress printed, the accepted kept."""
        from ..roofit.data.dataset import RooDataSet

        poi = RooArgSet(list(self._model.GetParametersOfInterest()))
        accepted = RooDataSet("pointsInInterval", "points in interval", list(self._points.get(0)))
        npass = 0
        for i in range(self._points.numEntries()):
            point = self._points.get(i)
            poi.assign(point)
            self._sampler.SetParametersForTestStat(poi)
            statistic = self._sampler.EvaluateTestStatistic(self._data, poi)
            if self._adaptive:
                dist = self._adaptive_distribution(point, statistic)
            else:
                dist = self._sampler.GetSamplingDistribution(point)
            if dist is None:
                log(None, ERROR, "Eval", "Neyman Construction: error generating sampling "
                    "distribution")  # fmt: skip
                return None
            lower = dist.quantile(self._left * self._size)[0]
            upper = dist.quantile(1.0 - (1.0 - self._left) * self._size)[0]
            if self._belt is not None and self._create_belt:
                self._belt.AddAcceptanceRegion(point, i, lower, upper)
            inside = lower <= statistic <= upper
            _progress(f"NeymanConstruction: Prog: {i + 1}/{self._points.numEntries()} total MC = "
                      f"{dist.GetSize()} this test stat = {g(statistic)}\n")  # fmt: skip
            values = "".join(f"{one.GetName()}={g(one.getVal())} " for one in point)
            _progress(f" {values}[{g(lower)}, {g(upper)}]  in interval = {int(inside)}\n\n")
            if inside:
                accepted.add(point)
                npass += 1
        log(None, INFO, "Eval", f"{npass} points in interval")
        return PointSetInterval("ClassicalConfidenceInterval", accepted)


def _generation(text: str) -> None:
    log_plain(None, PROGRESS, "Generation", text + "\n")


class FeldmanCousins:
    """Feldman and Cousins' unified intervals: a Neyman construction of the profile likelihood
    ratio, its region open to the left, over a scan of the parameters of interest."""

    def __init__(self, data: Any, model: Any) -> None:
        self._data, self._model = data, model
        self._size = 0.05
        self._sampler: Any = None
        self._points: Any = None
        self._poi_to_test: Any = None
        self._belt: Any = None
        self._adaptive = False
        self._factor = 1.0
        self._bins = 10
        self._fluctuate = True
        self._profile = True
        self._create_belt = False

    def SetTestSize(self, size: float) -> None:
        self._size = float(size)

    def SetConfidenceLevel(self, cl: float) -> None:
        self._size = 1.0 - float(cl)

    def Size(self) -> float:
        return self._size

    def ConfidenceLevel(self) -> float:
        return 1.0 - self._size

    def SetModel(self, model: Any) -> None:
        self._model = model

    def SetData(self, data: Any) -> None:
        self._data = data

    def UseAdaptiveSampling(self, flag: bool = True) -> None:
        self._adaptive = bool(flag)

    def AdditionalNToysFactor(self, factor: float) -> None:
        self._factor = float(factor)

    def SetNBins(self, bins: int) -> None:
        self._bins = int(bins)

    def FluctuateNumDataEntries(self, flag: bool = True) -> None:
        self._fluctuate = bool(flag)

    def DoProfileConstruction(self, flag: bool = True) -> None:
        self._profile = bool(flag)

    def SetPOIPointsToTest(self, points: Any) -> None:
        self._poi_to_test = points

    def SetParameterPointsToTest(self, points: Any) -> None:
        self._points = points

    def SaveBeltToFile(self, flag: bool = True) -> None:
        NeymanConstruction(self._data, self._model).SaveBeltToFile(flag)

    def CreateConfBelt(self, flag: bool = True) -> None:
        self._create_belt = bool(flag)

    def GetConfidenceBelt(self) -> Any:
        return self._belt

    def GetPointsToScan(self) -> Any:
        return self._points

    def GetTestStatSampler(self) -> Any:
        if self._sampler is None:
            self._create_sampler()
        return self._sampler

    def _create_sampler(self) -> None:
        """``CreateTestStatSampler``: the profile likelihood ratio, sampled ``50 / size`` times."""
        from .teststats import ProfileLikelihoodTestStat
        from .toymc import ToyMCSampler

        ntoys = int(self._factor * 50.0 / self._size)
        self._sampler = ToyMCSampler(ProfileLikelihoodTestStat(self._model.GetPdf()), ntoys)
        self._sampler.SetParametersForTestStat(self._model.GetParametersOfInterest())
        if self._model.GetObservables() is not None:
            self._sampler.SetObservables(self._model.GetObservables())
        self._sampler.SetPdf(self._model.GetPdf())
        _generation(f"FeldmanCousins: ntoys per point = {ntoys}" if not self._adaptive
                    else "FeldmanCousins: ntoys per point: adaptive")  # fmt: skip
        if self._fluctuate:
            _generation("FeldmanCousins: nEvents per toy will fluctuate about  expectation")
        else:
            _generation("FeldmanCousins: nEvents per toy will not fluctuate, will always be "
                        f"{self._data.numEntries()}")  # fmt: skip
            self._sampler.SetNEventsPerToy(self._data.numEntries())

    def _create_points(self) -> None:
        """``CreateParameterPoints``: a scan of the parameters of interest - and, with nuisance
        parameters, each point with them at the profile's conditional best fit."""
        from ..roofit.data.datahist import RooDataHist

        poi = list(self._model.GetParametersOfInterest())
        nuisance = list(self._model.GetNuisanceParameters() or ())
        parameters = RooArgSet(poi + nuisance)
        if nuisance and len(parameters) != len(poi) and self._profile:
            self._points = self._profile_points(poi, parameters)
            return
        _generation("FeldmanCousins: Model has no nuisance parameters")
        for one in parameters:
            one.setBins(self._bins)
        self._points = RooDataHist("parameterScan", "", list(parameters))
        _generation(f"FeldmanCousins: # points to test = {self._points.numEntries()}")

    def _profile_points(self, poi: list[Any], parameters: Any) -> Any:
        """The profile construction: each point of the scan, the nuisance parameters profiled."""
        from ..roofit.cmdargs import RooCmdArg
        from ..roofit.data.datahist import RooDataHist
        from ..roofit.data.dataset import RooDataSet
        from ..roofit.messages import FATAL
        from .modelconfig import quieted

        _generation("FeldmanCousins: Model has nuisance parameters, will do profile construction")
        for one in poi:
            one.setBins(self._bins)
        scan = self._poi_to_test or RooDataHist("parameterScan", "", poi)
        _generation(f"FeldmanCousins: # points to test = {scan.numEntries()}")
        with quieted(FATAL):
            nll = self._model.GetPdf().createNLL(self._data, RooCmdArg("CloneData", False))
            profile = nll.createProfile(poi)
            points = RooDataSet("profileConstruction", "profileConstruction", list(parameters))
            for i in range(scan.numEntries()):
                parameters.assign(scan.get(i))
                profile.getVal()
                points.add(parameters)
        return points

    def GetInterval(self) -> Any:
        """The unified interval: the points the construction accepts."""
        self._model.GuessObsAndNuisance(self._data.get())
        if self._sampler is None:
            self._create_sampler()
        self._sampler.SetObservables(self._model.GetObservables())
        if not self._fluctuate:
            self._sampler.SetNEventsPerToy(self._data.numEntries())
        self._create_points()
        construction = NeymanConstruction(self._data, self._model)
        construction.SetTestStatSampler(self._sampler)
        construction.SetTestSize(self._size)
        construction.SetParameterPointsToTest(self._points)
        construction.SetLeftSideTailFraction(0.0)
        construction.UseAdaptiveSampling(self._adaptive)
        construction.AdditionalNToysFactor(self._factor)
        construction.CreateConfBelt(self._create_belt)
        if self._create_belt:
            self._belt = construction.GetConfidenceBelt()
        return construction.GetInterval()
