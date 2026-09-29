"""``HypoTestInverter``: a confidence interval by inverting hypothesis tests.

It runs its calculator's test with the null model's snapshot at each value
of the parameter of interest - a fixed scan, or an automatic search for
where ``CLs`` (or ``CLs+b``) crosses the test size - and collects the
results in a :class:`~.inverterresult.HypoTestInverterResult`, whose limits
are where the curve crosses. Toy-based points can be run again with more
toys until their ``CLs`` is precise enough near the target.
"""

from __future__ import annotations

import math
from typing import Any

from ..roofit.collections import RooArgSet
from ..roofit.messages import ERROR, FATAL, INFO, PROGRESS, WARNING, log
from ..roofit.printing import g

__all__ = ["HypoTestInverter"]

#: ``ECalculatorType``.
kUndefined, kHybrid, kFrequentist, kAsymptotic = 0, 1, 2, 3
#: ``fgCLAccuracy``, ``fgNToys``: how precise an adaptive point's CLs is made, toys at a time.
CL_ACCURACY, N_TOYS = [0.005], [500]
#: ``fgAbsAccuracy``, ``fgRelAccuracy``, ``fgAlgo``: when the automatic search stops, and how.
ABS_ACCURACY, REL_ACCURACY, ALGORITHM = [0.05], [0.05], ["logSecant"]


def _kind(calc: Any) -> int:
    for kind, name in ((kHybrid, "HybridCalculator"), (kFrequentist, "FrequentistCalculator"),
                       (kAsymptotic, "AsymptoticCalculator")):  # fmt: skip
        if any(klass.__name__ == name for klass in type(calc).__mro__):
            return kind
    return kUndefined


def variable_to_scan(calc: Any) -> Any:
    """``GetVariableToScan``: the null model's first parameter of interest - or the alternate's."""
    for model in (calc.GetNullModel(), calc.GetAlternateModel()):
        poi = model.GetParametersOfInterest() if model is not None else None
        if poi is not None and len(poi):
            return next(iter(poi))
    return None


def check_input_models(calc: Any, var: Any) -> None:
    """``CheckInputModels``: which model is which, said - and a warning for a background model
    whose parameter of interest is not at zero."""
    sb, b = calc.GetNullModel(), calc.GetAlternateModel()
    if sb is None or b is None:
        log(None, FATAL, "InputArguments", "HypoTestInverter - model are not existing")
        return
    log(None, INFO, "InputArguments", "HypoTestInverter ---- Input models: \n\t\t using as S+B "
        f"(null) model     : {sb.GetName()}\n\t\t using as B (alternate) model  : "
        f"{b.GetName()}\n")  # fmt: skip
    if b.GetPdf() is None or b.GetObservables() is None:
        log(None, ERROR, "InputArguments", "HypoTestInverter - B model has no pdf or observables "
            "defined")  # fmt: skip
        return
    params = b.GetPdf().getParameters(b.GetObservables())
    if params.find(var.GetName()) is None:
        return
    snapshot = b.GetSnapshot()
    found = snapshot.find(var.GetName()) if snapshot is not None else None
    if found is None or found.getVal() != 0:
        log(None, WARNING, "InputArguments", f"HypoTestInverter - using a B model  with POI "
            f"{var.GetName()} not equal to zero  user must check input model configurations ")


class HypoTestInverter:
    """The inversion: a calculator, the variable it scans, and the results of each point."""

    kUndefined, kHybrid, kFrequentist, kAsymptotic = kUndefined, kHybrid, kFrequentist, kAsymptotic

    def __init__(self, *args: Any) -> None:
        self._toys_run, self._max_toys = 0, 0
        self._calc: Any = None
        self._var: Any = None
        self._results: Any = None
        self._use_cls = self._scan_log = False
        self._size, self._verbose, self._kind = 0.05, 0, kUndefined
        self._nbins, self._xmin, self._xmax = 0, 1.0, 1.0
        self._num_err = 0.0
        self._limit_plot: Any = None
        if args and hasattr(args[0], "numEntries"):
            self._from_models(*args)
        elif args:
            self._from_calculator(*args)

    def _from_calculator(self, calc: Any, var: Any = None, size: float = 0.05) -> None:
        self._calc, self._var, self._size = calc, var, float(size)
        self._scan_variable()
        self._kind = _kind(calc)
        if self._kind == kUndefined:
            log(None, ERROR, "InputArguments", "HypoTestInverter - Type of hypotest calculator is "
                "not supported ")  # fmt: skip

    def _from_models(self, data: Any, sb: Any, b: Any, var: Any = None,
                     kind: int = kFrequentist, size: float = 0.05) -> None:  # fmt: skip
        from .asymcalc import AsymptoticCalculator
        from .calculators import FrequentistCalculator, HybridCalculator

        self._var, self._kind, self._size = var, int(kind), float(size)
        made = {kFrequentist: FrequentistCalculator, kHybrid: HybridCalculator,
                kAsymptotic: AsymptoticCalculator}.get(self._kind)  # fmt: skip
        self._calc = made(data, b, sb) if made is not None else None
        self._scan_variable()

    def _scan_variable(self) -> None:
        if self._var is None and self._calc is not None:
            self._var = variable_to_scan(self._calc)
        if self._var is None:
            log(None, ERROR, "InputArguments", "HypoTestInverter - Cannot guess the variable to "
                "scan ")  # fmt: skip
        else:
            check_input_models(self._calc, self._var)

    # -- the settings ---------------------------------------------------------------

    def SetFixedScan(self, nBins: int, xMin: float = 1.0, xMax: float = -1.0,
                     scanLog: bool = False) -> None:  # fmt: skip
        self._nbins, self._xmin, self._xmax, self._scan_log = int(nBins), xMin, xMax, scanLog

    def SetAutoScan(self) -> None:
        self.SetFixedScan(0)

    def UseCLs(self, on: bool = True) -> None:
        self._use_cls = bool(on)
        if self._results is not None:
            self._results.UseCLs(on)

    def SetVerbose(self, level: int = 1) -> None:
        self._verbose = int(level)

    def SetMaximumToys(self, ntoys: int) -> None:
        self._max_toys = int(ntoys)

    def SetNumErr(self, err: float) -> None:
        self._num_err = float(err)

    def SetTestSize(self, size: float) -> None:
        self._size = float(size)
        if self._results is not None:
            self._results.SetTestSize(size)

    def SetConfidenceLevel(self, cl: float) -> None:
        self._size = 1.0 - float(cl)
        if self._results is not None:
            self._results.SetConfidenceLevel(cl)

    def Size(self) -> float:
        return self._size

    def ConfidenceLevel(self) -> float:
        return 1.0 - self._size

    def GetHypoTestCalculator(self) -> Any:
        return self._calc

    def GetTestStatistic(self) -> Any:
        sampler = self._calc.GetTestStatSampler() if self._calc is not None else None
        return sampler.GetTestStatistic() if sampler is not None else None

    def SetTestStatistic(self, stat: Any) -> bool:
        sampler = self._calc.GetTestStatSampler() if self._calc is not None else None
        if sampler is None:
            return False
        sampler.SetTestStatistic(stat)
        return True

    def Clear(self) -> None:
        self._results = self._limit_plot = None

    def SetData(self, data: Any) -> None:
        if self._calc is not None:
            self._calc.SetData(data)

    def GetLimitPlot(self) -> Any:
        return self._limit_plot

    @staticmethod
    def SetCloseProof(flag: bool) -> None:
        """PROOF closes after each toy run in ROOT; nothing runs on PROOF here."""

    # -- the results ----------------------------------------------------------------

    def _create_results(self) -> Any:
        """``CreateResults``: ``result_<var>``, told CLs or not, the level, and whether the test
        is two-sided."""
        from .inverterresult import HypoTestInverterResult

        if self._results is None:
            name = self._var.GetName()
            self._results = HypoTestInverterResult(f"result_{name}", self._var,
                                                   self.ConfidenceLevel())  # fmt: skip
            self._results.SetTitle(f"HypoTestInverter Result For {name}")
        self._results.UseCLs(self._use_cls)
        self._results.SetConfidenceLevel(1.0 - self._size)
        if self._calc is not None:
            self._results._two_sided = self._two_sided()
        return self._results

    def _two_sided(self) -> bool:
        if self._kind == kAsymptotic:
            return bool(self._calc.IsTwoSided())
        sampler = self._calc.GetTestStatSampler()
        stat = sampler.GetTestStatistic() if sampler is not None else None
        return bool(getattr(stat, "IsTwoSided", lambda: False)())

    def GetInterval(self) -> Any:
        """The inverted interval: a copy of the results - of a fixed scan, or of an automatic
        search - run now, if none were run before."""
        if self._results is not None and self._results.ArraySize() >= 1:
            log(None, INFO, "Eval", "HypoTestInverter::GetInterval - return an already existing "
                "interval ")  # fmt: skip
            return self._results.Clone()
        if self._nbins > 0:
            log(None, INFO, "Eval", "HypoTestInverter::GetInterval - run a fixed scan")
            if not self.RunFixedScan(self._nbins, self._xmin, self._xmax, self._scan_log):
                log(None, ERROR, "Eval", "HypoTestInverter::GetInterval - error running a fixed "
                    "scan ")  # fmt: skip
        else:
            log(None, INFO, "Eval", "HypoTestInverter::GetInterval - run an automatic scan")
            if not self.RunLimit()[0]:
                log(None, ERROR, "Eval", "HypoTestInverter::GetInterval - error running an auto "
                    "scan ")  # fmt: skip
        return self._results.Clone()

    def GetLowerLimitDistribution(self, rebuild: bool = False, nToys: int = 100) -> Any:
        return self._limit_distribution(rebuild, "Lower")

    def GetUpperLimitDistribution(self, rebuild: bool = False, nToys: int = 100) -> Any:
        return self._limit_distribution(rebuild, "Upper")

    def _limit_distribution(self, rebuild: bool, side: str) -> Any:
        if rebuild:
            from ..errors import UnsupportedFeatureError

            raise UnsupportedFeatureError("HypoTestInverter gives the expected limits of its "
                                          "scan here; rebuilding them from new toys is not here "
                                          "yet")  # fmt: skip
        if self._results is None:
            log(None, ERROR, "InputArguments", f"HypoTestInverter::Get{side}LimitDistribution("
                "false) - result not existing")  # fmt: skip
            return None
        return getattr(self._results, f"Get{side}LimitDistribution")()

    def RebuildDistributions(self, *args: Any) -> Any:
        return self._limit_distribution(True, "Upper")

    # -- the points -----------------------------------------------------------------

    def _cls(self, result: Any) -> tuple[float, float]:
        if self._use_cls:
            return result.CLs(), result.CLsError()
        return result.CLsplusb(), result.CLsplusbError()

    def _eval(self, adaptive: bool, target: float) -> Any:
        """``Eval``: the calculator's test at the snapshot - with the background as alternate,
        the data's statistic moved by ``NumErr`` - run again with more toys while adaptive and
        not precise enough near ``target``."""
        result = self._calc.GetHypoTest()
        if result is None:
            log(None, ERROR, "Eval", "HypoTestInverter::Eval - HypoTest failed")
            return None
        result.SetBackgroundAsAlt(True)
        shift = -self._num_err if result.GetPValueIsRightTail() else self._num_err
        result.SetTestStatisticData(result.GetTestStatisticData() + shift)
        mid, err = self._cls(result)
        if adaptive and self._kind in (kHybrid, kFrequentist):
            self._calc.SetToys(N_TOYS[0] if self._use_cls else 1, 4 * N_TOYS[0])
        while adaptive and err >= CL_ACCURACY[0] and (target == -1 or abs(mid - target) < 3 * err):
            result.Append(self._calc.GetHypoTest())
            mid, err = self._cls(result)
            if self._verbose:
                log(None, PROGRESS, "Eval", ("\tCLs = " if self._use_cls else "\tCLsplusb = ")
                    + f"{g(mid)} +/- {g(err)}")  # fmt: skip
        if self._verbose:
            self._said(result)
        if self._kind in (kFrequentist, kHybrid):
            self._rename(result)
        return result

    def _said(self, result: Any) -> None:
        log(None, PROGRESS, "Eval", f"P values for  {self._var.GetName()} =  "
            f"{g(self._var.getVal())}\n\tCLs      = {g(result.CLs())} +/- {g(result.CLsError())}"
            f"\n\tCLb      = {g(result.CLb())} +/- {g(result.CLbError())}\n\tCLsplusb = "
            f"{g(result.CLsplusb())} +/- {g(result.CLsplusbError())}\n")  # fmt: skip

    def _rename(self, result: Any) -> None:
        """The toys counted, and each distribution named for the point: ``<name>_<var>_1.00``."""
        null, alt = result.GetNullDistribution(), result.GetAltDistribution()
        if null is None or alt is None:  # RooStats would read through the missing ones
            return
        self._toys_run += alt.GetSize() + null.GetSize()
        at = f"{self._var.GetName()}_{self._var.getVal():4.2f}"
        null.SetName(f"{null.GetName()}_{at}")
        alt.SetName(f"{alt.GetName()}_{at}")

    def _scan_range(self, nbins: int, xmin: float, xmax: float,
                    scan_log: bool) -> Any:  # fmt: skip
        """The scan's points and range, as ``RunFixedScan`` checks them - or ``None``."""
        if nbins <= 0:
            log(None, ERROR, "InputArguments", "HypoTestInverter::RunFixedScan - Please provide "
                "nBins>0")  # fmt: skip
            return None
        if nbins == 1 and xmin != xmax:
            log(None, WARNING, "InputArguments", "HypoTestInverter::RunFixedScan - nBins==1 -> I "
                f"will run for xMin ({g(xmin)})")  # fmt: skip
        if xmin == xmax and nbins > 1:
            log(None, WARNING, "InputArguments", "HypoTestInverter::RunFixedScan - xMin==xMax -> I "
                "will enforce nBins==1")  # fmt: skip
            nbins = 1
        if xmin > xmax:
            log(None, ERROR, "InputArguments", "HypoTestInverter::RunFixedScan - Please provide "
                f"xMin ({g(xmin)}) smaller than xMax ({g(xmax)})")  # fmt: skip
            return None
        if xmin < self._var.getMin():
            xmin = self._var.getMin()
            log(None, WARNING, "InputArguments", "HypoTestInverter::RunFixedScan - xMin < lower "
                f"bound, using xmin = {g(xmin)}")  # fmt: skip
        if xmax > self._var.getMax():
            xmax = self._var.getMax()
            log(None, WARNING, "InputArguments", "HypoTestInverter::RunFixedScan - xMax > upper "
                f"bound, using xmax = {g(xmax)}")  # fmt: skip
        if xmin <= 0 and scan_log:
            log(None, ERROR, "InputArguments", "HypoTestInverter::RunFixedScan - cannot go in log "
                "steps if xMin <= 0")  # fmt: skip
            return None
        return nbins, xmin, xmax

    def RunFixedScan(self, nBins: int, xMin: float, xMax: float, scanLog: bool = False) -> bool:
        """Each of ``nBins`` points from ``xMin`` to ``xMax`` - evenly, or evenly in the log."""
        results = self._create_results()
        results._fitted = [False, False]
        checked = self._scan_range(int(nBins), float(xMin), float(xMax), bool(scanLog))
        if checked is None:
            return False
        nbins, xmin, xmax = checked
        for i in range(nbins):
            x = xmin
            if i > 0 and scanLog:
                x = math.exp(math.log(xmin) + i * (math.log(xmax) - math.log(xmin)) / (nbins - 1))
            elif i > 0:
                x = xmin + i * (xmax - xmin) / (nbins - 1)
            if not self.RunOnePoint(x):
                log(None, WARNING, "Eval", f"HypoTestInverter::RunFixedScan - The hypo test for "
                    f"point {g(x)} failed. Skipping.")  # fmt: skip
        return True

    def _clamped(self, x: float) -> float:
        low, high = self._var.getMin(), self._var.getMax()
        if x < low:
            log(None, ERROR, "InputArguments", "HypoTestInverter::RunOnePoint - Out of range: "
                f"using the lower bound {g(low)} on the scanned variable rather than {g(x)}")
            return low
        if x > high:
            if x > high * (1.0 + 1e-12):
                log(None, ERROR, "InputArguments", "HypoTestInverter::RunOnePoint - Out of range: "
                    f"using the upper bound {g(high)} on the scanned variable rather than {g(x)}")
            return high
        return x

    def RunOnePoint(self, thisX: float, adaptive: bool = False, clTarget: float = -1) -> bool:
        """The test at ``thisX`` - the null snapshot moved there - added to the results, or merged
        with the last point's if it is the same value."""
        results = self._create_results()
        x = self._clamped(float(thisX))
        before = self._var.getVal()
        self._var.setVal(x)
        sb = self._calc.GetNullModel()
        poi = RooArgSet(list(sb.GetParametersOfInterest()))
        poi.assign(RooArgSet([self._var]))
        sb.SetSnapshot(poi)
        name = self._var.GetName()
        if self._verbose > 0:
            log(None, PROGRESS, "Eval", f"Running for {name} = {g(self._var.getVal())}")
        result = self._eval(adaptive, float(clTarget))
        if result is None:
            log(None, ERROR, "Eval", f"HypoTestInverter - Error running point {name} = "
                f"{g(self._var.getVal())}")  # fmt: skip
            return False
        null, alt = result.NullPValue(), result.AlternatePValue()
        if not (math.isfinite(null) and 0 <= null <= 1 and math.isfinite(alt) and 0 <= alt <= 1):
            log(None, WARNING, "Eval", f"HypoTestInverter - Skipping invalid result for  point "
                f"{name} = {g(self._var.getVal())}. null p-value={g(null)}, alternate "
                f"p-value={g(alt)}")  # fmt: skip
            return False
        self._keep(results, x, result)
        self._var.setVal(before)
        return True

    @staticmethod
    def _keep(results: Any, x: float, result: Any) -> None:
        last = results._x[-1] if results._x else -999.0
        if not _same_point(x, last):
            results._x.append(x)
            results._results.append(result)
            return
        log(None, INFO, "Eval", f"HypoTestInverter::RunOnePoint - Merge with previous result for "
            f"{results._parameters.first().GetName()} = {g(x)}")  # fmt: skip
        previous = results._results[-1]
        if previous.GetNullDistribution() is not None and previous.GetAltDistribution() is not None:
            previous.Append(result)
        else:
            log(None, INFO, "Eval", "HypoTestInverter::RunOnePoint - replace previous empty result")
            results._results[-1] = result


    # -- the automatic search -------------------------------------------------------

    def RunLimit(self, limit: Any = None, limitErr: Any = None, absTol: float = 0.0,
                 relTol: float = 0.0, hint: Any = None) -> Any:  # fmt: skip
        """``RunLimit``: the upper limit searched for - bracketed, bisected, then fitted - into
        ``limit`` and ``limitErr`` when given as references; returns whether it was found (or,
        called bare, that and the limit and its error)."""
        from .inverterscan import run_limit

        found, value, error = run_limit(self, absTol, relTol, hint)
        if limit is None:
            return found, value, error
        limit.value, limitErr.value = value, error
        return found


def _same_point(x: float, last: float) -> bool:
    """``RunOnePoint``'s test: an absolute 1e-12 below one, a relative one from one up."""
    if abs(x) < 1:
        return abs(x - last) < 1e-12
    return abs(x - last) <= 0.5e-12 * (abs(x) + abs(last))
