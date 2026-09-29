"""The other test statistics: two fixed points' likelihoods, two profiles, a count, an estimate.

``SimpleLikelihoodRatioTestStat`` is ``-log L(null) + log L(alt)`` - the LEP
statistic - each model stripped of its constraints first;
``RatioOfProfiledLikelihoodsTestStat`` the difference of two profile
likelihood ratios - the Tevatron's; ``NumEventsTestStat`` the number of
events; ``MaxLikelihoodEstimateTestStat`` a parameter's fitted value.
"""

from __future__ import annotations

from typing import Any

from ..roofit.cmdargs import RooCmdArg
from ..roofit.collections import RooArgSet, as_list
from ..roofit.messages import ERROR, FATAL, WARNING, log
from .modelconfig import quieted
from .teststats import ProfileLikelihoodTestStat, TestStatistic
from .utils import MakeUnconstrainedPdf, RemoveConstantParameters

__all__ = [
    "MaxLikelihoodEstimateTestStat",
    "NumEventsTestStat",
    "RatioOfProfiledLikelihoodsTestStat",
    "SimpleLikelihoodRatioTestStat",
]

#: ``fgAlwaysReuseNll`` of the simple likelihood ratio.
_ALWAYS_REUSE = [True]


class SimpleLikelihoodRatioTestStat(TestStatistic):
    """``-log(L(null) / L(alt))``, each at its own fixed parameters."""

    def __init__(self, nullPdf: Any = None, altPdf: Any = None, nullParameters: Any = None,
                 altParameters: Any = None) -> None:  # fmt: skip
        self._null_pdf, self._alt_pdf = nullPdf, altPdf
        if nullPdf is not None and nullParameters is None:
            nullParameters, altParameters = nullPdf.getVariables(), altPdf.getVariables()
        self._null = RooArgSet(as_list(nullParameters)).snapshot() if nullPdf else None
        self._alt = RooArgSet(as_list(altParameters)).snapshot() if nullPdf else None
        self._cond, self._glob = RooArgSet(), RooArgSet()
        self._first, self._reuse = True, False
        self._detailed = False
        self._output: Any = None
        self._nll_null: Any = None
        self._nll_alt: Any = None

    @staticmethod
    def SetAlwaysReuseNLL(flag: bool) -> None:
        _ALWAYS_REUSE[0] = bool(flag)

    def SetReuseNLL(self, flag: bool) -> None:
        self._reuse = bool(flag)

    def SetNullParameters(self, items: Any) -> None:
        self._first, self._null = True, RooArgSet(as_list(items)).snapshot()

    def SetAltParameters(self, items: Any) -> None:
        self._first, self._alt = True, RooArgSet(as_list(items)).snapshot()

    def ParamsAreEqual(self) -> bool:
        null, alt = list(self._null), list(self._alt)
        if sorted(p.GetName() for p in null) != sorted(p.GetName() for p in alt):
            return False
        return all(a.getVal() == b.getVal() for a, b in zip(null, alt))

    def SetConditionalObservables(self, items: Any) -> None:
        self._cond = RooArgSet(as_list(items))

    def SetGlobalObservables(self, items: Any) -> None:
        self._glob = RooArgSet(as_list(items))

    def GetVarName(self) -> str:
        return "log(L(#mu_{1}) / L(#mu_{0}))"

    def EnableDetailedOutput(self, flag: bool = True) -> None:
        self._detailed, self._output = bool(flag), None

    def GetDetailedOutput(self) -> Any:
        return self._output

    def Evaluate(self, data: Any, nullPOI: Any) -> float:
        if self._first and self.ParamsAreEqual():
            log(None, WARNING, "InputArguments", "Same RooArgSet used for null and alternate, so "
                "you must explicitly SetNullParameters and SetAlternateParameters or the "
                "likelihood ratio will always be 1.")  # fmt: skip
        if self._first:
            null, alt = self._null_pdf, self._alt_pdf
            self._null_pdf = MakeUnconstrainedPdf(null, null.getObservables(data))
            self._alt_pdf = MakeUnconstrainedPdf(alt, alt.getObservables(data))
        self._first = False
        reuse = self._reuse or _ALWAYS_REUSE[0]
        with quieted(FATAL):
            self._nll_null = self._nll(self._nll_null, self._null_pdf, data, reuse)
            attached = RooArgSet(list(self._nll_null.getVariables()))
            attached.assign(self._null)
            attached.assign(RooArgSet(as_list(nullPOI)))
            null = float(self._nll_null.getVal())
            self._nll_alt = self._nll(self._nll_alt, self._alt_pdf, data, reuse)
            RooArgSet(list(self._nll_alt.getVariables())).assign(self._alt)
            alt = float(self._nll_alt.getVal())
        if not reuse:
            self._nll_null = self._nll_alt = None
        if self._detailed:
            rows = [("nullNLL", "null NLL", null), ("altNLL", "alternate NLL", alt)]
            self._output = _values("detailedOut_SLRTS", rows)
        return null - alt

    def _nll(self, nll: Any, pdf: Any, data: Any, reuse: bool) -> Any:
        """The likelihood, made the first time - and given the new data after that."""
        if nll is None or not reuse:
            return pdf.createNLL(data, RooCmdArg("CloneData", False),
                                 RooCmdArg("Constrain", RooArgSet(list(pdf.getParameters(data)))),
                                 RooCmdArg("GlobalObservables", self._glob),
                                 RooCmdArg("ConditionalObservables", self._cond))  # fmt: skip
        nll.setData(data, False)
        return nll


def _values(name: str, rows: list[tuple[str, str, float]]) -> RooArgSet:
    """A named set of variables holding ``rows``' values: a statistic's detailed output."""
    from ..roofit.variables import RooRealVar

    return RooArgSet([RooRealVar(n, t, v) for n, t, v in rows], name)


class RatioOfProfiledLikelihoodsTestStat(TestStatistic):
    """The null's profile likelihood ratio less the alternate's - or, without subtracting the
    maxima, the two conditional minima's difference."""

    def __init__(self, nullPdf: Any, altPdf: Any, altPOI: Any = None) -> None:
        self._null = ProfileLikelihoodTestStat(nullPdf)
        self._alt = ProfileLikelihoodTestStat(altPdf)
        self._alt_poi = RooArgSet(as_list(altPOI)).snapshot() if altPOI is not None else RooArgSet()
        self._subtract = True
        self._detailed = False
        self._output: Any = None

    def EnableDetailedOutput(self, flag: bool = True) -> None:
        self._detailed = bool(flag)
        self._both("EnableDetailedOutput", self._detailed)

    def GetDetailedOutput(self) -> Any:
        return self._output

    def _both(self, method: str, *args: Any) -> None:
        for one in (self._null, self._alt):
            getattr(one, method)(*args)

    def SetReuseNLL(self, flag: bool) -> None:
        self._both("SetReuseNLL", flag)

    def SetMinimizer(self, name: str) -> None:
        self._both("SetMinimizer", name)

    def SetStrategy(self, strategy: int) -> None:
        self._both("SetStrategy", strategy)

    def SetTolerance(self, tolerance: float) -> None:
        self._both("SetTolerance", tolerance)

    def SetPrintLevel(self, level: int) -> None:
        self._both("SetPrintLevel", level)

    def SetConditionalObservables(self, items: Any) -> None:
        self._both("SetConditionalObservables", items)

    def SetGlobalObservables(self, items: Any) -> None:
        self._both("SetGlobalObservables", items)

    def SetSubtractMLE(self, subtract: bool) -> None:
        self._subtract = bool(subtract)

    @staticmethod
    def SetAlwaysReuseNLL(flag: bool) -> None:
        ProfileLikelihoodTestStat.SetAlwaysReuseNLL(flag)

    def GetVarName(self) -> str:
        return "log(L(#mu_{1},#hat{#nu}_{1}) / L(#mu_{0},#hat{#nu}_{0}))"

    def ProfiledLikelihood(self, data: Any, poi: Any, pdf: Any) -> float:
        kind = 0 if self._subtract else 2
        for one in (self._null, self._alt):
            if pdf is one.GetPdf():
                return one.EvaluateProfileLikelihood(kind, data, poi)
        log(None, ERROR, "InputArguments", "RatioOfProfiledLikelihoods::ProfileLikelihood - "
            "invalid pdf used for computing the profiled likelihood - return NaN")  # fmt: skip
        return float("nan")

    def Evaluate(self, data: Any, nullParamsOfInterest: Any) -> float:
        kind = 0 if self._subtract else 2
        null = self._null.EvaluateProfileLikelihood(kind, data, nullParamsOfInterest)
        null_set = self._null.GetDetailedOutput()
        alt = self._alt.EvaluateProfileLikelihood(kind, data, self._alt_poi)
        self._output = None
        if self._detailed:
            rows = [(f"nullprof_{v.GetName()}", f"{v.GetTitle()} for null", v.getVal())
                    for v in (null_set or ())]  # fmt: skip
            rows += [(f"altprof_{v.GetName()}", f"{v.GetTitle()} for null", v.getVal())
                     for v in (self._alt.GetDetailedOutput() or ())]  # fmt: skip
            self._output = _values("", rows)
        return null - alt


class NumEventsTestStat(TestStatistic):
    """The number of events - their weight, if weighted; a counting model's one event's sum."""

    def __init__(self, pdf: Any = None) -> None:
        self._pdf = pdf

    def GetTestStatistic(self) -> Any:
        return self._pdf

    def GetVarName(self) -> str:
        return "Number of events"

    def Evaluate(self, data: Any, paramsOfInterest: Any = None) -> float:
        if data.isWeighted():
            return float(data.sumEntries())
        if self._pdf is None or self._pdf.canBeExtended():
            return float(data.numEntries())
        if data.numEntries() == 1:
            return float(sum(one.getVal() for one in data.get(0)))
        from ..roofit import cout

        cout.write("Data set is invalid\n")
        return 0.0


class MaxLikelihoodEstimateTestStat(TestStatistic):
    """A parameter's maximum-likelihood estimate - ``-1`` if the fit fails four times over."""

    def __init__(self, pdf: Any = None, parameter: Any = None) -> None:
        from ..fit.defaults import default

        self._pdf, self._parameter = pdf, parameter
        self._cond = RooArgSet()
        self._upper = True
        self._minimizer = ""
        self._strategy = int(default("Strategy"))
        self._print_level = int(default("PrintLevel"))

    def GetVarName(self) -> str:
        return f"Maximum Likelihood Estimate of {self._parameter.GetName()}"

    def PValueIsRightTail(self, isright: Any = None) -> Any:  # type: ignore[override]
        if isright is None:
            return self._upper
        self._upper = bool(isright)
        return None

    def SetConditionalObservables(self, items: Any) -> None:
        self._cond = RooArgSet(as_list(items))

    def Evaluate(self, data: Any, nullPOI: Any = None) -> float:
        from ..roofit import cout
        from ..roofit.fitting.minimizer import RooMinimizer

        with quieted(FATAL):
            params = RooArgSet(list(self._pdf.getParameters(data)))
            RemoveConstantParameters(params)
            nll = self._pdf.createNLL(data, RooCmdArg("CloneData", False),
                                      RooCmdArg("Constrain", params),
                                      RooCmdArg("ConditionalObservables", self._cond))  # fmt: skip
            minim = RooMinimizer(nll)
            minim.setStrategy(self._strategy)
            minim.setPrintLevel(self._print_level - 1)
            status = -1
            for tries in range(5):
                status = minim.minimize(self._minimizer, "Minimize")
                if status == 0:
                    break
                if tries > 1:
                    cout.write("    ----> Doing a re-scan first\n")
                    minim.minimize(self._minimizer, "Scan")
                if tries > 2:
                    cout.write("    ----> trying with strategy = 1\n")
                    minim.setStrategy(1)
        return -1.0 if status != 0 else float(self._parameter.getVal())
