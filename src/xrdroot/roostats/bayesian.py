"""``BayesianCalculator``: a credible interval from the posterior of one parameter of interest.

The posterior is the likelihood times the prior, integrated over the
nuisance parameters. RooStats first finds the likelihood's minimum along the
parameter of interest - ``BrentMinimizer1D`` - to keep ``exp(-nll)`` in
range, then, without nuisance parameters, the posterior is a RooFit formula
of the likelihood of the model times the prior; with them it is integrated
numerically at each value. The interval's ends are where the cumulative
posterior crosses the tail fractions - GSL's Brent solver on GSL's QAGS - or,
for the shortest interval, from a scan.
"""

from __future__ import annotations

import math
from typing import Any

from ..numerics.kronrod import DBL_MIN
from ..numerics.minimize1d import BrentMinimizer1D
from ..numerics.roots import brent_root
from ..roofit.cmdargs import RooCmdArg
from ..roofit.collections import RooArgSet, as_list
from ..roofit.messages import ERROR, INFO, WARNING, log
from ..roofit.printing import g
from .bayesfuncs import CdfFunction, Functor, Posterior, PosteriorFunction
from .intervals import Named, SimpleInterval
from .utils import RemoveConstantParameters

__all__ = ["BayesianCalculator"]


class BayesianCalculator(Named):
    """Bayesian credible intervals in one parameter of interest."""

    def __init__(self, data: Any = None, model: Any = None, *args: Any) -> None:
        super().__init__("")
        self._data, self._pdf, self._prior = data, None, None
        self._poi, self._nuisance = RooArgSet(), RooArgSet()
        self._conditional, self._global = RooArgSet(), RooArgSet()
        self._size, self._left = 0.05, 0.5
        self._precision, self._scan_bins, self._iterations = 0.00005, -1, 0
        self._integration = ""
        self._nuisance_pdf: Any = None
        self._clear()
        if model is not None and hasattr(model, "GetPdf"):
            self.SetModel(model)
        elif model is not None:  # (data, pdf, poi, prior, nuisance)
            self._pdf, self._poi = model, RooArgSet(as_list(args[0]))
            self._prior = args[1] if len(args) > 1 else None
            if len(args) > 2 and args[2]:
                self._nuisance = RooArgSet(as_list(args[2]))
                RemoveConstantParameters(self._nuisance)

    def _clear(self) -> None:
        self._log_like: Any = None
        self._likelihood: Any = None
        self._integrated: Any = None
        self._posterior_function: Any = None
        self._nll_min = 0.0
        self._lower = self._upper = 0.0
        self._valid = False
        self._scanned = 0

    def SetModel(self, model: Any) -> None:
        self._pdf, self._prior = model.GetPdf(), model.GetPriorPdf()
        self._poi = RooArgSet(as_list(model.GetParametersOfInterest() or ()))
        self._nuisance = RooArgSet(as_list(model.GetNuisanceParameters() or ()))
        self._conditional = RooArgSet(as_list(model.GetConditionalObservables() or ()))
        self._global = RooArgSet(as_list(model.GetGlobalObservables() or ()))
        RemoveConstantParameters(self._nuisance)
        self._clear()

    def SetData(self, data: Any) -> None:
        self._data = data
        self._clear()

    def SetPriorPdf(self, prior: Any) -> None:
        self._prior = prior
        self._clear()

    def SetParameters(self, poi: Any) -> None:
        self._poi = RooArgSet(as_list(poi))

    def SetTestSize(self, size: float) -> None:
        self._size = float(size)
        self._valid = False

    def SetConfidenceLevel(self, cl: float) -> None:
        self.SetTestSize(1.0 - float(cl))

    def Size(self) -> float:
        return self._size

    def ConfidenceLevel(self) -> float:
        return 1.0 - self._size

    def SetLeftSideTailFraction(self, fraction: float) -> None:
        self._left = float(fraction)

    def SetShortestInterval(self) -> None:
        self._left = -1.0

    def SetBrfPrecision(self, precision: float) -> None:
        self._precision = float(precision)

    def SetScanOfPosterior(self, bins: int = 100) -> None:
        self._scan_bins = int(bins)

    def SetNumIters(self, iterations: int) -> None:
        self._iterations = int(iterations)

    def SetIntegrationType(self, kind: str) -> None:
        self._integration = str(kind).upper()

    def ForceNuisancePdf(self, pdf: Any) -> None:
        self._nuisance_pdf = pdf

    # -- the posterior ------------------------------------------------------------

    def _nll_of(self, pdf: Any) -> Any:
        constrained = RooArgSet(self._pdf.getParameters(self._data))
        RemoveConstantParameters(constrained)
        return pdf.createNLL(self._data, RooCmdArg("Constrain", constrained),
                             RooCmdArg("ConditionalObservables", self._conditional),
                             RooCmdArg("GlobalObservables", self._global))  # fmt: skip

    def GetPosteriorFunction(self) -> Any:
        """The posterior - not normalised - as a function of the parameter of interest."""
        if self._integrated is not None:
            return self._integrated
        if self._pdf is None or len(self._poi) != 1:
            text = "missing pdf model" if self._pdf is None else (
                "missing parameter of interest" if not len(self._poi) else
                "current implementation works only on 1D intervals")  # fmt: skip
            log(self, ERROR, "InputArguments", f"BayesianCalculator::GetPosteriorPdf - {text}")
            return None
        self._log_like = self._nll_of(self._pdf)
        value = float(self._log_like.getVal())
        poi = self._poi[0]
        log(self, INFO, "Eval", f"BayesianCalculator::GetPosteriorFunction :  nll value {g(value)} "
            f"poi value = {g(poi.getVal())}")  # fmt: skip
        functor = Functor(self._log_like, [poi])
        minimizer = BrentMinimizer1D()
        minimizer.SetFunction(lambda x: functor([x]), poi.getMin(), poi.getMax())
        self._nll_min = minimizer.FValMinimum() if minimizer.Minimize(100, 1e-3, 1e-3) else 0.0
        log(self, INFO, "Eval", "BayesianCalculator::GetPosteriorFunction : minimum of NLL vs POI "
            f"for POI =  {g(poi.getVal())} min NLL = {g(self._nll_min)}")  # fmt: skip
        self._integrated = self._integrated_posterior(poi)
        return self._integrated

    def _integrated_posterior(self, poi: Any) -> Any:
        """The posterior with the nuisance parameters integrated out: by RooFit - or with none
        to integrate - or by the adaptive integrator."""
        if not len(self._nuisance) or "ROOFIT" in self._integration:
            return self._roofit_posterior()
        if self._integration and self._integration not in ("ADAPTIVE", ""):
            from ..errors import UnsupportedFeatureError

            raise UnsupportedFeatureError(
                f"BayesianCalculator integrates the nuisance parameters numerically here; its "
                f"{self._integration} integration is not here yet"
            )
        self._posterior_function = PosteriorFunction(
            self._log_like, poi, list(self._nuisance), self._prior, 1.0, self._nll_min,
            self._iterations)  # fmt: skip
        return Posterior(f"posteriorfunction_from_{self._log_like.GetName()}",
                         self._posterior_function, poi)  # fmt: skip

    def _roofit_posterior(self) -> Any:
        """``exp(-nll + min)`` of the model times the prior - integrated over the nuisance
        parameters, if any, by RooFit."""
        from ..roofit.functions import RooFormulaVar
        from ..roofit.pdfs.prodpdf import RooProdPdf

        both = self._pdf
        if self._prior is not None:
            name = f"product_{self._pdf.GetName()}_{self._prior.GetName()}"
            both = RooProdPdf(name, "", [self._pdf, self._prior])
        self._log_like = self._nll_of(both)
        name = f"likelihood_times_prior_{both.GetName()}"
        found = RooFormulaVar(name, f"exp(-@0+{g(self._nll_min)})", [self._log_like])
        if not len(self._nuisance):
            return found
        return found.createIntegral(list(self._nuisance))

    def GetPosteriorPdf(self) -> Any:
        from ..roofit.pdfs.generic import RooGenericPdf

        posterior = self.GetPosteriorFunction()
        if posterior is None:
            return None
        return RooGenericPdf(f"{self._name}_posteriorPdf_{posterior.GetName()}", "@0", [posterior])

    # -- the interval -------------------------------------------------------------

    def GetInterval(self) -> Any:
        """The central (or left-sided, or shortest) credible interval."""
        if self._valid:
            log(self, WARNING, "Eval", "BayesianCalculator::GetInterval - recomputing interval "
                "for the same CL and same model")  # fmt: skip
        if not len(self._poi):
            log(self, ERROR, "Eval", "BayesianCalculator::GetInterval - no parameter of interest "
                "is set ")  # fmt: skip
            return None
        poi = self._poi[0]
        self.GetPosteriorFunction()
        if self._left < 0:
            self._shortest()
        else:
            lower, upper = self._left * self._size, 1.0 - (1.0 - self._left) * self._size
            if self._scan_bins > 0:
                self._from_scan(lower, upper)
            else:
                self._from_cdf(lower, upper)
                if not self._valid:
                    self._scan_bins = 100
                    log(self, WARNING, "Eval", "BayesianCalculator::GetInterval - computing "
                        "integral from cdf failed - do a scan in 100 nbins ")  # fmt: skip
                    self._from_scan(lower, upper)
        if not self._valid:
            self._lower, self._upper = 1.0, 0.0
            log(self, ERROR, "Eval", "BayesianCalculator::GetInterval - cannot compute a valid "
                "interval - return a dummy [1,0] interval")  # fmt: skip
        else:
            log(self, INFO, "Eval", "BayesianCalculator::GetInterval - found a valid interval : "
                f"[{g(self._lower)} , {g(self._upper)} ]")  # fmt: skip
        interval = SimpleInterval(f"BayesianInterval_a{self._name}", poi, self._lower,
                                  self._upper, self.ConfidenceLevel())  # fmt: skip
        interval.SetTitle("SimpleInterval from BayesianCalculator")
        return interval

    def _from_cdf(self, lower_cut: float, upper_cut: float) -> None:
        """``ComputeIntervalFromCdf``: the cumulative posterior's crossings, by GSL's Brent."""
        self._valid = False
        log(self, INFO, "InputArguments", "BayesianCalculator:GetInterval Compute the interval "
            "from the posterior cdf ")  # fmt: skip
        poi = self._poi[0]
        if self.GetPosteriorFunction() is None:
            log(self, ERROR, "InputArguments", "BayesianCalculator::GetInterval() cannot make "
                "posterior Function ")  # fmt: skip
            return
        cdf = CdfFunction(self._log_like, [poi, *self._nuisance], self._prior, self._nll_min)
        if cdf.error:
            log(self, ERROR, "Eval", "BayesianCalculator: Numerical error computing CDF integral - "
                "try a different method ")  # fmt: skip
            return
        lower = poi.getMin() if lower_cut <= 0 else self._crossing(cdf, lower_cut, poi.getMin(),
                                                                   "lower limit")  # fmt: skip
        if lower is None:
            return
        upper = poi.getMax() if upper_cut >= 1.0 else self._crossing(cdf, upper_cut, lower,
                                                                     "upper limit")  # fmt: skip
        if upper is None:
            return
        self._lower, self._upper = lower, upper
        self._valid = True

    def _crossing(self, cdf: Any, cut: float, start: float, side: str) -> Any:
        """Where the cumulative posterior reaches ``cut``, from ``start`` up - or ``None``."""
        cdf.SetOffset(cut)
        precision = self._precision
        ok, root, _ = brent_root(cdf, start, self._poi[0].getMax(), 200, precision, precision)
        if cdf.error:
            log(self, WARNING, "Eval", "BayesianCalculator: Numerical error integrating the  CDF  "
                " ")  # fmt: skip
        if not ok:
            log(self, ERROR, "NumericIntegration", "BayesianCalculator::GetInterval - Error from "
                f"root finder when searching {side} !")  # fmt: skip
            return None
        return root

    # -- the scanned posterior ----------------------------------------------------

    def _approximate(self) -> int:
        """``ApproximatePosterior``: the posterior as the clone of a ``TF1`` of ``nbins``
        points - which ROOT's streamer saves as the posterior at the ``nbins + 1`` edges,
        interpolated linearly between them - taking the posterior's place, as RooStats does."""
        bins = self._scan_bins if self._scan_bins > 0 else 100
        if self._scanned >= bins:
            return self._scanned
        if self.GetPosteriorFunction() is None:
            return 0
        log(self, INFO, "Eval", f"BayesianCalculator - scan posterior function in nbins = {bins}")
        self._scanned = bins
        poi = self._poi[0]
        exact = self._integrated
        low, high = float(poi.getMin()), float(poi.getMax())
        saved = _saved(lambda x: _at(exact, poi, x), low, high, bins)
        self._integrated = Posterior(f"{exact.GetName()}_approx", saved, poi)
        return bins

    def _posterior_at(self, x: float) -> float:
        return _at(self._integrated, self._poi[0], x)

    def _integral(self, a: float, b: float) -> float:
        """``TF1::Integral(a, b, 0)``: GSL's QAGS at the default tolerances."""
        from ..numerics.qags import qags

        return qags(self._posterior_at, a, b, 1e-9, 1e-9, 1000)[0]

    def _from_scan(self, lower_cut: float, upper_cut: float) -> None:
        """``ComputeIntervalFromApproxPosterior``: ``TF1::GetQuantiles`` of the scan."""
        scanned = self._approximate()
        if not scanned:
            return
        npx = max(scanned, 4)
        poi = self._poi[0]
        low, high = float(poi.getMin()), float(poi.getMax())
        cuts = [lower_cut, upper_cut]
        self._lower, self._upper = _quantiles(self._integral, low, high, npx, cuts)
        self._valid = True

    def _histogram(self) -> list[float]:
        """``TF1::GetHistogram``'s contents: the posterior at each of the scan's bin centres."""
        poi = self._poi[0]
        npx = self._approximate()
        low, width = float(poi.getMin()), (float(poi.getMax()) - float(poi.getMin())) / npx
        return [self._posterior_at(low + (i + 0.5) * width) for i in range(npx)]

    def _shortest(self) -> None:
        """``ComputeShortestInterval``: the highest bins of the scan, down to the probability."""
        log(self, INFO, "Eval", "BayesianCalculator - computing shortest interval with CL = "
            f"{g(1.0 - self._size)}")  # fmt: skip
        if not self._approximate():
            return
        bins = self._histogram()
        poi = self._poi[0]
        low, high = float(poi.getMin()), float(poi.getMax())
        width = (high - low) / len(bins)
        order = sorted(range(len(bins)), key=lambda i: -bins[i])
        norm = math.fsum(bins)
        total = actual = 0.0
        upper, lower = low, high
        for index in order:
            # ROOT sorts the bins but reads the probability one bin down - the underflow's zero
            # for the first - from ``TH1::GetArray``, which starts at the underflow.
            p = (bins[index - 1] if index else 0.0) / norm
            total += p
            if total > 1.0 - self._size:
                actual = total - p
                break
            lower, upper = min(lower, low + index * width), max(upper, low + (index + 1) * width)
        if lower < upper:
            self._lower, self._upper = lower, upper
            if abs(actual - (1.0 - self._size)) > 0.1 * (1.0 - self._size):
                log(self, WARNING, "Eval", "BayesianCalculator::ComputeShortestInterval - actual "
                    f"interval CL = {g(actual)} differs more than 10% from desired CL value - "
                    f"must increase nbins {len(bins)} to an higher value ")  # fmt: skip
        else:
            log(self, ERROR, "Eval", f"BayesianCalculator::ComputeShortestInterval {len(bins)} "
                "bins are not sufficient ")  # fmt: skip
        self._valid = True

    def GetPosteriorPlot(self, norm: bool = False, precision: float = 0.01) -> Any:
        """The posterior over the parameter's range, the interval filled beneath it."""
        self.GetPosteriorFunction()
        if self._scan_bins > 0:
            self._approximate()
        posterior = self.GetPosteriorPdf() if norm else self._integrated
        if posterior is None:
            return None
        if not self._valid:
            self.GetInterval()
        return _posterior_plot(self, posterior, precision)

    def GetMode(self) -> float:
        bins = self._histogram()
        poi = self._poi[0]
        width = (float(poi.getMax()) - float(poi.getMin())) / len(bins)
        return float(poi.getMin()) + (bins.index(max(bins)) + 0.5) * width


def _at(function: Any, poi: Any, x: float) -> float:
    """``function`` with the parameter of interest set to ``x``."""
    poi.setVal(x)
    return float(function.getVal())


def _saved(function: Any, low: float, high: float, npx: int) -> Any:
    """``TF1::Save`` and ``TF1::GetSave``: ``function`` at ``npx + 1`` equidistant points,
    and a straight line between the two about each ``x`` - nothing outside the range."""
    dx = (high - low) / npx
    values = [function(low + dx * i) for i in range(npx + 1)]

    def interpolated(x: float) -> float:
        if x < low or x > high:
            return 0.0
        bin_ = min(npx - 1, int((x - low) / dx))
        xlow = low + bin_ * dx
        xup, ylow, yup = xlow + dx, values[bin_], values[bin_ + 1]
        return ((xup * ylow - xlow * yup) + x * (yup - ylow)) / dx

    return interpolated


def _quantiles(integral: Any, low: float, high: float, npx: int, probs: list[float]) -> list[float]:
    """``TF1::GetQuantiles``: each bin's integral, a parabola through each bin's cumulative
    integral, and the ``x`` each probability falls at."""
    dx = (high - low) / npx
    cumulative = [0.0]
    for i in range(npx):
        cumulative.append(cumulative[-1] + abs(integral(low + i * dx, low + i * dx + dx)))
    total = cumulative[-1]
    cumulative = [0.0] + [c / total for c in cumulative[1:]]
    alpha, beta, gamma = [], [], []
    for i in range(npx):
        x0 = low + dx * i
        r2 = cumulative[i + 1] - cumulative[i]
        r1 = integral(x0, x0 + 0.5 * dx) / total
        c = (2 * r2 - 4 * r1) / (dx * dx)
        alpha.append(x0)
        beta.append(r2 / dx - c * dx)
        gamma.append(2 * c)
    return [_quantile(cumulative, alpha, beta, gamma, npx, high, r) for r in probs]


def _quantile(cumulative: list[float], alpha: list[float], beta: list[float], gamma: list[float],
              npx: int, high: float, r: float) -> float:  # fmt: skip
    import bisect

    bin_ = max(bisect.bisect_right(cumulative, r) - 1, 0)
    if bin_ == npx:
        return high
    while bin_ < npx - 1 and _equal(cumulative[bin_ + 1], r) and _equal(cumulative[bin_ + 2], r):
        bin_ += 1
    rr = r - cumulative[bin_]
    if rr == 0.0:
        # ROOT adds ``dx`` here if the bin's upper cumulative is ``r`` too - which the search,
        # finding the last point not above ``r``, never leaves it.
        return alpha[bin_]
    return alpha[bin_] + _within_bin(beta[bin_], gamma[bin_], rr)


def _within_bin(beta: float, gamma: float, rr: float) -> float:
    """How far into its bin ``r`` falls: the parabola's root - or, flat, the line's."""
    import numpy as np

    with np.errstate(all="ignore"):  # C's division: a flat bin's zero slope gives inf or nan
        fac = float(np.float64(-2.0 * gamma * rr) / beta / beta)
    if fac != 0 and fac <= 1:
        return (-beta + math.sqrt(beta * beta + 2 * gamma * rr)) / gamma
    return rr / beta if beta != 0.0 else 0.0


def _equal(a: float, b: float) -> bool:
    """``TMath::AreEqualRel(a, b, 1e-12)``."""
    return a == b or abs(a - b) <= 0.5e-12 * (abs(a) + abs(b)) or abs(a - b) < DBL_MIN


def _posterior_plot(calculator: Any, posterior: Any, precision: float) -> Any:
    """``GetPosteriorPlot``'s frame: the interval filled grey behind the whole posterior."""
    poi = calculator._poi[0]
    plot = poi.frame()
    plot.SetTitle(f'Posterior probability of parameter "{poi.GetName()}"')
    ranged = RooCmdArg("Range", calculator._lower, calculator._upper, False)
    posterior.plotOn(plot, ranged, RooCmdArg("VLines"), RooCmdArg("DrawOption", "F"),
                     RooCmdArg("MoveToBack"), RooCmdArg("FillColor", 920),
                     RooCmdArg("Precision", precision))  # fmt: skip
    posterior.plotOn(plot)
    plot.GetYaxis().SetTitle("posterior function")
    return plot
