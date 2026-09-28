"""``MethodBase::GetEfficiency``: the signal efficiency at a background efficiency.

TMVA counts, for every cut on a classifier's output in 10000 steps, the
fraction of signal and of background above it; draws straight lines through
those (``TSpline1``); finds, by Brent's method (``RootFinder``), the cut
giving each of 100 signal efficiencies; and so builds the background
efficiency as a function of the signal efficiency. The efficiencies TMVA
prints - at a background efficiency of 0.01, 0.10 and 0.30 - are read off
that curve by scanning it in 1000 steps, and its average is the ``area``
TMVA ranks by. The training-sample efficiencies are found the same way, but
with the cuts found from the *test* sample's signal efficiency, as TMVA's
own code does it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..hist import Histogram
from . import hists
from .evaluation import NBINS_HIGH, NBINS_MVA_OUTPUT, NBINS_ROC, norm_hist
from .log import Logger
from .pdf import spline1

__all__ = ["Efficiencies", "Spline1", "root"]

#: The machine epsilon, as ``std::numeric_limits<double>::epsilon()``.
EPS = float(np.finfo(np.float64).eps)


class Spline1:
    """``TMVA::TSpline1`` over the bin centres and contents of a histogram's ``TGraph``."""

    def __init__(self, histogram: Histogram) -> None:
        self.x = histogram.axes[0].root_centers()[1:-1]
        self.y = hists.bins(histogram)[1:-1]

    def __call__(self, x: Any) -> Any:
        return spline1(self.x, self.y, np.asarray(x, dtype=np.float64))


def _brent_step(a: float, b: float, c: float, fa: float, fb: float, fc: float, state: Any) -> Any:
    """One step of ``RootFinder::Root``: the next step, by interpolation or by bisection."""
    tol, m, d, e, same = state
    if abs(e) < tol or abs(fa) <= abs(fb):
        return m, m
    s = fb / fa
    if same:
        p, q = 2 * m * s, 1 - s
    else:
        q, r = fa / fc, fb / fc
        p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
        q = (q - 1) * (r - 1) * (s - 1)
    q, p = (-q, p) if p > 0 else (q, -p)
    if 2 * p < min(3 * m * q - abs(tol * q), abs(e * q)):
        return p / q, d
    return m, m


def root(function: Callable[[float], float], low: float, high: float, target: float) -> float:
    """``RootFinder::Root``: where ``function`` reaches ``target``, by Brent's method.

    With no change of sign between the ends, TMVA warns and answers 1.
    """
    a, b = low, high
    fa, fb = function(a) - target, function(b) - target
    if fb * fa > 0:
        Logger("RootFinder").warning(
            f"<Root> initial interval w/o root: (a={a:g}, b={b:g}), refValue = {target:g}"
        )
        return 1.0
    fc, c, d, e = fb, 0.0, 0.0, 0.0
    same = False
    for _ in range(101):
        if (fb < 0 and fc < 0) or (fb > 0 and fc > 0):
            same, c, fc = True, a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            same = True
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 0.5 * 2.2204460492503131e-16 * abs(b)
        m = 0.5 * (c - b)
        if fb == 0 or abs(m) <= tol:
            return b
        d, e = _brent_step(a, b, c, fa, fb, fc, (tol, m, d, e, same))
        a, fa = b, fb
        b += d if abs(d) > tol else (tol if m > 0 else -tol)
        fb = function(b) - target
    Logger("RootFinder").warning("<Root> maximum iterations (100) reached before convergence")
    return b


def _cumulative(values: Any, weights: Any, low: float, high: float, positive: bool) -> Any:
    """The weight above (or below) each of the 10000 cuts, as ``GetEfficiency``'s loop adds it."""
    top = np.trunc((values - low) / (high - low) * NBINS_HIGH).astype(np.int64) + 1
    if positive:
        keep = top <= NBINS_HIGH
        counts = np.bincount(np.maximum(top[keep], 1), weights[keep], NBINS_HIGH + 1)
        return np.cumsum(counts[::-1])[::-1][1:]
    keep = top >= 1
    counts = np.bincount(np.minimum(top[keep], NBINS_HIGH), weights[keep], NBINS_HIGH + 1)
    return np.concatenate(([0.0], np.cumsum(counts[1:])))[:-1]


def _efficiency_hist(name: str, title: str, low: float, high: float, contents: Any) -> Histogram:
    made = hists.book(name, title, NBINS_HIGH, low, high, kind="D")
    peak = max(EPS, float(np.max(contents)))
    hists.set_bins(made, np.concatenate(([0.0], contents / peak, [0.0])), entries=len(contents))
    return made


def _scan(curve: Callable[[Any], Any], reference: float) -> float:
    """The signal efficiency where ``curve`` - background against signal - crosses ``reference``."""
    effs = (np.arange(1, 1001) - 0.5) / np.float32(1000)
    effb = curve(effs)
    previous_s = previous_b = 0.0
    for eff_s, eff_b in zip(effs, effb):
        if (eff_b - reference) * (previous_b - reference) <= 0:
            return 0.5 * (float(eff_s) + previous_s)
        previous_s, previous_b = float(eff_s), float(eff_b)
    return 0.5 * (float(effs[-1]) + previous_s)


@dataclass
class Efficiencies:
    """One classifier's efficiency curves, worked out once and read at each efficiency asked for."""

    testvar: str
    xmin: float
    xmax: float
    positive: bool
    histograms: dict[str, Histogram] = field(default_factory=dict)

    def _value_for_root(self, spline: Spline1) -> Callable[[float], float]:
        """``GetValueForRoot``: the signal efficiency at a cut, forced to 1 and 0 at the ends."""

        def value(cut: float) -> float:
            if cut - self.xmin < 1.0e-5:
                return 1.0 if self.positive else 0.0
            if self.xmax - cut < 1.0e-5:
                return 0.0 if self.positive else 1.0
            return float(spline(cut))

        return value

    def _curve(self, spline_b: Spline1, name: str, title: str) -> tuple[Histogram, Any]:
        """Background efficiency at each of 100 signal efficiencies, with its histogram."""
        made = hists.book(name, title, NBINS_ROC, 0.0, 1.0, kind="D")
        centres = made.axes[0].root_centers()[1:-1]
        function = self._value_for_root(self.spline_s)
        effb = np.array([float(spline_b(root(function, self.xmin, self.xmax, s))) for s in centres])
        hists.set_bins(made, np.concatenate(([0.0], effb, [0.0])), entries=NBINS_ROC)
        return made, effb

    def test(self, values: Any, signal: Any, weights: Any, low: float, high: float) -> None:
        """The test sample's curves: ``MVA_EFF_S``, ``MVA_EFF_B``, ``effBvsS`` and the rest."""
        values = np.asarray(values, dtype=np.float32).astype(np.float64)
        weights = np.asarray(weights, dtype=np.float32).astype(np.float64)
        self.signal_total = float(np.sum(weights[signal]))
        for side, mask, label in (("S", signal, "signal"), ("B", ~signal, "background")):
            counts = _cumulative(values[mask], weights[mask], low, high, self.positive)
            name = f"{self.testvar}_eff{side}"
            title = f"{self.testvar} ({label})"
            self.histograms[f"MVA_EFF_{side}"] = _efficiency_hist(name, title, low, high, counts)
        self.spline_s = Spline1(self.histograms["MVA_EFF_S"])
        spline_b = Spline1(self.histograms["MVA_EFF_B"])
        curve, effb = self._curve(spline_b, f"{self.testvar}_effBvsS", self.testvar)
        self.histograms["MVA_EFF_BvsS"] = curve
        self.histograms["MVA_REJ_BvsS"] = self._companion("_rejBvsS", 1.0 - effb)
        inverse = np.where(effb > EPS, 1.0 / np.where(effb > EPS, effb, 1.0), 0.0)
        self.histograms["MVA_INVEFF_BvsS"] = self._companion("_invBeffvsSeff", inverse)
        self.curve = Spline1(curve)

    def _companion(self, suffix: str, contents: Any) -> Histogram:
        made = hists.book(self.testvar + suffix, self.testvar, NBINS_ROC, 0.0, 1.0, kind="D")
        return hists.set_bins(made, np.concatenate(([0.0], contents, [0.0])), entries=NBINS_ROC)

    def efficiency(self, reference: float) -> tuple[float, float]:
        """The test signal efficiency at background efficiency ``reference``, and its error."""
        eff_s = _scan(self.curve, np.float32(reference))
        error = np.sqrt(eff_s * (1.0 - eff_s) / self.signal_total) if self.signal_total > 0 else 0.0
        return eff_s, float(error)

    def area(self) -> float:
        """The average background rejection over the signal efficiencies: TMVA's ``effArea``."""
        effs = (np.arange(1, 1001) - 0.5) / np.float32(1000)
        return float(np.sum(1.0 - self.curve(effs))) / 1000

    def train(self, values: Any, signal: Any, weights: Any, low: float, high: float) -> None:
        """The training sample's curves, ``MVA_TRAIN_S`` and the rest, as TMVA makes them."""
        values = np.asarray(values, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        top = self.xmax + 0.00001
        for side, mask, label in (("S", signal, "signal"), ("B", ~signal, "background")):
            made = hists.book(
                f"{self.testvar}_Train_{side}",
                f"{self.testvar}_Train_{side}",
                NBINS_MVA_OUTPUT,
                self.xmin,
                top,
                kind="D",
            )
            made.sumw2(True)
            made.fill(values[mask], weight=weights[mask])
            norm_hist(made)
            self.histograms[f"MVA_TRAIN_{side}"] = made
            counts = _cumulative(values[mask], weights[mask], low, high, self.positive)
            name, title = f"{self.testvar}_trainingEff{side}", f"{self.testvar} ({label})"
            self.histograms[f"MVA_TRAINEFF_{side}"] = _efficiency_hist(
                name, title, low, high, counts
            )
        spline_b = Spline1(self.histograms["MVA_TRAINEFF_B"])
        curve, effb = self._curve(spline_b, f"{self.testvar}_trainingEffBvsS", self.testvar)
        self.histograms["EFF_BVSS_TR"] = curve
        rejection = hists.book(
            f"{self.testvar}_trainingRejBvsS", self.testvar, NBINS_ROC, 0.0, 1.0, kind="D"
        )
        self.histograms["REJ_BVSS_TR"] = hists.set_bins(
            rejection, np.concatenate(([0.0], 1.0 - effb, [0.0])), entries=NBINS_ROC
        )
        self.train_curve = Spline1(curve)

    def training_efficiency(self, reference: float) -> float:
        """The training signal efficiency at background efficiency ``reference``."""
        return _scan(self.train_curve, np.float32(reference))
