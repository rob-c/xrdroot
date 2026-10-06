"""``TSpectrumFit``: Morhac's AWMI and Stiefel peak fits, number for number ROOT 6.40's.

Four Gaussian peaks of sigma 2.5, over a background and a ripple of
``0.5 * (7i mod 5)`` so that no fit comes out exact, fitted by every kind of
statistic, step and power ROOT has; the references are ROOT's own
(:mod:`spectrumfitrefs`), bit for bit on ROOT's machine.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from refmachine import roots
from spectrumfitrefs import CASES, EDGES
from xrdroot.spectrum.fit import PeakSetup, fit
from xrdroot.spectrum.fitengine import FitSettings

#: The fitted channels, 0 to 119.
N = 120

#: Where the fits start: positions a channel or two out, amplitudes rounded.
POSITIONS, AMPLITUDES = [19.0, 47.0, 60.0, 91.0], [60.0, 150.0, 40.0, 90.0]


def gaus(x: float, mean: float, sigma: float) -> float:
    """``TMath::Gaus(x, mean, sigma, kTRUE)``, as ROOT rounds it."""
    arg = (x - mean) / sigma
    return math.exp((-0.5 * arg) * arg) / (2.50662827463100024 * sigma)


def spectrum(background: bool = True) -> np.ndarray:
    """The spectrum ROOT fitted, added up as the macro added it."""
    out = []
    for i in range(N):
        v = 5.0 + 0.02 * i if background else 0.0
        v = v + 0.5 * ((i * 7) % 5)
        for mean, area in zip((20.3, 45.7, 61.2, 90.0), (400.0, 900.0, 300.0, 600.0), strict=False):
            v += area * gaus(i, mean, 2.5)
        out.append(v)
    return np.array(out)


def setup_for(background: bool, tails: bool, fixmode: int) -> PeakSetup:
    """The peaks as a case sets them: 1 fixes a position and an amplitude, 2 sigma."""
    setup = PeakSetup(list(POSITIONS), list(AMPLITUDES), [False, fixmode == 1, False, False],
                      [False, False, fixmode == 1, False])  # fmt: skip
    setup.init["sigma"], setup.fixed["sigma"] = 3.0, fixmode == 2
    if background:
        setup.init.update(a0=3.0, a1=0.01, a2=0.0)
        setup.fixed.update(a0=False, a1=False, a2=False)
    if tails:
        setup.init.update(t=0.1, b=1.5, s=0.05)
        setup.fixed.update(t=False, b=False, s=False)
    return setup


def found(result) -> list[float]:
    """A fit's numbers in the order the references keep them."""
    values, errors = result.values, result.errors
    out = [result.chi, values["sigma"][0], errors["sigma"][0]]
    for k in range(4):
        for kind in ("pos", "amp", "area"):
            out += [values[kind][k], errors[kind][k]]
    return out


@pytest.mark.parametrize("label", list(CASES))
def test_each_way_of_fitting_finds_what_root_finds(label):
    (stiefel, statistic, optim, power, taylor, background, tails, fixmode, iterations,
     alpha), tolerance, expected = CASES[label]  # fmt: skip
    settings = FitSettings(0, N - 1, iterations, alpha, statistic, optim, power, taylor)
    result = fit(spectrum(background), setup_for(background, tails, fixmode), settings, stiefel)
    assert found(result) == roots(expected, rel=tolerance)


def printed(fitter, source, peaks: int) -> list[list[float]]:
    """What the edge macro printed, asked of ``fitter`` as a script asks ROOT's."""
    cells = [ctypes.c_double() for _ in range(14)]
    fitter.GetSigma(cells[0], cells[1])
    fitter.GetBackgroundParameters(*cells[2:8])
    fitter.GetTailParameters(*cells[8:14])
    rows = [[fitter.GetChi()], [c.value for c in cells[0:2]], [c.value for c in cells[2:8]],
            [c.value for c in cells[8:14]]]  # fmt: skip
    arrays = (fitter.GetPositions(), fitter.GetPositionsErrors(), fitter.GetAmplitudes(),
              fitter.GetAmplitudesErrors(), fitter.GetAreas(), fitter.GetAreasErrors())  # fmt: skip
    rows += [[float(a[k]) for a in arrays] for k in range(peaks)]
    return [*rows, [float(source[k]) for k in (0, 20, 46, N - 1)]]


#: The edge cases' tolerances, measured as the cases' are; the sloping tail's is
#: large because its ``b`` sits at the 0.001 it is held to.
EDGE_TOLERANCE = {"slope": 4e-4}

#: The peaks the edge cases start from, the third between two channels.
EDGE_POSITIONS = [19.0, 47.0, 60.5, 91.0]

NO, YES, ZERO = [False] * 4, [True] * 4, [0.0] * 4


def edge(label: str, fitter, source, peaks: int = 4) -> None:
    expected = EDGES[label]
    tolerance = EDGE_TOLERANCE.get(label, 1e-12)
    assert printed(fitter, source, peaks) == [roots(row, rel=tolerance) for row in expected]


def test_needle_thin_peaks_fit_only_where_they_fall_on_a_channel():
    source = spectrum()
    f = ROOT.TSpectrumFit(4)
    f.SetFitParameters(0, N - 1, 5, 0.5, f.kFitOptimChiCounts, f.kFitAlphaHalving, f.kFitPower2,
                       f.kFitTaylorOrderSecond)  # fmt: skip
    f.SetPeakParameters(0.00005, True, EDGE_POSITIONS, YES, AMPLITUDES, NO)
    f.SetTailParameters(0.1, True, 0.001, True, 0.0, True)
    f.SetBackgroundParameters(3.0, False, 0.0, True, 0.0, True)
    f.FitAwmi(source)
    edge("needles", f, source)


def test_a_step_under_no_peaks_is_left_with_the_error_it_had():
    source = spectrum()
    f = ROOT.TSpectrumFit(4)
    f.SetFitParameters(0, N - 1, 5, 0.5, 0, 0, 2, 0)
    f.SetPeakParameters(2.0, True, EDGE_POSITIONS, YES, ZERO, YES)
    f.SetTailParameters(0.0, True, 1.0, True, 0.1, False)
    f.SetBackgroundParameters(3.0, False, 0.0, True, 0.0, True)
    f.FitAwmi(source)
    edge("stepless", f, source)


def test_steps_searched_in_millionths_all_improve_and_run_out():
    source = spectrum()
    f = ROOT.TSpectrumFit(4)
    f.SetFitParameters(0, N - 1, 2, 0.000001, 0, f.kFitAlphaOptimal, 2, 0)
    f.SetPeakParameters(3.0, False, EDGE_POSITIONS, NO, AMPLITUDES, NO)
    f.FitAwmi(source)
    edge("tiny alpha", f, source)


def test_a_fitter_of_no_peaks_fits_the_background_alone():
    source = spectrum()
    f = ROOT.TSpectrumFit()
    f.SetFitParameters(0, N - 1, 20, 0.5, 0, 0, 2, 0)
    f.SetBackgroundParameters(3.0, False, 0.01, False, 0.0, True)
    f.FitStiefel(source)
    edge("background", f, source, peaks=0)


def test_a_tail_whose_slope_starts_at_its_floor_is_fitted_by_stiefel():
    source = spectrum()
    f = ROOT.TSpectrumFit(4)
    f.SetFitParameters(0, N - 1, 10, 0.5, 0, 0, 2, 0)
    f.SetPeakParameters(3.0, False, EDGE_POSITIONS, NO, AMPLITUDES, NO)
    f.SetTailParameters(0.2, False, 0.001, False, 0.0, True)
    f.FitStiefel(source)
    edge("slope", f, source)
