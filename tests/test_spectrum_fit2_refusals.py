"""``TSpectrum2Fit``'s refusals, in ROOT's words, and its getters' smaller corners."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import xrdroot.pyroot as ROOT


def fitter():
    f = ROOT.TSpectrum2Fit(1)
    f.SetFitParameters(0, 9, 0, 9, 3, 0.5, 0, 0, 2, 0)
    return f


def peaks(**changed: float) -> list:
    """``SetPeakParameters``' fourteen arrays for one peak at (4, 5), its ridges beside it."""
    starts = {"x": 4.0, "y": 5.0, "x1": 3.0, "y1": 6.0, "amp": 50.0, "ampx": 2.0, "ampy": 1.0}
    starts.update(changed)
    out: list = []
    for kind in ("x", "y", "x1", "y1", "amp", "ampx", "ampy"):
        out += [[starts[kind]], [False]]
    return out


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((-1, 9, 0, 9, 3, 0.5, 0, 0, 2, 0), "Wrong range"),
        ((0, 9, 3, 3, 3, 0.5, 0, 0, 2, 0), "Wrong range"),
        ((0, 9, 0, 9, 0, 0.5, 0, 0, 2, 0), "Invalid number of iterations, must be positive"),
        ((0, 9, 0, 9, 3, 0.5, 0, 0, 5, 0), "Wrong power"),
    ],
)
def test_fit_settings_root_would_not_take_are_refused(capsys, arguments, message):
    f = fitter()
    f.SetFitParameters(*arguments)
    assert capsys.readouterr().err == f"Error in <TSpectrum2Fit::SetFitParameters>: {message}\n"


@pytest.mark.parametrize(
    ("sigmas", "ro", "changed", "message"),
    [
        ((0.0, 1.0), 0.0, {}, "Invalid sigma, must be > than 0"),
        ((1.0, -1.0), 0.0, {}, "Invalid sigma, must be > than 0"),
        ((1.0, 1.0), 1.5, {}, "Invalid ro, must be from region <-1,1>"),
        ((1.0, 1.0), -1.5, {}, "Invalid ro, must be from region <-1,1>"),
        ((1.0, 1.0), 0.0, {"x": 10.0}, "Invalid peak position, must be in the range fXmin, fXmax"),
        ((1.0, 1.0), 0.0, {"y": -1.0}, "Invalid peak position, must be in the range fYmin, fYmax"),
        ((1.0, 1.0), 0.0, {"x1": 9.5}, "Invalid ridge position, must be in the range fXmin, fXmax"),
        ((1.0, 1.0), 0.0, {"y1": 11.0}, "Invalid ridge position, must be in the range fYmin, "
         "fYmax"),
        ((1.0, 1.0), 0.0, {"amp": -1.0}, "Invalid peak amplitude, must be > than 0"),
        ((1.0, 1.0), 0.0, {"ampx": -1.0}, "Invalid x ridge amplitude, must be > than 0"),
        ((1.0, 1.0), 0.0, {"ampy": -1.0}, "Invalid y ridge amplitude, must be > than 0"),
    ],
)
def test_peaks_root_would_not_take_are_refused(capsys, sigmas, ro, changed, message):
    f = fitter()
    f.SetPeakParameters(sigmas[0], False, sigmas[1], False, ro, True, *peaks(**changed))
    assert capsys.readouterr().err == f"Error in <TSpectrum2Fit::SetPeakParameters>: {message}\n"


def test_a_fitter_of_no_peaks_is_refused_and_one_made_bare_has_none(capsys):
    ROOT.TSpectrum2Fit(-2)
    assert capsys.readouterr().err == (
        "Error in <TSpectrum2Fit::TSpectrum2Fit>: Invalid number of peaks, must be > than 0\n"
    )
    bare = ROOT.TSpectrum2Fit()
    assert (bare.fNPeaks, bare.GetName(), bare.GetChi()) == (0, "Spectrum2Fit", 0.0)


def test_a_fit_leaves_the_fitted_spectrum_in_the_rows_it_was_handed():
    source = np.full((10, 10), 1.0)
    source[4, 5] = 40.0
    f = fitter()
    f.SetPeakParameters(1.0, False, 1.0, False, 0.0, True, *peaks())
    before = source.copy()
    f.FitAwmi(source)
    assert not np.array_equal(source, before)
    cells = [ctypes.c_double() for _ in range(2)]
    assert f.GetSigmaX(*cells) == (cells[0].value, cells[1].value) == f.GetSigmaX()
    volumes = np.zeros(1)
    f.GetVolumes(volumes)
    assert volumes[0] > 0
