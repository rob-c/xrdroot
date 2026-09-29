"""``TSpectrumFit``'s refusals, in ROOT's words, and the fitter's smaller corners."""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from refmachine import roots
from xrdroot.spectrum.fit import _held
from xrdroot.spectrum.fitpeaks import shape

#: A spectrum of one peak on a flat floor.
SOURCE = np.array([2.0 + 50.0 * math.exp(-(d * d)) for d in ((i - 10.0) / 2.0 for i in range(21))])


def fitter(peaks: int = 1):
    f = ROOT.TSpectrumFit(peaks)
    f.SetFitParameters(0, 20, 5, 0.5, 0, 0, 2, 0)
    return f


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((-1, 20, 5, 0.5, 0, 0, 2, 0), "Wrong range"),
        ((5, 5, 5, 0.5, 0, 0, 2, 0), "Wrong range"),
        ((0, 20, 0, 0.5, 0, 0, 2, 0), "Invalid number of iterations, must be positive"),
        ((0, 20, 5, 0.0, 0, 0, 2, 0), "Invalid step coefficient alpha, must be > than 0 and <=1"),
        ((0, 20, 5, 1.5, 0, 0, 2, 0), "Invalid step coefficient alpha, must be > than 0 and <=1"),
        ((0, 20, 5, 0.5, 3, 0, 2, 0), "Wrong type of statistic"),
        ((0, 20, 5, 0.5, 0, 2, 2, 0), "Wrong optimization algorithm"),
        ((0, 20, 5, 0.5, 0, 0, 3, 0), "Wrong power"),
        ((0, 20, 5, 0.5, 0, 0, 2, 2), "Wrong order of Taylor development"),
    ],
)
def test_fit_settings_root_would_not_take_are_refused_and_left_as_they_were(
    capsys, arguments, message
):
    f = fitter()
    f.SetFitParameters(*arguments)
    assert capsys.readouterr().err == f"Error in <TSpectrumFit::SetFitParameters>: {message}\n"
    assert f._settings.xmax == 20


@pytest.mark.parametrize(
    ("sigma", "position", "amplitude", "message"),
    [
        (0.0, 10.0, 50.0, "Invalid sigma, must be > than 0"),
        (2.0, 21.0, 50.0, "Invalid peak position, must be in the range fXmin, fXmax"),
        (2.0, -1.5, 50.0, "Invalid peak position, must be in the range fXmin, fXmax"),
        (2.0, 10.0, -1.0, "Invalid peak amplitude, must be > than 0"),
    ],
)
def test_peaks_root_would_not_take_are_refused(capsys, sigma, position, amplitude, message):
    f = fitter()
    f.SetPeakParameters(sigma, False, [position], [False], [amplitude], [False])
    assert capsys.readouterr().err == f"Error in <TSpectrumFit::SetPeakParameters>: {message}\n"


def test_a_fitter_of_no_peaks_is_refused_and_has_none(capsys):
    f = ROOT.TSpectrumFit(0)
    err = capsys.readouterr().err
    assert err == (
        "Error in <TSpectrumFit::TSpectrumFit>: Invalid number of peaks, must be > than 0\n"
    )
    assert (f.fNPeaks, len(f.GetPositions()), f.GetName()) == (0, 0, "SpectrumFit")


def test_a_fit_of_nothing_free_or_of_more_than_the_channels_is_refused(capsys):
    f = fitter()
    f.SetPeakParameters(2.0, True, [10.0], [True], [50.0], [True])
    f.FitAwmi(SOURCE.copy())
    assert capsys.readouterr().err == "Error in <TSpectrumFit::FitAwmi>: All parameters are fixed\n"
    f.SetFitParameters(0, 1, 5, 0.5, 0, 0, 2, 0)
    f.SetPeakParameters(2.0, False, [1.0], [False], [50.0], [False])
    source = SOURCE.copy()
    f.FitStiefel(source)
    assert capsys.readouterr().err == (
        "Error in <TSpectrumFit::FitAwmi>: "
        "Number of fitted parameters is larger than # of fitted points\n"
    )
    assert np.array_equal(source, SOURCE)


def test_a_fit_writes_the_fitted_spectrum_back_over_its_channels_only():
    f = ROOT.TSpectrumFit(1)
    f.SetFitParameters(5, 15, 5, 0.5, 0, 0, 2, 0)
    f.SetPeakParameters(2.0, False, [10.0], [False], [50.0], [False])
    source = list(SOURCE)
    f.FitAwmi(source)
    assert source[:5] == list(SOURCE[:5]) and source[16:] == list(SOURCE[16:])
    assert source[10] != SOURCE[10]
    assert f.GetPositions()[0] == pytest.approx(10.0, abs=0.01)


def test_the_getters_hand_back_what_they_put_by_reference():
    f = fitter()
    f.SetPeakParameters(2.0, False, [10.0], [False], [50.0], [False])
    f.FitAwmi(SOURCE.copy())
    sigma, error = ctypes.c_double(), ctypes.c_double()
    assert f.GetSigma(sigma, error) == (sigma.value, error.value) == f.GetSigma()
    assert f.GetTailParameters() == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert len(f.GetBackgroundParameters()) == 6


def test_each_kind_of_parameter_is_held_where_root_holds_it():
    assert [_held("amp", v, (0, 9)) for v in (-1.0, 2.0)] == [0.0, 2.0]
    assert [_held("pos", v, (0, 9)) for v in (-1.0, 4.5, 10.0)] == [0.0, 4.5, 9.0]
    assert [_held("sigma", v, (0, 9)) for v in (0.0, 2.0)] == [0.001, 2.0]
    assert [_held("b", v, (0, 9)) for v in (-0.0001, 0.0001, 0.5)] == [-0.001, 0.001, 0.5]
    assert _held("t", -3.0, (0, 9)) == -3.0


#: What ROOT printed for two fits of :data:`SOURCE`: chi, sigma and its error, the
#: position, the amplitude and their errors, the tail's slope and its error, and
#: the fitted spectrum's peak.
FITTED = {
    "likelihood searched": [
        3.9036620849249348, 2.748106151750179, 1.0483383968960533, 9.2217586294062581,
        1.2820210593475734, 33.206692701386615, 21.861718961347854, 1.0, 0.0, 30.647582882111013,
    ],
    "slope of no tail": [
        2.1990505604585713, 2.5507850905533438, 0.8880469065151555, 9.7299395471716004,
        1.0060132411489286, 34.847949778174701, 18.3628371736507, 1.0, 0.0, 34.459513571394567,
    ],
}  # fmt: skip


def fitted(label: str, f) -> None:
    source = SOURCE.copy()
    f.SetPeakParameters(2.5, False, [9.0], [False], [30.0], [False])
    f.FitAwmi(source)
    tails = f.GetTailParameters()
    peak = (f.GetPositions(), f.GetPositionsErrors(), f.GetAmplitudes(), f.GetAmplitudesErrors())
    got = [f.GetChi(), *f.GetSigma(), *(a[0] for a in peak), tails[2], tails[3], source[10]]
    assert got == roots(FITTED[label], rel=1e-12)


def test_a_spectrum_the_model_draws_exactly_is_fitted_where_it_starts():
    channels = np.arange(21, dtype=np.float64)
    exact = shape(channels, np.array([50.0]), np.array([10.0]), 2.0, 0.0, 0.0, 1.0, (0.0,) * 3)
    for method in ("FitAwmi", "FitStiefel"):
        f = fitter()
        f.SetPeakParameters(2.0, False, [10.0], [False], [50.0], [False])
        source = exact.copy()
        getattr(f, method)(source)
        # Beyond three sigma the peak is 0, and ROOT's chi counts an empty channel as one.
        assert (f.GetChi(), f.GetPositions()[0], f.GetSigma()[0]) == (10 / 18, 10.0, 2.0)
        assert np.array_equal(source, exact)


def test_a_likelihood_whose_best_step_is_the_first_takes_it():
    f = ROOT.TSpectrumFit(1)
    f.SetFitParameters(0, 20, 3, 0.5, f.kFitOptimMaxLikelihood, f.kFitAlphaOptimal, 2, 0)
    fitted("likelihood searched", f)


def test_a_slope_fitted_under_no_tail_does_not_move_and_keeps_its_error():
    f = fitter()
    f.SetFitParameters(0, 20, 3, 0.5, 0, 0, 2, 0)
    f.SetTailParameters(0.0, True, 1.0, False, 0.0, True)
    fitted("slope of no tail", f)
