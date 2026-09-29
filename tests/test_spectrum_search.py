"""``TSpectrum::SearchHighRes`` and ``Search``: peaks found as ROOT 6.40 finds them.

The references are ROOT 6.40.04's (``tests/data/spectrum-6.40.txt``). The
search deconvolves by a Gaussian whose ``exp`` is truncated to an integer,
and smooths with a Markov chain of ``exp`` weights - the C library's, which
on ROOT's machine are ROOT's to the bit. Elsewhere a weight may round the
other way in the last place; five Gold iterations carry that forward
without growing it much, and a peak's centroid by as little: ``1e-10``
relative holds the deconvolved spectrum and the positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from refmachine import roots
from spectrumcases import reference, tags, y128
from xrdroot.pyroot import TH1F, TH3F, TCanvas, TSpectrum, TVirtualFitter
from xrdroot.spectrum.extend import left_slope
from xrdroot.spectrum.search import Found, _centroid, _insert, search_high_res


@pytest.mark.parametrize("tag", tags("sd"))
def test_every_search_finds_roots_peaks_and_leaves_roots_deconvolved_spectrum(tag):
    sigma, remove, markov = float(tag.split()[1]), tag.split()[2] == "1", tag.split()[3] == "1"
    s, dest = TSpectrum(), np.zeros(128)
    found = s.SearchHighRes(y128(), dest, 128, sigma, 5, remove, 5, markov, 3)
    positions = reference()[tag.replace("sd", "sp")]
    assert (found, s.GetNPeaks()) == (len(positions), len(positions))
    assert s.GetPositionX()[:found].tolist() == roots(positions, rel=1e-10)
    assert dest.tolist() == roots(reference()[tag], rel=1e-10, abs=1e-300)


@pytest.mark.parametrize(
    ("sigma", "threshold", "remove", "markov", "window", "size", "message"),
    [(0.5, 5, False, False, 3, 128, "Invalid sigma, must be greater than or equal to 1"),
     (2, 0, False, False, 3, 128, "Invalid threshold, must be positive and less than 100"),
     (2, 100, False, False, 3, 128, "Invalid threshold, must be positive and less than 100"),
     (102.3, 5, False, False, 3, 128, "Too large sigma"),
     (2, 5, False, True, 0, 128, "Averaging window must be positive"),
     (2, 5, True, False, 3, 20, "Too large clipping window")],
)  # fmt: skip
def test_a_search_root_refuses_says_why_and_finds_nothing(
    capsys, sigma, threshold, remove, markov, window, size, message
):
    dest = np.full(size, 7.0)
    assert TSpectrum().SearchHighRes(y128(), dest, size, sigma, threshold, remove, 3, markov,
                                     window) == 0  # fmt: skip
    assert capsys.readouterr().err == f"Error in <TSpectrum::SearchHighRes>: {message}\n"
    assert dest.tolist() == [7.0] * size


def test_a_search_that_fills_every_place_for_peaks_says_so(capsys):
    s = TSpectrum(2)
    assert s.SearchHighRes(y128(), np.zeros(128), 128, 2, 5, False, 5, False, 3) == 2
    assert s.GetPositionX().tolist() == reference()["sp 2 0 0"][:2]
    assert "Warning in <TSpectrum::SearchHighRes>: Peak buffer full" in capsys.readouterr().err


def test_a_search_smoothing_nothing_finds_nothing_and_writes_nothing():
    s, dest = TSpectrum(), np.full(64, 3.0)
    assert s.SearchHighRes(np.zeros(64), dest, 64, 2, 5, False, 5, True, 3) == 0
    assert dest.tolist() == [3.0] * 64
    assert search_high_res(np.zeros(64), 2, 5, False, 5, True, 3, 10) == Found()


def test_search1highres_is_searchhighres():
    assert TSpectrum.Search1HighRes is TSpectrum.SearchHighRes


def test_the_padding_continues_a_falling_start_and_never_a_rising_one():
    assert left_slope(np.array([9.0, 7.0, 5.0, 3.0]), 1.0) == -2.0
    assert left_slope(np.array([1.0, 2.0, 3.0]), 1.0) == 0.0
    assert left_slope(np.array([9.0, 7.0]), 0.5) == 0.0


def test_a_centroid_is_kept_inside_the_spectrum():
    assert _centroid(np.array([0.0, 1.0, 1.0, 1.0]), 1, 2, 5) == 0.0
    assert _centroid(np.array([1.0, 1.0, 1.0, 0.0]), 2, -9, 5) == 4.0


def test_a_peak_lower_than_every_kept_one_is_dropped_when_every_place_is_taken():
    positions = [1.0, 2.0]
    _insert(positions, 3.0, np.array([0.0, 5.0, 4.0, 1.0]), 0, 2)
    assert positions == [1.0, 2.0]
    _insert(positions, 3.0, np.array([0.0, 5.0, 4.0, 9.0]), 0, 2)
    assert positions == [3.0, 1.0]


def _histogram(name: str = "hs") -> TH1F:
    h = TH1F(name, "h", 128, 0, 256)
    for i, value in enumerate(y128()):
        h.SetBinContent(i + 1, value)
    return h


def test_a_histogram_search_finds_roots_peaks_at_bin_centres_with_their_contents():
    s = TSpectrum()
    assert s.Search(_histogram(), 2, "goff", 0.1) == 3
    assert s.GetPositionX()[:3].tolist() == reference()["hx"]
    assert s.GetPositionY()[:3].tolist() == reference()["hy"]


def test_a_search_over_an_axis_range_with_its_own_sigma_and_no_background_or_smoothing():
    s, h = TSpectrum(), _histogram()
    h.GetXaxis().SetRange(11, 120)
    found = s.Search(h, 0, "goff nobackground nomarkov", 0.02)
    assert s.GetPositionX()[:found].tolist() == roots(reference()["hrx"], rel=1e-10)
    assert s.GetPositionY()[:found].tolist() == reference()["hry"]


def test_the_static_search_is_a_search_by_a_spectrum_of_its_own():
    assert TSpectrum.StaticSearch(_histogram(), 2, "goff", 0.1) == 3


def test_a_search_marks_its_peaks_on_the_histogram_and_draws_it():
    TCanvas("csearch", "", 200, 200)
    s, h = TSpectrum(), _histogram()
    assert s.Search(h, 2, "L,", 0.1) == 3
    s.Search(h, 2, "L", 0.1)
    markers = [f for f in h.GetListOfFunctions() if f.ClassName() == "TPolyMarker"]
    assert len(markers) == 1
    marker = markers[0]
    assert (marker.GetN(), marker.GetMarkerStyle(), marker.GetMarkerColor()) == (3, 23, 2)
    assert marker.GetMarkerSize() == pytest.approx(1.3)
    assert marker.GetX().tolist() == reference()["hx"]
    assert h.GetListOfFunctions().FindObject("TPolyMarker") is marker
    from xrdroot.pyroot.graphics.pads import current

    assert [(obj, how) for obj, how in current().primitives] == [(h, "l")]


def test_a_search_told_nodraw_marks_the_histogram_without_drawing_it():
    TCanvas("cnodraw", "", 200, 200)
    h = _histogram()
    TSpectrum().Search(h, 2, "nodraw", 0.1)
    from xrdroot.pyroot.graphics.pads import current

    assert current().primitives == []
    assert h.GetListOfFunctions().FindObject("TPolyMarker").GetN() == 3


def test_a_search_finding_nothing_marks_nothing():
    h = TH1F("hflat", "", 64, 0, 64)
    assert TSpectrum().Search(h, 2, "", 0.1) == 0
    assert h.GetListOfFunctions().FindObject("TPolyMarker") is None


def test_a_threshold_outside_zero_to_one_is_taken_as_five_per_cent(capsys):
    assert TSpectrum().Search(_histogram(), 2, "goff", 1.5) == TSpectrum().Search(
        _histogram(), 2, "goff", 0.05
    )
    assert (
        "Warning in <TSpectrum::Search>: threshold must 0<threshold<1, threshold=0.05 assumed"
        in (capsys.readouterr().err)
    )


def test_a_histogram_of_more_than_two_dimensions_is_refused_and_one_of_two_finds_nothing(capsys):
    s = TSpectrum()
    assert s.Search(TH3F("h3s", "", 2, 0, 1, 2, 0, 1, 2, 0, 1)) == 0
    assert "Error in <TSpectrum::Search>: Only implemented for 1-d and 2-d histograms" in (
        capsys.readouterr().err
    )
    from xrdroot.pyroot import TH2F

    assert s.Search(TH2F("h2s", "", 8, 0, 1, 8, 0, 1)) == 0
    assert s.Search(None) == 0


def test_a_spectrum_prints_its_peaks_as_root_does(capsys):
    s = TSpectrum()
    s.Search(_histogram(), 2, "goff", 0.1)
    s.Print()
    assert capsys.readouterr().out == (
        "\nNumber of positions = 3\n x[0] = 61, y[0] = 526.14\n x[1] = 75, y[1] = 343.375\n"
        " x[2] = 181, y[2] = 203.982\n"
    )


def test_a_spectrum_is_roots_named_object_with_its_statics_and_resolution():
    s = TSpectrum(0, 3.5)
    assert (s.GetName(), s.GetTitle(), s.ClassName()) == (
        "Spectrum",
        "Miroslav Morhac peak finder",
        "TSpectrum",
    )
    assert (len(s.GetPositionX()), s.fResolution, s.GetHistogram(), s.GetNPeaks()) == (
        1,
        3.5,
        None,
        0,
    )
    s.SetResolution(0.5)
    assert s.fResolution == 1.0
    TSpectrum.SetAverageWindow(5)
    TSpectrum.SetDeconIterations(7)
    assert (TSpectrum.fgAverageWindow, TSpectrum.fgIterations) == (5, 7)
    TSpectrum.SetAverageWindow()
    TSpectrum.SetDeconIterations()
    assert (TSpectrum.fgAverageWindow, TSpectrum.fgIterations) == (3, 3)


def test_the_fitter_root_makes_room_in_is_the_latest_fits(monkeypatch):
    from xrdroot.pyroot.core import fitters

    monkeypatch.setitem(fitters.LATEST, "result", None)
    assert TVirtualFitter.Fitter(None, 40) is None
