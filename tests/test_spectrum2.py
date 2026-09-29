"""``TSpectrum2``: two-dimensional clipping, smoothing, deconvolution and peaks, as ROOT 6.40's.

The references are ROOT 6.40.04's (``tests/data/spectrum-6.40.txt``), for a
24 by 18 plane. Clipping and unboosted deconvolution are arithmetic alone
and ROOT's to the bit everywhere; the Markov chain's ``exp``, a boost's
``pow`` and the search's Gaussian are the C library's, ROOT's to the bit on
ROOT's machine, and elsewhere held to ``1e-10`` relative - a last-place
difference in a weight, carried through a few hundred products and ten
iterations of quotients, moves an answer by far less.
"""

from __future__ import annotations

import numpy as np
import pytest

from refmachine import roots
from spectrumcases import reference, tags, z2
from xrdroot.pyroot import TH1F, TH2F, TCanvas, TSpectrum2
from xrdroot.spectrum.markov2 import smooth_markov2
from xrdroot.spectrum.search2 import _centroid

NX, NY = 24, 18
RESPONSE = [[10.0 / (1 + (i - 1) * (i - 1) + j * j) if (i < 4 and j < 3) else 0.0
             for j in range(NY)] for i in range(NX)]  # fmt: skip


@pytest.mark.parametrize("tag", tags("b2"))
def test_every_direction_and_filter_clips_a_plane_as_roots(tag):
    direction, filtering = (int(word) for word in tag.split()[1:])
    rows = [row.copy() for row in z2()]
    assert TSpectrum2().Background(rows, NX, NY, 4, 3, direction, filtering) is None
    assert np.ravel(rows).tolist() == reference()[tag]


def test_the_constants_are_roots_enum():
    s = TSpectrum2
    assert (s.kBackIncreasingWindow, s.kBackDecreasingWindow) == (0, 1)
    assert (s.kBackSuccessiveFiltering, s.kBackOneStepFiltering) == (0, 1)


@pytest.mark.parametrize(
    ("sizes", "iterations", "message"),
    [
        ((0, NY), (4, 3), "Wrong parameters"),
        ((NX, NY), (0, 3), "Width of Clipping Window Must Be Positive"),
        ((NX, 6), (4, 3), "Too Large Clipping Window"),
    ],
)
def test_clipping_root_refuses_leaves_the_plane(sizes, iterations, message):
    plane = z2()
    assert TSpectrum2().Background(plane, *sizes, *iterations, 1, 0) == message
    assert plane.tolist() == z2().tolist()


def test_markov_smoothing_of_a_plane_is_roots():
    plane = z2()
    assert TSpectrum2().SmoothMarkov(plane, NX, NY, 3) is None
    assert plane.ravel().tolist() == roots(reference()["m2"], rel=1e-10)


def test_markov_smoothing_of_a_plane_refuses_no_window_and_leaves_nothing_alone():
    plane = z2()
    assert TSpectrum2().SmoothMarkov(plane, NX, NY, 0) == "Averaging Window must be positive"
    assert smooth_markov2(np.zeros((3, 3)), 2) is None
    assert TSpectrum2().SmoothMarkov(plane, 0, NY, 3) is None


@pytest.mark.parametrize("repetitions", [1, 2])
def test_gold_deconvolution_of_a_plane_is_roots(repetitions):
    plane = z2()
    assert TSpectrum2().Deconvolution(plane, RESPONSE, NX, NY, 10, repetitions, 1.5) is None
    expected = reference()[f"g2 {repetitions}"]
    assert plane.ravel().tolist() == (expected if repetitions == 1 else roots(expected, rel=1e-10))


def test_a_plane_deconvolution_refuses_what_root_refuses():
    s, plane = TSpectrum2(), z2()
    assert s.Deconvolution(plane, RESPONSE, 0, NY, 10, 1, 1.0) == "Wrong parameters"
    assert (
        s.Deconvolution(plane, RESPONSE, NX, NY, 0, 1, 1.0)
        == "Number of iterations must be positive"
    )
    assert (
        s.Deconvolution(plane, RESPONSE, NX, NY, 10, 0, 1.0)
        == "Number of repetitions must be positive"
    )
    assert s.Deconvolution(plane, np.zeros((NX, NY)), NX, NY, 10, 1, 1.0) == "Zero response data"
    assert plane.tolist() == z2().tolist()


@pytest.mark.parametrize("tag", tags("s2"))
def test_every_plane_search_finds_roots_peaks_and_leaves_roots_deconvolved_plane(tag):
    remove, markov = (word == "1" for word in tag.split()[1:])
    s, dest = TSpectrum2(), np.zeros((NX, NY))
    found = s.SearchHighRes(z2(), dest, NX, NY, 1.5, 10, remove, 4, markov, 3)
    xs, ys = reference()[tag.replace("s2", "x2")], reference()[tag.replace("s2", "y2")]
    assert found == len(xs)
    assert s.GetPositionX()[:found].tolist() == roots(xs, rel=1e-10)
    assert s.GetPositionY()[:found].tolist() == roots(ys, rel=1e-10)
    assert dest.ravel().tolist() == roots(reference()[tag], rel=1e-10, abs=1e-300)


@pytest.mark.parametrize(
    ("sigma", "threshold", "markov", "window", "message"),
    [(0.5, 10, False, 3, "Invalid sigma, must be greater than or equal to 1"),
     (1.5, 100, False, 3, "Invalid threshold, must be positive and less than 100"),
     (102.3, 10, False, 3, "Too large sigma"),
     (1.5, 10, True, 0, "Averaging window must be positive")],
)  # fmt: skip
def test_a_plane_search_root_refuses_says_why(capsys, sigma, threshold, markov, window, message):
    assert TSpectrum2().SearchHighRes(z2(), np.zeros((NX, NY)), NX, NY, sigma, threshold, False,
                                      4, markov, window) == 0  # fmt: skip
    assert capsys.readouterr().err == f"Error in <TSpectrum2::SearchHighRes>: {message}\n"


def test_a_plane_search_keeps_only_as_many_peaks_as_it_has_places_for():
    s = TSpectrum2(1)
    assert s.SearchHighRes(z2(), np.zeros((NX, NY)), NX, NY, 1.5, 10, False, 4, False, 3) == 1
    assert s.GetPositionX().tolist() == roots(reference()["x2 0 0"][:1], rel=1e-10)


def test_a_plane_search_smoothing_nothing_finds_nothing_and_writes_nothing():
    dest = np.full((NX, NY), 2.0)
    assert TSpectrum2().SearchHighRes(np.zeros((NX, NY)), dest, NX, NY, 1.5, 10, False, 4, True,
                                      3) == 0  # fmt: skip
    assert dest.tolist() == np.full((NX, NY), 2.0).tolist()


def test_a_plane_centroid_is_kept_inside_the_plane():
    assert _centroid(np.array([0.0, 1.0, 1.0, 1.0]), 1, 2, 5) == 0.0
    assert _centroid(np.array([1.0, 1.0, 1.0, 0.0]), 2, -9, 5) == 4.0


def _histogram(name: str = "h2s") -> TH2F:
    h = TH2F(name, "h2", NX, 0, 48, NY, -9, 9)
    for i, row in enumerate(z2()):
        for j, value in enumerate(row):
            h.SetBinContent(i + 1, j + 1, value)
    return h


def test_a_plane_histogram_search_finds_roots_peaks_at_bin_centres_and_prints_them(capsys):
    s = TSpectrum2()
    assert s.Search(_histogram(), 1.5, "goff nomarkov", 0.1) == 2
    assert s.GetPositionX()[:2].tolist() == reference()["hx2"]
    assert s.GetPositionY()[:2].tolist() == reference()["hy2"]
    s.Print()
    assert capsys.readouterr().out == (
        "\nNumber of positions = 2\n x[0] = 15, y[0] = -3.5\n x[1] = 33, y[1] = 3.5\n"
    )
    assert TSpectrum2.StaticSearch(_histogram(), 1.5, "goff nomarkov nobackground", 0.1) >= 1


def test_a_plane_search_marks_and_draws_the_histogram_with_the_option_as_given():
    TCanvas("csearch2", "", 200, 200)
    h = _histogram()
    assert TSpectrum2().Search(h, 1.5, "COL nomarkov", 0.1) == 2
    assert h.GetListOfFunctions().FindObject("TPolyMarker").GetX().tolist() == reference()["hx2"]
    from xrdroot.pyroot.graphics.pads import current

    assert [(obj, how) for obj, how in current().primitives] == [(h, "COL nomarkov")]
    assert TSpectrum2().Search(TH2F("hflat2", "", 8, 0, 1, 8, 0, 1), 1.5, "col", 0.1) == 0


def test_only_a_plane_histogram_is_searched_or_clipped(capsys):
    s = TSpectrum2()
    assert s.Search(TH1F("h1s2", "", 8, 0, 1)) == 0
    assert s.Background(TH1F("h1b2", "", 8, 0, 1)) is None
    err = capsys.readouterr().err
    assert "Error in <TSpectrum2::Search>: Must be a 2-d histogram" in err
    assert "Error in <TSpectrum2::Background>: Only implemented for 2-d histograms" in err
    assert (s.Search(None), s.Background(None)) == (0, None)


def test_the_background_of_a_plane_histogram_is_its_clone_over_its_axis_ranges():
    h = _histogram()
    h.GetXaxis().SetRange(3, 20)
    hb = TSpectrum2().Background(h, 4, 3, "BackOneStepFiltering BackIncreasingWindow")
    rows = np.array([[h.GetBinContent(i + 3, j + 1) for j in range(NY)] for i in range(18)])
    TSpectrum2().Background(rows, 18, NY, 4, 3, 0, 1)
    got = [[hb.GetBinContent(i + 3, j + 1) for j in range(NY)] for i in range(18)]
    assert got == rows.astype(np.float32).astype(np.float64).tolist()
    assert (hb.GetName(), hb.GetEntries(), hb.GetBinContent(1, 1)) == ("h2s_background", 324.0, 0.0)
    same = TSpectrum2.StaticBackground(h, 4, 3, "backonestepfiltering backincreasingwindow")
    assert same.GetBinContent(5, 5) == hb.GetBinContent(5, 5)


def test_a_plane_background_drawn_same_replaces_the_one_drawn_before():
    TCanvas("cbg2", "", 200, 200)
    h = _histogram()
    h.Draw("col")
    TSpectrum2().Background(h, 4, 3, "same")
    second = TSpectrum2().Background(h, 4, 3, "same")
    from xrdroot.pyroot.graphics.pads import current

    assert [obj for obj, _ in current().primitives if obj.GetName() == "h2s_background"] == [second]


def test_roots_copy_for_the_chain_stops_at_the_end_of_its_working_row():
    """A plane more than eight times as wide as it is tall copies past the end of ROOT's
    working row - which in C++ is memory not its own - and here stops there."""
    from xrdroot.spectrum.prepare2 import through_rows

    values = np.arange(40.0).reshape(20, 2)
    read, base = through_rows(values, values + 100)
    assert read.tolist() == [[0.0, 0.0]] * 20
    assert base.tolist() == (values + 100).tolist()
