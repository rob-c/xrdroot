"""``TSpectrum::Background``: SNIP clipping of every order, direction and smoothing, as ROOT's.

The references are ROOT 6.40.04's (``tests/data/spectrum-6.40.txt``). The
clipping is sums, halves and comparisons alone - no library function - so
it is ROOT's to the last bit on every machine, and held to that.
"""

from __future__ import annotations

from array import array

import numpy as np
import pytest

from spectrumcases import reference, tags, y64, y128
from xrdroot.pyroot import TH1F, TH2F, TCanvas, TSpectrum
from xrdroot.spectrum.background import background


@pytest.mark.parametrize("tag", tags("bg"))
def test_every_filter_order_direction_smoothing_and_compton_edge_clips_as_roots(tag):
    direction, order, smoothing, window, compton = (int(word) for word in tag.split()[1:])
    spectrum = y64()
    assert TSpectrum().Background(spectrum, 64, 6, direction, order, bool(smoothing), window,
                                  bool(compton)) is None  # fmt: skip
    assert spectrum.tolist() == reference()[tag]


def test_the_constants_are_roots_enum():
    s = TSpectrum
    assert (s.kBackOrder2, s.kBackOrder4, s.kBackOrder6, s.kBackOrder8) == (0, 1, 2, 3)
    assert (s.kBackIncreasingWindow, s.kBackDecreasingWindow) == (0, 1)
    assert [getattr(s, f"kBackSmoothing{w}") for w in range(3, 16, 2)] == list(range(3, 16, 2))


def test_a_spectrum_is_read_and_written_as_an_array_array_or_a_list_too():
    expected = reference()["bg 1 0 0 3 0"]
    for kind in (lambda values: array("d", values), list):
        spectrum = kind(y64().tolist())
        TSpectrum().Background(spectrum, 64, 6, 1, 0, False, 3, False)
        assert list(spectrum) == expected


def test_only_the_size_given_is_clipped_and_the_rest_left():
    spectrum = np.concatenate([y64(), [7.0, 8.0]])
    TSpectrum().Background(spectrum, 64, 6, 1, 0, False, 3, False)
    assert spectrum[:64].tolist() == reference()["bg 1 0 0 3 0"]
    assert spectrum[64:].tolist() == [7.0, 8.0]


@pytest.mark.parametrize(
    ("size", "iterations", "window", "message"),
    [(0, 6, 3, "Wrong Parameters"), (64, 0, 3, "Width of Clipping Window Must Be Positive"),
     (12, 6, 3, "Too Large Clipping Window"), (64, 6, 4, "Incorrect width of smoothing window")],
)  # fmt: skip
def test_arguments_root_refuses_leave_the_spectrum_and_hand_back_its_message(
    size, iterations, window, message
):
    spectrum = y64()
    assert TSpectrum().Background(spectrum, size, iterations, 1, 0, True, window, False) == message
    assert spectrum.tolist() == y64().tolist()


def test_a_filter_order_root_has_no_constant_for_clips_nothing():
    spectrum = y64()
    TSpectrum().Background(spectrum, 64, 6, 1, 7, False, 3, False)
    assert spectrum.tolist() == y64().tolist()


def test_the_smoothing_window_is_not_checked_when_there_is_no_smoothing():
    assert background(y64(), 6, True, 2, False, 4, False).tolist() == reference()["bg 1 0 0 3 0"]


def _histogram() -> TH1F:
    h = TH1F("hbg", "h", 128, 0, 256)
    for i, value in enumerate(y128()):
        h.SetBinContent(i + 1, value)
    return h


def test_the_background_of_a_histogram_is_its_red_clone_over_its_axis_range():
    h = _histogram()
    h.GetXaxis().SetRange(11, 120)
    hb = TSpectrum().Background(h, 5, "BackOrder4 BackSmoothing5 compton")
    assert [hb.GetBinContent(i + 1) for i in range(128)] == reference()["hb"]
    assert (hb.GetName(), hb.GetLineColor(), hb.GetEntries()) == ("hbg_background", 2, 110.0)


def test_the_static_background_is_a_background_by_a_spectrum_of_its_own():
    h = _histogram()
    h.GetXaxis().SetRange(11, 120)
    hb = TSpectrum.StaticBackground(h, 5, "backorder4 backsmoothing5 compton")
    assert [hb.GetBinContent(i + 1) for i in range(128)] == reference()["hb"]


def test_a_background_drawn_same_replaces_the_one_drawn_before():
    TCanvas("cbg", "", 200, 200)
    h = _histogram()
    h.Draw()
    TSpectrum().Background(h, 5, "same nosmoothing backincreasingwindow backorder6")
    second = TSpectrum().Background(h, 5, "same backsmoothing15 backorder8")
    from xrdroot.pyroot.graphics.pads import current

    drawn = [obj for obj, _ in current().primitives if obj.GetName() == "hbg_background"]
    assert drawn == [second]


def test_a_background_drawn_same_with_no_pad_is_only_drawn(monkeypatch):
    from xrdroot.pyroot.graphics import pads

    monkeypatch.setattr(pads, "_CURRENT", [None])
    hb = TSpectrum().Background(_histogram(), 5, "same")
    assert hb.GetName() == "hbg_background"


def test_only_a_one_dimensional_histogram_has_a_background_here(capsys):
    assert TSpectrum().Background(TH2F("h2bg", "", 4, 0, 1, 4, 0, 1)) is None
    assert "Error in <TSpectrum::Background>: Only implemented for 1-d histograms" in (
        capsys.readouterr().err
    )


def test_no_histogram_has_no_background():
    assert TSpectrum().Background(None) is None


def test_a_compton_edge_the_background_never_meets_again_runs_to_the_last_channel():
    from xrdroot.spectrum.edges import compton_edges

    spectrum = np.array([0.0, 0.0, 10.0, 20.0, 30.0, 40.0])
    clipped = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    scale = 4.0 / 100.0
    expected = [0.0, 0.0, scale * 10.0, scale * 30.0, scale * 60.0, scale * 100.0]
    assert compton_edges(clipped, spectrum).tolist() == expected
