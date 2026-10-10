"""``ROOT.TCandle``: its settings for every candle, and one candle's numbers from a slice."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.plot.candle import SETTINGS


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    kept = dict(SETTINGS)
    yield from fresh(tmp_path)
    SETTINGS.update(kept)


def test_the_statics_set_what_every_candle_is_drawn_with():
    ROOT.TCandle.SetWhiskerRange(2.0)
    ROOT.TCandle.SetBoxRange(0.6)
    ROOT.TCandle.SetScaledCandle(True)
    ROOT.TCandle.SetScaledViolin(False)
    assert (SETTINGS["whisker_range"], SETTINGS["box_range"]) == (2.0, 0.6)
    assert ROOT.TCandle.IsCandleScaled() and not ROOT.TCandle.IsViolinScaled()


def test_an_option_is_read_into_digits_each_asked_about_as_root_asks():
    made = ROOT.TCandle("candley2")
    assert made.GetOption() == 112321 + made.kHorizontal and made.GetDrawOption() == "candley2"
    assert made.IsHorizontal() and not made.IsVertical()
    assert made.IsOption(made.kMedianNotched) and not made.IsOption(made.kMedianLine)
    assert made.IsOption(made.kWhisker15) and made.IsOption(made.kPointsOutliers)
    assert not made.IsOption(made.kHistoViolin) and not made.IsOption(made.kNoOption)
    made.SetOption(0)
    assert made.IsOption(made.kNoOption) and made.IsVertical()
    assert ROOT.TCandle().ParseOption("violinx(3000010)") == 3000010
    with pytest.raises(UnsupportedFeatureError, match="raw points"):
        ROOT.TCandle(1.0, 0.5, 3, [1.0, 2.0, 3.0])


def test_a_candle_of_a_slice_works_out_its_numbers_once_and_takes_those_set_by_hand():
    h = ROOT.TH1D("slice", "slice", 20, 0, 20)
    for at, count in enumerate([0, 0, 1, 4, 10, 20, 30, 20, 10, 4, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]):
        h.SetBinContent(at + 1, count)
    made = ROOT.TCandle(3.0, 1.0, h)
    made.ParseOption("candlex1")
    assert made.GetQ1() > made.GetMedian() > made.GetQ3()  # ROOT's Q1 is the box's upper end
    assert made.GetQ2() == made.GetMedian() and made.GetMean() == pytest.approx(h.GetMean())
    made.SetMean(1.0)
    made.SetQ1(9.0)
    made.SetQ3(2.0)
    made.SetQ2(5.0)
    assert (made.GetMean(), made.GetQ1(), made.GetQ3(), made.GetQ2()) == (1.0, 9.0, 2.0, 5.0)
    made.SetMedian(4.0)
    assert made.GetMedian() == 4.0
    empty = ROOT.TCandle()
    empty.SetHistogram(ROOT.TH1D("none", "none", 2, 0, 1))
    empty.SetAxisPosition(2.0)
    empty.SetCandleWidth(0.5)
    empty.SetHistoWidth(0.5)
    empty.SetLog(0, 0, 0)
    assert empty.GetMean() == 0.0 and ROOT.TCandle().GetMedian() == 0.0


def test_a_histogram_draws_its_candles_and_violins_through_the_canvas(tmp_path):
    pytest.importorskip("matplotlib")
    from xrdroot.pyroot.graphics import hook

    hook.install()
    h = ROOT.TH2I("h", "sin", 6, 0, 6, 30, -2, 2)
    rng = ROOT.TRandom3(3)
    for i in range(6):
        for _ in range(100):
            h.Fill(i + 0.5, rng.Gaus(0.3 * i, 0.3))
    canvas = ROOT.TCanvas("c", "c", 300, 300)
    canvas.Divide(2, 1)
    canvas.cd(1)
    h.Draw("candlex2")
    canvas.cd(2)
    h.Draw("violiny(112000000)")
    canvas.SaveAs(str(tmp_path / "candles.png"))
    assert (tmp_path / "candles.png").stat().st_size > 2000
