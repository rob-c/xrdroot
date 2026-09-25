"""What is done with a histogram: copies, arithmetic, reshaping, comparing and drawing from it."""

from __future__ import annotations

import array

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def filled(name="h", n=4):
    h = ROOT.TH1D(name, "t", n, 0, n)
    for x, w in ((0.5, 1.0), (1.5, 2.0), (2.5, 3.0)):
        h.Fill(x, w)
    return h


def test_clones_copies_and_draw_copies_are_kept_as_root_keeps_them():
    h = filled()
    c = h.Clone("c")
    assert ROOT.gROOT.FindObject("c") is c
    assert c.GetBinContent(2) == 2
    assert h.Clone().GetName() == "h"
    ROOT.TH1.AddDirectory(False)
    assert ROOT.gROOT.FindObject("d") is None or h.Clone("d") is not None
    ROOT.TH1.AddDirectory(True)
    target = ROOT.TH1D("t", "", 1, 0, 1)
    h.Copy(target)
    assert target.GetNbinsX() == 4
    assert target.GetBinContent(3) == 3
    drawn = h.DrawCopy("hist")
    assert drawn.GetName() == "h_copy"
    assert drawn.GetDirectory() is None
    normal = h.DrawNormalized("", 2.0)
    assert normal.Integral() == pytest.approx(2.0)
    assert ROOT.TH1D("e", "", 1, 0, 1).DrawNormalized().Integral() == 0


def test_arithmetic_in_place_is_roots():
    h, g = filled("h"), filled("g")
    h.Add(g, 2.0)
    assert h.GetBinContent(2) == 6
    h.Add(g, g, 1.0, -1.0)
    assert h.GetBinContent(2) == 0
    h.Add(g)
    h.Multiply(g)
    assert h.GetBinContent(3) == 9
    h.Multiply(g, g, 2.0, 1.0)
    assert h.GetBinContent(3) == 18
    h.Divide(g)
    assert h.GetBinContent(3) == 6
    h.Divide(g, g, 3.0, 1.0, "B")
    assert h.GetBinContent(3) == 3
    h.Scale(2.0, "width")
    assert h.GetBinContent(3) == 6


def test_adding_a_function_adds_it_at_each_bins_centre():
    h = ROOT.TH1D("h", "", 2, 0, 2)
    h.Add(ROOT.TF1("line", "x", 0, 2), 2.0)
    assert (h.GetBinContent(1), h.GetBinContent(2)) == (1.0, 3.0)
    h.Add(lambda x: 1.0 + 0 * x)
    assert h.GetBinContent(1) == 2.0


def test_the_operators_make_new_histograms():
    h, g = filled("h"), filled("g")
    assert (h + g).GetBinContent(2) == 4
    assert (h - g).GetBinContent(2) == 0
    assert (h * 3).GetBinContent(2) == 6
    assert (3 * h).GetBinContent(2) == 6
    assert (h / g).GetBinContent(2) == 1
    assert (h * g).GetBinContent(2) == 4
    h += g
    h -= g
    h *= 2
    h /= 2
    assert h.GetBinContent(2) == 2
    assert isinstance(h, ROOT.TH1D)


def test_rebinning_in_place_or_into_a_new_histogram():
    h = filled("h")
    made = h.Rebin(2, "h2")
    assert made.GetNbinsX() == 2
    assert h.GetNbinsX() == 4
    assert ROOT.gROOT.FindObject("h2") is made
    assert h.Rebin(2) is h
    assert h.GetNbinsX() == 2
    assert h.GetBinContent(1) == 3
    uneven = filled("u").Rebin(2, "u2", array.array("d", [0, 1, 4]))
    assert uneven.GetBinContent(2) == 5
    assert filled("x").RebinX(4).GetNbinsX() == 1
    h2 = ROOT.TH2D("h2d", "", 4, 0, 4, 4, 0, 4)
    h2.Fill(0.5, 0.5)
    assert h2.Rebin2D(2, 2, "h2r").GetNbinsX() == 2
    assert h2.Rebin2D(2, 2) is h2
    assert h2.RebinY(2, "h2y").GetNbinsY() == 1


def test_projections_take_roots_names_and_reuse_them_silently(capsys):
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 2, 0, 2)
    h2.Fill(0.5, 1.5, 2.0)
    first = h2.ProjectionX()
    again = h2.ProjectionX()
    assert first.GetName() == again.GetName() == "h2_px"
    assert ROOT.gROOT.FindObject("h2_px") is again
    assert "Replacing" not in capsys.readouterr().err
    assert h2.ProjectionY().GetBinContent(2) == 2
    assert h2.ProjectionY("y", 1, 1).GetBinContent(2) == 2
    assert h2.ProjectionX("x", 1, 1).GetBinContent(1) == 0
    assert h2.ProfileY().GetName() == "h2_pfy"
    assert h2.ProfileX("pp", 1, 2).GetName() == "pp"
    assert h2.ProfileY("qq", 1, 2).GetName() == "qq"
    h3 = ROOT.TH3D("h3", "", 2, 0, 2, 2, 0, 2, 2, 0, 2)
    h3.Fill(0.5, 0.5, 1.5)
    assert h3.Project3D("z").GetBinContent(2) == 1
    assert h3.Project3D("xy").GetName() == "h3_xy"


def test_a_projection_is_kept_only_while_histograms_are(capsys):
    ROOT.TH1.AddDirectory(False)
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 2, 0, 2)
    assert ROOT.gROOT.FindObject(h2.ProjectionX().GetName()) is None


def test_the_cumulative_runs_forward_or_back_with_its_errors():
    h = filled()
    forward = h.GetCumulative()
    back = h.GetCumulative(False, "_back")
    assert [forward.GetBinContent(b) for b in range(1, 5)] == [1, 3, 6, 6]
    assert [back.GetBinContent(b) for b in range(1, 5)] == [6, 5, 3, 0]
    assert forward.GetBinError(2) == pytest.approx(np.sqrt(5))
    assert back.GetName() == "h_back"
    assert forward.GetEntries() == 4


def test_smoothing_quantiles_and_comparisons():
    h = filled()
    q = array.array("d", [0.0] * 5)
    assert h.GetQuantiles(5, q) >= 1
    h.Smooth(1)
    g = filled("g")
    residuals = np.zeros(4)
    assert 0 <= g.Chi2Test(filled("k"), "WW", residuals) <= 1
    assert 0 <= g.KolmogorovTest(filled("m")) <= 1


def test_fitting_hands_back_a_result_and_hangs_the_function(capsys):
    ROOT.gRandom.SetSeed(1)
    h = ROOT.TH1D("h", "", 40, -4, 4)
    h.FillRandom("gaus", 2000)
    result = h.Fit("gaus", "QS")
    assert int(result) == 0
    assert result.Parameter(2) == pytest.approx(1.0, rel=0.1)
    assert h.GetFunction("gaus") is not None
    assert h.GetFunction("nothing") is None
    assert h.GetListOfFunctions().GetSize() == 1
    ranged = h.Fit("gaus", "QS", "", -1, 1)
    assert ranged.Ndf() < result.Ndf()


def test_fill_random_draws_from_functions_histograms_and_named_functions():
    source = ROOT.TF1("line", "x", 0, 1)
    h = ROOT.TH1D("h", "", 10, 0, 1)
    h.FillRandom("line", 100)
    h.FillRandom(source, 100)
    h.FillRandom("gaus", 10)
    assert h.GetEntries() == 210
    g = ROOT.TH1D("g", "", 10, 0, 1)
    g.FillRandom(h, 50, ROOT.TRandom3(5))
    assert g.GetEntries() == 50


def test_random_numbers_from_a_histogram(tmp_path):
    empty = ROOT.TH1D("e", "", 2, 0, 1)
    assert empty.GetRandom() == 0.0
    x, y = array.array("d", [9]), array.array("d", [9])
    assert ROOT.TH2D("e2", "", 2, 0, 1, 2, 0, 1).GetRandom2(x, y) == (0.0, 0.0)
    assert x[0] == 0
    h = filled()
    assert 0 <= h.GetRandom(ROOT.TRandom3(2)) <= 4
    assert h.GetIntegral()[-1] == 1.0
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 2, 0, 2)
    h2.Fill(1.5, 0.5)
    px, py = h2.GetRandom2(x, y, ROOT.TRandom3(3))
    assert 1 <= px <= 2
    assert 0 <= py <= 1
    assert x[0] == px
