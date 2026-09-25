"""``TAxis``, a histogram's statistics and extremes, and ``TProfile``."""

from __future__ import annotations

import array

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.pyroot.core import axes as core_axes


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_an_axis_stands_alone_in_every_constructor():
    assert ROOT.TAxis().GetNbins() == 1
    assert ROOT.TAxis(4, 0, 2).GetBinWidth(1) == 0.5
    uneven = ROOT.TAxis(3, [0, 1, 3, 6, 99])
    assert (
        uneven.GetXmax() == 6
        and uneven.IsVariableBinSize()
        and list(uneven.GetXbins()) == [0, 1, 3, 6]
    )


def test_an_axis_names_and_styles_itself_in_its_members():
    axis = ROOT.TH1D("h", "", 4, 0, 4).GetXaxis()
    axis.SetName("xx")
    axis.SetTitle("the x")
    assert axis.GetName() == "xx" and axis.GetTitle() == "the x"
    axis.SetTitleOffset(1.4)
    axis.SetLabelSize()
    axis.SetNdivisions(505, False)
    assert (
        axis.GetTitleOffset() == 1.4
        and axis.GetLabelSize() == 0.035
        and axis.GetNdivisions() == -505
    )
    axis.SetNdivisions(510)
    assert axis.GetNdivisions() == 510
    axis.ResetAttAxis()
    assert axis.GetTitleOffset() == 1.0
    for setter, bit in (
        ("CenterTitle", core_axes.CENTER_TITLE),
        ("RotateTitle", core_axes.ROTATE_TITLE),
        ("CenterLabels", core_axes.CENTER_LABELS),
        ("SetNoExponent", core_axes.NO_EXPONENT),
        ("SetMoreLogLabels", core_axes.MORE_LOG_LABELS),
    ):
        getattr(axis, setter)()
        assert axis.TestBit(bit)
    assert axis.GetCenterTitle()


def test_the_bins_of_an_even_and_an_uneven_axis_are_roots():
    even = ROOT.TAxis(4, 0, 2)
    assert (even.GetBinLowEdge(0), even.GetBinUpEdge(5), even.GetBinCenter(5)) == (-0.5, 2.5, 2.25)
    assert even.FindBin(-1) == 0 and even.FindBin(2) == 5 and even.FindBin(0.6) == 2
    assert even.GetBinCenterLog(2) == pytest.approx(np.sqrt(0.5 * 1.0))
    assert even.GetBinCenterLog(1) == 0.25
    uneven = ROOT.TAxis(3, [0.0, 1.0, 3.0, 6.0])
    assert (
        uneven.GetBinLowEdge(3) == 3 and uneven.GetBinUpEdge(3) == 6 and uneven.GetBinCenter(2) == 2
    )
    assert uneven.GetBinWidth(9) == 3 and uneven.GetBinWidth(0) == 1 and uneven.FindBin(4.0) == 3
    assert uneven.GetBinLowEdge(0) == -2 and uneven.GetBinCenter(0) == -1
    lows, centres = np.zeros(3), np.zeros(3)
    uneven.GetLowEdge(lows)
    uneven.GetCenter(centres)
    assert list(lows) == [0, 1, 3] and list(centres) == [0.5, 2, 4.5]


def test_ranges_are_set_as_root_sets_them():
    axis = ROOT.TAxis(10, 0, 10)
    axis.SetRange(3, 5)
    assert (axis.GetFirst(), axis.GetLast(), axis.GetRangeActual()) == (3, 5, True)
    assert axis._row["fFirst"] == 3 and axis._row["fLast"] == 5
    axis.SetRange(0, 99)
    assert (axis.GetFirst(), axis.GetLast()) == (0, 11)
    for first, last in ((5, 3), (-1, -2), (20, 30), (0, 0)):
        axis.SetRange(first, last)
        assert (axis.GetFirst(), axis.GetLast(), axis.GetRangeActual()) == (1, 10, False)
    axis.SetRangeUser(2.0, 4.0)
    assert (axis.GetFirst(), axis.GetLast()) == (3, 4)
    axis.SetLimits(-1, 1)
    assert axis.GetXmin() == -1
    axis.Set(2, 0.0, 4.0)
    assert axis.GetNbins() == 2 and axis.GetBinWidth(1) == 2
    axis.Set(2, array.array("d", [0, 1, 5]))
    assert axis.GetBinUpEdge(2) == 5


def test_the_value_axis_range_is_the_histograms_extremes():
    h = ROOT.TH1D("h", "", 2, 0, 2)
    h.GetYaxis().SetRangeUser(-1, 5)
    assert (h.GetMinimum(), h.GetMaximum()) == (-1, 5)
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 2, 0, 2)
    h2.GetZaxis().SetRangeUser(0, 3)
    assert h2.GetMaximum() == 3


def test_unzoom_with_a_pad_resets_the_range(monkeypatch):
    axis = ROOT.TAxis(10, 0, 10)
    axis.SetRange(2, 3)
    monkeypatch.setitem(ROOT.__dict__, "gPad", object())
    axis.UnZoom()
    assert not axis.GetRangeActual()


def test_labels_time_and_extending_are_kept_for_the_drawing():
    axis = ROOT.TAxis(3, 0, 3)
    axis.SetBinLabel(2, "two")
    axis.LabelsOption("v")
    axis.ChangeLabel(1, -1, -1, -1, -1, -1, "one")
    assert (
        axis.GetBinLabel(2) == "two"
        and axis.GetBinLabel(1) == ""
        and axis._row["_labels_option"] == "v"
    )
    axis.SetTimeDisplay(True)
    axis.SetTimeFormat("%H:%M")
    axis.SetTimeOffset(100)
    assert axis.GetTimeDisplay() and axis.GetTimeFormat() == "%H:%M%F100"
    axis.SetCanExtend(True)
    assert not axis.CanExtend()


def test_statistics_extremes_and_their_errors():
    h = ROOT.TH1D("h", "", 4, 0, 4)
    for x in (0.5, 1.5, 1.5, 3.5):
        h.Fill(x)
    assert h.GetMean(11) == h.GetMeanError() and h.GetStdDev(11) == h.GetStdDevError()
    assert h.GetRMSError() == h.GetStdDevError() and h.GetSkewness(11) > 0 and h.GetKurtosis(11) > 0
    h.SetEntries(10)
    assert h.GetEntries() == 10
    h.SetMaximum(9)
    h.SetMinimum(-9)
    assert (h.GetMaximum(), h.GetMinimum(), h.GetMaximumStored(), h.GetMinimumStored()) == (
        9,
        -9,
        9,
        -9,
    )
    h.SetMaximum()
    h.SetMinimum()
    assert (
        h.GetMaximum() == 2
        and h.GetMinimum() == 0
        and h.GetMaximum(2) == 1
        and h.GetMinimum(0) == 1
    )
    assert h.GetMaximum(0) == -ROOT.TMath.Infinity() or h.GetMaximum(0) < -1e38
    assert h.GetMinimum(5) > 1e38
    bx = array.array("i", [0])
    assert h.GetMaximumBin(bx, array.array("i", [0]), array.array("i", [0])) == 2 and bx[0] == 2
    assert h.GetBinLowEdge(2) == 1 and h.Interpolate(1.0) == pytest.approx(1.5)
    h.GetXaxis().SetRange(3, 3)
    assert h.GetMean() == 0.0 and h.GetStdDev() == 0.0


def test_covariance_of_two_axes():
    h2 = ROOT.TH2D("h2", "", 10, 0, 10, 10, 0, 10)
    for x in range(10):
        h2.Fill(x + 0.5, x + 0.5)
    assert h2.GetCorrelationFactor() == pytest.approx(1.0)
    assert h2.GetCovariance(1, 1) == pytest.approx(h2.GetStdDev(1) ** 2)
    assert ROOT.TH2D("e", "", 1, 0, 1, 1, 0, 1).GetCovariance() == 0.0
    assert ROOT.TH2D("f", "", 1, 0, 1, 1, 0, 1).GetCorrelationFactor() == 0.0
    h3 = ROOT.TH3D("h3", "", 2, 0, 2, 2, 0, 2, 2, 0, 2)
    h3.Fill(0.5, 0.5, 0.5)
    h3.Fill(1.5, 1.5, 1.5)
    assert h3.GetCovariance(1, 3) == pytest.approx(0.25) and h3.GetCovariance(
        3, 2
    ) == pytest.approx(0.25)


def test_integrals_over_ranges_with_and_without_widths():
    h = ROOT.TH1D("h", "", 4, 0, 8)
    for x in (1, 3, 5, 7):
        h.Fill(x)
    err = array.array("d", [0])
    assert h.Integral(2, 3) == 2 and h.Integral(2, 3, "width") == 4
    assert h.IntegralAndError(1, 4, err, "") == 4 and err[0] == 2


def test_a_profile_is_the_mean_in_each_bin():
    p = ROOT.TProfile("p", "prof", 2, 0, 2, 0, 10, "s")
    assert p.ClassName() == "TProfile" and p.GetErrorOption() == "s" and p.GetYmax() == 10
    assert p.Fill(0.5, 2.0) == 1 and p.Fill(0.5, 4.0, 3.0) == 1 and p.Fill(1.5, 20.0) == 2
    assert p.GetBinContent(1) == pytest.approx(3.5) and p.GetBinEntries(1) == 4
    assert p.GetBinEffectiveEntries(1) == pytest.approx(16 / 10) and p.GetSumOfWeights() == 4
    p.SetErrorOption("")
    assert p.GetErrorOption() == "" and p.GetBinError(1) > 0 and p.GetYmin() == 0
    p.FillN(2, [1.5, 1.5], [1.0, 3.0])
    p.FillN(1, [0.5], [1.0], [1.0])
    projected = p.ProjectionX()
    assert projected.GetName() == "p_px" and projected.GetBinContent(2) == pytest.approx(2.0)
    assert p.ProjectionX("entries", "b").GetBinContent(2) == 2
    assert (
        ROOT.TProfile().GetNbinsX() == 1
        and ROOT.TProfile("q", "", 2, array.array("d", [0, 1, 3])).GetNbinsX() == 2
    )


def test_two_and_three_dimensional_profiles():
    p2 = ROOT.TProfile2D("p2", "", 2, 0, 2, 2, 0, 2, 0, 100)
    p2.Fill(0.5, 0.5, 3.0)
    p2.Fill(0.5, 0.5, 5.0)
    assert p2.GetBinContent(1, 1) == 4 and p2.ClassName() == "TProfile2D" and p2.GetYmin() == 0
    p3 = ROOT.TProfile3D("p3", "", 1, 0, 1, 1, 0, 1, 1, 0, 1)
    p3.Fill(0.5, 0.5, 0.5, 7.0)
    assert p3.GetBinContent(1, 1, 1) == 7
