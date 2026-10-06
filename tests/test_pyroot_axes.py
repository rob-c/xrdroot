"""``TAxis``, a histogram's statistics and extremes, and ``TProfile``."""

from __future__ import annotations

import array

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.pyroot.core import axes as core_axes


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_an_axis_stands_alone_in_every_constructor():
    expect(
        (ROOT.TAxis().GetNbins(), 1),
        (ROOT.TAxis(4, 0, 2).GetBinWidth(1), 0.5),
    )
    uneven = ROOT.TAxis(3, [0, 1, 3, 6, 99])
    expect(
        (uneven.GetXmax(), 6),
        (bool(uneven.IsVariableBinSize()), True),
        (list(uneven.GetXbins()), [0, 1, 3, 6]),
    )


def test_an_axis_names_and_styles_itself_in_its_members():
    axis = ROOT.TH1D("h", "", 4, 0, 4).GetXaxis()
    axis.SetName("xx")
    axis.SetTitle("the x")
    expect(
        (axis.GetName(), "xx"),
        (axis.GetTitle(), "the x"),
    )
    axis.SetTitleOffset(1.4)
    axis.SetLabelSize()
    axis.SetNdivisions(505, False)
    expect(
        (axis.GetTitleOffset(), 1.4),
        (axis.GetLabelSize(), 0.035),
        (axis.GetNdivisions(), -505),
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
    expect(
        ((even.GetBinLowEdge(0), even.GetBinUpEdge(5), even.GetBinCenter(5)), (-0.5, 2.5, 2.25)),
        (even.FindBin(-1), 0),
        (even.FindBin(2), 5),
        (even.FindBin(0.6), 2),
        (even.GetBinCenterLog(2), pytest.approx(np.sqrt(0.5 * 1.0))),
        (even.GetBinCenterLog(1), 0.25),
    )
    uneven = ROOT.TAxis(3, [0.0, 1.0, 3.0, 6.0])
    expect(
        (uneven.GetBinLowEdge(3), 3),
        (uneven.GetBinUpEdge(3), 6),
        (uneven.GetBinCenter(2), 2),
        (uneven.GetBinWidth(9), 3),
        (uneven.GetBinWidth(0), 1),
        (uneven.FindBin(4.0), 3),
        (uneven.GetBinLowEdge(0), -2),
        (uneven.GetBinCenter(0), -1),
    )
    lows, centres = np.zeros(3), np.zeros(3)
    uneven.GetLowEdge(lows)
    uneven.GetCenter(centres)
    expect(
        (list(lows), [0, 1, 3]),
        (list(centres), [0.5, 2, 4.5]),
    )


def test_ranges_are_set_as_root_sets_them():
    axis = ROOT.TAxis(10, 0, 10)
    axis.SetRange(3, 5)
    expect(
        ((axis.GetFirst(), axis.GetLast(), axis.GetRangeActual()), (3, 5, True)),
        (axis._row["fFirst"], 3),
        (axis._row["fLast"], 5),
    )
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
    expect(
        (axis.GetNbins(), 2),
        (axis.GetBinWidth(1), 2),
    )
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
    expect(
        (axis.GetBinLabel(2), "two"),
        (axis.GetBinLabel(1), ""),
        (axis._row["_labels_option"], "v"),
    )
    axis.SetTimeDisplay(True)
    axis.SetTimeFormat("%H:%M")
    axis.SetTimeOffset(100)
    expect(
        (bool(axis.GetTimeDisplay()), True),
        (axis.GetTimeFormat(), "%H:%M%F100"),
    )
    axis.SetCanExtend(True)
    assert axis.CanExtend()


def test_statistics_extremes_and_their_errors():
    h = ROOT.TH1D("h", "", 4, 0, 4)
    for x in (0.5, 1.5, 1.5, 3.5):
        h.Fill(x)
    expect(
        (h.GetMean(11), h.GetMeanError()),
        (h.GetStdDev(11), h.GetStdDevError()),
        (h.GetRMSError(), h.GetStdDevError()),
        (bool(h.GetSkewness(11) > 0), True),
        (bool(h.GetKurtosis(11) > 0), True),
    )
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
    expect(
        (h.GetMaximum(), 2),
        (h.GetMinimum(), 0),
        (h.GetMaximum(2), 1),
        (h.GetMinimum(0), 1),
        (bool(h.GetMaximum(0) == -ROOT.TMath.Infinity() or h.GetMaximum(0) < -1e38), True),
        (bool(h.GetMinimum(5) > 1e38), True),
    )
    bx = array.array("i", [0])
    expect(
        (h.GetMaximumBin(bx, array.array("i", [0]), array.array("i", [0])), 2),
        (bx[0], 2),
        (h.GetBinLowEdge(2), 1),
        (h.Interpolate(1.0), pytest.approx(1.5)),
    )
    h.GetXaxis().SetRange(3, 3)
    expect(
        (h.GetMean(), 0.0),
        (h.GetStdDev(), 0.0),
    )


def test_covariance_of_two_axes():
    h2 = ROOT.TH2D("h2", "", 10, 0, 10, 10, 0, 10)
    for x in range(10):
        h2.Fill(x + 0.5, x + 0.5)
    expect(
        (h2.GetCorrelationFactor(), pytest.approx(1.0)),
        (h2.GetCovariance(1, 1), pytest.approx(h2.GetStdDev(1) ** 2)),
        (ROOT.TH2D("e", "", 1, 0, 1, 1, 0, 1).GetCovariance(), 0.0),
        (ROOT.TH2D("f", "", 1, 0, 1, 1, 0, 1).GetCorrelationFactor(), 0.0),
    )
    h3 = ROOT.TH3D("h3", "", 2, 0, 2, 2, 0, 2, 2, 0, 2)
    h3.Fill(0.5, 0.5, 0.5)
    h3.Fill(1.5, 1.5, 1.5)
    expect(
        (h3.GetCovariance(1, 3), pytest.approx(0.25)),
        (h3.GetCovariance(3, 2), pytest.approx(0.25)),
    )


def test_integrals_over_ranges_with_and_without_widths():
    h = ROOT.TH1D("h", "", 4, 0, 8)
    for x in (1, 3, 5, 7):
        h.Fill(x)
    err = array.array("d", [0])
    expect(
        (h.Integral(2, 3), 2),
        (h.Integral(2, 3, "width"), 4),
        (h.IntegralAndError(1, 4, err, ""), 4),
        (err[0], 2),
    )


def test_a_profile_is_the_mean_in_each_bin():
    p = ROOT.TProfile("p", "prof", 2, 0, 2, 0, 10, "s")
    expect(
        (p.ClassName(), "TProfile"),
        (p.GetErrorOption(), "s"),
        (p.GetYmax(), 10),
        (p.Fill(0.5, 2.0), 1),
        (p.Fill(0.5, 4.0, 3.0), 1),
        (p.Fill(1.5, 20.0), 2),
        (p.GetBinContent(1), pytest.approx(3.5)),
        (p.GetBinEntries(1), 4),
        (p.GetBinEffectiveEntries(1), pytest.approx(16 / 10)),
        (p.GetSumOfWeights(), 3.5),
    )
    p.SetErrorOption("")
    expect(
        (p.GetErrorOption(), ""),
        (bool(p.GetBinError(1) > 0), True),
        (p.GetYmin(), 0),
    )
    p.FillN(2, [1.5, 1.5], [1.0, 3.0])
    p.FillN(1, [0.5], [1.0], [1.0])
    projected = p.ProjectionX()
    expect(
        (projected.GetName(), "p_px"),
        (projected.GetBinContent(2), pytest.approx(2.0)),
        (p.ProjectionX("entries", "b").GetBinContent(2), 2),
        (ROOT.TProfile().GetNbinsX(), 1),
        (ROOT.TProfile("q", "", 2, array.array("d", [0, 1, 3])).GetNbinsX(), 2),
    )


def test_two_and_three_dimensional_profiles():
    p2 = ROOT.TProfile2D("p2", "", 2, 0, 2, 2, 0, 2, 0, 100)
    p2.Fill(0.5, 0.5, 3.0)
    p2.Fill(0.5, 0.5, 5.0)
    expect(
        (p2.GetBinContent(1, 1), 4),
        (p2.ClassName(), "TProfile2D"),
        (p2.GetYmin(), 0),
    )
    p3 = ROOT.TProfile3D("p3", "", 1, 0, 1, 1, 0, 1, 1, 0, 1)
    p3.Fill(0.5, 0.5, 0.5, 7.0)
    assert p3.GetBinContent(1, 1, 1) == 7


def test_set_axis_range_takes_the_bins_of_its_ends_or_past_the_axes_the_contents_range():
    h = ROOT.TH1F("hsar", "", 10, 0, 10)
    h.SetAxisRange(2.5, 6.5)
    assert (h.GetXaxis().GetFirst(), h.GetXaxis().GetLast()) == (3, 7)
    h.SetAxisRange(1, 50, "Y")
    assert (h.GetMinimum(), h.GetMaximum()) == (1.0, 50.0)
    h.SetAxisRange(0.5, 1.5, "w")
    h.SetAxisRange(0.5, 1.5, "")
    assert (h.GetXaxis().GetFirst(), h.GetXaxis().GetLast()) == (3, 7)
    h2 = ROOT.TH2F("h2sar", "", 4, 0, 4, 8, 0, 8)
    h2.SetAxisRange(2.5, 5.5, "y")
    assert (h2.GetYaxis().GetFirst(), h2.GetYaxis().GetLast()) == (3, 6)


def test_a_polyline_is_named_as_its_class_until_it_is_given_a_name():
    marker = ROOT.TPolyMarker(2, [0.0, 1.0], [2.0, 3.0])
    assert marker.GetName() == "TPolyMarker"
    marker.SetName("mine")
    assert (marker.GetName(), ROOT.TPolyLine().GetName()) == ("mine", "TPolyLine")


def test_a_histogram_sets_an_attribute_on_the_axes_it_names():
    h = ROOT.TH2F("axesset", "", 2, 0, 1, 2, 0, 1)
    h.SetNdivisions(505, "xy")
    h.SetLabelSize(0.07, "YQ")  # a letter that names no axis is passed over
    h.SetTitleOffset(1.5)
    assert (h.GetXaxis().GetNdivisions(), h.GetYaxis().GetNdivisions()) == (505, 505)
    assert h.GetYaxis().GetLabelSize() == pytest.approx(0.07)
    assert h.GetXaxis().GetTitleOffset() == pytest.approx(1.5)
