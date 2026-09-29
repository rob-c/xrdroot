"""``SamplingDistPlot`` and ``HypoTestPlot``: sampling distributions as histograms on a frame.

Distributions of a few values each: histogrammed over their range and a
bin and a half, normalised or not, shaded beyond a cut, styled one by one,
and drawn together on a ``RooPlot`` with lines, a legend and log axes; a
hypothesis test's two distributions shaded beyond its data's value.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.graphics.style import gStyle

NULL = [0.1, 0.5, 0.7, 1.2, 2.0, 2.5]
ALT = [1.5, 2.2, 3.1, 3.3, 4.0, 5.2]


@pytest.fixture(autouse=True)
def _style() -> Iterator[None]:
    ROOT.TCanvas("c", "c", 400, 300)
    logs = (gStyle.GetOptLogx(), gStyle.GetOptLogy())
    yield
    gStyle.SetOptLogx(logs[0])
    gStyle.SetOptLogy(logs[1])


def _dist(name: str, values: list[float], *rest: Any) -> Any:
    return ROOT.RooStats.SamplingDistribution(name, f"{name} title", values, *rest)


def test_a_distribution_is_a_normalised_histogram_over_its_range_and_a_half_bin() -> None:
    plot = ROOT.RooStats.SamplingDistPlot(10)
    scale = plot.AddSamplingDistribution(_dist("null", NULL, [1, 1, 1, 1, 1, 2], "q"))
    hist = plot.GetTH1F()
    width = (2.5 - 0.1) / 10
    axis = hist.GetXaxis()
    assert (axis.GetXmin(), axis.GetXmax()) == (0.1 - 1.5 * width, 2.5 + 1.5 * width)
    assert (axis.GetTitle(), hist.Integral("width")) == ("q", pytest.approx(1.0))
    assert scale == pytest.approx(1 / (7 * (axis.GetXmax() - axis.GetXmin()) / 10))
    assert plot.AddSamplingDistribution(_dist("raw", ALT), "HIST") == 1.0
    assert plot.GetTH1F(_dist("null", [])).GetName() == "null"
    assert plot.GetTH1F(_dist("none", [])) is None


def test_empty_or_single_valued_distributions_are_said(capsys: Any) -> None:
    plot = ROOT.RooStats.SamplingDistPlot(10, -2.0, 2.0)
    assert plot.AddSamplingDistribution(_dist("e", [])) == 0.0
    assert plot.AddSamplingDistributionShaded(_dist("e", []), 0, 1) == 0.0
    plot.AddSamplingDistribution(_dist("one", [1.0, math.inf]))
    axis = plot.GetTH1F().GetXaxis()
    assert (axis.GetXmin(), axis.GetXmax()) == (-2.0, 2.0)
    out = capsys.readouterr().out
    assert out.count("Empty sampling distribution given to plot. Skipping.") == 2
    assert "Could not determine xmin and xmax of sampling distribution" in out


def test_a_shaded_copy_keeps_only_the_bins_between_its_ends() -> None:
    plot = ROOT.RooStats.SamplingDistPlot(10)
    plot.AddSamplingDistributionShaded(_dist("null", NULL), 1.0, 2.0)
    shaded = plot._items[-1]
    assert (shaded.GetName(), shaded.GetFillStyle()) == ("null_shaded", 3004)
    kept = [shaded.GetBinCenter(i) for i in range(12) if shaded.GetBinContent(i)]
    assert kept and all(1.0 <= x <= 2.0 for x in kept)
    null = _dist("null", [])
    plot.SetLineColor(2, null)
    plot.SetLineWidth(3)
    plot.SetLineStyle(2, null)
    plot.SetMarkerStyle(21)
    plot.SetMarkerColor(4)
    plot.SetMarkerSize(2)
    first = plot._items[0]
    assert (shaded.GetFillColor(), first.GetLineColor(), first.GetLineWidth()) == (2, 2, 3)
    assert (first.GetMarkerStyle(), first.GetMarkerColor(), first.GetMarkerSize()) == (21, 4, 2)


def test_everything_is_drawn_on_one_frame_with_its_legend_and_log_axes(capsys: Any) -> None:
    plot = ROOT.RooStats.SamplingDistPlot(10)
    legend = ROOT.TLegend(0.6, 0.6, 0.9, 0.9)
    plot.AddTH1(ROOT.TH1F("untitled", "", 10, 0, 3))
    plot.SetLogXaxis(True)
    plot.SetLogXaxis(False)
    plot.SetLegend(legend)
    plot.SetAxisTitle("t")
    plot.AddSamplingDistribution(_dist("null", NULL))
    plot.AddLine(1.0, 0.0, 1.0, 0.5, "cut")
    plot.AddLine(2.0, 0.0, 2.0, 0.5)
    h = ROOT.TH1F("h", "a histogram", 10, 0, 3)
    plot.AddTH1(h)
    plot.AddTF1(ROOT.TF1("f", "x", 0, 3), "a line")
    plot.AddTF1(ROOT.TF1("g", "x", 0, 3))
    plot.SetXRange(0.0, 3.0)
    plot.SetYRange(0.01, 2.0)
    plot.SetLogYaxis(True)
    plot.Draw()
    frame = plot.GetPlot()
    assert (frame.GetMinimum(), gStyle.GetOptLogy()) == (0.01, 1)
    assert [e.GetLabel() for e in legend.GetListOfPrimitives()] == [
        "null title", "cut", "a histogram", "a line"]  # fmt: skip
    plot.SetApplyStyle(False)
    plot.SetLogYaxis(False)
    plot.SetLogXaxis(False)
    plot.Draw()
    assert "gStyle will be changed to adjust SetOptLogy(...)" in capsys.readouterr().out
    plot.ApplyDefaultStyle()


def _result(data: bool, right: bool = True) -> Any:
    result = ROOT.RooStats.HypoTestResult("r")
    result.SetNullDistribution(_dist("null", NULL))
    result.SetAltDistribution(_dist("alt", ALT))
    result.SetPValueIsRightTail(right)
    if data:
        result.SetTestStatisticData(2.1)
    return result


def test_a_tests_distributions_are_shaded_beyond_its_data_with_a_line_at_it() -> None:
    plot = ROOT.RooStats.HypoTestPlot(_result(True), 20)
    names = [h.GetName() for h in plot._items]
    assert names == ["alt", "alt_shaded", "null", "null_shaded"]
    assert (plot._items[0].GetLineColor(), plot._items[2].GetLineColor()) == (600, 632)
    assert plot._others[0].GetX1() == 2.1
    left = ROOT.RooStats.HypoTestPlot(_result(True, right=False), 20, 0.0, 6.0, "HIST")
    assert left.GetTH1F().GetXaxis().GetXmin() == 0.0
    plain = ROOT.RooStats.HypoTestPlot(_result(False))
    assert [h.GetName() for h in plain._items] == ["alt", "null"]
    lone = ROOT.RooStats.HypoTestResult("lone")
    lone.SetTestStatisticData(1.0)
    assert ROOT.RooStats.HypoTestPlot(lone)._items == []
    plain = ROOT.RooStats.HypoTestResult("plain")
    assert ROOT.RooStats.HypoTestPlot(plain)._items == []


def test_a_plot_without_a_legend_or_ranges_spans_its_histograms() -> None:
    plot = ROOT.RooStats.SamplingDistPlot(10)
    plot.AddSamplingDistribution(_dist("null", NULL))
    plot.Draw()
    axis = plot.GetPlot().GetXaxis()
    hist = plot.GetTH1F().GetXaxis()
    assert (axis.GetXmin(), axis.GetXmax()) == (hist.GetXmin(), hist.GetXmax())


def test_no_pad_leaves_the_log_axes_alone() -> None:
    from xrdroot.pyroot.roostats import sdplot

    sdplot._logged(None, True, True)
    hist = ROOT.TH1F("b", "", 2, 0, 1)
    sdplot._bounded(hist, 0.5, math.nan)  # a minimum alone
    assert (hist.GetMinimum(), hist.GetMaximum()) == (0.5, 0.0)
