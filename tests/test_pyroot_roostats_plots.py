"""``LikelihoodIntervalPlot``: the profile of one parameter with its cut, or the contour of two.

The Gaussian of ten values that the ProfileLikelihoodCalculator tests use:
its mean's profile drawn on a frame or as a ``TF1``, with the interval's
ends and the cut at half the chi-square quantile, and the contour of the
mean and the width with the best fit marked.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.fit import defaults
from xrdroot.pyroot.roostats import plots
from xrdroot.roofit.messages import service

VALUES = [0.3, 1.2, -0.4, 2.1, 0.8, 1.5, 0.1, 1.9, 0.6, 1.1]


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    saved = dict(defaults.DEFAULTS)
    service().reset()
    ROOT.TCanvas("c", "c", 400, 300)
    yield
    service().reset()
    defaults.DEFAULTS.clear()
    defaults.DEFAULTS.update(saved)


def _interval(poi: str = "mu", level: float = 0.6827) -> tuple[Any, Any]:
    w = ROOT.RooWorkspace("w")
    w.factory("Gaussian::g(x[-10,10],mu[0,-5,5],s[1,0.1,5])")
    x = w.var("x")
    data = ROOT.RooDataSet("d", "d", ROOT.RooArgSet(x))
    for value in VALUES:
        x.setVal(value)
        data.add(ROOT.RooArgSet(x))
    mc = ROOT.RooStats.ModelConfig("mc", w)
    mc.SetPdf("g")
    mc.SetParametersOfInterest(poi)
    mc.SetObservables("x")
    if poi == "mu":
        mc.SetNuisanceParameters("s")
    plc = ROOT.RooStats.ProfileLikelihoodCalculator(data, mc, 1 - level)
    return w, plc.GetInterval()


def _lines(frame: Any) -> list[tuple[float, ...]]:
    found = [frame.getObject(i) for i in range(int(frame.numItems()))]
    return [(o.GetX1(), o.GetY1(), o.GetX2(), o.GetY2()) for o in found if o.ClassName() == "TLine"]


def test_one_parameter_is_its_profile_on_a_frame_with_the_ends_and_the_cut() -> None:
    w, interval = _interval()
    plot = ROOT.RooStats.LikelihoodIntervalPlot(interval)
    plot.SetRange(0.0, 2.0)
    plot.SetPrecision(1e-4)
    plot.SetNPoints(50)
    plot.SetLineColor("kRed")
    plot.Draw()
    frame = plot.GetPlottedObject()
    mu = w.var("mu")
    low, high = interval.LowerLimit(mu), interval.UpperLimit(mu)
    level = 0.5 * plots._chi2_quantile(0.6827, 1)
    assert _lines(frame) == [(low, 0.0, low, level), (high, 0.0, high, level),
                             (0.0, level, 2.0, level)]  # fmt: skip
    assert frame.GetYaxis().GetTitle() == "- log #lambda(mu)"
    assert (mu.getBins(), frame.GetMinimum()) == (100, 0.0)


def test_the_profile_as_a_function_runs_over_twice_the_interval_or_to_the_maximum() -> None:
    w, interval = _interval()
    plot = ROOT.RooStats.LikelihoodIntervalPlot(interval)
    plot.Draw("tf1")
    hist = plot.GetPlottedObject()
    mu = w.var("mu")
    low, high = interval.LowerLimit(mu), interval.UpperLimit(mu)
    axis = hist.GetXaxis()
    assert (axis.GetXmin(), axis.GetXmax()) == pytest.approx((2 * low - high, 2 * high - low))
    capped = ROOT.RooStats.LikelihoodIntervalPlot(interval)
    capped.SetMaximum(1.0)
    capped.Draw("tf1")
    axis = capped.GetPlottedObject().GetXaxis()
    assert low - 0.5 < axis.GetXmin() < low < high < axis.GetXmax() < high + 0.5
    ranged = ROOT.RooStats.LikelihoodIntervalPlot(interval)
    ranged.SetRange(0.5, 1.5)
    ranged.SetMaximum(1.0)
    ranged.Draw("tf1")
    axis = ranged.GetPlottedObject().GetXaxis()
    assert (axis.GetXmin(), axis.GetXmax()) == (0.5, 1.5)


def test_a_maximum_the_profile_never_reaches_below_leaves_the_range(monkeypatch: Any) -> None:
    _, interval = _interval()
    capped = ROOT.RooStats.LikelihoodIntervalPlot(interval)
    capped.SetMaximum(1.0)
    monkeypatch.setattr(ROOT.TF1, "GetX", lambda self, *args: -100.0)
    capped.Draw("tf1")
    assert capped.GetPlottedObject() is not None


def test_parameters_not_the_intervals_are_dropped_and_three_refused(capsys: Any) -> None:
    w, interval = _interval()
    plot = ROOT.RooStats.LikelihoodIntervalPlot(interval)
    plot.SetPlotParameters(ROOT.RooArgSet(w.var("mu"), w.var("x")))
    plot.Draw()
    assert "Parameter xis not in the list of LikelihoodInterval parameters" in (
        capsys.readouterr().out)  # fmt: skip
    w, interval = _interval("mu,s")
    plot = ROOT.RooStats.LikelihoodIntervalPlot()
    plot.SetLikelihoodInterval(interval)
    params = interval.GetParameters()
    params.add(w.var("x"))
    plot._params = params
    interval.GetParameters = lambda: params
    plot.Draw()
    assert "contours for more than 2 dimensions not implemented!" in capsys.readouterr().out


def test_two_parameters_are_minuits_contour_closed_with_the_best_fit_marked() -> None:
    _, interval = _interval("mu,s")
    plot = ROOT.RooStats.LikelihoodIntervalPlot(interval)
    plot.SetNPoints(8)
    plot.SetRange(0.0, 0.2, 2.0, 2.0)
    plot.SetContourColor("kBlack")
    plot.SetFillStyle(3004)
    plot.Draw()
    graph = plot.GetPlottedObject()
    assert (graph.GetName(), graph.GetN()) == (f"Graph_of_{interval.GetName()}", 9)
    assert (graph.GetX()[0], graph.GetY()[0]) == (graph.GetX()[8], graph.GetY()[8])
    axes = plot._kept_frame.GetXaxis()
    assert (axes.GetXmin(), axes.GetXmax(), plot._kept_frame.GetTitle()) == (
        0.0, 2.0, "Contour of s vs mu")  # fmt: skip
    again = ROOT.RooStats.LikelihoodIntervalPlot(interval)
    again.SetNPoints(8)
    again.SetContourColor(2)
    again.Draw("same C")
    assert again.GetPlottedObject().GetFillStyle() == 4050
    with pytest.raises(UnsupportedFeatureError, match="not here yet"):
        again.Draw("nominuit")
    plain = ROOT.RooStats.LikelihoodIntervalPlot(interval)
    plain.SetNPoints(8)
    plain.SetLineColor(3)
    plain.Draw("same")
    assert plain.GetPlottedObject().GetLineColor() == 3


def test_a_contour_short_of_points_is_said_and_no_best_fit_is_no_marker(capsys: Any,
                                                                       monkeypatch: Any) -> None:
    _, interval = _interval("mu,s")
    monkeypatch.setattr(type(interval), "GetBestFitParameters", lambda self: None)

    def short(self: Any, px: Any, py: Any, xs: list[float], ys: list[float], n: int) -> int:
        xs[:3], ys[:3] = [0.5, 1.0, 1.5], [0.5, 1.0, 0.5]
        return 3

    monkeypatch.setattr(type(interval), "GetContourPoints", short)
    plot = ROOT.RooStats.LikelihoodIntervalPlot(interval)
    plot.SetNPoints(6)
    plot.Draw()
    assert plot.GetPlottedObject().GetN() == 5  # ROOT's removal skips as the points shift
    assert "Less points calculated in contours np = 3 / 6" in capsys.readouterr().out


def test_one_of_two_parameters_is_the_profile_of_the_likelihood_in_it() -> None:
    w, interval = _interval("mu,s")
    plot = ROOT.RooStats.LikelihoodIntervalPlot(interval)
    plot.SetPlotParameters(ROOT.RooArgSet(w.var("mu")))
    plot.SetNPoints(20)
    plot.Draw()
    assert plot.GetPlottedObject().ClassName() == "RooPlot"
