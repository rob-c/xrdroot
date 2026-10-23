"""Contour computations hand their public graphics to declared adapters."""

from __future__ import annotations

from types import SimpleNamespace

from statisticalproviders import install_provider
from xrdroot.roofit.fitting import contour
from xrdroot.roofit.fitting.minimizer import RooMinimizer
from xrdroot.roofit.functions import RooAddition, RooFormulaVar
from xrdroot.roofit.variables import RooRealVar


def test_a_marked_frame_before_migrad_uses_the_declared_marker_factory(monkeypatch):
    x, y = RooRealVar("x", "x", 0, -2, 2), RooRealVar("y", "y", 1, -2, 2)
    formula = RooFormulaVar("formula", "x*x+y*y", [x, y])
    objective = RooAddition("objective", "objective", [formula])
    minimizer = RooMinimizer(objective)
    markers = []

    def marker(*values):
        markers.append(values)
        return values

    install_provider(monkeypatch, "root.class", marker, "TMarker")
    frame = minimizer.contour(x, y)
    assert markers == [(0.0, 1.0, 8)]
    assert frame.items == [(markers[0], "", False)]
    assert frame.GetTitle() == "A RooPlot"


def test_contour_points_and_their_level_reach_the_graph_adapter_unchanged(monkeypatch):
    received = []
    expected = object()

    def graph(*values):
        received.append(values)
        return expected

    install_provider(monkeypatch, "roofit.contour_graph", graph)
    minimizer = object()
    points = [(0.0, 1.0), (1.0, 0.0)]
    assert contour._graph(minimizer, points, 2.0, 3) is expected
    assert received == [(minimizer, points, 2.0, 3)]
    assert received[0][1] is points


def test_the_root_contour_adapter_closes_the_graph_and_preserves_its_style():
    from xrdroot.pyroot.core.graphs import TGraph
    from xrdroot.pyroot.roofit.contour import make_graph

    minimizer = SimpleNamespace(function=SimpleNamespace(GetName=lambda: "nll"))
    graph = make_graph(minimizer, [(0.0, 1.0), (1.0, 0.0)], 2.0, 3)
    assert isinstance(graph, TGraph)
    assert graph.GetName() == "contour_nll_n2.000000"
    assert list(graph.GetX()) == [0.0, 1.0, 0.0]
    assert list(graph.GetY()) == [1.0, 0.0, 1.0]
    assert (graph.GetLineColor(), graph.GetLineStyle(), graph.GetLineWidth()) == (600, 4, 2)
