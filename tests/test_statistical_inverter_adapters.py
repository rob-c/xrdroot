"""Inverter state and search rules stay in the engine while its ROOT fits use declared targets."""

from __future__ import annotations

import math
from types import SimpleNamespace

from statisticalproviders import install_provider
from xrdroot.roofit.variables import RooRealVar
from xrdroot.roostats import inverterlimits, inverterscan
from xrdroot.roostats.hypotest import HypoTestResult
from xrdroot.roostats.inverterresult import HypoTestInverterResult


def test_error_graph_and_fit_requests_retain_their_arguments_through_downward_providers(
    monkeypatch,
):
    result, graph = object(), object()
    received = []

    def points(*args):
        received.append(args)
        return graph, 3

    def fitted(*args):
        received.append(args)
        return 0.125

    install_provider(monkeypatch, "roostats.inverter_error_graph", points)
    install_provider(monkeypatch, "roostats.inverter_error_fit", fitted)
    assert inverterlimits._error_graph(result, 1.0, 3.0) == (graph, 3)
    assert inverterlimits._fitted_error(result, graph, 0.05, False, 2.0, (1.0, 3.0)) == 0.125
    assert received == [(result, 1.0, 3.0), (result, graph, 0.05, False, 2.0, (1.0, 3.0))]


def test_search_fitting_keeps_the_same_fit_options_and_updates_the_same_bracket(monkeypatch):
    calls = []
    expo = SimpleNamespace(GetParameter=lambda index: 2.0, GetParError=lambda index: 0.01)
    plot = SimpleNamespace(
        GetX=lambda: [1.0, 2.0, 3.0],
        Sort=lambda: calls.append("sort"),
        Fit=lambda *args: calls.append(args),
    )
    install_provider(monkeypatch, "roostats.inverter_plot", lambda result: plot)
    install_provider(monkeypatch, "roostats.inverter_exponential", lambda *args: expo)
    mu = RooRealVar("mu", "mu", 2, 0, 10)
    it = SimpleNamespace(_var=mu, _verbose=0, _results=HypoTestInverterResult("r", mu))
    search = inverterscan._Search(it, 0.05, (0.1, 0.1))
    search.r_min, search.r_max, search.limit = 1.0, 3.0, 2.0
    search.fit_range = (1.0, 3.0)
    assert search.fit() is expo
    assert it._limit_plot is plot
    assert calls == ["sort", (expo, "QNR EX0")]
    assert (search.limit, search.error) == (2.0, 0.01)


def test_the_public_error_graph_keeps_its_sorting_and_only_the_valid_error_points():
    from xrdroot.pyroot.core.graphs import TGraphErrors

    mu = RooRealVar("mu", "mu", 1, 0, 10)
    result = HypoTestInverterResult("r", mu)
    for x, y, error in ((3.0, 0.01, 0.03), (1.0, 0.4, 0.01), (2.0, 0.2, 0.0)):
        point = HypoTestResult("p", y, 0.5)
        point.SetBackgroundAsAlt(True)
        point.SetNullPValueError(error)
        result.Add(x, point)
    graph, inside = inverterlimits._error_graph(result, 1.0, 3.0)
    assert isinstance(graph, TGraphErrors)
    assert inside == 3
    assert list(graph.GetX()) == [1.0, 3.0]
    assert list(graph.GetY()) == [0.4, 0.01]
    assert list(graph.GetEY()) == [0.01, 0.03]


def test_the_exponential_adapter_keeps_the_target_initial_slope_and_limit():
    from xrdroot.pyroot.core.funcs import TF1

    search = inverterscan._Search(object(), 0.05, (0.1, 0.1))
    search.r_min, search.r_max, search.limit = 1.0, 3.0, 2.0
    search.cls_min, search.cls_max = (0.5, 0.01), (0.01, 0.01)
    expo = search._exponential(1.0, 3.0)
    assert isinstance(expo, TF1)
    assert expo.GetName() == "expoFit"
    assert [expo.GetParameter(i) for i in range(3)] == [0.05, math.log(0.01 / 0.5) / 2, 2.0]
    assert expo.GetRange() == (1.0, 3.0)
