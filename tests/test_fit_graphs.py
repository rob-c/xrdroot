"""Fitting graphs: the error bars decide the chi-square, as ``GetDataType`` decides it.

A graph without errors is fitted with errors of one and its errors scaled
afterwards; errors in y weigh the points; errors in x as well make it the
effective-variance chi-square, the y error widened by the x error times the
slope; asymmetric errors take the side facing the function. Each is checked
against the sum written out by hand, and a straight line against the normal
equations where the fit is linear.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot import Function, Graph, Histogram, MultiGraph, UnsupportedFeatureError
from xrdroot.fit import cost
from xrdroot.fit.data import (
    ASYM_ERROR,
    COORD_ERROR,
    NO_ERROR,
    VALUE_ERROR,
    DataOptions,
    from_graphs,
)

X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
Y = np.array([0.1, 2.0, 3.9, 1.2, 3.1])


def multigraph(*graphs):
    members = {"TNamed": {"fName": "mg", "fTitle": ""}, "fGraphs": list(graphs)}
    return MultiGraph("TMultiGraph", members)


def test_a_graph_without_errors_is_fitted_with_unit_errors_scaled_by_the_chi_square(capsys):
    graph = Graph.new("g", X[::-1], Y[::-1])  # given out of order: sorted by x
    result = graph.fit("pol1", "Q")
    design = np.stack([np.ones(5), X], axis=1)
    expected, *_ = np.linalg.lstsq(design, Y, rcond=None)
    np.testing.assert_allclose(result.parameters, expected, rtol=1e-12)
    raw = np.sqrt(np.diag(np.linalg.inv(design.T @ design)))
    np.testing.assert_allclose(result.errors, raw * math.sqrt(result.chi2 / 3), rtol=1e-12)
    stored = graph.functions[0]
    assert stored.range == pytest.approx((0.0, 4.4)) and len(stored.members["fSave"]) == 103


def test_y_errors_weigh_the_points_and_zero_errors_leave_a_point_out():
    ey = np.array([0.2, 0.5, 0.0, 0.3, 0.4])
    graph = Graph.new("g", X, Y, yerr=ey)
    result = graph.fit("pol1", "Q")
    kept = ey > 0
    design = np.stack([np.ones(4), X[kept]], axis=1) / ey[kept, None]
    expected, *_ = np.linalg.lstsq(design, Y[kept] / ey[kept], rcond=None)
    np.testing.assert_allclose(result.parameters, expected, rtol=1e-12)
    assert result.npoints == 4 and result.ndf == 2
    assert graph.fit("pol1", "WQ").npoints == 5  # W: every error one, every point in


def test_x_errors_make_it_the_effective_variance_chi_square():
    ex, ey = np.full(5, 0.3), np.full(5, 0.4)
    graph = Graph.new("g", X, Y, xerr=ex, yerr=ey)
    result = graph.fit("pol1", "Q")
    assert result.minimizer == "Minuit2 / Migrad"
    a, b = result.parameters
    expected = np.sum((Y - a - b * X) ** 2 / (ey**2 + (ex * b) ** 2))
    assert result.chi2 == pytest.approx(expected, rel=1e-9)
    ignoring = graph.fit("pol1", "EX0 Q")  # without the x errors: plain, and linear
    assert ignoring.minimizer == "Linear"
    zero_x = Graph.new("z", X, Y, xerr=np.zeros(5), yerr=ey)
    assert zero_x.fit("pol1", "Q").minimizer == "Linear"
    assert graph.fit("pol0", "Q").minimizer == "Linear"  # a constant has no slope to widen by


def test_asymmetric_errors_take_the_side_facing_the_function():
    low, high = np.array([0.2, 0.3, 0.2, 0.5, 0.2]), np.array([0.6, 0.2, 0.4, 0.2, 0.3])
    graph = Graph.new("a", X, Y, yerr=(low, high))
    data = from_graphs([graph], DataOptions(), None)
    assert data.kind == ASYM_ERROR and not data.options.coord_errors
    result = graph.fit("pol1", "Q")
    a, b = result.parameters
    residual = Y - a - b * X
    chosen = np.where(residual < 0, high, low)
    assert result.chi2 == pytest.approx(np.sum((residual / chosen) ** 2), rel=1e-9)
    with_x = Graph.new("b", X, Y, xerr=(np.full(5, 0.1), np.full(5, 0.3)), yerr=(low, high))
    assert from_graphs([with_x], DataOptions(), None).kind == ASYM_ERROR
    only_x = Graph.new("c", X, Y, xerr=(np.full(5, 0.1), np.full(5, 0.3)), yerr=(0 * X, 0 * X))
    assert from_graphs([only_x], DataOptions(), None).kind == COORD_ERROR
    none = Graph.new("d", X, Y, xerr=(0 * X, 0 * X), yerr=(0 * X, 0 * X))
    assert from_graphs([none], DataOptions(), None).kind == NO_ERROR


def test_the_kind_of_data_is_roots_for_every_class_of_graph():
    plain = Graph.new("p", X, Y)
    even = Graph.new("e", X, Y, yerr=np.full(5, 0.2))
    assert from_graphs([plain], DataOptions(), None).kind == NO_ERROR
    assert from_graphs([even], DataOptions(), None).kind == VALUE_ERROR
    zero = Graph.new("z", X, Y, yerr=np.zeros(5))
    assert from_graphs([zero], DataOptions(), None).kind == NO_ERROR
    assert from_graphs([even], DataOptions(errors1=True), None).kind == NO_ERROR


def test_a_multigraph_fits_all_its_graphs_the_most_elaborate_kind_of_any():
    plain = Graph.new("p", X, Y + 1.0)
    even = Graph.new("e", X, Y, yerr=np.full(5, 0.2))
    mg = multigraph(plain, even)
    result = mg.fit("pol1", "Q")
    # a TGraph among graphs with errors has no errors to give, and so no points
    assert result.npoints == 5 and mg.functions[0].range == (0.0, 4.0)
    assert mg.fit("pol1", "WQ").npoints == 10


def test_a_range_keeps_the_points_inside_it_ends_included():
    graph = Graph.new("g", X, Y, yerr=np.full(5, 0.2))
    assert graph.fit("pol1", "Q", (1.0, 3.0)).npoints == 3
    f = Function("pol1", "pol1", range=(0.5, 4.0))
    assert graph.fit(f, "RQ").npoints == 4


def test_a_graph_keeps_the_range_its_own_histogram_was_drawn_with():
    graph = Graph.new("g", X, Y, yerr=np.full(5, 0.2))
    frame = graph.fit("pol1", "Q")
    assert frame.function.range == pytest.approx((0.0, 4.4))
    below = Graph.new("n", -X, Y, yerr=np.full(5, 0.2))
    assert below.fit("pol1", "Q").function.range == pytest.approx((-4.4, 0.0))
    drawn = Graph.new("d", X, Y, yerr=np.full(5, 0.2))
    drawn._core["fHistogram"] = Histogram.book("frame", (50, -1.0, 6.0))
    assert drawn.fit("pol1", "Q").function.range == pytest.approx((-1.0, 6.0))
    single = Graph.new("s", [2.0, 2.0], [1.0, 3.0], yerr=[0.5, 0.5])
    assert single.fit("pol0", "Q").function.range == pytest.approx((1.9, 3.1))


def test_what_a_graph_fit_cannot_do_is_refused_by_name():
    graph = Graph.new("g", X, Y)
    with pytest.raises(UnsupportedFeatureError, match="robust fit"):
        graph.fit("pol1", "ROB")
    members = dict(graph.members)
    layered = Graph("TGraphMultiErrors", members)
    with pytest.raises(UnsupportedFeatureError, match="TGraphMultiErrors"):
        layered.fit("pol1", "Q")


def test_the_effective_chi_square_uses_roots_derivative_and_caps_a_nan():
    graph = Graph.new("g", X, Y, xerr=np.full(5, 0.3), yerr=np.full(5, 0.4))
    data = from_graphs([graph], DataOptions(), None)
    fcn = cost.effective_chi2(data, Function("f", "[0]*x*x"))
    # the slope of p x^2 at x is 2 p x, which Richardson's derivative has exactly
    expected = np.sum((Y - X * X) ** 2 / (0.16 + (0.3 * 2 * X) ** 2))
    assert fcn([1.0]) == pytest.approx(expected, rel=1e-9)
    assert math.isfinite(cost.effective_chi2(data, Function("f", "sqrt([0]-x)"))([1.0]))
