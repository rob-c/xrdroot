"""``TGraph::Eval``, ``Integral``, ``Sort`` and the layered errors, from graphs built in code.

A graph is made with ``Graph.new`` or ``Graph.from_layers``, so its arithmetic is testable without a
file: ``Eval`` draws straight lines between the points and along them past the ends, ``Integral`` is

A graph is made with ``Graph.new`` or ``Graph.from_layers``, so its arithmetic is testable
without a file: ``Eval`` draws straight lines between the points and along them past the ends,
``Integral`` is ROOT's shoelace over the closed polygon, ``GetMean``/``GetRMS`` are the points'
own, and ``Sort`` carries the error bars with the points. Every number below was measured from a
graph in code before it was asserted - which is how the *degenerate* closed polygon was found: a
straight line encloses no area at all.

No file is opened, which is what lets this test sit inside the ``hist`` component's own scope.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot import UnsupportedFeatureError
from xrdroot.graph import Graph
from xrdroot.graphmath import (
    ABS_SUM,
    ONLY_FIRST,
    SQUARE_SUM,
    error_y,
    evaluate,
    integral,
    layer_sum,
    mean,
    rms,
    sort,
)


def _line() -> Graph:
    """``y = 2x`` through four points, which ``Eval`` reads between and past."""
    return Graph.new("g", [0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0])


def test_eval_draws_straight_lines_and_carries_the_end_line_past_it():
    """``TGraph::Eval`` is the point's own ``y`` on a point, the line's value between two, and the
    end line carried on outside them - ROOT's extrapolation, not a clamp."""
    line = _line()
    assert [evaluate(line, x) for x in (1.0, 0.5, 1.5, -1.0, 4.0)] == [2.0, 1.0, 3.0, -2.0, 8.0]
    assert evaluate(line, np.array([0.5, 2.5])).tolist() == [1.0, 5.0]


def test_eval_walks_points_that_are_not_in_order_and_a_nan_takes_the_first_point():
    """A graph whose ``x`` is not increasing is walked rather than searched, and the answer is the
    line the points make; a NaN is on no side of any point, so it takes the first point's ``y``."""
    walked = Graph.new("u", [2.0, 0.0, 1.0], [20.0, 0.0, 10.0])
    assert (evaluate(walked, 0.5), evaluate(walked, 1.5), evaluate(walked, 0.0)) == (5.0, 15.0, 0.0)
    assert evaluate(walked, float("nan")) == 20.0
    assert evaluate(_line(), np.array([0.5]), sorted_x=False).tolist() == [1.0]


def test_a_graph_of_one_point_answers_that_point_and_of_none_answers_zero():
    """ROOT's two degenerate graphs: one point is its own ``y`` everywhere, and none is 0."""
    assert evaluate(Graph.new("one", [1.0], [7.0]), 5.0) == 7.0
    assert evaluate(Graph.new("none", [], []), 5.0) == 0.0


def test_the_integral_is_root_s_shoelace_over_the_closed_polygon():
    """The last point is joined back to the first, so a triangle of area one reads 1 - while a
    straight line, whose closed polygon is degenerate, reads 0 however many points it has."""
    triangle = Graph.new("t", [0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    assert integral(triangle, 0, -1) == pytest.approx(1.0)  # last = -1 means the last point
    assert integral(triangle, 0, 1) == pytest.approx(0.0)  # a range of two points has no area
    assert integral(_line(), 0, -1) == pytest.approx(0.0)  # and a line encloses none


def test_the_integral_clamps_a_range_that_reaches_outside_the_graph():
    """``first`` below zero is the first point and ``last`` past the end the last, so a range wider
    than the graph is the whole of it."""
    triangle = Graph.new("t", [0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    assert integral(triangle, -5, 99) == pytest.approx(1.0)


def test_the_mean_and_the_rms_are_the_points_own():
    """``GetMean`` on the points' ``x`` and their ``y``, and ``GetRMS`` the spread about that mean:
    four points from 0 to 3 are a mean of 1.5 and a spread of sqrt(1.25)."""
    line = _line()
    assert (mean(line, 0), mean(line, 1)) == (1.5, 3.0)
    assert (rms(line, 0), rms(line, 1)) == pytest.approx((np.sqrt(1.25), np.sqrt(5.0)))
    nothing = Graph.new("none", [], [])
    assert (mean(nothing, 0), rms(nothing, 0)) == (0.0, 0.0)


def test_an_axis_that_is_not_one_of_a_graph_s_two_is_refused():
    """A graph has ``x`` at 0 and ``y`` at 1 - ROOT's own 1 and 2 - and anything else says so."""
    with pytest.raises(ValueError, match="axis=2 is not an axis of a graph"):
        mean(_line(), 2)


def test_sort_puts_the_points_in_order_and_carries_the_error_bars_with_them():
    """``TGraph::Sort`` reorders every per-point column, so a bar still belongs to its point."""
    made = Graph.new("s", [2.0, 0.0, 1.0], [20.0, 0.0, 10.0], yerr=[0.2, 0.0, 0.1])
    sort(made)
    assert made.x.tolist() == [0.0, 1.0, 2.0]
    assert made.y.tolist() == [0.0, 10.0, 20.0]
    assert made.yerr[0].tolist() == [0.0, 0.1, 0.2]  # each bar followed its own point


def test_a_layered_graph_s_errors_add_up_by_its_summation_mode():
    """``kOnlyFirst`` is the first layer alone, ``kSquareSum`` the root of the squares and
    ``kAbsSum`` their sum."""
    layered = Graph.from_layers(
        "m", [0.0, 1.0], [1.0, 2.0],
        layers=[([0.1, 0.2], [0.3, 0.2]), ([0.4, 0.5], [0.5, 0.6])],
        mode=SQUARE_SUM,
    )  # fmt: skip
    assert layered.classname == "TGraphMultiErrors"
    assert layer_sum(layered.layers, 0, ONLY_FIRST, 0) == pytest.approx(0.1)
    assert layer_sum(layered.layers, 0, SQUARE_SUM, 0) == pytest.approx(np.hypot(0.1, 0.4))
    assert layer_sum(layered.layers, 0, ABS_SUM, 0) == pytest.approx(0.5)
    assert layer_sum((), 0, ABS_SUM, 0) == -1.0  # no layers at all is ROOT's -1


def test_error_y_is_the_mean_of_the_two_sides_squares_in_the_graph_s_mode():
    """``GetErrorY`` sums the low bars and the high bars by the graph's mode and answers the root of
    the mean of their squares: low 0.1 and 0.4, high 0.3 and 0.5, so sqrt((0.17 + 0.34) / 2)."""
    layered = Graph.from_layers(
        "m", [0.0], [1.0], layers=[([0.1], [0.3]), ([0.4], [0.5])], mode=SQUARE_SUM
    )
    assert error_y(layered, 0) == pytest.approx(np.sqrt(0.255))


def test_a_point_a_graph_does_not_have_has_no_error_y_and_a_bad_mode_is_refused():
    """``GetErrorY`` of a point out of range is -1, as ROOT answers; a summation mode that is none
    of ROOT's three is refused rather than answered with a number the file cannot have meant."""
    layered = Graph.from_layers("m", [0.0], [1.0], layers=[([0.1], [0.2])])
    assert error_y(layered, 9) == -1.0
    layered.members["fSumErrorsMode"] = 9
    with pytest.raises(UnsupportedFeatureError, match="kOnlyFirst, kSquareSum and kAbsSum"):
        error_y(layered, 0)
