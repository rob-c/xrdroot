"""What ROOT works out from a graph's points: ``Eval``, ``Integral``, ``Sort``, ``GetMean``.

``TGraph::Eval`` walks the points in the order they were added rather than
assuming them sorted, and extrapolates past an end along the two points its
walk found last - which, for points out of order, are not always the two
nearest. That is checked here as ROOT's code has it, beside the answers any
straight line gives.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from xrdroot import Graph, open_root

DATA = pathlib.Path(__file__).parent / "data"


def test_eval_is_the_straight_line_between_points_and_past_the_ends_along_them():
    g = Graph.new("g", [0, 1, 2], [0, 10, 40])
    assert g.eval([0.5, 1.5, 1.0]).tolist() == [5.0, 25.0, 10.0]
    assert g.eval(-1.0) == -10.0 and g.eval(3.0) == 70.0
    assert isinstance(g.eval(0.25), float)
    assert g.eval(np.array([[0.5], [3.0]])).tolist() == [[5.0], [70.0]]
    assert g.eval(math.nan) == 0.0  # a NaN is never beside a point, and ROOT gives the first


def test_eval_of_points_out_of_order_walks_them_as_root_does():
    g = Graph.new("g", [2, 0, 1], [40, 0, 10])
    assert g.eval([0.5, 1.5, 1.0]).tolist() == [5.0, 25.0, 10.0]
    # Past either end ROOT's walk has kept x = 0 and x = 2 as the two nearest,
    # x = 1 never bettering the one it came after, and extrapolates along them.
    assert g.eval(3.0) == 60.0 and g.eval(-1.0) == -20.0
    assert g.eval(math.nan) == 40.0


def test_eval_of_points_on_top_of_each_other_is_the_first_of_them():
    g = Graph.new("g", [1, 1, 0], [5, 7, 0])
    assert g.eval(2.0) == 7.0  # the two it extrapolates along share an x
    assert g.eval(0.5) == 2.5


def test_eval_of_no_points_is_zero_and_of_one_point_its_y():
    assert Graph.new("g", [], []).eval([1.0, 2.0]).tolist() == [0.0, 0.0]
    assert Graph.new("g", [3], [4]).eval(9.0) == 4.0


def test_the_integral_is_the_area_of_the_polygon_the_points_make():
    square = Graph.new("s", [0, 1, 1, 0], [0, 0, 1, 1])
    assert square.integral() == 1.0
    triangle = Graph.new("t", [0, 4, 0, 9], [0, 0, 3, 9])
    assert triangle.integral(0, 2) == 6.0 and triangle.integral(-5, 2) == 6.0
    assert triangle.integral(1, 99) == triangle.integral(1, 3)
    assert triangle.integral(2, 2) == 0.0 and triangle.integral(3, 1) == 0.0


def test_the_mean_and_rms_are_of_the_points_themselves():
    g = Graph.new("g", [1, 2, 3, 6], [2, 2, 5, 7], yerr=[9, 9, 9, 9])
    assert g.mean() == 3.0 and g.mean(1) == 4.0
    assert g.rms() == math.sqrt(3.5) and g.rms(1) == pytest.approx(math.sqrt(4.5))
    empty = Graph.new("e", [], [])
    assert (empty.mean(), empty.rms(1)) == (0.0, 0.0)
    with pytest.raises(ValueError, match="axis=2 is not an axis of a graph"):
        g.mean(2)


def test_sorting_puts_the_points_in_order_and_their_bars_with_them():
    g = Graph.new(
        "g", [3, 1, 2, 1], [30, 10, 20, 11], yerr=[3, 1, 2, 1.1], xerr=[0.3, 0.1, 0.2, 0.4]
    )
    g.sort()
    assert g.x.tolist() == [1, 1, 2, 3] and g.y.tolist() == [10, 11, 20, 30]  # ties kept in order
    assert g.yerr[0].tolist() == [1, 1.1, 2, 3] and g.xerr[1].tolist() == [0.1, 0.4, 0.2, 0.3]
    assert g.members["TGraph"]["fX"].tolist() == [1, 1, 2, 3]
    uneven = Graph.new("u", [2, 1], [4, 1], yerr=([0.1, 0.2], [0.3, 0.4]))
    uneven.sort()
    assert uneven.yerr[0].tolist() == [0.2, 0.1] and uneven.yerr[1].tolist() == [0.4, 0.3]
    plain = Graph.new("p", [2, 1], [4, 1])
    plain.sort()
    assert plain.points() == [(1.0, 1.0), (2.0, 4.0)]


def test_sorting_a_graph_of_layered_errors_moves_every_layer():
    with open_root(str(DATA / "tgme.root")) as root:
        gme = root["gme"]
        stat, _syst = gme.layers
        order = [1, 0, 2, 3, 4]
        gme.x[:] = gme.x[order]  # swap the first two, so that sorting swaps them back
        gme._core["fX"] = gme.x.copy()
        gme.sort()
        assert gme.x.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
        again, _ = gme.layers
        assert again[0].tolist() == [stat[0][1], stat[0][0], *stat[0][2:]]
