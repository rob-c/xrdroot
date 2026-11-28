"""A graph as a picture and a sequence: its points, its attached functions, and its text drawing.

``Graph`` is ROOT's ``TGraph`` in this kit: a pair of arrays plus whatever bars the class adds, and
the things a reader touches first - how many points it has, what the n-th one is, iterating it, the
functions hung on it, and the character grid ``text`` draws for a log file. Each is reachable from a
graph built in code, so nothing here opens a file and the test may sit inside the ``hist``
component's own isolation scope.

The refusals are ROOT's own rules, each asserted as the full sentence it is: a graph written without
its points, one whose ``fNpoints`` outruns its arrays, a bar count that is not one per point, and a
layered graph whose y errors are not one pair until its mode says how they add.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot import UnsupportedFeatureError
from xrdroot.errors import FormatError
from xrdroot.graph import Graph


def _line() -> Graph:
    """``y = 2x`` through four points, which every test below reads."""
    return Graph.new("g", [0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0])


def test_a_graph_is_a_sequence_of_its_points():
    """``len``, ``g[i]``, iteration and ``GetPoint``-as-``points()`` are the same four pairs, and
    they are the arrays the graph holds rather than a copy of them."""
    line = _line()
    assert (len(line), line[0], line[-1]) == (4, (0.0, 0.0), (3.0, 6.0))
    assert list(line) == [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]
    assert line.points() == list(line)
    assert repr(line) == "<TGraph 'g' of 4 points>"


def test_a_graph_prints_the_class_its_name_and_its_point_count():
    """ROOT names the class it really is, so a graph with even bars says ``TGraphErrors``."""
    made = Graph.new("e", [0.0, 1.0], [1.0, 2.0], yerr=[0.1, 0.2])
    assert repr(made) == "<TGraphErrors 'e' of 2 points>"
    assert (made.name, made.title) == ("e", "")


def test_the_functions_hung_on_a_graph_are_listed_with_it():
    """``GetListOfFunctions`` is where a fit lands after ``Fit``; ``attach`` is that list's append,
    and a graph of none has an empty one rather than nothing."""
    line = _line()
    assert line.functions == []
    marker = object()
    line.attach(marker)
    assert line.functions == [marker]
    assert line.functions == line.functions  # the same functions, listed twice


def test_a_graph_written_without_its_points_is_refused_by_name():
    """A row that is not a graph at all, and one whose ``fNpoints`` outruns its arrays, each say so
    in a sentence naming the class - ROOT's own ``Error`` line rather than a bare ``IndexError``."""
    with pytest.raises(FormatError, match="a TGraph was written without its points"):
        Graph("TGraph", {"TNamed": {"fName": "g", "fTitle": ""}})
    short = {"TGraph": {"fNpoints": 3, "fX": np.array([1.0, 2.0]), "fY": np.array([1.0, 2.0])}}
    with pytest.raises(FormatError, match="a TGraph of 3 points holds only 2 of them"):
        Graph("TGraph", short)


def test_the_class_picks_itself_from_the_bars_it_is_given():
    """No bar makes a ``TGraph``, one per point a ``TGraphErrors``, and an uneven pair a
    ``TGraphAsymmErrors`` with the even one carried on both sides."""
    assert _line().classname == "TGraph"
    even = Graph.new("e", [0.0, 1.0], [1.0, 2.0], xerr=[0.5, 0.5], yerr=[0.1, 0.2])
    assert even.classname == "TGraphErrors"
    assert even.yerr[0].tolist() == [0.1, 0.2] and even.yerr[1].tolist() == [0.1, 0.2]
    uneven = Graph.new("a", [0.0, 1.0], [1.0, 2.0], yerr=[[0.1, 0.2], [0.3, 0.4]])
    assert uneven.classname == "TGraphAsymmErrors"
    assert uneven.yerr[1].tolist() == [0.3, 0.4]


def test_a_bar_count_that_is_not_one_per_point_is_refused_with_the_count():
    """ROOT has no meaning for two bars over three points, and the refusal says how many of each
    there are rather than truncating."""
    with pytest.raises(ValueError, match="yerr has 2 bars for 3 points"):
        Graph.new("g", [0.0, 1.0, 2.0], [1.0, 2.0, 3.0], yerr=[0.1, 0.2])
    with pytest.raises(ValueError, match="xerr has 1 low and 1 high bars for 3 points"):
        Graph.new("g", [0.0, 1.0, 2.0], [1.0, 2.0, 3.0], xerr=[[0.1], [0.2]])


def test_x_and_y_of_different_lengths_are_not_points():
    """A graph is a pair of arrays of one length, and the refusal counts both."""
    with pytest.raises(ValueError, match="4 x values and 3 y values are not points"):
        Graph.new("g", [0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0])


def test_a_layered_graph_s_y_errors_are_one_pair_only_once_the_mode_says_how():
    """``TGraphMultiErrors`` keeps a layer of bars per kind of uncertainty, so its ``yerr`` is not
    one pair until the summation mode is chosen - and the refusal points at ``.layers``.
    """
    layers = [([0.1, 0.2], [0.3, 0.2]), ([0.4, 0.5], [0.5, 0.6])]
    layered = Graph.from_layers("m", [0.0, 1.0], [1.0, 2.0], layers=layers)
    assert len(layered.layers) == 2
    with pytest.raises(UnsupportedFeatureError, match="keeps its y errors in 2 layers"):
        assert layered.yerr is not None  # the refusal, not a pair


def test_a_graph_draws_itself_with_characters_for_a_log_file():
    """``text`` is ROOT's character drawing as this kit has it: a row per height with its y label,
    the points marked, and the x range printed under the grid."""
    lines = _line().text(width=20, height=5).splitlines()
    assert len(lines) == 6  # five rows and the x axis
    assert all(line.endswith("|") for line in lines[:-1])
    assert "*" in "".join(lines)
    assert lines[-1].strip().startswith("0") and lines[-1].rstrip().endswith("3")


def test_a_graph_of_no_points_says_so_instead_of_drawing_an_empty_grid():
    """A graph of nothing has no range to draw, and ROOT's own answer is a sentence."""
    assert Graph.new("none", [], []).text() == "(a graph of no points)"
