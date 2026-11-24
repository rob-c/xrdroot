"""A graph sorted into x order, and ``std::is_sorted``, which is how a program checks it.

ROOT 6.40's own ``test_TGraph_sorting`` sorts a million-point graph of each kind it has and then
asks ``std::is_sorted`` over the graph's own arrays - ``graph.GetX(), graph.GetX() + n`` - so what
is tested here is both halves: the algorithm over a range this kit can name, and the sort that
carries every array a graph keeps.
"""

from __future__ import annotations

import contextlib
import io

import pytest

import xrdroot.pyroot as ROOT
from cintfake import fake
from pyrootgraphics import fresh_session, gaussian  # noqa: F401
from xrdroot.cppstd.objects import sorted_range
from xrdroot.pycint import Refusal, translate
from xrdroot.pycint.execute import run_source
from xrdroot.pycxx.infer import _named_type


def _printed(body: str) -> str:
    """What a translated program prints, its C++ run as the kit runs it."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        run_source("void t() {\n" + body + "\n}\n", "t.C", root=fake())
    return buffer.getvalue()


def test_a_sorted_range_is_what_cpp_means_by_it_and_an_equal_pair_is_not_out_of_order():
    """``std::is_sorted``: no element less than the one before it - which equal neighbours do
    not break, and one step down does."""
    assert sorted_range([1.0, 2.0, 2.0, 3.0], 0, 4) is True
    assert sorted_range([1.0, 3.0, 2.0], 0, 3) is False
    assert sorted_range([9.0, 1.0, 2.0], 1, None) is True  # the range stops at the container's end


def test_a_range_is_sorted_by_the_comparison_the_program_gave():
    """A third argument is C++'s predicate - ``less(a, b)`` - and the range is in order when no
    later element is less than the one before it, so a reversed comparison asks for descending."""
    assert sorted_range([3, 2, 1], 0, 3, lambda a, b: a > b) is True  # descending asked for
    assert sorted_range([1, 2, 3], 0, 3, lambda a, b: a > b) is False


def test_std_is_sorted_over_a_c_array_translates_and_runs():
    """``std::is_sorted(a, a + n)``: a C array's two ends, answered by the one pass."""
    printed = _printed(
        "int a[3] = {1, 2, 3}; int b[3] = {3, 2, 1};"
        'printf("%d %d\\n", std::is_sorted(a, a + 3), std::is_sorted(b, b + 3));'
    )
    assert printed == "1 0\n"


def test_std_is_sorted_with_the_wrong_number_of_arguments_is_refused():
    """Two ends and optionally a comparison: anything else is not a range, and is said so."""
    with pytest.raises(Refusal, match=r"std::is_sorted with 1 arguments"):
        translate("void t() { int a[2]; std::is_sorted(a); }", "t.C")


def test_a_return_type_a_dialect_names_with_a_star_is_a_pointer():
    """A graph's ``GetX`` answers ``Double_t *``, and a pointer is what makes a range of it: the
    entry says so with the ``*`` a declaration has, and the inferred type carries the pointer."""
    assert _named_type("Double_t*").pointer == 1
    assert _named_type("Double_t*").name == "Double_t"
    plain = _named_type("TMatrixD")
    assert (plain.name, plain.pointer) == ("TMatrixD", 0)


def test_a_graph_sorted_by_x_carries_its_points_and_its_bars_with_them():
    """``TGraph::Sort``: ROOT sorts a permutation of the point numbers and moves every array by
    it, so a point's y and its error bars stay its own."""
    graph = ROOT.TGraphErrors(3)
    for at, (x, y) in enumerate([(3.0, 30.0), (1.0, 10.0), (2.0, 20.0)]):
        graph.SetPoint(at, x, y)
        graph.SetPointError(at, 0.1 * (at + 1), 0.5 * (at + 1))
    graph.Sort()
    assert list(graph.GetX()) == [1.0, 2.0, 3.0]
    assert list(graph.GetY()) == [10.0, 20.0, 30.0]
    assert list(graph.GetEX()) == [0.2, 0.30000000000000004, 0.1]  # pytest.approx below
    assert list(graph.GetEY()) == pytest.approx([1.0, 1.5, 0.5])
    assert graph.TestBit(graph.kIsSortedX) is True


def test_a_graph_sorts_by_y_and_the_other_way_round_with_roots_comparators():
    """``TGraph::CompareY`` and ``ascending`` false are ROOT's other two ways: the comparator
    answers "is left greater", so a *descending* order is ``ascending = kFALSE`` over it."""
    graph = ROOT.TGraph(3)
    for at, (x, y) in enumerate([(1.0, 30.0), (2.0, 10.0), (3.0, 20.0)]):
        graph.SetPoint(at, x, y)
    graph.Sort(ROOT.TGraph.CompareY, False)
    assert list(graph.GetY()) == [30.0, 20.0, 10.0]
    assert list(graph.GetX()) == [1.0, 3.0, 2.0]  # each point's x went with its y


def test_a_sort_of_a_window_leaves_the_points_outside_it_where_they_are():
    """``Sort(..., low, high)`` is ROOT's sub-range: ``std::stable_sort`` over that window of the
    point numbers, and ``UpdateArrays`` writing the window back - the rest untouched."""
    graph = ROOT.TGraph(4)
    for at, value in enumerate([9.0, 7.0, 1.0, 3.0]):
        graph.SetPoint(at, value, value * 10.0)
    graph.Sort(None, True, 1, 2)
    assert list(graph.GetX()) == [9.0, 1.0, 7.0, 3.0]
    assert list(graph.GetY()) == [90.0, 10.0, 70.0, 30.0]


def test_only_the_whole_graph_ascending_by_x_is_marked_sorted():
    """ROOT sets ``kIsSortedX`` in the one case ``Eval`` may binary-search on - the default
    comparator, ascending, over the whole graph - and leaves it alone otherwise."""
    whole = ROOT.TGraph(2)
    whole.SetPoint(0, 2.0, 1.0)
    whole.SetPoint(1, 1.0, 2.0)
    whole.Sort()
    assert whole.TestBit(whole.kIsSortedX) is True
    by_y = ROOT.TGraph(2)
    by_y.SetPoint(0, 2.0, 1.0)
    by_y.SetPoint(1, 1.0, 2.0)
    by_y.Sort(ROOT.TGraph.CompareY, False)
    assert by_y.TestBit(by_y.kIsSortedX) is False


def test_a_multi_error_graph_sorts_every_layer_and_the_x_errors_with_its_points():
    """``TGraphMultiErrors::Sort``: its own x errors and each layer of y errors are arrays ROOT's
    ``UpdateArrays`` moves too, so a sort that reordered only the points would give every point
    another point's bars."""
    graph = ROOT.TGraphMultiErrors(
        3, [3.0, 1.0, 2.0], [30.0, 10.0, 20.0],
        [0.3, 0.1, 0.2], [0.6, 0.4, 0.5], [30.0, 10.0, 20.0], [60.0, 40.0, 50.0],
    )  # fmt: skip
    assert list(graph.GetEXlow()) == [0.3, 0.1, 0.2]  # the constructor's own order
    graph.Sort()
    assert list(graph.GetX()) == [1.0, 2.0, 3.0]
    assert list(graph.GetEXlow()) == pytest.approx([0.1, 0.2, 0.3])
    assert list(graph.GetEXhigh()) == pytest.approx([0.4, 0.5, 0.6])
    assert [list(layer[0]) for layer in graph._layers][-1] == pytest.approx([10.0, 20.0, 30.0])
    assert [list(layer[1]) for layer in graph._layers][-1] == pytest.approx([40.0, 50.0, 60.0])
