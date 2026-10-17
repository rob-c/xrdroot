"""Tree scatter ownership, weights and drawing options from ROOT 6.40.04."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.pyroot.trees import scatter


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    from xrdroot.pyroot.graphics import hook

    with contextmanager(fresh)(tmp_path):
        hook.install()
        yield


def _tree():
    tree = ROOT.TTree("tree", "tree")
    values = {name: np.zeros(1) for name in ("x", "y", "w")}
    for name, value in values.items():
        tree.Branch(name, value, name + "/D")
    for row in ((1, 3, 1), (2, 8, 0), (3, 15, -2), (4, 24, .5)):
        for value, number in zip(values.values(), row, strict=False):
            value[0] = number
        tree.Fill()
    return tree


@pytest.mark.parametrize("named", ("", ">>h", ">>h(5,0,5,5,0,25)"))
@pytest.mark.parametrize("option", ("", "p", "l", "*", "goffp", "same", "p same"))
def test_a_two_variable_draw_keeps_the_selected_points_and_roots_histogram(named, option):
    canvas = ROOT.TCanvas("canvas", "canvas", 400, 300)
    tree = _tree()
    tree.SetMarkerStyle(5)
    tree.SetMarkerColor(2)
    tree.SetMarkerSize(1.25)
    tree.SetLineColor(4)
    tree.SetLineStyle(2)
    tree.SetLineWidth(3)
    assert tree.Draw("y:x" + named, "w", option) == 3
    np.testing.assert_array_equal(tree.GetV1(), [3, 15, 24])
    np.testing.assert_array_equal(tree.GetV2(), [1, 3, 4])
    assert tree.GetSelectedRows() == 3
    histogram = tree.GetHistogram()
    _check_histogram(histogram, named, option)
    graph = canvas.FindObject("Graph")
    if "goff" in option:
        assert graph is None and len(canvas.GetListOfPrimitives()) == 0
        return
    _check_graph(graph, option)
    _check_frame(canvas, histogram, graph, named, option)


def _check_histogram(histogram, named, option):
    assert histogram.ClassName() == "TH2F"
    assert histogram.GetEntries() == (3 if named else 0)
    assert histogram.TestBit(ROOT.TH1.kNoStats) == (not named)
    if "same" in option:
        assert (histogram.GetXaxis().GetXmin(), histogram.GetXaxis().GetXmax()) == (0, 1)
        assert histogram.GetSumOfWeights() == 0
    else:
        assert histogram.GetSumOfWeights() == (-.5 if named else 0)


def _check_graph(graph, option):
    assert isinstance(graph, ROOT.TGraph)
    assert graph.GetTitle() == "Graph"
    np.testing.assert_array_equal(graph.GetX(), [1, 3, 4])
    np.testing.assert_array_equal(graph.GetY(), [3, 15, 24])
    assert (graph.GetMarkerColor(), graph.GetMarkerSize()) == (2, 1.25)
    assert graph.GetMarkerStyle() == (3 if option == "*" else 5)
    assert (graph.GetLineColor(), graph.GetLineStyle(), graph.GetLineWidth()) == (4, 2, 3)


def _check_frame(canvas, histogram, graph, named, option):
    if "same" in option:
        assert list(canvas.GetListOfPrimitives()) == [graph]
    else:
        frame = canvas.FindObject("h" if named else "htemp")
        assert (frame.GetEntries(), frame.TestBit(ROOT.TH1.kNoStats)) == (0, True)
        assert (frame is histogram) == (not named)
        assert canvas.primitives[0][1] == "axis"
        assert canvas.primitives[-1][1] == ("p" if option in ("", "*") else option)


def test_a_named_scatter_adds_weighted_fills_without_adding_old_points():
    canvas = ROOT.TCanvas("canvas", "canvas", 400, 300)
    tree = _tree()
    assert tree.Draw("y:x>>h(5,0,5,5,0,25)", "w", "goff") == 3
    histogram = tree.GetHistogram()
    assert tree.Draw("y:x>>+h", "w") == 3
    assert tree.GetHistogram() is histogram
    assert (histogram.GetEntries(), histogram.GetSumOfWeights()) == (6, -1)
    assert canvas.FindObject("Graph").GetN() == 3
    assert canvas.FindObject("h").GetEntries() == 3


def test_same_preserves_the_pad_and_a_range_alias_and_tree_weight_select_the_points():
    canvas = ROOT.TCanvas("canvas", "canvas", 400, 300)
    background = ROOT.TH2F("background", "background", 10, 0, 10, 10, 0, 30)
    background.Draw("axis")
    before = list(canvas.GetListOfPrimitives())
    tree = _tree()
    tree.SetAlias("twice", "2*x")
    tree.SetWeight(2)
    assert tree.Draw("y:twice", "w", "same", 2, 1) == 1
    graph = canvas.FindObject("Graph")
    assert list(canvas.GetListOfPrimitives()) == [*before, graph]
    np.testing.assert_array_equal(graph.GetX(), [6])
    np.testing.assert_array_equal(graph.GetY(), [15])


def test_the_graph_can_own_its_axes_when_option_a_requests_them():
    canvas = ROOT.TCanvas("canvas", "canvas", 400, 300)
    tree = _tree()
    assert tree.Draw("y:x", "w", "ap") == 3
    graph = canvas.FindObject("Graph")
    assert list(canvas.GetListOfPrimitives()) == [graph]
    assert tree.GetHistogram() is graph.GetHistogram()


def test_get_histogram_is_empty_before_drawing_and_tracks_an_ordinary_draw():
    tree = _tree()
    assert tree.GetHistogram() is None
    assert tree.Draw("x", "", "goff") == 4
    assert tree.GetHistogram().GetEntries() == 4


def test_a_kept_histograms_existing_statistics_setting_survives_a_scatter():
    tree = _tree()
    histogram = ROOT.TH2F("kept", "kept", 5, 0, 5, 5, 0, 25)
    histogram.SetStats(False)
    assert tree.Draw("y:x>>kept", "w") == 3
    assert tree.GetHistogram() is histogram
    assert histogram.TestBit(ROOT.TH1.kNoStats)


@pytest.mark.parametrize("option", ("goff", "col", "hist", "prof", "box", "surf", "scat"))
def test_binned_options_keep_the_ordinary_histogram_or_profile(option):
    canvas = ROOT.TCanvas("canvas", "canvas", 400, 300)
    tree = _tree()
    assert tree.Draw("y:x", "w", option) == 3
    assert canvas.FindObject("Graph") is None
    assert tree.GetHistogram().GetEntries() == 3


def test_a_preexisting_profile_is_filled_as_a_profile_with_the_default_option():
    canvas = ROOT.TCanvas("canvas", "canvas", 400, 300)
    tree = _tree()
    profile = ROOT.TProfile("profile", "profile", 5, 0, 5)
    assert tree.Draw("y:x>>profile", "w") == 3
    assert tree.GetHistogram() is profile
    assert canvas.FindObject("Graph") is None


def test_scatter_coordinates_are_evaluated_once_even_when_get_v_is_called(monkeypatch):
    calls = []
    original = scatter.treedraw.batches

    def batches(*args):
        calls.append(args)
        yield from original(*args)

    monkeypatch.setattr(scatter.treedraw, "batches", batches)
    tree = _tree()
    tree.Draw("y:x", "w")
    np.testing.assert_array_equal(tree.GetV1(), [3, 15, 24])
    np.testing.assert_array_equal(tree.GetV2(), [1, 3, 4])
    assert len(calls) == 1


def test_a_scatter_rejects_a_negative_first_entry_before_reading():
    with pytest.raises(ValueError, match="neither can be negative"):
        _tree().Draw("y:x", "w", "", 2, -1)


@pytest.mark.parametrize("option", ("", "p", "goffp", "same"))
def test_an_empty_selection_has_no_points_and_no_histogram_fills(option):
    canvas = ROOT.TCanvas("canvas", "canvas", 400, 300)
    tree = _tree()
    assert tree.Draw("y:x", "0", option) == 0
    assert tree.GetSelectedRows() == 0
    assert tree.GetHistogram().GetEntries() == 0
    assert canvas.FindObject("Graph") is None


@pytest.mark.parametrize("option", ("", "p", "goffp", "same"))
def test_an_empty_selection_clears_old_points_unless_drawing_is_preserved(option):
    canvas = ROOT.TCanvas("canvas", "canvas", 400, 300)
    tree = _tree()
    assert tree.Draw("y:x", "w") == 3
    graph = canvas.FindObject("Graph")
    assert graph.GetN() == 3
    assert tree.Draw("y:x", "0", option) == 0
    assert canvas.FindObject("Graph") is (graph if option in ("goffp", "same") else None)


def test_a_live_histogram_drawn_as_axes_does_not_create_a_statistics_box():
    pytest.importorskip("matplotlib")
    from xrdroot.pyroot.graphics.snapshot import model

    canvas = ROOT.TCanvas("canvas", "canvas", 400, 300)
    histogram = ROOT.TH2F("axes", "axes", 5, 0, 5, 5, 0, 25)
    histogram.Fill(1, 3)
    histogram.Draw("axis")
    saved = model(canvas)
    assert not any(obj.classname == "TPaveStats" for obj, _ in saved.primitives)
    figure = saved.plot()
    assert len(figure.axes) == 1
    assert not figure.axes[0].collections
