"""``TGraphMultiErrors``: layers of y errors, the sum they make, and the file they go in.

The numbers asserted here are ROOT 6.40's own - the ones
``hist/hist/test/TGraphMultiErrorsTests.cxx`` expects of its five points - and
the bytes the layers are written as were compared with ``tests/data/tgme.root``,
a file ROOT wrote, record for record.
"""

from __future__ import annotations

import io
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot import Graph, UnsupportedFeatureError, create, open_root

DATA = pathlib.Path(__file__).parent / "data"

#: The points and the two layers of y errors ROOT's own test fills.
X = [0, 1, 2, 3, 4]
Y = [0, 2, 4, 1, 3]
STAT = ([1, 0.5, 1, 0.5, 1], [0.5, 1, 0.5, 1, 2.0])
SYS = ([0.5, 0.4, 0.8, 0.3, 1.2], [0.6, 0.7, 0.6, 0.4, 0.8])


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def layered(mode: int = 0) -> Graph:
    return Graph.from_layers(
        "gme", X, Y, xerr=([0.3] * 5, [0.3] * 5), layers=[STAT, SYS], mode=mode
    )


def written(graph: Graph) -> bytes:
    buf = io.BytesIO()
    with create(buf) as out:
        out["gme"] = graph
    return buf.getvalue()


# -- what the layers sum to, against ROOT's own numbers ---------------------


def test_the_first_layer_alone_is_what_a_graph_answers_by_default():
    graph = layered()
    assert graph.classname == "TGraphMultiErrors"
    assert graph.sum_errors_mode == 0
    assert graph.error_y(0) == 0.7905694150420949
    assert graph.error_y(4) == 1.5811388300841898


def test_the_square_sum_adds_every_layer_in_quadrature():
    graph = layered(1)
    assert graph.error_y(1) == 0.9746794344808963
    assert graph.error_y(3) == 0.8660254037844387


def test_the_absolute_sum_adds_the_layers_as_they_stand():
    assert layered(2).error_y(2) == 1.4916433890176297


def test_a_point_the_graph_does_not_have_answers_minus_one():
    assert layered().error_y(5) == -1.0


def test_a_graph_written_without_a_sum_mode_answers_with_its_first_layer():
    plain = Graph.new("g", X, Y, yerr=[0.5] * 5)
    assert plain.sum_errors_mode == 0


def test_a_sum_mode_that_is_not_one_of_roots_three_is_refused():
    with pytest.raises(UnsupportedFeatureError, match="kSquareSum and kAbsSum"):
        layered(7).error_y(0)


# -- the file the layers go in ---------------------------------------------


def test_a_graph_of_layers_read_back_keeps_them():
    back = open_root(io.BytesIO(written(layered(1))))["gme"]
    assert back.classname == "TGraphMultiErrors"
    assert [list(map(float, side)) for side in back.layers[0]] == [[1, 0.5, 1, 0.5, 1], STAT[1]]
    assert [list(map(float, side)) for side in back.layers[1]] == [SYS[0], SYS[1]]
    assert (list(back.xerr[0]), list(back.xerr[1])) == ([0.3] * 5, [0.3] * 5)
    assert back.sum_errors_mode == 1
    assert back.error_y(1) == 0.9746794344808963


def test_the_bytes_the_layers_are_written_as_are_the_ones_root_writes():
    """``tgme.root`` is ROOT's own file: written again from its members, it is the same bytes.

    The layer records are the last thing in the object, so what is compared is everything
    from the first layer's record to the end of the record.
    """
    donor = open_root(str(DATA / "tgme.root"))
    with donor:
        held = donor["gme"]
        theirs = donor.key("gme").payload(donor._source)
    ours = open_root(io.BytesIO(written(Graph("TGraphMultiErrors", dict(held.members)))))
    mine = ours.key("gme").payload(ours._source)
    head = bytes.fromhex("4000005e0009")  # the record round a container of TArrayD
    assert mine[mine.index(head) :] == theirs[theirs.index(head) :]


def test_the_file_describes_the_arrays_its_layers_are():
    back = open_root(io.BytesIO(written(layered())))
    described = set(back._source.streamers())
    assert {"TGraphMultiErrors", "TArrayD", "TArray"} <= described


# -- the class ROOT's own test drives --------------------------------------


def test_the_class_sums_its_layers_the_way_root_does():
    graph = ROOT.TGraphMultiErrors(5, X, Y, [0.3] * 5, [0.3] * 5, *STAT)
    graph.AddYError(5, *SYS)
    assert graph.GetNYErrors() == 2
    assert graph.GetErrorY(0) == 0.7905694150420949
    graph.SetSumErrorsMode(ROOT.TGraphMultiErrors.kSquareSum)
    assert graph.GetErrorY(1) == 0.9746794344808963
    graph.SetSumErrorsMode(ROOT.TGraphMultiErrors.kAbsSum)
    assert graph.GetErrorY(2) == 1.4916433890176297
    assert graph.GetSumErrorsMode() == ROOT.TGraphMultiErrors.kAbsSum


def test_a_layer_deleted_leaves_the_ones_after_it_and_the_last_is_never_deleted():
    graph = ROOT.TGraphMultiErrors(5, X, Y, [0.3] * 5, [0.3] * 5, *STAT)
    graph.AddYError(5, *SYS)
    graph.SetSumErrorsMode(ROOT.TGraphMultiErrors.kAbsSum)
    graph.DeleteYError(1)
    assert graph.GetNYErrors() == 1
    graph.DeleteYError(0)  # the only layer left stays, as ROOT's does
    assert graph.GetNYErrors() == 1
    graph.DeleteYError(4)  # and one that was never there changes nothing
    graph.AddYError(5, *SYS)
    assert graph.GetNYErrors() == 2
    assert graph.GetErrorY(0) == 1.3152946437965904


def test_a_point_set_after_the_graph_was_cut_down_bars_and_all():
    graph = ROOT.TGraphMultiErrors(5, X, Y, [0.3] * 5, [0.3] * 5, *STAT)
    graph.AddYError(5, *SYS)
    graph.SetSumErrorsMode(ROOT.TGraphMultiErrors.kAbsSum)
    graph.Set(4)
    graph.Set(5)
    graph.SetPoint(4, X[4], Y[4])
    graph.SetPointEX(4, 0.3, 0.3)
    graph.SetPointEY(4, 0, STAT[0][4], STAT[1][4])
    graph.SetPointEY(4, 1, SYS[0][4], SYS[1][4])
    assert graph.GetErrorY(4) == 2.5179356624028344


def test_a_point_or_a_layer_the_graph_does_not_have_is_left_alone():
    graph = ROOT.TGraphMultiErrors(5, X, Y, [0.3] * 5, [0.3] * 5, *STAT)
    graph.SetPointEX(-1, 9.0, 9.0)
    graph.SetPointEY(0, 5, 9.0, 9.0)
    assert graph.GetErrorXlow(0) == 0.3
    assert graph.GetErrorX(-1) == -1.0
    assert graph.GetErrorXlow(9) == -1.0
    assert graph.GetErrorXhigh(9) == -1.0
    assert graph.GetErrorY(0, 5) == -1.0
    assert graph.GetErrorY(9, 0) == -1.0
    assert graph.GetErrorY(0, 0) == 0.7905694150420949
    graph.SetPointEY(9, 0, 0.1, 0.2)  # a point past the end grows the graph to it
    assert graph.GetN() == 10
    assert graph.GetErrorY(9, 0) == float(np.sqrt((0.01 + 0.04) / 2))


def test_a_graph_of_no_points_says_so_and_keeps_no_layer():
    empty = ROOT.TGraphMultiErrors()
    assert empty.GetN() == 0 and empty.GetNYErrors() == 0
    assert empty.GetAttLine(0) is None and empty.GetAttFill(0) is None


def test_a_layer_has_a_line_and_a_fill_of_its_own():
    graph = ROOT.TGraphMultiErrors(5, X, Y, [0.3] * 5, [0.3] * 5, *STAT)
    graph.AddYError(5, *SYS)
    graph.SetLineColor(ROOT.kRed)
    graph.GetAttLine(0).SetLineColor(ROOT.kRed)
    graph.GetAttLine(1).SetLineColor(ROOT.kBlue)
    graph.GetAttFill(1).SetFillStyle(3004)
    members = graph._xrd.members
    assert members["TGraph"]["TAttLine"]["fLineColor"] == 632
    assert [one["fLineColor"] for one in members["fAttLine"]] == [632, 600]
    assert [one["fFillStyle"] for one in members["fAttFill"]] == [0, 3004]


def test_a_graph_written_to_a_file_still_open_is_read_back_out_of_it():
    holder = ROOT.TFile("open.root", "RECREATE")
    graph = ROOT.TGraphMultiErrors(5, X, Y, [0.3] * 5, [0.3] * 5, *STAT)
    graph.AddYError(5, *SYS)
    graph.SetLineColor(ROOT.kRed)
    graph.GetAttLine(1).SetLineColor(ROOT.kBlue)
    graph.GetAttFill(1).SetFillStyle(0)
    assert graph.Write("gme") == 1
    back = holder.Get("gme")
    assert isinstance(back, ROOT.TGraphMultiErrors)
    back.SetSumErrorsMode(ROOT.TGraphMultiErrors.kSquareSum)
    assert back.GetErrorY(1) == 0.9746794344808963
    assert back.GetAttLine(1).GetLineColor() == 600
    holder.Close()


def test_a_graph_read_back_and_written_again_keeps_its_layers():
    holder = ROOT.TFile("again.root", "RECREATE")
    graph = ROOT.TGraphMultiErrors(5, X, Y, [0.3] * 5, [0.3] * 5, *STAT)
    graph.AddYError(5, *SYS)
    graph.Write("gme")
    holder.Get("gme")
    holder.Close()
    with open_root("again.root") as f:
        assert [len(one[0]) for one in f["gme"].layers] == [5, 5]


def test_roots_own_file_of_layers_is_the_class_its_colours_and_its_numbers():
    """``tgme.root`` is C++ ROOT's - read through the class, not through the plain reader."""
    holder = ROOT.TFile(str(DATA / "tgme.root"), "READ")
    graph = holder.Get("gme")
    assert isinstance(graph, ROOT.TGraphMultiErrors)
    assert (graph.GetN(), graph.GetNYErrors()) == (5, 2)
    assert [one.GetLineColor() for one in (graph.GetAttLine(0), graph.GetAttLine(1))] == [632, 600]
    assert graph.GetErrorY(0) == pytest.approx(0.79056941504209488)
    graph.SetSumErrorsMode(ROOT.TGraphMultiErrors.kSquareSum)
    assert graph.GetErrorY(1) == pytest.approx(0.97467943448089633)
    holder.Close()


def test_a_point_set_past_the_end_grows_the_graph_as_root_grows_it():
    graph = ROOT.TGraphMultiErrors(5, X, Y, [0.3] * 5, [0.3] * 5, *STAT)
    graph.SetPointEX(9, 1.0, 2.0)
    assert graph.GetN() == 10
    assert (graph.GetErrorXlow(9), graph.GetErrorXhigh(9)) == (1.0, 2.0)


def test_the_x_error_is_the_mean_of_the_two_sides_squares():
    graph = ROOT.TGraphMultiErrors(5, X, Y, [0.3] * 5, [0.6] * 5, *STAT)
    assert graph.GetErrorX(0) == pytest.approx(float(np.sqrt((0.3**2 + 0.6**2) / 2.0)))
    assert graph.GetErrorX(99) == -1.0


def test_a_name_the_file_does_not_hold_is_no_object_at_all():
    holder = ROOT.TFile("absent.root", "RECREATE")
    assert holder.Get("nothing_by_that_name") is None
    holder.Close()


def test_a_layer_the_file_wrote_no_look_for_keeps_the_ones_it_starts_with():
    graph = ROOT.TGraphMultiErrors(5, X, Y, [0.3] * 5, [0.3] * 5, *STAT)
    graph.AddYError(5, *SYS)
    held = graph._holds_attributes(SimpleNamespace(members={"fAttFill": None, "fAttLine": None}))
    assert len(held) == 2
    assert all(set(layer) == {"TAttFill", "TAttLine"} for layer in held)


def test_saving_its_layers_as_text_is_refused_by_name_rather_than_written_short():
    graph = ROOT.TGraphMultiErrors(5, X, Y, [0.3] * 5, [0.3] * 5, *STAT)
    graph.AddYError(5, *SYS)
    with pytest.raises(UnsupportedFeatureError, match="keeps a bar per layer"):
        graph.SaveAs("layers.csv")


def test_the_points_alone_are_a_text_file_that_loses_no_layer():
    graph = ROOT.TGraphMultiErrors(5, X, Y, [0.3] * 5, [0.3] * 5, *STAT)
    graph.SaveAs("points.csv", "noerrors")
    assert pathlib.Path("points.csv").read_text().splitlines()[0] == "0,0"
