"""ROOT's own ``SavePrimitive`` text: the lines a macro is made of, and the classes that write them.

ROOT writes a canvas as ``TCanvas::SaveSource`` does - a header, the ``gStyle`` lines, the
canvas, then each object's own ``SavePrimitive`` - so what a histogram, a profile, a ``TH2Poly``
and a graph write here is checked against the 6.40 oracle's own ``C_Output.C``, the macro it
writes for ROOT's ``testMapCppName`` canvas.
"""

from __future__ import annotations

from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from pyrootgraphics import fresh_session, gaussian  # noqa: F401
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.core.objects import SAVE_STATE, TObject, reset_save_state


def _lines(body: Any) -> list[str]:
    """What ``body`` writes into an empty macro, its state started fresh."""
    reset_save_state()
    out: list[str] = []
    body(out)
    return out


def test_a_class_is_spelled_out_once_and_the_second_object_of_it_is_not():
    """``SavePrimitiveConstructor``: ``gROOT->ClassSaved`` - ROOT's second ``TGraph`` of a macro
    is ``graph = new TGraph(...)`` with no type in front of it, and an empty line first when it
    is asked for."""
    out = _lines(lambda out: (
        TObject.SavePrimitiveConstructor(out, "TGraph", "a", "1"),
        TObject.SavePrimitiveConstructor(out, "TGraph", "b", "2", True),
    ))  # fmt: skip
    assert out == [
        "   TGraph *a = new TGraph(1);",
        "   ",
        "   b = new TGraph(2);",
    ]
    assert SAVE_STATE["classes"] == {"TGraph"}


def test_an_array_is_a_vector_of_doubles_ten_to_a_line_once_it_is_long():
    """``SavePrimitiveVector``: ROOT's own spelling of the numbers, and its break after the
    tenth once there are more than fifteen of them."""
    short = _lines(lambda out: TObject.SavePrimitiveVector(out, "graph_x", [0, 0.123, 1.5]))
    assert short == ["   std::vector<Double_t> graph_x_vect0{ 0, 0.123, 1.5 };"]
    long = _lines(lambda out: TObject.SavePrimitiveVector(out, "v", list(range(17))))
    assert long == [
        "   std::vector<Double_t> v_vect0{\n"
        "      0, 1, 2, 3, 4, 5, 6, 7, 8, 9,\n"
        "      10, 11, 12, 13, 14, 15, 16\n"
        "   };"
    ]


def test_an_array_asked_for_its_data_is_skipped_when_it_holds_only_zeros():
    """``SavePrimitiveVector``'s ``flag`` 111: ``nullptr`` for an array of nothing but zeros, and
    otherwise the name to call ``data()`` on - with an empty line first when ``flag`` is 1."""
    out: list[str] = []
    reset_save_state()
    zeros = TObject.SavePrimitiveVector(out, "empty", [0.0, 0.0], 111)
    given = TObject.SavePrimitiveVector(out, "kept", [0.0, 2.0], 111)
    TObject.SavePrimitiveVector(out, "spaced", [1.0], 1)
    # an array of nothing is not numbered: ROOT answers before it takes a name of its own
    assert (zeros, given) == ("nullptr", "kept_vect0.data()")
    assert out == [
        "   std::vector<Double_t> kept_vect0{ 0, 2 };",
        "   ",
        "   std::vector<Double_t> spaced_vect1{ 1 };",
    ]


def test_the_name_and_title_a_macro_gives_back_are_one_line_each():
    """``TNamed::SavePrimitiveNameTitle``: both, with a quote escaped as C++ needs it."""
    named = ROOT.TNamed('a"b', "t")
    out = _lines(lambda out: named.SavePrimitiveNameTitle(out, "graph"))
    assert out == ['   graph->SetName("a\\"b");', '   graph->SetTitle("t");']


def test_a_graph_refuses_to_be_saved_as_a_macro_when_it_carries_errors():
    """ROOT writes a graph with errors by its own ``SavePrimitive``, a ``SetPointError`` per
    point; until that is here, the class is refused rather than written without them."""
    graph = ROOT.TGraphErrors(1)
    graph.SetPoint(0, 1.0, 2.0)
    with pytest.raises(UnsupportedFeatureError, match="A TGraphErrors is not yet written"):
        graph.SavePrimitive([])


def test_a_profile_whose_bins_were_filled_writes_its_entries_before_its_means():
    """``TProfile::SavePrimitive``: the constructor is given the error option, and a bin's count
    (``SetBinEntries``) comes before its mean - a mean is nothing without the count."""
    profile = ROOT.TProfile("prof", "a profile", 2, 0.0, 2.0)
    profile.Fill(0.5, 2.0)
    profile.Fill(0.5, 4.0)
    profile.Fill(1.5, 6.0)
    profile.SetDirectory(None)
    out = _lines(lambda out: profile.SavePrimitive(out, ""))
    name = out[1].split()[1].lstrip("*")
    assert out[0] == "   "
    assert out[1].strip() == f'TProfile *{name} = new TProfile("prof", "a profile", 2, 0, 2, "");'
    assert f"   {name}->SetBinEntries(1,2);" in out
    assert f"   {name}->SetBinEntries(2,1);" in out
    counted, mean = f"   {name}->SetBinEntries(1,2);", f"   {name}->SetBinContent(1,3);"
    assert out.index(counted) < out.index(mean)
    assert f"   {name}->Draw();" in out


def test_a_two_dimensional_profile_is_refused_by_name_rather_than_written_wrongly():
    """ROOT's ``TProfile2D::SavePrimitive`` writes two axes' bin entries, which is not written
    here yet: the class says so instead of writing a one-dimensional profile's text."""
    with pytest.raises(UnsupportedFeatureError, match="A TProfile2D is not yet written"):
        ROOT.TProfile2D("p2", "p2", 2, 0.0, 2.0, 2, 0.0, 2.0).SavePrimitive([])


def test_a_th2poly_writes_a_graph_for_each_of_its_bins_polygons():
    """``TH2Poly::SavePrimitive``: a bin's polygon goes in as a ``TGraph`` and is added with
    ``AddBin``, and the contents follow the bins - ROOT's own order."""
    poly = ROOT.TH2Poly("poly", "a poly", 4, 0.0, 2.0, 4, 0.0, 2.0)
    poly.AddBin(0.0, 0.0, 1.0, 1.0)
    poly.AddBin(1.0, 1.0, 2.0, 2.0)
    poly.SetBinContent(2, 3.0)
    poly.Sumw2()
    poly.SetBinError(2, 0.5)
    out = _lines(lambda out: poly.SavePrimitive(out, "same"))
    name = out[1].split()[1].lstrip("*")
    assert out[1].strip() == f'TH2Poly *{name} = new TH2Poly("poly", "a poly", 4, 0, 2, 4, 0, 2);'
    assert "   std::vector<Double_t> graph_x_vect0{ 0, 0, 1, 1, 0 };" in out
    assert f"   {name}->AddBin(graph);" in out
    assert f"   {name}->SetBinContent(2,3);" in out
    assert f"   {name}->SetBinError(2,0.5);" in out
    assert out[-1] == f'   {name}->Draw("same");'


def test_a_th2poly_has_no_directory_and_no_draw_option_of_its_own():
    """``GetDirectory`` and ``GetOption`` as ROOT's ``TH1`` has them, and ``SetStats(kFALSE)``
    setting the ``kNoStats`` bit a macro writes as ``SetStats(0)``."""
    poly = ROOT.TH2Poly("poly", "p", 2, 0.0, 1.0, 2, 0.0, 1.0)
    assert (poly.GetDirectory(), poly.GetOption()) == (None, "")
    poly.SetStats(False)
    assert poly.TestBit(poly.kNoStats) is True
    out = _lines(lambda out: poly.SavePrimitive(out, ""))
    assert f"   {out[1].split()[1].lstrip('*')}->SetStats(0);" in out


def test_a_graph_writes_a_title_for_each_axis_that_was_given_one():
    """The axis titles a graph carries, which ROOT reaches through the frame's histogram: here
    they are the two ``SetTitle`` calls that put them back."""
    graph = ROOT.TGraph(1)
    graph.SetPoint(0, 1.0, 2.0)
    graph.GetXaxis().SetTitle("energy")
    graph.GetYaxis().SetTitle("counts")
    out = _lines(lambda out: graph.SavePrimitive(out, ""))
    assert '   graph->GetXaxis()->SetTitle("energy");' in out
    assert '   graph->GetYaxis()->SetTitle("counts");' in out


def test_a_canvas_is_written_as_savesource_writes_one(tmp_path):
    """``TCanvas::SaveSource``: the cling pragma, ``void`` the name the file gives, the comment
    naming the canvas, the style, the palette, the canvas' own frame, and the tail that marks it
    drawn and selected."""
    canvas = ROOT.TCanvas("framed", "a canvas", 300, 200)
    ROOT.TLine(0.1, 0.2, 0.9, 0.8).Draw()
    path = tmp_path / "framed.C"
    canvas.SaveAs(str(path))
    lines = path.read_text().splitlines()
    assert lines[:5] == [
        "#ifdef __CLING__", "#pragma cling optimize(0)", "#endif",
        "void framed()", "{",
    ]  # fmt: skip
    assert lines[5] == "//=========Macro generated from canvas: framed/a canvas"
    assert lines[6].startswith("//=========  (") and lines[6].endswith(") by ROOT version 6.40.04")
    assert lines[7] == '   TCanvas *framed = new TCanvas("framed", "a canvas", 10, 10, 300, 200);'
    assert lines[8:13] == [
        "   gStyle->SetOptFit(0);",
        "   gStyle->SetOptStat(1111);",
        "   gStyle->SetOptTitle(1);",
        "   TColor::SetPalette(57, nullptr);",
        "   framed->Range(0,0,1,1);",
    ]
    assert "   framed->Modified();" in lines
    assert "   framed->SetSelected(framed);" in lines
    assert lines[-1] == "}"


def test_a_macro_for_a_file_name_that_starts_with_a_digit_is_left_unnamed(tmp_path):
    """``SaveSource``: a function name that is no C++ name is left out, and the macro's
    statements stand in the braces alone."""
    canvas = ROOT.TCanvas("digits", "d", 300, 200)
    path = tmp_path / "2fast.C"
    canvas.SaveAs(str(path))
    lines = path.read_text().splitlines()
    assert lines[:4] == ["#ifdef __CLING__", "#pragma cling optimize(0)", "#endif", "{"]


def test_a_canvas_whose_fill_and_border_are_roots_own_writes_neither(tmp_path):
    """``TPad::SavePrimitive``: a fill of -1 and a border of 1 and 4 are ROOT's defaults, and a
    ``Set`` for one of them is written only when it was changed."""
    canvas = ROOT.TCanvas("plain", "p", 300, 200)
    canvas.members.update(fFillColor=-1, fFillStyle=-1, fBorderMode=1, fBorderSize=4)
    path = tmp_path / "plain.C"
    canvas.SaveAs(str(path))
    written = path.read_text()
    assert "plain->SetFillColor" not in written
    assert "plain->SetFillStyle" not in written
    assert "plain->SetBorderMode" not in written
    assert "plain->SetBorderSize" not in written


def test_a_graph_of_no_points_is_made_empty_and_written_with_no_arrays():
    """``TGraph::SavePrimitive``: a graph with no points has no arrays to hand the constructor,
    and ROOT puts an empty line before it as it does for a histogram's."""
    graph = ROOT.TGraph()
    out = _lines(lambda out: graph.SavePrimitive(out, ""))
    assert out[:2] == ["   ", "   TGraph *graph = new TGraph();"]
    assert not [line for line in out if "graph_x_vect" in line]


def test_a_profile_keeps_its_squared_weights_from_its_constructor():
    """ROOT's ``TProfile`` allocates them when it is booked, so its ``SavePrimitive`` has no
    ``save_errors`` to ask about - measured: an unfilled profile's ``GetSumw2N()`` is its cells,
    not zero."""
    profile = ROOT.TProfile("empty", "an empty profile", 2, 0.0, 2.0)
    assert profile.GetSumw2N() == 4
