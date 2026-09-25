"""Canvases and pads made and drawn on as a ROOT macro does, in batch mode.

Each test builds a canvas the way the ``hist`` and ``graphics`` tutorials
do - ``TCanvas``, ``Divide``, ``cd``, ``Draw`` with and without ``same`` -
and checks what the pads hold, where they are, and what drawing them puts
on the figure.
"""

from __future__ import annotations

import pytest
from pyrootgraphics import fresh_session, gaussian, graph  # noqa: F401

import xrdroot.pyroot as ROOT
from xrdroot.canvas import Canvas, Pad
from xrdroot.pyroot import graphics
from xrdroot.pyroot.graphics import pads, snapshot


def test_a_canvas_is_its_window_less_the_decoration_as_in_batch_mode():
    c = ROOT.TCanvas("c", "c", 800, 600)
    assert (c.GetWw(), c.GetWh(), c.GetWindowWidth(), c.GetWindowHeight()) == (796, 572, 800, 600)
    assert ROOT.gPad.GetName() == "c" and bool(ROOT.gPad) and ROOT.gPad == c


@pytest.mark.parametrize(
    ("arguments", "size", "top"),
    [
        ((), (696, 472), (10, 10)),
        (("t",), (696, 472), (10, 10)),
        (("t", 2), (496, 472), (20, 20)),
        (("t", 5, 6, 300, 200), (296, 172), (5, 6)),
        (("t", -300, 200), (300, 200), (10, 10)),
        ((300, 200), (296, 172), (10, 10)),
    ],
)
def test_a_canvas_takes_every_size_roots_constructors_do(arguments, size, top):
    c = ROOT.TCanvas("c", *arguments)
    assert (c.GetWw(), c.GetWh()) == size
    assert (c.GetWindowTopX(), c.GetWindowTopY()) == top


def test_a_canvas_of_a_name_already_taken_replaces_the_one_before():
    first = ROOT.TCanvas("c", "c")
    second = ROOT.TCanvas("c", "c")
    assert pads.CANVASES == [second] and first is not second


def test_a_canvas_is_resized_placed_and_closed():
    c = ROOT.TCanvas("c", "c")
    assert c.SetCanvasSize(400, 300) and (c.GetWw(), c.GetWh()) == (400, 300)
    c.SetWindowSize(500, 400)
    assert (c.GetWw(), c.GetWh()) == (496, 372)
    c.SetWindowPosition(3, 4)
    assert (c.GetWindowTopX(), c.GetWindowTopY()) == (3, 4)
    c.SetBatch()
    c.Show()
    c.Iconify()
    assert c.IsBatch() and c.SetRealAspectRatio() and repr(c) == "<TCanvas 'c' 496x372>"
    c.Close()
    assert pads.CANVASES == [] and not ROOT.gPad and ROOT.gPad == None  # noqa: E711
    c.Close()  # twice is nothing


def test_gpad_with_no_canvas_says_it_is_null():
    with pytest.raises(AttributeError, match="gPad is null"):
        ROOT.gPad.GetName()
    assert repr(ROOT.gPad) == "None" and hash(ROOT.gPad)


def test_divide_makes_pads_numbered_from_the_top_left_and_cd_goes_to_them():
    c = ROOT.TCanvas("c", "c", 700, 500)
    c.Divide(2, 2)
    assert ROOT.gPad == c  # Divide leaves the current pad where it was
    assert [p.GetName() for p in c.pads()] == ["c_1", "c_2", "c_3", "c_4"]
    one, _two, three, _four = c.pads()
    assert one.members["fXlowNDC"] == pytest.approx(0.01)
    assert one.members["fYlowNDC"] == pytest.approx(0.51)
    assert three.members["fYlowNDC"] == pytest.approx(0.01)
    assert c.cd(3) is three and ROOT.gPad == three
    assert three.GetMother() is c and three.GetCanvas() is c and three.GetNumber() == 3
    assert c.cd(9) is c and ROOT.gPad == three  # no such pad: gPad stays
    assert c.GetMother() is c and c.GetPad(9) is None


def test_divide_with_no_margins_tiles_the_pad():
    c = ROOT.TCanvas("c", "c")
    c.Divide(3, 1, 0, 0)
    assert [p.members["fWNDC"] for p in c.pads()] == pytest.approx([1 / 3] * 3)


def test_a_pad_made_by_hand_is_drawn_in_the_current_pad_once():
    c = ROOT.TCanvas("c", "c")
    pad = ROOT.TPad("p", "p", 0.1, 0.1, 0.5, 0.5, 5, 2, 1)
    pad.Draw()
    pad.Draw()
    c.Draw()
    assert c.pads() == [pad] and pad.GetFillColor() == 5 and pad.GetBorderMode() == 1
    pad.cd()
    graphics.hook.draw(ROOT.TPad("q", "q"))  # a pad handed to the hook draws itself
    assert len(pad.pads()) == 1


def test_drawing_with_no_canvas_makes_c1_and_then_c1_n2():
    gaussian().Draw()
    assert ROOT.gPad.GetName() == "c1" and pads.current().GetWw() == 696
    graphics.canvas.default_canvas()
    assert [c.GetName() for c in pads.CANVASES] == ["c1", "c1_n2"]


def test_data_drawn_without_same_clears_the_pad_and_with_it_is_added():
    c = ROOT.TCanvas("c", "c")
    first, second = gaussian("a"), gaussian("b")
    first.Draw()
    ROOT.TLine(0, 0, 1, 1).Draw()
    second.Draw("hist")
    assert [o for o, _ in c.primitives] == [second]
    first.Draw("same")
    assert [option for _, option in c.primitives] == ["hist", "same"]
    g = graph()
    g.Draw("P")  # a graph without axes is drawn over what is there
    g.Draw("AP")
    assert [o for o, _ in c.primitives] == [g]
    assert c.IsModified()
    c.Update()
    assert not c.IsModified()


def test_find_object_and_get_primitive_look_through_the_pads():
    c = ROOT.TCanvas("c", "c")
    c.Divide(2)
    c.cd(2)
    h = gaussian("wanted")
    h.Draw()
    text = ROOT.TLatex(0.5, 0.5, "x")
    text.SetName("label")
    text.Draw()
    assert c.FindObject("wanted") is h and c.FindObject(text) is text
    assert c.FindObject("nothing") is None
    assert c.cd(2).GetPrimitive("label") is text and c.GetPrimitive("label") is None
    assert pads.find_anywhere("wanted") is h
    c.Update()
    assert c.cd(2).GetPrimitive("stats").GetName() == "stats"


def test_a_pad_is_logged_gridded_ticked_and_margined_as_it_is_told():
    c = ROOT.TCanvas("c", "c")
    c.SetLogy()
    c.SetLogx(0)
    c.SetGrid()
    c.SetTicks(1, 0)
    c.SetGridx(0)
    c.SetLeftMargin(0.15)
    c.SetMargin(0.2, 0.05, 0.12, 0.08)
    c.SetFrameFillColor(19)
    c.SetFillColor(18)
    c.SetPad(0.0, 0.0, 1.0, 1.0)
    assert (c.GetLogy(), c.GetLogx(), c.GetGridx(), c.GetGridy()) == (1, 0, 0, 1)
    assert (c.GetTickx(), c.GetTicky()) == (1, 0)
    assert (c.GetLeftMargin(), c.GetRightMargin(), c.GetTopMargin()) == (0.2, 0.05, 0.08)
    assert c.GetFrameFillColor() == 19 and c.GetFillColor() == 18
    c.SetEditable(False)
    c.SetFixedAspectRatio()
    assert not c.IsEditable()
    with pytest.raises(AttributeError, match="TCanvas has SetNothing"):
        c.SetNothing(1)


def test_range_places_a_pad_with_nothing_framing_it():
    c = ROOT.TCanvas("c", "c")
    c.Range(0, 0, 20, 10)
    assert c.GetRange() == (0.0, 0.0, 20.0, 10.0)
    assert (c.GetUxmin(), c.GetUymax()) == (0.0, 10.0)

    class Cell:
        value = 0.0

    x1, y1 = Cell(), Cell()
    c.GetRange(x1, y1)
    assert (x1.value, y1.value) == (0.0, 0.0)


def test_a_pads_frame_is_what_its_first_histogram_draws_and_logarithmic_ends_are_powers():
    c = ROOT.TCanvas("c", "c")
    gaussian().Draw()
    assert (c.GetUxmin(), c.GetUxmax()) == (-4.0, 4.0)
    c.SetLogy()
    assert c.GetUymax() < 3 and c.GetUymin() < 1
    c.SetLogx()
    assert c.GetUxmin() == pytest.approx(-3 + 0.60206, abs=1e-4)  # a thousandth of 4


def test_get_frame_restyles_the_frame_it_is_drawn_with():
    c = ROOT.TCanvas("c", "c")
    gaussian().Draw()
    c.GetFrame().SetFillColor(5)
    frame = snapshot.model(c).primitives[0][0]
    assert (frame.classname, frame.get("fFillColor")) == ("TFrame", 5)
    assert c.GetFrame() is c.GetFrame()


def test_a_canvas_becomes_the_canvas_model_a_saved_one_is_read_as():
    c = ROOT.TCanvas("c", "c", 600, 400)
    c.Divide(2)
    c.cd(1)
    h = gaussian()
    h.Draw("E1")
    c.cd(2)
    ROOT.TLatex(0.5, 0.5, "#alpha").Draw()
    made = snapshot.model(c)
    assert isinstance(made, Canvas) and (made.width, made.height) == (596, 372)
    left, right = made.pads
    assert [type(obj).__name__ for obj, _ in left.primitives] == [
        "Primitive", "Histogram", "Primitive", "Primitive",
    ]  # fmt: skip
    assert [obj.classname for obj, _ in right.primitives] == ["TLatex"]
    assert isinstance(snapshot.model(c.cd(1)), Pad) and not isinstance(
        snapshot.model(c.cd(1)), Canvas
    )


def test_clear_empties_a_pad_and_makes_it_current():
    c = ROOT.TCanvas("c", "c")
    c.Divide(2)
    c.cd(2)
    c.Clear()
    assert c.primitives == [] and ROOT.gPad == c


def test_ls_lists_what_a_pad_holds(capsys):
    c = ROOT.TCanvas("c", "the title")
    gaussian().Draw("hist")
    c.ls()
    c.Paint()
    c.RedrawAxis()
    c.Flush()
    out = capsys.readouterr().out
    assert "TCanvas c: the title" in out and "Wrapped 'h' 'hist'" in out
    assert repr(ROOT.TPad("p", "p")) == "<TPad 'p' of 0 primitives>"
