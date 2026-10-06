"""ROOT 7's canvas: primitives drawn on pads, frames, styles, and what batch makes of showing."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.rcanvas.lengths import RPadLength


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_canvas_draws_primitives_by_class_by_name_and_of_root_6_objects() -> None:
    canvas = ROOT.RCanvas.Create("lines")
    line = canvas.Draw["RLine"]().SetP1([0.1, 0.1]).SetP2([0.9, 0.9])
    assert (line.GetP1().x, line.GetP2().y) == (RPadLength(0.1), RPadLength(0.9))
    box = canvas.Draw[ROOT.RBox](ROOT.RPadPos(0.1, 0.3), ROOT.RPadPos(0.3, 0.6))
    text = canvas.Add["ROOT::Experimental::RText"]([0.5, 0.5], "hello")
    marker = canvas.Draw["RMarker"]().SetP([0.2, 0.2])
    assert marker.GetP() == ROOT.RPadPos(0.2, 0.2)
    hist = ROOT.TH1F("h", "h", 4, 0, 1)
    drawn = canvas.Draw(hist, "same")
    assert (drawn.GetObject(), drawn.GetOption(), drawn.CSS_TYPE) == (hist, "same", "th1f")
    assert canvas.GetPrimitives() == [line, box, text, marker, drawn]
    assert canvas.NumPrimitives() == 5 and "5 primitives" in repr(canvas)
    with pytest.raises(TypeError, match="not a drawable"):
        canvas.Draw["RNothing"]()
    with pytest.raises(TypeError, match="not a drawable"):
        canvas.Draw["Primitive"]()
    canvas.Wipe()
    assert canvas.NumPrimitives() == 0


def test_a_primitive_has_its_id_class_and_place_on_the_frame() -> None:
    text = ROOT.RText("label")
    assert text.GetText() == "label" and text.GetPos() == ROOT.RPadPos()
    text.SetText("other").SetPos([0.2, 0.3])
    assert text.GetText() == "other" and text.GetPos() == ROOT.RPadPos(0.2, 0.3)
    assert text.SetId("t").GetId() == "t" and text.SetCssClass("c").GetCssClass() == "c"
    assert text.SetOnFrame().onFrame and text.SetClipping().clipping
    canvas = ROOT.RCanvas()
    canvas.Draw[ROOT.RText]("x").SetId("found")
    assert canvas.FindPrimitive("found").GetText() == "x" and canvas.FindPrimitive("no") is None


def test_paves_titles_axes_and_fonts_hold_what_they_are_given() -> None:
    pave = ROOT.RPaveText()
    pave.AddLine("one")
    pave.AddLine("two")
    assert (pave.GetNumLines(), pave.GetLine(1), pave.corner) == (2, "two", ROOT.RPave.kTopRight)
    pave.ClearLines()
    assert pave.GetNumLines() == 0
    title = ROOT.RFrameTitle("title")
    assert title.SetText("new").GetText() == "new"
    axis = ROOT.RAxisDrawable().SetLabels(["a", "b"])
    assert axis.labels == ["a", "b"] and axis.length == RPadLength()
    font = ROOT.RFont("Comic", "comic.woff2")
    assert (font.family, font.src) == ("Comic", "comic.woff2")
    style = ROOT.TObjectDrawable(ROOT.TObjectDrawable.kStyle)
    assert style.kind == 3 and style.CSS_TYPE == "" and style.GetObject() is None


def test_a_pad_has_a_frame_behind_its_primitives_and_divides_into_pads() -> None:
    canvas = ROOT.RCanvas.Create("pads")
    first = canvas.Draw["RLine"]()
    frame = canvas.AddFrame()
    assert canvas.AddFrame() is frame and canvas.GetPrimitives()[:2] == [frame, first]
    frame.x.min, frame.drawAxes = 5, True
    assert canvas.GetFrame().x.min == 5 and ROOT.RPad().GetFrame() is None
    pads = canvas.Divide(1, 3)
    assert len(pads) == 1 and len(pads[0]) == 3
    assert pads[0][2].pos == ROOT.RPadPos(0, 2 / 3) and pads[0][2].GetParent() is canvas
    assert pads[0][1].size == ROOT.RPadPos(1, 1 / 3)
    sheet = ROOT.RStyle.Parse("line { line_width: 2; }")
    canvas.UseStyle(sheet)
    assert canvas.GetStyle() is sheet


def test_in_batch_a_canvas_is_shown_nowhere_and_saves_no_image(capsys) -> None:
    canvas = ROOT.RCanvas.Create("batch")
    called = []
    canvas.Show()
    canvas.Hide()
    canvas.Modified()
    assert canvas.IsModified() and not canvas.IsShown()
    canvas.Update(True, called.append)
    assert not canvas.IsModified() and called == []
    canvas.Run(0)
    canvas.SetSize(900, 700)
    canvas.SetTitle("renamed")
    assert (canvas.GetSize(), canvas.GetTitle()) == ([900, 700], "renamed")
    canvas.ClearOnClose(object())
    assert canvas.SaveAs("line.png") is False
    assert "Warning in <RCanvas::SaveAs>: the image line.png" in capsys.readouterr().err
    assert canvas in ROOT.RCanvas.GetCanvases()
    canvas.Remove()
    canvas.Remove()
    assert canvas not in ROOT.RCanvas.GetCanvases()


def test_a_style_gives_a_drawable_the_values_of_the_blocks_that_pick_it_out() -> None:
    style = ROOT.RStyle.Parse("line { line_color: red; line_width: 2; } .group1 "
                              "{ marker_style: 8; onframe: true; size: 0.5; }")  # fmt: skip
    style.AddBlock("#obj7").AddString("line_color", "#0000FF").AddDouble("line_width", 5.0)
    style.AddBlock(".user").AddInt("line_style", 4).AddBool("hidden", 1)
    line = ROOT.RLine().SetId("obj7")
    assert style.Eval("line_color", line) == "#0000FF" and style.Eval("line_width", line) == 5.0
    assert style.Eval("line_color", ROOT.RLine()) == "red"
    marker = ROOT.RMarker().SetCssClass("group1")
    assert (style.Eval("marker_style", marker), style.Eval("onframe", marker)) == (8, True)
    assert style.Eval("size", marker) == 0.5 and style.Eval("line_style", marker) is None
    assert ROOT.RStyle.Parse("line { width 2; }") is None
    assert ROOT.RStyle.Parse("stray text") is None


def test_the_web_window_manager_is_there_but_serves_nothing() -> None:
    manager = ROOT.RWebWindowsManager.Instance()
    assert ROOT.RWebWindowsManager.Instance() is manager
    with pytest.raises(UnsupportedFeatureError, match="RWebWindowsManager::CreateWindow"):
        manager.CreateWindow()
    with pytest.raises(AttributeError):
        manager.__wrapped__  # noqa: B018
    assert ROOT.Experimental.RCanvas is ROOT.RCanvas
    with pytest.raises(UnsupportedFeatureError, match="RCanvas and what it draws"):
        ROOT.Experimental.RNothing  # noqa: B018
