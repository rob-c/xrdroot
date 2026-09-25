"""Text, shapes, paves and legends made, styled and drawn as ROOT macros do."""

from __future__ import annotations

import numpy as np
import pytest
from pyrootgraphics import fresh_session, gaussian  # noqa: F401

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.graphics import snapshot


def _figure(c):
    return snapshot.model(c).plot()


def test_attributes_start_as_gstyle_says_and_are_set_and_got_by_roots_names():
    ROOT.gStyle.SetTextFont(62)
    text = ROOT.TLatex(0.1, 0.2, "#sqrt{2}")
    assert text.GetTextFont() == 62 and text.GetTextSize() == 0.05
    text.SetTextColor(ROOT.kRed if hasattr(ROOT, "kRed") else 632)
    text.SetTextAlign(22)
    text.SetTextAngle(45)
    text.SetX(0.3)
    assert (text.GetTextAlign(), text.GetTextAngle(), text.GetX()) == (22, 45.0, 0.3)
    with pytest.raises(AttributeError, match="TLatex has SetMarkerColor"):
        text.SetMarkerColor(2)


def test_tobjects_methods_name_bit_and_clone():
    line = ROOT.TLine(0, 0, 1, 1)
    line.SetName("l")
    line.SetTitle("a line")
    line.SetUniqueID(3)
    assert (line.GetName(), line.GetTitle(), line.GetUniqueID()) == ("l", "a line", 3)
    assert line.ClassName() == "TLine" and line.InheritsFrom("TLine")
    assert ROOT.TArrow().InheritsFrom(ROOT.TLine) and not line.InheritsFrom("TBox")
    line.SetBit(1 << 3)
    assert line.TestBit(1 << 3)
    line.ResetBit(1 << 3)
    assert not line.TestBit(1 << 3)
    copy = line.Clone("m")
    copy.SetLineColor(4)
    assert (copy.GetName(), line.GetLineColor(), copy.GetLineColor()) == ("m", 1, 4)
    assert line.Clone().GetName() == "l" and repr(line) == "<TLine 'l'>"
    assert repr(ROOT.TBox()) == "<TBox>"
    line.Paint()
    line.SetFillColorAlpha(2, 0.5)
    line.SetLineColorAlpha(2, 0.5)
    assert line.members["fLineColor"] > 900


def test_draw_helpers_draw_a_styled_copy_at_the_new_place():
    c = ROOT.TCanvas("c", "c")
    c.Range(0, 0, 1, 1)
    base = ROOT.TLine()
    base.SetLineColor(2)
    made = [
        base.DrawLine(0.1, 0.1, 0.9, 0.9), base.DrawLineNDC(0, 0, 1, 1),
        ROOT.TArrow(0, 0, 1, 1).DrawArrow(0.1, 0.5, 0.9, 0.5, 0.02, "<|>"),
        ROOT.TArrow().DrawArrow(0.1, 0.4, 0.9, 0.4),
        ROOT.TBox().DrawBox(0.2, 0.2, 0.4, 0.4),
        ROOT.TEllipse().DrawEllipse(0.5, 0.5, 0.1, 0),
        ROOT.TArc().DrawArc(0.5, 0.5, 0.2, 0, 90),
        ROOT.TMarker(0, 0, 20).DrawMarker(0.3, 0.3),
        ROOT.TPolyLine().DrawPolyLine(3, [0.1, 0.2, 0.3], [0.1, 0.3, 0.1], "f"),
        ROOT.TText().DrawText(0.1, 0.1, "$5"), ROOT.TText().DrawTextNDC(0.1, 0.9, "top"),
        ROOT.TLatex().DrawLatex(0.5, 0.1, "#alpha"), ROOT.TLatex().DrawLatexNDC(0.5, 0.9, "#beta"),
        ROOT.TMathText().DrawMathText(0.5, 0.5, "x^2"),
        ROOT.TPaveLabel().DrawPaveLabel(0.6, 0.6, 0.9, 0.7, "label", "NDC"),
        ROOT.TGaxis().DrawAxis(0.1, 0.05, 0.9, 0.05, 0, 100),
    ]  # fmt: skip
    assert [obj for obj, _ in c.primitives] == made
    assert made[0].GetLineColor() == 2 and made[1].GetNDC() and not made[0].GetNDC()
    assert made[2].GetOption() == "<|>" and made[3].GetOption() == ">"
    assert made[5].GetR2() == 0.1
    figure = _figure(c)
    assert len(figure.axes[0].texts) > 6


def test_shapes_hold_what_roots_constructors_give_them():
    wbox = ROOT.TWbox(0, 0, 1, 1, 5, 3, -1)
    assert (wbox.GetFillColor(), wbox.GetBorderSize(), wbox.GetBorderMode()) == (5, 3, -1)
    crown = ROOT.TCrown(0.5, 0.5, 0.1, 0.2, 0, 180)
    assert (crown.GetR1(), crown.GetR2(), crown.GetPhimax()) == (0.1, 0.2, 180.0)
    arrow = ROOT.TArrow(0, 0, 1, 1, 0.03, "|>")
    arrow.SetOption()
    arrow.SetArrowSize(0.1)
    assert (arrow.GetOption(), arrow.GetArrowSize()) == (">", 0.1)
    marker = ROOT.TMarker(1, 2, 21)
    marker.SetMarkerSize(2)
    assert (marker.GetMarkerStyle(), marker.GetMarkerSize(), marker.GetY()) == (21, 2.0, 2.0)


def test_a_polyline_grows_point_by_point():
    poly = ROOT.TPolyLine(2, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert poly.GetN() == 2 and list(poly.GetX()) == [0.0, 1.0]
    poly.SetPoint(4, 3.0, 3.0)
    assert poly.GetN() == 5 and poly.GetY()[4] == 3.0
    assert poly.SetNextPoint(5.0, 5.0) == 5 and poly.GetN() == 6
    poly.SetPoint(0, 9.0, 9.0)
    assert poly.GetX()[0] == 9.0
    poly.SetPolyLine(1, [1.0], [2.0])
    assert poly.GetN() == 1
    empty = ROOT.TPolyMarker(3)
    assert list(empty.GetX()) == [0.0, 0.0, 0.0] and empty.GetMarkerStyle() == 1


def test_text_is_placed_and_restated():
    text = ROOT.TText(0.1, 0.2, "one")
    text.SetText(0.3, 0.4, "two")
    text.SetNDC()
    assert (text.GetX(), text.GetY(), text.GetTitle(), text.GetNDC()) == (0.3, 0.4, "two", True)
    text.SetNDC(False)
    assert not text.GetNDC()


def test_an_axis_of_its_own_is_graduated_by_numbers_or_a_functions_range():
    axis = ROOT.TGaxis(0, 0, 1, 0, -5, 5, 505, "+L")
    assert (axis.GetWmin(), axis.GetWmax(), axis.GetNdiv(), axis.GetChopt()) == (-5, 5, 505, "+L")
    axis.SetNdivisions(510)
    axis.SetOption("-")
    axis.CenterTitle()
    axis.SetLabelSize(0.05)
    axis.SetTitle("scale")
    assert (axis.GetNdiv(), axis.GetChopt(), axis.GetLabelSize()) == (510, "-C", 0.05)
    axis.CenterTitle(False)
    assert axis.GetChopt() == "-"

    class Function:
        range = (2.0, 8.0)

        def GetName(self):
            return "f"

    c = ROOT.TCanvas("c", "c")
    c.add(Function(), "")
    by_function = ROOT.TGaxis(0, 0, 1, 0, "f", 505, "G")
    assert (by_function.GetWmin(), by_function.GetWmax(), by_function.GetNdiv()) == (2, 8, 505)
    assert ROOT.TGaxis(0, 0, 1, 0, "nothing").GetWmax() == 1.0
    assert ROOT.TGaxis().GetWmax() == 1.0


def test_a_pave_text_stacks_lines_that_take_the_paves_attributes():
    pave = ROOT.TPaveText(0.1, 0.1, 0.5, 0.5, "NDC")
    first = pave.AddText("first")
    pave.AddText(0.5, 0.5, "placed")
    pave.AddLine(0, 0.5, 1, 0.5)
    assert pave.GetOption() == "brNDC" and pave.GetX1NDC() == 0.1
    assert (first.GetTextSize(), first.GetTextFont(), pave.GetTextAlign()) == (0.0, 0, 22)
    assert pave.GetSize() == 3 and pave.GetLine(0) is first and pave.GetLine(7) is None
    assert pave.GetLineWith("plac").GetX() == 0.5 and pave.GetLineWith("none") is None
    pave.SetAllWith("first", "color", 2)
    pave.SetAllWith("first", "tilt", 2)
    assert first.GetTextColor() == 2 and len(pave.GetListOfLines()) == 3
    pave.SetLabel("label")
    pave.SetMargin(0.1)
    pave.Clear()
    assert pave.GetSize() == 0


def test_a_pave_moved_in_ndc_keeps_its_place_as_its_corners():
    placed = ROOT.TPave(0.1, 0.1, 0.4, 0.4, 4, "brNDC")
    placed.SetX1NDC(0.2)
    placed.SetY1NDC(0.2)
    placed.SetX2NDC(0.6)
    placed.SetY2NDC(0.7)
    assert (placed.GetX1(), placed.GetY2()) == (0.2, 0.7)
    in_units = ROOT.TPave(1, 1, 4, 4)
    in_units.SetX1NDC(0.2)
    in_units.SetOption("tr")
    assert (in_units.GetX1(), in_units.GetX1NDC(), in_units.GetOption()) == (1.0, 0.2, "tr")
    assert not in_units.is_ndc() and in_units.GetShadowColor() == 1


def test_a_pave_label_is_one_label():
    label = ROOT.TPaveLabel(0.1, 0.1, 0.9, 0.3, "hello", "NDC")
    label.SetLabel("world")
    assert (label.GetLabel(), label.GetBorderSize(), label.GetTextAlign()) == ("world", 3, 22)


def test_a_stats_box_says_its_own_options_and_parent():
    stats = ROOT.TPaveStats(0.7, 0.7, 0.9, 0.9)
    stats.SetOptStat(11)
    stats.SetOptFit(1)
    stats.SetParent("h")
    assert (stats.GetOptStat(), stats.GetOptFit(), stats.GetParent()) == (11, 1, "h")
    assert stats.GetName() == "stats"
