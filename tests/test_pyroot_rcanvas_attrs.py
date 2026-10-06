"""ROOT 7's drawing attributes: pad lengths, colours, and groups assigned into whole."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.cint.runtime import user_literal
from xrdroot.pyroot.rcanvas.attrs import Attrs
from xrdroot.pyroot.rcanvas.lengths import RPadLength, position


def test_a_pad_length_sums_its_parts_and_scales_them() -> None:
    mixed = user_literal("_normal", 0.5) - user_literal("_px", 5) + user_literal("_user", 2)
    assert (mixed.GetNormal(), mixed.GetPixel(), mixed.GetUser()) == (0.5, -5.0, 2.0)
    assert mixed.HasNormal() and mixed.HasPixel() and mixed.HasUser()
    assert repr(mixed) == "RPadLength(0.5_normal + -5_px + 2_user)"
    assert 2 * RPadLength.Normal(0.1) == RPadLength(0.2) and RPadLength(0.2) != 0.2
    assert -RPadLength.Pixel(3) == RPadLength(pixel=-3)
    assert RPadLength.User(4) / 2 == RPadLength(user=2)
    assert 0.25 + RPadLength(0.25) == RPadLength(0.5)
    assert 1 - RPadLength(0.25) == RPadLength(0.75)
    assert not RPadLength().HasNormal()


def test_a_position_is_two_lengths_from_lengths_numbers_or_braces() -> None:
    pos = ROOT.RPadPos(0.1, user_literal("_px", 20))
    assert pos.x == RPadLength(0.1) and pos.y == RPadLength(pixel=20)
    assert "RPadPos(" in repr(pos)
    assert position([0.1, user_literal("_px", 20)]) == pos and position(pos) is pos
    assert ROOT.RPadExtent is ROOT.RPadPos


def test_a_colour_is_its_components_or_a_name_a_browser_knows() -> None:
    green = ROOT.RColor(0, 255, 0, 127)
    assert (green.GetRed(), green.GetGreen(), green.GetBlue(), green.GetAlpha()) == (0, 255, 0, 127)
    assert green.AsString() == "#00FF00" and green.AsHex() == "00FF00"
    assert ROOT.RColor.kBlue.AsString() == "blue" and ROOT.RColor("Orange").GetGreen() == 165
    assert ROOT.RColor("nonesuch").rgba == (0, 0, 0, 255)
    assert ROOT.RColor.kRed == ROOT.RColor(255, 0, 0) and repr(ROOT.RColor.kRed) == "RColor(red)"


def test_a_group_knows_its_members_and_refuses_others() -> None:
    line = ROOT.RAttrLine(ROOT.RColor.kBlue, 3.0, ROOT.RAttrLine.kDashed)
    assert (line.width, line.style, line.color) == (3.0, 2, ROOT.RColor.kBlue)
    assert ROOT.RAttrLine().color == ROOT.RColor("black") and ROOT.RAttrLine.kStyle6 == 6
    with pytest.raises(AttributeError, match="has no attribute 'thickness'; it has color"):
        line.thickness = 2
    with pytest.raises(TypeError, match="set its members"):
        ROOT.RBox().fill = ROOT.RColor.kRed  # a colour where a group belongs
    with pytest.raises(TypeError, match="set its members"):
        ROOT.RAxisDrawable().axis = 3
    assert "RAttrFill(color=" in repr(ROOT.RAttrFill()) and ROOT.RAttrFill.k3004 == 3004
    with pytest.raises(AttributeError, match="it has none"):
        Attrs().x = 1


def test_a_group_given_a_value_whole_keeps_it_its_own_way() -> None:
    text = ROOT.RAttrText()
    text.font = ROOT.RAttrFont.kArialOblique
    assert text.font.GetFullName() == "Arial oblique"
    text.font.family = "Comic"
    assert text.font.family == "Comic"
    with pytest.raises(ValueError, match="no font 13"):
        text.font = 13
    frame = ROOT.RFrame()
    frame.margins = user_literal("_normal", 0.2)
    frame.margins.top = user_literal("_normal", 0.25)
    assert (frame.margins.left, frame.margins.top) == (RPadLength(0.2), RPadLength(0.25))
    box = ROOT.RBox()
    border = ROOT.RAttrBorder()
    border.rx = 10
    box.border = border
    assert box.border.rx == 10 and box.border is not border


def test_an_axis_has_its_range_scale_title_ticks_and_ending() -> None:
    axis = ROOT.RAxisDrawable([0.1, 0.1], True, 0.8).axis
    axis.title = "vertical"
    assert axis.title.SetCenter().position == "center" and axis.title.value == "vertical"
    assert axis.title.SetLeft().position == "left" and axis.title.SetRight().position == "right"
    assert axis.ticks.SetInvert().side == "invert" and axis.ticks.SetBoth().side == "both"
    assert axis.ticks.SetNormal().side == "normal"
    for style in ("Arrow", "Circle", "Square", "Diamond", "None"):
        getattr(axis.ending, f"Set{style}")()
        assert axis.ending.style == ("" if style == "None" else style.lower())
    assert axis.SetMinMax(1, 100).max == 100.0
    assert axis.SetTimeDisplay("%H:%M").time and axis.timeFormat == "%H:%M"
    axis.labels.center = True
    assert axis.labels.center
