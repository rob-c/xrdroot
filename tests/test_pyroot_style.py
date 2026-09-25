"""``gStyle``, ROOT's named styles, ``TColor`` and the palettes, as macros use them."""

from __future__ import annotations

import pytest
from pyrootgraphics import fresh_session  # noqa: F401

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.graphics import colors, style


def test_every_field_of_a_style_has_a_set_and_a_get():
    ROOT.gStyle.SetStatX(0.9)
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetPadGridX(True)
    ROOT.gStyle.SetStatFormat("4.2f")
    assert ROOT.gStyle.GetStatX() == 0.9 and ROOT.gStyle.GetOptTitle() == 0
    assert ROOT.gStyle.GetPadGridX() == 1 and ROOT.gStyle.GetStatFormat() == "4.2f"
    ROOT.gStyle.SetHistLineWidth()  # the default, as a Set with no argument keeps it
    assert ROOT.gStyle.GetHistLineWidth() == 1
    with pytest.raises(AttributeError, match="TStyle has SetNothing"):
        ROOT.gStyle.SetNothing(1)


def test_an_axis_field_is_set_for_each_axis_named_and_the_title_for_none():
    ROOT.gStyle.SetLabelSize(0.05, "xy")
    ROOT.gStyle.SetTitleSize(0.07, "")
    ROOT.gStyle.SetTitleFontSize(0.06)
    assert ROOT.gStyle.GetLabelSize("X") == 0.05 and ROOT.gStyle.GetLabelSize("Y") == 0.05
    assert ROOT.gStyle.GetLabelSize("Z") == 0.035 and ROOT.gStyle.GetLabelSize() == 0.05
    assert ROOT.gStyle.GetTitleFontSize() == 0.06 and ROOT.gStyle.GetTitleSize("") == 0.06


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(0, 0), (1111, 1111), ("nemr", 1111), ("ne", 11), ("neMRuoI", 2112211)],
)
def test_opt_stat_takes_digits_or_roots_letters(mode, expected):
    ROOT.gStyle.SetOptStat(mode)
    assert ROOT.gStyle.GetOptStat() == expected


def test_opt_stat_refuses_a_letter_it_does_not_know():
    with pytest.raises(ValueError, match="'x' is not one"):
        ROOT.gStyle.SetOptStat("nex")


def test_set_style_makes_the_named_style_current_and_get_style_finds_it():
    ROOT.set_style("Plain")
    assert ROOT.gStyle.GetName() == "Plain" and ROOT.gStyle.GetCanvasColor() == 0
    assert ROOT.gStyle.GetLabelFont("x") == 62 and repr(ROOT.gStyle) == "<TStyle 'Plain'>"
    atlas = ROOT.get_style("ATLAS")
    assert atlas.GetOptStat() == 0 and atlas.GetTitleOffset("Y") == 1.4
    assert ROOT.get_style("Nothing") is None
    with pytest.raises(ValueError, match="there is no style 'Nothing'"):
        ROOT.set_style("Nothing")
    mine = ROOT.TStyle("Mine", "my own")
    assert ROOT.get_style("Mine") is mine and mine.GetTitle() == "my own"
    mine.cd()
    assert ROOT.gStyle.GetName() == "Mine"


def test_a_style_is_copied_and_reset():
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetLabelSize(0.1, "x")
    ROOT.gStyle.SetPalette(55)
    other = ROOT.TStyle("Other")
    ROOT.gStyle.Copy(other)
    assert other.GetOptStat() == 0 and other.GetLabelSize("x") == 0.1
    assert other.custom_palette() == ROOT.gStyle.custom_palette()
    other.Reset()
    assert other.GetOptStat() == 1111 and other.GetLabelSize("x") == 0.035


def test_set_opt_fit_and_pad_ticks():
    ROOT.gStyle.SetOptFit()
    ROOT.gStyle.SetPadTickX(1)
    assert ROOT.gStyle.GetOptFit() == 1 and ROOT.gStyle.GetPadTickX() == 1


def test_a_palette_is_one_of_roots_by_number_or_name_or_one_of_ones_own():
    assert ROOT.gStyle.custom_palette() is None  # kBird, until told
    assert ROOT.gStyle.GetNumberOfColors() == 255
    first = ROOT.gStyle.GetColorPalette(0)
    assert colors.rgb_of(first) == pytest.approx((53 / 255, 42 / 255, 134 / 255), abs=0.01)
    ROOT.gStyle.SetPalette(ROOT.kRainBow)
    rainbow = ROOT.gStyle.custom_palette()
    assert len(rainbow) == 255 and colors.rgb_of(rainbow[0]) == pytest.approx((0, 0, 99 / 255))
    ROOT.gStyle.SetPalette(ROOT.kRainBow)
    assert ROOT.gStyle.custom_palette() == rainbow  # laid once
    ROOT.gStyle.SetPalette(1)
    assert ROOT.gStyle.custom_palette() == list(range(51, 101))
    ROOT.gStyle.SetPalette(3, [2, 3, 4])
    assert ROOT.gStyle.palette() == [2, 3, 4] and ROOT.gStyle.GetColorPalette(4) == 3
    ROOT.gStyle.SetPalette(ROOT.kBird)
    assert ROOT.gStyle.custom_palette() is None
    ROOT.gStyle.SetPalette(ROOT.kSunset)  # not among the gradients here: kBird's stops
    assert len(ROOT.gStyle.custom_palette()) == 255


def test_tcolor_finds_a_colour_by_value_or_makes_it():
    assert ROOT.TColor.GetColor("#ff0000") == 2 and ROOT.TColor.GetColor(1.0, 0.0, 0.0) == 2
    made = ROOT.TColor.GetColor(10, 20, 30)
    assert made == ROOT.TColor.GetFreeColorIndex() - 1
    assert ROOT.TColor.GetColor(10, 20, 30) == made
    with pytest.raises(ValueError, match="'red' is not"):
        ROOT.TColor.GetColor("red")
    with pytest.raises(TypeError, match="not 2 arguments"):
        ROOT.TColor.GetColor(1, 2)


def test_a_tcolor_is_a_colour_of_the_table():
    c = ROOT.TColor(1500, 0.5, 0.25, 1.0, "mine", 0.5)
    assert (c.GetNumber(), c.GetName(), c.GetAlpha()) == (1500, "mine", 0.5)
    assert (c.GetRed(), c.GetGreen(), c.GetBlue()) == (0.5, 0.25, 1.0)
    c.SetRGB(1.0, 1.0, 1.0)
    c.SetAlpha(1.0)
    assert c.AsHexString() == "#ffffff" and repr(c) == "<TColor 1500 #ffffff>"
    assert ROOT.TColor().GetName() == "Color1501"


def test_colours_made_brighter_darker_or_transparent():
    bright, dark = ROOT.TColor.GetColorBright(2), ROOT.TColor.GetColorDark(2)
    assert colors.rgb_of(bright)[0] == 1.0 and colors.rgb_of(dark)[0] == pytest.approx(0.7)
    see_through = ROOT.TColor.GetColorTransparent(4, 0.3)
    assert colors.ALPHA[see_through] == 0.3


def test_a_gradient_colour_table_becomes_the_palette():
    first = ROOT.TColor.CreateGradientColorTable(
        3, [0.0, 0.5, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 50
    )
    assert ROOT.TColor.GetNumberOfColors() == 50 and ROOT.TColor.GetPalette()[0] == first
    assert colors.rgb_of(first) == (1.0, 0.0, 0.0)
    ROOT.TColor.CreateGradientColorTable(2, [0, 1], [0, 1], [0, 1], [0, 1], 10, 1.0, False)
    assert ROOT.TColor.GetNumberOfColors() == 50
    ROOT.TColor.SetPalette(2, [5, 6])
    assert ROOT.gStyle.palette() == [5, 6]


def test_the_style_module_knows_its_names():
    assert style.gStyle.GetName() == "Modern" and "Classic" in style.STYLES
