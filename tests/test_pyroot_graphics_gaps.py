"""Pads' places, ratios and paves, axes' titles and labels by value, transparent text.

canvas2, canvas3, file, gaxis3 and transparency use them; the pictures are
the harness's to compare, so what is checked here is what each call keeps
and what the painter is handed.
"""

from __future__ import annotations

import ctypes

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.canvas.axis import Axis, Label, _changed


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_pads_place_in_its_canvas_is_its_mothers_times_its_own() -> None:
    canvas = ROOT.TCanvas("c", "c", 400, 400)
    canvas.Divide(2, 1, 0, 0)
    inner = canvas.cd(2)
    inner.Divide(1, 2, 0, 0)
    corner = inner.cd(2)
    assert (corner.GetAbsXlowNDC(), corner.GetAbsYlowNDC()) == (0.5, 0.0)
    assert (corner.GetAbsWNDC(), corner.GetAbsHNDC()) == (0.5, 0.5)
    assert (canvas.GetAbsWNDC(), canvas.GetAbsHNDC()) == (1.0, 1.0)
    found = [ctypes.c_double() for _ in range(4)]
    corner.GetPadPar(*found)
    assert [one.value for one in found] == [0.0, 0.0, 1.0, 0.5]


def test_divide_ratios_lays_rows_and_columns_out_by_their_ratios_from_the_top_left() -> None:
    canvas = ROOT.TCanvas("r", "r", 400, 400)
    canvas.DivideRatios(2, 2, [0.2, 0.8], [0.4, 0.6], 0.0, 0.0)
    first, last = canvas.GetPad(1), canvas.GetPad(4)
    assert (first.GetXlowNDC(), first.GetYlowNDC(), first.GetWNDC()) == pytest.approx(
        (0.0, 0.6, 0.2))  # fmt: skip
    assert (last.GetXlowNDC(), last.GetYlowNDC(), last.GetHNDC()) == pytest.approx((0.2, 0.0, 0.6))
    canvas.DivideRatios(2, 3)
    assert canvas.GetPad(6).GetWNDC() == 0.5 and canvas.GetPad(6).GetHNDC() == pytest.approx(1 / 3)


def test_a_pave_drawn_again_is_a_new_one_in_the_default_fill() -> None:
    ROOT.TCanvas("p", "p", 400, 400)
    pave = ROOT.TPave(2.5, 8.5, 3.3, 11, 1)
    pave.SetFillColor(33)
    again = pave.DrawPave(7.7, 11, 6.9, 8.5, 1)
    assert (again.GetX1(), again.GetX2(), again.GetBorderSize()) == (6.9, 7.7, 1)
    assert again.GetFillColor() != 33


def test_an_axis_title_takes_its_text_colour_and_font_and_a_label_is_changed_by_value() -> None:
    axis = ROOT.TGaxis(-5.5, 0.0, 5.5, 0.0, 0.0, 100, 510, "")
    axis.SetTitleColor(ROOT.kBlue)
    axis.SetTitleFont(42)
    assert (axis.GetTextColor(), axis.GetTextFont()) == (ROOT.kBlue, 42)
    axis.ChangeLabelByValue(30.0, -1, -1, -1, ROOT.kRed, -1, "thirty")
    axis.SetTimeOffset(1325376000)
    assert axis.members["fTimeFormat"].endswith("%F1325376000")
    painted = Axis(0, 0, 1, 0, 0, 100, changed=tuple(axis.members["_changed_labels"]))
    label = Label("30", 0.3, 0.0, 23, 0.0, 42, 0.04, 1)
    assert _changed(painted, label, 4, 10, 30.0).text == "thirty"
    assert _changed(painted, label, 5, 10, 40.0) == label
    numbered = Axis(0, 0, 1, 0, 0, 100, changed=((4, -1, -1, -1, 2, -1, ""),))
    assert _changed(numbered, label, 4, 10, 30.0).color == 2


def test_text_and_markers_take_a_colour_at_an_opacity() -> None:
    latex = ROOT.TLatex(0.1, 0.1, "x")
    latex.SetTextColorAlpha(9, 0.476)
    marker = ROOT.TMarker(0.5, 0.5, 20)
    marker.SetMarkerColorAlpha(2, 0.3)
    # each a colour of its own, made transparent, as GetColorTransparent makes one per call
    assert ROOT.gROOT.GetColor(latex.GetTextColor()).GetAlpha() == pytest.approx(0.476)
    assert ROOT.gROOT.GetColor(marker.GetMarkerColor()).GetAlpha() == pytest.approx(0.3)
