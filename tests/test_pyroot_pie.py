"""``TPie``: slices of a circle, each a value's share, labelled and painted as ROOT's are."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootgraphics import fresh_session  # noqa: F401
from xrdroot.canvas.pie import _label, _text_style
from xrdroot.pyroot.graphics.snapshot import primitive

image = pytest.importorskip("matplotlib.image")


def _pie():
    return ROOT.TPie("pie", "TPie:", 4, [1.0, 2.0, 3.0, 4.0], [2, 3, 4, 5], ["a", "b", "", "d"])


def test_a_pie_is_made_of_slices_with_values_colours_and_labels():
    pie = _pie()
    assert pie.GetEntries() == 4 and pie.GetEntryVal(3) == 4.0 and pie.GetEntryFillColor(1) == 3
    assert (pie.GetEntryLabel(0), pie.GetEntryLabel(2)) == ("a", "pie_slice_2")
    pie.SetEntryRadiusOffset(2, 0.05)
    pie.SetEntryFillStyle(2, 3004)
    pie.SetEntryLineWidth(1, 3)
    pie.SetEntryVal(0, 10.0)
    pie.SetEntryLabel(0, "ten")
    piece = pie.GetSlice(2)
    assert (pie.GetEntryRadiusOffset(2), piece.GetFillStyle(), pie.GetEntryLineWidth(1)) == (
        0.05, 3004, 3)
    assert (pie.GetEntryVal(0), piece.GetValue(), pie.GetEntryLabel(0)) == (10.0, 3.0, "ten")
    pie.SetFillColors([6, 7, 8, 9])
    pie.SetLabels(["w", "x", "y", "z"])
    pie.SetCircle(0.4, 0.6, 0.3)
    pie.SetLabelsOffset(-0.08)
    assert (pie.GetSlice(3).GetFillColor(), pie.GetSlice(3).GetTitle()) == (9, "z")
    assert (pie.GetX(), pie.GetY(), pie.GetRadius()) == (0.4, 0.6, 0.3)
    assert pie.GetLabelsOffset() == pytest.approx(-0.08)
    with pytest.raises(AttributeError, match="ROOT's TPie has Nothing"):
        pie.Nothing()
    assert ROOT.TPie("e", "e", 2).GetEntryVal(1) == 0.0


def test_a_pie_of_a_histogram_takes_a_slice_per_bin_titled_by_its_label():
    h = ROOT.TH1D("h", "counts", 3, 0, 3)
    for at, label in enumerate(("one", "two", "three")):
        h.GetXaxis().SetBinLabel(at + 1, label)
        h.SetBinContent(at + 1, at + 1.0)
    pie = ROOT.TPie(h)
    assert (pie.GetName(), pie.GetEntries(), pie.GetEntryLabel(2), pie.GetEntryVal(1)) == (
        "h", 3, "three", 2.0)
    legend = pie.MakeLegend(0.1, 0.1, 0.4, 0.4, "slices")
    assert pie.GetLegend() is legend and legend.GetNRows() == 4


def test_labels_are_made_by_the_format_and_turned_as_the_option_says():
    pie = _pie()
    pie.SetLabelFormat("%txt: %val (%frac, %perc)")
    pie.SetValueFormat("%.1f")
    pie.SetPercentFormat("%.0f")
    prim = primitive(pie)
    assert _label(prim, prim.get("fPieSlices")[0], 1.0, 10.0) == "a: 1.0 (0.10, 10 %)"
    assert _text_style("RSC", 45.0, prim.get("fPieSlices")[1]) == {"angle": 45.0, "align": 12,
                                                                    "color": 3}
    assert _text_style("r", 180.0, None) == {"angle": 360.0, "align": 32}
    assert _text_style("T", 30.0, None)["angle"] == -60.0 and _text_style("t", 200.0, None) == {
        "angle": 290.0, "align": 22}
    assert _text_style("", 100.0, None) == {"align": 32} and _text_style("", 0.0, None) == {
        "align": 12}


@pytest.mark.parametrize("option", ["rsc", "t", "", "nol", "3d"])
def test_a_pie_is_painted_slice_by_slice_with_its_labels(option, tmp_path):
    import warnings

    canvas = ROOT.TCanvas("c", "c", 200, 200)
    pie = _pie()
    pie.SetEntryRadiusOffset(1, 0.1)
    pie.SetAngularOffset(30.0)
    pie.Draw(option)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        canvas.SaveAs(str(tmp_path / "pie.png"))
    picture = np.asarray(image.imread(str(tmp_path / "pie.png")))
    assert min(picture.shape[:2]) > 100 and picture[..., :3].std() > 0.05  # something drawn


def test_a_pie_of_nothing_draws_no_slice(tmp_path):
    canvas = ROOT.TCanvas("c", "c", 100, 100)
    ROOT.TPie("none", "none", 2).Draw()
    canvas.SaveAs(str(tmp_path / "none.png"))
    assert (tmp_path / "none.png").exists()
