"""Smaller graphics classes the tutorials use: ``TPavesText`` and its kin.

``TPavesText`` is ROOT's ``archi.C`` and ``framework.C`` boxes: a pave of
text on ``npaves - 1`` paves stacked three borders apart behind it.
"""

from __future__ import annotations

from typing import Any

import xrdroot.pyroot as ROOT
from geomsupport import geometry_session  # noqa: F401
from pyrootsupport import expect


def stacked_boxes(tmp_path: Any, option: str) -> int:
    """How many box outlines a pad with one ``TPavesText`` of ``option`` is drawn with."""
    canvas = ROOT.TCanvas("c", "c", 200, 200)
    paves = ROOT.TPavesText(0.2, 0.2, 0.8, 0.8, 4, option)
    paves.AddText("stacked")
    paves.Draw()
    canvas.SaveAs(str(tmp_path / f"paves-{option}.png"))
    return paves.GetNpaves()


def test_a_stack_of_paves_is_drawn_behind_its_text(tmp_path: Any) -> None:
    for option in ("br", "tl"):
        assert stacked_boxes(tmp_path, option) == 4
    paves = ROOT.TPavesText(0, 0, 1, 1)
    paves.SetNpaves(2)
    expect((paves.GetNpaves(), 2), (paves.members["fBorderSize"], 1),
           (ROOT.TPavesText().ClassName(), "TPavesText"))  # fmt: skip


def test_a_colour_with_an_opacity_is_one_of_its_own_found_again_by_both() -> None:
    see_through = ROOT.TColor.GetColor(0.25, 0.25, 0.25, 0.55)
    again = ROOT.TColor.GetColor(0.25, 0.25, 0.25, 0.55)
    opaque = ROOT.TColor.GetColor(0.25, 0.25, 0.25, 1.0)
    expect((again, see_through), (opaque != see_through, True),
           (ROOT.gROOT.GetColor(see_through).GetAlpha(), 0.55))  # fmt: skip


def test_a_canvas_says_it_is_drawn_without_opengl_whatever_the_style_prefers() -> None:
    ROOT.gStyle.SetCanvasPreferGL(True)
    canvas = ROOT.TCanvas("c", "c", 100, 100)
    expect((ROOT.gStyle.GetCanvasPreferGL(), True), (canvas.UseGL(), False),
           (canvas.IsWeb(), False))  # fmt: skip
    ROOT.gStyle.SetCanvasPreferGL(False)


def test_groot_finds_a_colour_made_or_roots_own_and_none_for_one_not_there() -> None:
    red = ROOT.gROOT.GetColor(2)
    expect((red.GetRed(), 1.0), (red.GetNumber(), 2), (ROOT.gROOT.GetColor(2), red),
           (ROOT.gROOT.GetColor(99999), None))  # fmt: skip
