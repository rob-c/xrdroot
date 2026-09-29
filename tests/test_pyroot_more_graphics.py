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
