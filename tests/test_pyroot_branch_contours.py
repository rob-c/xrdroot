"""Contour lines at the levels a user set, and ``CONT2``'s styles from a saved palette.

``SetContour(n, levels)`` sets ``TH1::kUserContour`` and ``fContour``; drawn
``CONT``, those levels are the ones ``THistPainter`` walks, in place of
gStyle's even twenty. ``CONT2`` gives each level the line style of its
palette colour's index modulo five - ``kBird``'s, from 924, when the canvas
saved no palette of its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from test_canvas_contour import _cone
from test_canvas_draw import FRAME, make, prim
from xrdroot.canvas import contour
from xrdroot.plot.drawers import USER_CONTOUR, _user_contours


def spy(monkeypatch: pytest.MonkeyPatch) -> list[list[float]]:
    """The levels each painting of contour lines walked."""
    seen: list[list[float]] = []
    real = contour.contour_segments

    def recording(xs: Any, ys: Any, values: Any, levels: Any) -> Any:
        seen.append(list(levels))
        return real(xs, ys, values, levels)

    monkeypatch.setattr(contour, "contour_segments", recording)
    return seen


def test_a_users_levels_are_the_ones_drawn(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``kUserContour`` set, ``CONT3`` walks ``fContour``'s levels and no others."""
    h = _cone()
    h._core["fContour"] = np.array([0.8, 1.2, 1.6])
    h._core["fBits"] = int(h._core.get("fBits", 0)) | USER_CONTOUR
    seen = spy(monkeypatch)
    make([(h, "cont3")]).plot()
    assert seen == [[0.8, 1.2, 1.6]]


def test_levels_without_the_users_bit_are_remade(monkeypatch: pytest.MonkeyPatch) -> None:
    """``fContour`` without ``kUserContour`` - gStyle's own - is remade: twenty levels."""
    h = _cone()
    h._core["fContour"] = np.array([0.8, 1.2, 1.6])
    seen = spy(monkeypatch)
    make([(h, "cont3")]).plot()
    assert len(seen[0]) == 20
    assert _user_contours({"fBits": USER_CONTOUR}) == ()


def test_cont2_takes_its_styles_from_a_saved_palettes_indices() -> None:
    """A palette of colours 7 and 10 gives the styles 2 and 5 - ``10 % 5`` taken as 5."""
    colors = [prim("TColor", fNumber=n, fRed=n / 10, fGreen=0.5, fBlue=0.0) for n in (7, 10)]
    palette = [prim("TColor", fNumber=7, fRed=0.7, fGreen=0.5, fBlue=0.0),
               prim("TColor", fNumber=10, fRed=1.0, fGreen=0.5, fBlue=0.0)]  # fmt: skip
    ax = make([(colors, ""), (palette, ""), (_cone(), "cont2")]).plot().axes[0]
    drawn = [a for a in ax.get_children() if getattr(a, "pixel_clip", None) == FRAME]
    assert {tuple(a.dashes) for a in drawn} == {(3, 3), (5, 3, 1, 3)}  # styles 2 and 5


def test_set_contour_in_a_script_is_what_its_canvas_draws(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """``h.SetContour(2, levels)`` then ``Draw("CONT3")``: the canvas walks those two levels."""
    import array

    import xrdroot.pyroot as ROOT

    h = ROOT.TH2D("hscript", "h", 4, 0, 4, 4, 0, 4)
    for i in range(4):
        for j in range(4):
            h.SetBinContent(i + 1, j + 1, float(i + j))
    h.SetContour(2, array.array("d", [1.5, 3.5]))
    seen = spy(monkeypatch)
    canvas = ROOT.TCanvas("ccont", "c", 300, 200)
    h.Draw("CONT3")
    canvas.SaveAs(str(tmp_path / "cont.png"))
    assert seen == [[1.5, 3.5]]
