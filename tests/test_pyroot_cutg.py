"""``TCutG``: a polygon of the plane, points inside it, and histograms drawn within it.

``fit2d.C`` fills only the points ``IsInside`` its cut; ``hist018_TH2_cutg.C``
draws a histogram ``"col [cut]"``. The polygon's sums are ROOT's, worked
here on a unit square whose answers are known.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from geomsupport import geometry_session  # noqa: F401
from pyrootsupport import expect
from xrdroot.pyroot.graphics.cutg import cut_drawn

#: A unit square, closed, anticlockwise.
SQUARE = (np.array([0.0, 1.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0, 0.0]))


def square(name: str = "cut") -> Any:
    return ROOT.TCutG(name, 5, *SQUARE)


def test_a_point_is_inside_by_the_crossings_of_a_ray_from_it() -> None:
    cut = square()
    expect((cut.IsInside(0.5, 0.5), 1), (cut.IsInside(1.5, 0.5), 0), (cut.IsInside(0.5, -0.1), 0),
           (cut.IsInside(0.0, 0.5), 0), (cut.IsInside(1.0, 0.5), 1))  # fmt: skip
    # (a point on the left edge is out and one on the right in, as ROOT's crossing test has it)


def test_a_cut_s_area_and_centre_are_its_polygon_s() -> None:
    cut = square()
    cx, cy = np.zeros(1), np.zeros(1)
    expect((cut.Area(), 1.0), (cut.Center(cx, cy), (0.5, 0.5)), (float(cx[0]), 0.5))


def test_a_cut_is_kept_by_name_and_a_new_one_of_that_name_replaces_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = square()
    first.SetVarX("px")
    first.SetVarY("py")
    second = square()
    expect((ROOT.gROOT.GetListOfSpecials().FindObject("cut"), second),
           ((first.GetVarX(), first.GetVarY()), ("px", "py")))  # fmt: skip
    assert "Replacing existing TCutG: cut" in capsys.readouterr().err


def filled() -> Any:
    h = ROOT.TH2F("h", "h", 4, -1, 3, 4, -1, 3)
    for i in range(1, 5):
        for j in range(1, 5):
            h.SetBinContent(i, j, 1.0)
    return h


def test_a_histogram_s_integral_within_a_cut_is_of_the_cells_whose_centres_it_holds() -> None:
    cut = ROOT.TCutG("big", 5, SQUARE[0] * 2 - 0.1, SQUARE[1] * 2 - 0.1)
    h = filled()
    assert (cut.IntegralHist(h), cut.IntegralHist(h, "width")) == (4.0, 4.0)


def test_a_histogram_drawn_with_a_cut_keeps_only_the_cells_inside_or_outside_it() -> None:
    ROOT.TCutG("big", 5, SQUARE[0] * 2 - 0.1, SQUARE[1] * 2 - 0.1)
    h = filled()
    inside, option = cut_drawn(h, "col [big]")
    outside, _ = cut_drawn(h, "col [-big]")
    unknown, _ = cut_drawn(h, "col [nothing]")
    expect((option, "col "), (float(inside.values().sum()), 4.0),
           (float(outside.values().sum()), 12.0),
           (float(unknown.values().sum()), 16.0), (cut_drawn(h, "col"), (h, "col")),
           (cut_drawn(ROOT.TH1F("one", "one", 2, 0, 1), "[big]")[1], ""))  # fmt: skip


def test_a_histogram_with_a_cut_is_painted(tmp_path: Any) -> None:
    square()
    ROOT.TCanvas("c", "c", 200, 200)
    filled().Draw("col [cut]")
    ROOT.gPad.SaveAs(str(tmp_path / "cut.png"))
    assert (tmp_path / "cut.png").exists()
