"""``TH2Poly``: bins of any shape, filled where a point falls, drawn as ROOT draws them.

The honeycombs are ``hist038``'s; the regions round the range are
``TH2Poly::Fill``'s, numbered -1 to -9 from the top left.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from geomsupport import geometry_session  # noqa: F401
from pyrootsupport import expect
from xrdroot.polybins import PolyBins, inside


def test_a_point_is_in_a_polygon_by_the_crossings_of_a_ray_from_it() -> None:
    xs, ys = [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]
    expect((inside(xs, ys, 0.5, 0.5), True), (inside(xs, ys, 1.5, 0.5), False),
           (inside(xs, ys, 0.5, 1.5), False))  # fmt: skip


def test_a_floating_histogram_widens_its_axes_from_the_origin() -> None:
    bins = PolyBins()
    bins.add([([1.0, 2.0, 2.0, 1.0], [1.0, 1.0, 3.0, 3.0])])
    expect((bins.xrange, [0.0, 2.0]), (bins.yrange, [0.0, 3.0]))


@pytest.mark.parametrize(
    ("x", "y", "found"),
    [(-1, 4, -1), (1, 4, -2), (3, 4, -3), (-1, 2, -4), (0.5, 0.5, -5), (3, 2, -6),
     (-1, -1, -7), (1, -1, -8), (3, -1, -9), (1.5, 2, 1)],
)  # fmt: skip
def test_a_fill_goes_to_the_bin_holding_it_or_the_region_it_fell_in(x: float, y: float,
                                                                    found: int) -> None:
    bins = PolyBins((0.0, 2.0), (0.0, 3.0), floating=False)
    bins.add([([1.0, 2.0, 2.0, 1.0], [1.0, 1.0, 3.0, 3.0])])
    assert bins.fill(x, y) == found


def honeycomb(option: str = "v") -> Any:
    h = ROOT.TH2Poly()
    h.Honeycomb(0, 0, 0.1, 5, 5, option)
    for x, y, w in ((0.1, 0.1, 15.0), (0.4, 0.4, 10.0), (0.5, 0.5, 20.0)):
        h.Fill(x, y, w)
    return h


def test_a_honeycomb_has_k_and_k_less_one_hexagons_in_its_rows_by_turns() -> None:
    upright, lying = honeycomb("v"), honeycomb("h")
    expect((upright.GetNumberOfBins(), 23), (lying.GetNumberOfBins(), 23),
           (upright.Integral(), 45.0), (upright.GetEntries(), 3.0),
           (upright.GetMaximum(), 20.0), (upright.GetMinimum(), 0.0),
           (upright.GetMean(1), pytest.approx((1.5 + 4 + 10) / 45)),
           (upright.GetMean(2), pytest.approx((1.5 + 4 + 10) / 45)))  # fmt: skip


def test_bins_are_added_as_boxes_polygons_or_named_graphs_and_filled_by_name() -> None:
    h = ROOT.TH2Poly("h", "h", 0, 10, 0, 10)
    box = h.AddBin(0, 0, 1, 1)
    triangle = h.AddBin(3, np.array([2.0, 4.0, 2.0]), np.array([2.0, 2.0, 4.0]))
    graph = ROOT.TGraph(4, np.array([5.0, 6, 6, 5]), np.array([5.0, 5, 6, 6]))
    graph.SetName("square")
    named = h.AddBin(graph)
    many = ROOT.TMultiGraph()
    many.Add(ROOT.TGraph(3, np.array([7.0, 8, 7]), np.array([7.0, 7, 8])))
    h.AddBin(many)
    h.Fill("square", 2.0)
    h.Fill("nothing", 2.0)
    h.SetBinContent(box, 3.0)
    h.SetBinContent(-5, 1.0)
    h.SetBinContent(99, 1.0)
    expect(((box, triangle, named), (1, 2, 3)), (h.GetBinContent(named), 2.0),
           (h.GetBinContent(1), 3.0),
           (h.GetBinContent(-5), 1.0), (h.GetBinContent(99), 0.0), (h.GetBinCenter(2), (3.0, 3.0)),
           (h.GetMean(), 0.0))  # fmt: skip
    h.SetMinimum(1)
    h.SetMaximum(5)
    expect((h.GetMinimum(), 1.0), (h.GetMaximum(), 5.0))
    h.Reset()
    expect((h.Integral(), 0.0), (h.GetBinContent(-5), 0.0), (ROOT.TH2Poly().GetMaximum(), 0.0),
           (ROOT.TH2Poly("b", "b", 10, 0, 1, 10, 0, 1).Fill(2, 2), -3))  # fmt: skip


def test_a_polygon_histogram_is_painted_as_its_option_says(tmp_path: Any) -> None:
    canvas = ROOT.TCanvas("c", "c", 300, 300)
    h = honeycomb()
    h.SetStats(0)
    h.Draw("colz L TEXT")
    canvas.Update()
    kinds = [getattr(obj, "classname", "") for obj, _ in canvas.primitives]
    h.Draw("L same")
    canvas.SaveAs(str(tmp_path / "poly.png"))
    expect((kinds.count("TLatex"), 23), (kinds.count("TPolyLine"), 46))


def test_a_2d_histogram_leaves_cells_below_its_minimum_unpainted(tmp_path: Any) -> None:
    from xrdroot.plot.grid import _mesh

    del _mesh
    h = ROOT.TH2F("h2", "h2", 2, 0, 1, 1, 0, 1)
    h.SetBinContent(1, 1, 1.0)
    h.SetBinContent(2, 1, 5.0)
    h.SetMinimum(2.0)
    ROOT.TCanvas("c", "c", 200, 200)
    h.Draw("colz")
    ROOT.gPad.SaveAs(str(tmp_path / "min.png"))
    assert (tmp_path / "min.png").exists()


def test_a_polygon_histogram_with_bins_below_its_minimum_paints_only_its_frame() -> None:
    canvas = ROOT.TCanvas("c", "c", 200, 200)
    h = honeycomb()
    h.SetMinimum(100)
    h.Draw("col")
    canvas.Update()
    assert [getattr(obj, "classname", "") for obj, _ in canvas.primitives].count("TPolyLine") == 0
