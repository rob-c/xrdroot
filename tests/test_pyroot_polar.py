"""``TGraphPolar`` and ``TGraphPolargram``: ROOT's polar graphs, grid and points.

The labels, alignments and ranges are what ``TGraphPolargram`` and
``TGraphPolar::CreatePolargram`` work out, their arithmetic followed here
case by case; the pictures of ``gr012``-``gr014`` are the tutorial harness's.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from geomsupport import geometry_session  # noqa: F401
from pyrootsupport import expect
from xrdroot.canvas import polargram as grid

PI = math.pi


def test_a_polargram_reaches_a_tenth_past_the_radii_and_a_step_past_the_angles() -> None:
    assert grid.polar_range([0, 1, 2, 3], [1, 2, 3, 5]) == pytest.approx((0.6, 5.4, 0.0, 3.75))
    assert grid.polar_range([1.0], [2.0], [0.5], [0.5]) == pytest.approx((1.4, 2.6, 0.5, 2.5))
    assert grid.polar_range([1.0, 1.0], [2.0, 2.0]) == pytest.approx((1.9, 3.1, 1.0, 2.5))


@pytest.mark.parametrize(
    ("angle", "ortho", "align"),
    [(0.0, False, 12), (PI / 4, False, 11), (PI / 2, False, 21), (3 * PI / 4, False, 31),
     (PI, False, 32), (5 * PI / 4, False, 33), (3 * PI / 2, False, 23), (7 * PI / 4, False, 13),
     (-PI / 4, False, 13), (PI / 4, True, 12), (PI, True, 32), (7 * PI / 4, True, 12)],
)  # fmt: skip
def test_labels_are_aligned_by_the_angle_they_are_at(angle: float, ortho: bool, align: int) -> None:
    assert grid.find_align(angle, ortho) == align


def test_labels_along_a_spoke_are_never_upside_down() -> None:
    angles = [grid.find_text_angle(a) for a in (PI / 4, 3 * PI / 4, 5 * PI / 4, 7 * PI / 4)]
    assert angles == pytest.approx([45.0, 315.0, 45.0, 315.0])


def test_radian_labels_are_fractions_of_pi_in_their_lowest_terms() -> None:
    assert [grid.radian_label(i, 8) for i in range(5)] == [
        "0", "#frac{#pi}{4}", "#frac{#pi}{2}", "#frac{3#pi}{4}", "#pi"]  # fmt: skip
    assert grid.radian_label(4, 3) == "#frac{8#pi}{3}" and grid.radian_label(2, 2) == "2#pi"
    assert grid.number_label(0.5) == "0.5" and grid.number_label(-12.25) == "-12.2"


def test_a_line_leaving_the_circle_is_cut_where_it_crosses_and_picks_up_again() -> None:
    runs = grid.inside_runs([0.0, 0.5, 2.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.1, 0.1])
    assert len(runs) == 2
    assert runs[0][0][-1] == pytest.approx(1.0)
    assert math.hypot(runs[1][0][0], runs[1][1][0]) == pytest.approx(1.0)
    assert grid.inside_runs([2.0], [0.0]) == []


def test_a_circle_has_as_many_points_as_its_length_asks() -> None:
    small, big = grid.circle(0.001), grid.circle(1.0)
    expect((len(small[0]), 9), (len(big[0]), 201), (big[0][0], 1.0))


def polar_pad() -> Any:
    ROOT.TCanvas("c", "c", 300, 300)
    theta = np.linspace(0, 2 * PI, 8, endpoint=False)
    graph = ROOT.TGraphPolar(8, theta, 1 + 0.1 * np.arange(8), np.full(8, 0.1), np.full(8, 0.05))
    graph.Draw("AEPLF")
    return graph


def test_a_polar_graph_finds_or_makes_its_polargram_when_its_pad_is_painted(tmp_path: Any) -> None:
    graph = polar_pad()
    assert graph.GetPolargram() is None
    ROOT.gPad.Update()
    gram = graph.GetPolargram()
    again = ROOT.TGraphPolar(2, np.array([0.0, 1.0]), np.array([1.0, 1.2]))
    again.Draw("CP")
    ROOT.gPad.SaveAs(str(tmp_path / "polar.png"))
    expect((again.GetPolargram(), gram), (ROOT.gPad.members["fX2"], 1.25),
           (gram.IsRadian(), True), (len(graph.GetXpol()), 8))  # fmt: skip


def test_a_polargram_is_ranged_and_labelled_as_it_is_told(tmp_path: Any) -> None:
    graph = polar_pad()
    ROOT.gPad.Update()
    gram = graph.GetPolargram()
    graph.SetMinRadial(0.5)
    graph.SetMaxRadial(3.0)
    graph.SetMinPolar(0.1)
    graph.SetMaxPolar(6.0)
    gram.SetNdivPolar(703)
    gram.SetNdivRadial(-504)
    gram.SetToDegree()
    gram.SetAxisAngle(45)
    for label in ("Polar", "Radial"):
        getattr(gram, f"Set{label}LabelColor")(2)
        getattr(gram, f"Set{label}LabelFont")(42)
        getattr(gram, f"Set{label}LabelSize")(0.05)
        getattr(gram, f"Set{label}Offset")(0.03)
    ROOT.gPad.SaveAs(str(tmp_path / "degrees.png"))
    expect(((gram.GetRMin(), gram.GetRMax()), (0.5, 3.0)), (gram.GetTMax(), 360.0),
           (gram.GetNdivPolar(), 703), (gram.GetNdivRadial(), -504), (gram.IsDegree(), True),
           (gram.GetAngle(), pytest.approx(PI / 4)), (gram.GetPolarLabelSize(), 0.05),
           (gram.GetRadialLabelSize(), 0.05), (gram.scale(), pytest.approx(180 / PI)))  # fmt: skip


def test_a_polargram_may_count_in_grads_or_in_nothing_it_knows() -> None:
    gram = ROOT.TGraphPolargram("g", 0, 1, 0, 1, "G")
    expect((gram.IsGrad(), True), (gram.GetTMax(), 200.0), (gram.scale(), pytest.approx(100 / PI)))
    gram.SetToRadian()
    gram.SetTwoPi()
    gram.SetRangeRadial(2, 1)
    gram.SetNdivPolar(0)
    degrees = ROOT.TGraphPolargram("d", 0, 1, 0, 1, "D")
    expect((gram.IsRadian(), False), (gram.GetTMax(), pytest.approx(2 * PI)), (gram.GetRMax(), 1.0),
           (gram.GetNdivPolar(), 508), (degrees.GetTMax(), 360.0))


def test_a_polargram_drawn_alone_paints_only_what_its_option_asks(tmp_path: Any) -> None:
    ROOT.TCanvas("c", "c", 300, 300)
    for option in ("RN", "PO", "P"):
        ROOT.TGraphPolargram("g", 0, 2, 0, 10, "").Draw(option)
    ROOT.TGraphPolargram("r", 0, 2, 0, 1, "R").Draw("O")
    ROOT.gPad.SaveAs(str(tmp_path / "grams.png"))
    assert (tmp_path / "grams.png").stat().st_size > 0


def test_an_empty_polar_graph_has_no_polargram_and_no_limits_to_set() -> None:
    empty = ROOT.TGraphPolar()
    empty.SetMinRadial(0)
    empty.SetMaxRadial(1)
    empty.SetMinPolar(0)
    empty.SetMaxPolar(1)
    expect((empty.CreatePolargram(), None), (empty.GetPolargram(), None))


def test_a_polar_graph_with_no_bars_and_only_a_fill_paints_what_it_is_asked(tmp_path: Any) -> None:
    ROOT.TCanvas("c", "c", 300, 300)
    theta = np.linspace(0, 2 * PI, 6, endpoint=False)
    graph = ROOT.TGraphPolar(6, theta, np.ones(6), np.zeros(6), np.zeros(6))
    graph.Draw("EF")
    gram = ROOT.TGraphPolargram("mine", 0, 2, 0, 2 * PI, "")
    graph.SetPolargram(gram)
    gram.ChangeRangePolar(1, 0)
    gram.SetToGrad()
    ROOT.gPad.SaveAs(str(tmp_path / "fill.png"))
    expect((graph.GetPolargram() is not gram, True), (len(graph.GetYpol()), 6),
           (gram.GetTMax(), 200.0))  # fmt: skip
