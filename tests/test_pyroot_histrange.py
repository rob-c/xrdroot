"""``THistRange``: a histogram's bins as a range, the way ROOT's ``TBinIterator`` walks them.

ROOT's ``test_THBinIterator`` is the authority for the numbers here - it sums a histogram's
bins through the axis ranges, through the global bin numbers and through this range, and
wants one answer from all three.
"""

from __future__ import annotations

import xrdroot.pyroot as ROOT


def filled_1d(bins: int = 4) -> ROOT.TH1D:
    found = ROOT.TH1D("h", "h", bins, 0.0, float(bins))
    for number in range(1, bins + 1):
        found.SetBinContent(number, 10.0 * number)
    return found


def test_a_range_visits_the_bins_without_the_flow_bins() -> None:
    found = filled_1d()
    assert list(ROOT.THistRange(found)) == [1, 2, 3, 4]
    assert sum(found.GetBinContent(bin) for bin in ROOT.THistRange(found)) == found.Integral()


def test_a_narrowed_axis_range_visits_only_the_bins_it_draws() -> None:
    found = filled_1d()
    found.GetXaxis().SetRange(2, 3)
    assert list(ROOT.THistRange(found)) == [2, 3]


def test_the_four_range_kinds_visit_what_roothas_them_visit() -> None:
    found = filled_1d()
    kinds = ROOT.TBinIterator
    assert list(ROOT.THistRange(found, kinds.kAxisBins)) == [1, 2, 3, 4]
    assert list(ROOT.THistRange(found, kinds.kAllBins)) == [0, 1, 2, 3, 4, 5]
    assert list(ROOT.THistRange(found, kinds.kUnOfBins)) == [0, 1, 2, 3, 4, 5]
    assert (kinds.kHistRange, kinds.kAxisBins, kinds.kAllBins, kinds.kUnOfBins) == (0, 1, 2, 3)


def test_a_two_dimensional_range_numbers_its_bins_x_fastest() -> None:
    found = ROOT.TH2D("h", "h", 2, 0.0, 2.0, 3, 0.0, 3.0)
    assert list(ROOT.THistRange(found)) == [5, 6, 9, 10, 13, 14]
    found.GetYaxis().SetRange(2, 3)
    assert list(ROOT.THistRange(found)) == [9, 10, 13, 14]


def test_a_three_dimensional_range_walks_z_last() -> None:
    found = ROOT.TH3D("h", "h", 1, 0.0, 1.0, 1, 0.0, 1.0, 2, 0.0, 2.0)
    assert list(ROOT.THistRange(found)) == [13, 22]
    found.GetZaxis().SetRange(2, 2)
    assert list(ROOT.THistRange(found)) == [22]


def test_a_polygons_bins_are_one_dimension_of_their_own() -> None:
    poly = ROOT.TH2Poly("p", "p", 0.0, 3.0, 0.0, 1.0)
    for number in range(3):
        poly.AddBin(number, 0, number + 1, 1)
    assert list(ROOT.THistRange(poly)) == [1, 2, 3]
    # ROOT stops at the bin numbered -1, which is its own end sentinel, so a poly's last
    # flow bin is never visited - measured on 6.40, not guessed at.
    assert list(ROOT.THistRange(poly, ROOT.TBinIterator.kAllBins)) == list(range(-9, -1))
    assert list(ROOT.THistRange(poly, ROOT.TBinIterator.kUnOfBins)) == list(range(-9, -1))
    assert list(ROOT.THistRange(poly, ROOT.TBinIterator.kAxisBins)) == [1, 2, 3]


def test_an_iterator_stops_where_end_says_it_does() -> None:
    found = filled_1d()
    walking, end = ROOT.THistRange(found).begin(), ROOT.THistRange(found).end()
    assert (walking != end, walking == end) == (True, False)
    visited = []
    while walking != end:
        visited.append(next(walking))
    assert visited == [1, 2, 3, 4]
    assert (walking == end, walking == 1) == (True, False)
    assert repr(walking) == "TBinIterator(-1)"


def test_a_range_hands_out_a_fresh_walk_every_time() -> None:
    found = filled_1d()
    once = list(ROOT.THistRange(found))
    assert (list(ROOT.THistRange(found)), once) == (once, [1, 2, 3, 4])
    walking = ROOT.THistRange(found).begin()
    assert iter(walking) is walking
