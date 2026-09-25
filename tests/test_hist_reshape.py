"""Rebinning, projecting and profiling histograms, with ROOT's bookkeeping.

What goes where is checked with fills small enough to follow by hand: which
old bins a new one takes, where what falls off a shortened axis goes, and
when ROOT keeps the old running sums and entries and when it makes them
again - the rules of ``TH1::Rebin``, ``TH2::Rebin2D``, ``TH2::ProjectionX``
and ``Project3D``, and ``TH2::ProfileX``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot import Efficiency, Graph, Histogram, Profile, UnsupportedFeatureError


def four():
    h = Histogram.book("h", (4, 0.0, 4.0))
    h.fill([0.5, 1.5, 1.5, 2.5, 3.5, -1.0, 9.0])
    return h


def test_rebinning_by_a_divisor_merges_neighbours_and_keeps_the_sums():
    h = four()
    two = h.rebin(2)
    assert two.axes[0].edges().tolist() == [0, 2, 4] and two.axes[0].even
    assert two.values(flow=True).tolist() == [1, 3, 2, 1]
    assert two.entries == 7 and two.mean() == h.mean()
    assert two.name == "h" and h.rebin(4, name="one").name == "one"
    assert h.axes[0].nbins == 4  # a new histogram; this one is as it was


def test_bins_a_group_leaves_over_go_to_the_overflow_and_the_sums_are_made_again():
    h = Histogram.book("h", (5, 0.0, 5.0))
    h.fill([0.5, 1.5, 4.5])
    two = h.rebin(2)
    assert two.axes[0].edges().tolist() == [0, 2, 4]
    assert two.values(flow=True).tolist() == [0, 2, 0, 1]
    assert two.entries == 3
    assert two.mean() == 1.0  # from the bins, the fill at 4.5 being off the axis now


def test_an_uneven_axis_rebins_onto_its_own_edges_and_roots_last_one():
    h = Histogram.book("h", [0.0, 0.1, 0.2, 0.7])
    h.fill([0.05, 0.15, 0.5], weight=2.0)
    one = h.rebin(3)
    # ROOT's GetBinLowEdge past the last bin works the edge out as if even.
    assert one.axes[0].edges().tolist() == [0.0, 0.0 + 3 * ((0.7 - 0.0) / 3)]
    assert one.values().tolist() == [6.0]
    assert one.variances().tolist() == [math.sqrt(12.0) ** 2]


def test_rebinning_onto_edges_the_axis_has_merges_the_bins_between_them():
    h = four()
    made = h.rebin([0, 1, 4])
    assert not made.axes[0].even
    assert made.values(flow=True).tolist() == [1, 1, 4, 1]
    inner = h.rebin(np.array([1.0, 2.0, 3.0]))
    assert inner.values(flow=True).tolist() == [1 + 1, 2, 1, 1 + 1]


def test_rebinning_is_refused_when_it_would_split_a_bin_or_makes_no_sense():
    h = four()
    with pytest.raises(ValueError, match=r"1\.5 is not an edge of the axis"):
        h.rebin([0, 1.5, 4])
    for group in (0, 5, 1.5):
        with pytest.raises(ValueError, match="does not rebin an axis of 4"):
            h.rebin(group)
    with pytest.raises(ValueError, match="one group per axis"):
        h.rebin(2, 2)
    with pytest.raises(ValueError, match="one group per axis"):
        Histogram.book("h3", (2, 0, 1), (2, 0, 1), (2, 0, 1)).rebin(1, 1, 1)
    with pytest.raises(UnsupportedFeatureError, match="rebinning 'p' is not offered"):
        Profile.book("p", (2, 0, 1)).rebin(2)


def test_two_axes_rebin_in_blocks_with_the_leftovers_in_the_overflow():
    h = Histogram.book("h", (4, 0, 4), (3, 0, 3))
    h.fill([0.5, 1.5, 2.5, 3.5, 0.5, -1], [0.5, 0.5, 1.5, 2.5, -1, 0.5], weight=2.0)
    made = h.rebin(2, 2)
    assert made.shape == (2, 1) and made.axes[1].edges().tolist() == [0, 2]
    grid = made.values(flow=True)
    assert grid[1, 1] == 4 and grid[2, 1] == 2 and grid[2, 2] == 2  # y's third bin is overflow now
    assert grid[1, 0] == 2 and grid[0, 1] == 2
    assert grid.sum() == h.values(flow=True).sum()
    assert made.variances(flow=True).sum() == h.variances(flow=True).sum()
    assert made.entries == 6
    even = h.rebin(2, 3)
    assert even.shape == (2, 1) and even.mean(0) == h.mean(0)
    uneven = Histogram.book("u", [0, 1, 3], (2, 0, 2))
    uneven.fill([0.5, 2.0], [0.5, 1.5])
    assert uneven.rebin(2, 1).axes[0].edges().tolist() == [0, 3]


def plane():
    h = Histogram.book("h", (2, 0, 2), (2, 0, 2))
    h.fill([0.5, 1.5, 1.5, 0.5], [0.5, 0.5, 1.5, 5.0])
    return h


def test_a_projection_sums_every_bin_of_the_other_axis_flow_and_all():
    h = plane()
    px = h.projection_x()
    assert (px.name, px.classname) == ("h_px", "TH1D")
    assert px.values(flow=True).tolist() == [0, 2, 2, 0]
    # Its total, four, is not the three fills on the axes, so the sums are
    # made again from the bins, and the entries are the total rounded.
    assert px.entries == 4 and px.mean() == 1.0


def test_a_projection_of_the_bins_on_the_axis_keeps_the_running_sums():
    h = plane()
    inside = h.projection_x(y_range=(1, 2))
    assert inside.values(flow=True).tolist() == [0, 1, 2, 0]
    assert inside.mean() == h.mean(0) and inside.std() == h.std(0)
    assert inside.entries == 3
    along = h.projection_y("py", x_range=(1, 1))
    assert along.name == "py" and along.values(flow=True).tolist() == [0, 1, 0, 1]
    assert along.entries == 2
    backwards = h.projection_y(x_range=(2, 1))  # a range the wrong way round is all of it
    assert backwards.values(flow=True).tolist() == [0, 2, 1, 1]


def test_a_projection_of_everything_keeps_the_entries_when_nothing_fell_off():
    h = Histogram.book("h", (2, 0, 2), (2, 0, 2))
    h.fill([0.5, 1.5], [0.5, 1.5])
    px = h.projection_x()
    assert px.entries == 2 and px.mean() == h.mean(0)


def test_a_weighted_projection_counts_its_effective_entries():
    h = Histogram.book("h", (1, 0, 1), (2, 0, 2))
    h.fill([0.5, 0.5, 0.5], [0.5, 1.5, 7.0], weight=[1.0, 2.0, 3.0])
    px = h.projection_x()
    assert px.values().tolist() == [6.0]
    assert px.variances().tolist() == [math.sqrt(1 + 4 + 9) ** 2]
    assert px.entries == 36 / 14


def test_three_axes_project_onto_the_ones_named_in_the_order_named():
    h = Histogram.book("h", (2, 0, 2), (3, 0, 3), (4, 0, 4))
    h.fill([0.5, 1.5, 1.5], [0.5, 2.5, 2.5], [3.5, 0.5, 0.5])
    xy = h.projection("xy")
    assert xy.classname == "TH2D" and xy.shape == (2, 3) and xy.name == "h_xy"
    assert xy.values()[1, 2] == 2
    zy = h.projection("zy")
    assert zy.shape == (4, 3) and zy.values()[0, 2] == 2 and zy.values()[3, 0] == 1
    x = h.projection("x", ranges={"y": (1, 3), "z": (1, 4)})
    assert x.values().tolist() == [1, 2] and x.mean() == h.mean(0)
    yz = h.projection("yz", ranges={"x": (1, 2)})
    assert yz.members["TH2"]["fTsumwxy"] == h.members["TH3"]["fTsumwyz"]


def test_a_projection_must_name_fewer_axes_than_there_are_each_once():
    h = Histogram.book("h", (2, 0, 2), (2, 0, 2), (2, 0, 2))
    for axes in ("q", "xyz", "xx", ""):
        with pytest.raises(ValueError, match="is not a projection of 'h'"):
            h.projection(axes)
    with pytest.raises(UnsupportedFeatureError, match="projecting 'p' is not offered"):
        Profile.book("p", (2, 0, 2), (2, 0, 2)).projection("x")


def column():
    h = Histogram.book("h", (2, 0, 2), (2, 0, 4))
    h.fill([0.5, 0.5, 0.5, 1.5], [1.0, 3.0, 3.0, 1.0])
    return h


def test_a_profile_along_x_is_the_mean_of_y_in_each_bin_of_x():
    made = column().profile_x()
    assert isinstance(made, Profile) and made.name == "h_pfx"
    assert made.values().tolist() == [7 / 3, 1.0]
    assert made.bin_entries().tolist() == [3, 1]
    # The content two went in as a weight of two, so ROOT started keeping squares.
    assert made.members["fBinSumw2"].tolist() == [0, 5, 1, 0]
    assert made.entries == 16 / 6


def test_a_profile_along_y_is_the_mean_of_x_in_each_bin_of_y():
    made = column().profile_y()
    assert made.name == "h_pfy" and made.values().tolist() == [1.0, 0.5]
    narrow = column().profile_x(y_range=(2, 2))
    assert narrow.values().tolist() == [3.0, 0.0]
    clamped = column().profile_x(y_range=(-1, 99))
    assert clamped.values().tolist() == [7 / 3, 1.0]


def test_a_weighted_histogram_profiles_with_its_own_squares_of_weights():
    h = Histogram.book("h", (1, 0, 1), (2, 0, 2))
    h.fill([0.5, 0.5], [0.5, 1.5], weight=[2.0, 3.0])
    made = h.profile_x()
    assert made.values().tolist() == [(2 * 0.5 + 3 * 1.5) / 5]
    assert made.members["fBinSumw2"].tolist() == [0, 4 + 9, 0]


def test_only_a_histogram_of_two_axes_has_a_profile():
    with pytest.raises(ValueError, match="'h' is not one"):
        Histogram.book("h", (2, 0, 2)).profile_x()


def test_a_histogram_becomes_a_graph_of_its_bins_as_tgrapherrors_makes_one():
    h = Histogram.book("h", [0.0, 1.0, 3.0], title="spectrum")
    h.fill([0.5, 2.0, 2.0])
    graph = Graph.from_histogram(h)
    assert (graph.classname, graph.name, graph.title) == ("TGraphErrors", "h", "spectrum")
    assert graph.x.tolist() == [0.5, 2.0] and graph.y.tolist() == [1, 2]
    assert graph.xerr[0].tolist() == [0.5, 1.0]
    assert graph.yerr[0].tolist() == [1.0, math.sqrt(2)]
    p = Profile.book("p", (1, 0, 1))
    p.fill([0.5, 0.5], [1.0, 3.0])
    assert Graph.from_histogram(p, name="g").y.tolist() == [2.0]


def test_an_efficiency_becomes_a_graph_of_its_intervals_leaving_out_empty_bins():
    eff = Efficiency.book("e", (3, 0, 3))
    eff.fill([True, False, True], [0.5, 0.5, 2.5])
    graph = Graph.from_histogram(eff)
    assert graph.classname == "TGraphAsymmErrors"
    assert graph.x.tolist() == [0.5, 2.5] and graph.y.tolist() == [0.5, 1.0]
    low, high = eff.errors()
    assert graph.yerr[0].tolist() == [low[0], low[2]]
    assert graph.yerr[1].tolist() == [high[0], high[2]]


def test_only_something_of_one_axis_becomes_a_graph():
    with pytest.raises(ValueError, match="a TH2D of 2 axes is not one"):
        Graph.from_histogram(Histogram.book("h", (1, 0, 1), (1, 0, 1)))
    with pytest.raises(ValueError, match="a list of 0 axes is not one"):
        Graph.from_histogram([])
    with pytest.raises(ValueError, match="'e' has 2 axes"):
        Graph.from_histogram(Efficiency.book("e", (1, 0, 1), (1, 0, 1)))


def test_two_axes_rebin_to_uneven_edges_on_both_when_either_was_uneven():
    h = Histogram.book("h", [0, 1, 3, 4], (4, 0, 2))
    made = h.rebin(3, 2)
    assert not made.axes[1].even and made.axes[1].edges().tolist() == [0, 1, 2]
