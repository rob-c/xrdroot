"""Indexing a histogram as UHI says: bins, locators, cuts, rebinning and sums.

The rules are UHI's, and where ``boost-histogram`` implements them the same
index is given to one of its histograms filled alike, and the two must
agree - flow bins and all. The bookkeeping is ROOT's: an axis summed away is
a projection, with ``ProjectionX``'s rules for what the result keeps; a cut
keeps the entries and, once anything went into the flow, makes the running
sums again from the bins, as ``TH1::Rebin`` does when bins fall off the end.
"""

from __future__ import annotations

import pathlib

import boost_histogram as bh
import numpy as np
import pytest

from xrdroot import (
    Histogram,
    Profile,
    UnsupportedFeatureError,
    loc,
    open_root,
    overflow,
    rebin,
    underflow,
)

DATA = pathlib.Path(__file__).parent / "data"

#: The fills every one-axis example here is made of, off both ends included.
FILLS = [-1.0, 0.5, 1.5, 1.5, 2.5, 3.5, 4.5, 4.5, 4.5, 7.0]


def five(kind="D"):
    h = Histogram.book("h", (5, 0.0, 5.0), kind=kind)
    h.fill(FILLS)
    return h


def boosted():
    made = bh.Histogram(bh.axis.Regular(5, 0.0, 5.0))
    made.fill(FILLS)
    return made


def grid():
    h = Histogram.book("g", (3, 0.0, 3.0), (2, 0.0, 2.0))
    h.fill([0.5, 1.5, 2.5, 2.5, -1.0], [0.5, 1.5, 1.5, 3.0, 0.5])
    return h


# -- single bins --------------------------------------------------------------


def test_a_whole_number_is_a_bin_counted_from_zero_and_from_the_end_below_it():
    h = five()
    assert [h[i] for i in range(5)] == [1.0, 2.0, 1.0, 1.0, 3.0]
    assert h[-1] == 3.0 and h[np.int64(1)] == 2.0
    with pytest.raises(IndexError, match="bin 5 is not on an axis of 5 bins"):
        h[5]
    with pytest.raises(IndexError, match="bin -6"):
        h[-6]


def test_a_locator_finds_a_bin_by_coordinate_and_the_flow_bins_by_name():
    h = five()
    assert (h[underflow], h[overflow]) == (1.0, 1.0)
    assert h[loc(1.7)] == 2.0 and h[loc(1.7) + 1] == 1.0 and h[loc(1.7) - 1] == 1.0
    assert h[loc(-3.0)] == 1.0 and h[loc(9.0)] == 1.0  # off the ends, into the flow
    assert h[overflow - 1] == 3.0
    with pytest.raises(IndexError, match=r"locates bin 7.*from\s+-1, its underflow"):
        h[loc(9.0) + 2]
    assert (repr(loc(1.5)), repr(loc(1.5) + 2), repr(underflow), repr(overflow)) == (
        "loc(1.5)",
        "loc(1.5) + 2",
        "underflow",
        "overflow",
    )


def test_an_axis_says_which_bin_a_coordinate_is_in_as_uhi_numbers_them():
    axis = five().axes[0]
    assert [axis.index(x) for x in (-0.5, 0.0, 4.99, 5.0, float("nan"))] == [-1, 0, 4, 5, 5]


def test_boost_histograms_own_locators_and_rebin_work_here_as_they_work_there():
    h, b = five(), boosted()
    assert h[bh.loc(2.5)] == b[bh.loc(2.5)]
    assert h[bh.underflow] == b[bh.underflow] and h[bh.overflow] == b[bh.overflow]
    assert h[:: bh.rebin(2)].values(flow=True).tolist() == b[:: bh.rebin(2)].values(True).tolist()


def test_what_is_not_an_index_is_refused_by_name():
    h = five()
    with pytest.raises(TypeError, match="a float does not pick a bin"):
        h[1.5]
    with pytest.raises(TypeError, match="a float does not end a slice"):
        h[1.5:]
    with pytest.raises(TypeError, match="2 is not a step a histogram's slice takes"):
        h[::2]
    with pytest.raises(ValueError, match="rebin"):
        rebin(0)
    with pytest.raises(ValueError, match="rebin"):
        rebin(True)


# -- cutting and rebinning ------------------------------------------------------


def test_a_cut_keeps_what_it_cuts_away_in_the_flow_bins_as_boost_histogram_does():
    h, b = five(), boosted()
    for index in (slice(1, 4), slice(None, 3), slice(2, None), slice(0, len)):
        assert h[index].values(flow=True).tolist() == b[index].values(flow=True).tolist()
        assert h[index].edges().tolist() == b[index].axes[0].edges.tolist()
    # A negative start counts from the end, as UHI says and boost-histogram 1.7
    # on does; 1.6, which is the last for Python 3.9, counted it otherwise.
    assert h[-2:].values(flow=True).tolist() == [5.0, 1.0, 3.0, 1.0]
    assert h[-2:].edges().tolist() == [3.0, 4.0, 5.0]


def test_a_cut_by_coordinate_is_a_cut_from_the_bins_the_coordinates_are_in():
    h = five()
    cut = h[loc(1.5) : loc(3.0)]
    assert cut.edges().tolist() == [1.0, 2.0, 3.0]
    assert cut.values(flow=True).tolist() == [2.0, 2.0, 1.0, 5.0]
    assert h[underflow:overflow].values(flow=True).tolist() == h.values(flow=True).tolist()


def test_a_cut_keeps_the_entries_and_makes_the_running_sums_again_from_the_bins():
    h = five()
    cut = h[1:4]
    assert cut.entries == h.entries == 10
    assert cut.mean() == (2 * 1.5 + 2.5 + 3.5) / 4  # from the bins now on the axis
    whole = h[:]
    assert whole.mean() == h.mean() and whole is not h  # nothing moved: the sums stand
    assert whole.members["TH1"]["fTsumw"] == h.members["TH1"]["fTsumw"]


def test_rebinning_merges_bins_and_sends_what_a_group_leaves_over_to_the_overflow():
    h, b = five(), boosted()
    two = h[:: rebin(2)]
    assert two.values(flow=True).tolist() == [1.0, 3.0, 2.0, 4.0]
    assert two.values(flow=True).tolist() == b[:: bh.rebin(2)].values(flow=True).tolist()
    assert two.edges().tolist() == [0.0, 2.0, 4.0] and two.axes[0].even
    both = h[1 : 5 : rebin(2)]
    assert both.values(flow=True).tolist() == b[1 : 5 : bh.rebin(2)].values(flow=True).tolist()
    assert h[:: rebin(5)].mean() == h.mean()  # every bin in one, nothing moved
    with pytest.raises(ValueError, match="make no group of 3"):
        h[3 :: rebin(3)]
    with pytest.raises(ValueError, match="leaves no bins"):
        h[3:3]


def test_a_rebinned_histogram_is_the_same_as_roots_rebin_of_it():
    with open_root(str(DATA / "gauss-h1.root")) as f:
        h = f["h1d"]
        assert h[:: rebin(2)].values(flow=True).tolist() == h.rebin(2).values(flow=True).tolist()
        # ROOT squares each bin's error back from its root; the squares here add as kept.
        assert h[:: rebin(2)].variances() == pytest.approx(h.rebin(2).variances(), rel=1e-14)
        assert h[:: rebin(2)].entries == h.entries


def test_an_uneven_axis_is_cut_and_rebinned_onto_its_own_edges():
    h = Histogram.book("u", [0.0, 1.0, 3.0, 6.0, 10.0])
    h.fill([0.5, 2.0, 4.0, 8.0], weight=2.0)
    cut = h[1 :: rebin(2)]
    assert cut.edges().tolist() == [1.0, 6.0] and not cut.axes[0].even
    assert cut.values(flow=True).tolist() == [2.0, 4.0, 2.0]
    assert cut.variances(flow=True).tolist() == [4.0, 8.0, 4.0]


def test_an_even_axis_cut_in_the_middle_keeps_roots_edges():
    h = Histogram.book("e", (10, 0.0, 1.0))
    cut = h[3:7]
    assert (cut.axes[0].low, cut.axes[0].high) == (0.0 + 3 * 0.1, 0.0 + 7 * 0.1)


def test_a_cut_keeps_the_storage_and_its_limits():
    h = five("F")
    cut = h[1:3]
    assert cut.classname == "TH1F" and cut._bins.dtype == np.float32
    assert cut.values(flow=True).tolist() == [2.0, 2.0, 1.0, 5.0]


def test_a_cut_of_two_axes_cuts_each_the_way_it_cuts_one():
    g = grid()
    cut = g[1:, :1]
    assert cut.shape == (2, 1)
    assert cut.values(flow=True).sum() == g.values(flow=True).sum()
    assert cut.values().tolist() == [[0.0], [0.0]]
    assert cut.values(flow=True)[1:-1, -1].tolist() == [1.0, 2.0]


def test_a_profile_is_cut_as_its_sums_are_and_keeps_its_means():
    p = Profile.book("p", (4, 0.0, 4.0))
    p.fill([0.5, 1.5, 1.5, 2.5, 3.5], [1.0, 2.0, 4.0, 6.0, 8.0])
    cut = p[1:3]
    assert cut.values().tolist() == [3.0, 6.0] and cut.bin_entries().tolist() == [2.0, 1.0]
    merged = p[:: rebin(2)]
    assert merged.values().tolist() == [7 / 3, 7.0]
    assert p[1] == 3.0 and p[underflow] == 0.0  # a bin's mean


# -- summing axes away ----------------------------------------------------------


def test_summing_every_axis_gives_a_number_flow_and_all_unless_the_slice_says():
    h, b = five(), boosted()
    assert h[::sum] == b[::sum] == 10.0
    assert h[sum] == 10.0
    assert h[0:len:sum] == b[0:len:sum] == 8.0
    assert h[1:3:sum] == 3.0 and h[underflow:2:sum] == 4.0
    with pytest.raises(ValueError, match="sums no bins"):
        h[3:1:sum]


def test_summing_an_axis_away_is_roots_projection_with_its_range():
    g = grid()
    assert g[:, sum].values(flow=True).tolist() == g.projection_x().values(flow=True).tolist()
    assert g[{1: sum}].values(flow=True).tolist() == g[:, sum].values(flow=True).tolist()
    assert g[sum, :].values(flow=True).tolist() == g.projection_y().values(flow=True).tolist()
    ranged = g[:, 0:len:sum]
    assert ranged.values(flow=True).tolist() == g.projection_x(y_range=(1, 2)).values(True).tolist()
    assert g[:, sum].entries == g.entries  # all of y, flow and all: ROOT keeps the entries
    assert g[:, sum].name == "g"


def test_a_bin_picked_on_one_axis_is_that_bin_of_the_rest():
    g = grid()
    assert g[2, :].values(flow=True).tolist() == [0.0, 0.0, 1.0, 1.0]  # y of 1.5, and 3
    assert g[2, 1] == 1.0 and g[underflow, 0] == 1.0
    assert g[1:3, sum].values(flow=True).tolist() == g[:, sum][1:3].values(flow=True).tolist()
    assert g[::sum, ::sum] == g.values(flow=True).sum()


def test_three_axes_summed_down_to_two_keep_the_order_they_had():
    h = Histogram.book("c", (2, 0, 2), (3, 0, 3), (4, 0, 4))
    h.fill([0.5, 1.5], [0.5, 2.5], [3.5, 3.5])
    xz = h[:, sum, :]
    assert xz.shape == (2, 4) and xz.values()[1, 3] == 1.0
    assert h[..., sum].shape == (2, 3) and h[0, ...].shape == (3, 4)


def test_a_profile_is_not_summed_since_a_sum_of_means_is_not_a_thing():
    p = Profile.book("p", (2, 0.0, 2.0))
    p.fill([0.5], [1.0])
    with pytest.raises(UnsupportedFeatureError, match="a sum of means is not a thing"):
        p[::sum]
    two = Profile.book("q", (2, 0, 2), (2, 0, 2))
    with pytest.raises(UnsupportedFeatureError, match="is a profile"):
        two[:, sum]


# -- the shape of an index ------------------------------------------------------


def test_an_index_is_one_item_per_axis_with_ellipsis_for_the_rest_or_a_dict_by_axis():
    g = grid()
    assert g[..., 1].values().tolist() == g[:, 1].values().tolist()
    assert g[1, ...].values().tolist() == g[1].values().tolist()
    assert g[{0: 2, 1: 1}] == g[2, 1]
    with pytest.raises(IndexError, match=r"one \.\.\. and this one holds more"):
        g[..., ...]
    with pytest.raises(IndexError, match="an index of 3 items"):
        g[1, 1, 1]
    with pytest.raises(IndexError, match="2 is not an axis of 'g'"):
        g[{2: sum}]


# -- setting bins ---------------------------------------------------------------


def test_setting_a_bin_is_set_bin_content_one_entry_more_and_the_sums_from_the_bins():
    h = five()
    h[2] = 7.0
    assert h.values().tolist() == [1.0, 2.0, 7.0, 1.0, 3.0]
    assert h.entries == 11
    assert h.mean() == pytest.approx((0.5 + 3.0 + 17.5 + 3.5 + 13.5) / 14)
    h[underflow] = 0.0
    assert h.values(flow=True)[0] == 0.0 and h.entries == 12


def test_setting_a_slice_takes_one_value_per_bin_or_one_for_all():
    h = five()
    h[1:3] = [8.0, 9.0]
    assert h.values().tolist() == [1.0, 8.0, 9.0, 1.0, 3.0] and h.entries == 12
    h[...] = 4.0
    assert h.values(flow=True).tolist() == [1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0]
    h[:] = np.arange(7.0)  # two more than the axis: the flow bins too, as UHI has it
    assert h.values(flow=True).tolist() == list(range(7))
    with pytest.raises(ValueError, match=r"values of shape \(3,\) do not fit the \(2,\) bins"):
        h[1:3] = [1.0, 2.0, 3.0]


def test_setting_a_grid_takes_the_flow_along_whichever_axis_has_room_for_it():
    g = grid()
    g[:, 1] = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert g.values(flow=True)[:, 2].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    g[1:, :] = [[6.0, 7.0], [8.0, 9.0]]
    assert g.values()[1:].tolist() == [[6.0, 7.0], [8.0, 9.0]]


def test_setting_is_bins_and_slices_and_nothing_else():
    h = five()
    with pytest.raises(ValueError, match="sum sets none"):
        h[::sum] = 1.0
    with pytest.raises(ValueError, match="rebin sets none"):
        h[:: rebin(2)] = 1.0
    with pytest.raises(UnsupportedFeatureError, match="a mean cannot be set"):
        Profile.book("p", (2, 0, 2))[0] = 1.0
