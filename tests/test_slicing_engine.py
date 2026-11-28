"""``h[...]``: UHI indexing over a histogram, the way ``hist`` and ``boost-histogram`` do it.

A histogram is booked and filled in code, so the indexing rules are testable without a file: a whole
number is a bin counted from zero, a locator names the bin a coordinate falls in or a flow bin, a
slice cuts the axis and **adds what it cut away to the flow** rather than losing it, a rebin merges
bins and its leftovers go to the overflow, and a ``sum`` step sums an axis away. Every number below
was measured from a histogram booked here before it was asserted.

The bookkeeping is ROOT's where ROOT has the same operation: a cut or a rebinning is ``TH1::Rebin``
with the bins dropping into the flow, so the entries never change; setting through a slice is 6.40's
``SetSliceContent``, which resets the statistics, so the entries become the total weight.

No file is opened, which is what lets this test sit inside the ``hist`` component's own scope.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot import UnsupportedFeatureError
from xrdroot.hist import Histogram
from xrdroot.profile import Profile
from xrdroot.slicing import loc, overflow, rebin, underflow


def _filled() -> Histogram:
    """Six bins of 0..6 with one fill in the first, two in the second and one in the last, and one
    below and one above: the flow bins hold one each."""
    made = Histogram.book("h", (6, 0.0, 6.0))
    for value in (0.5, 1.5, 1.5, 5.5, -1.0, 9.0):
        made.fill(value)
    return made


def _plane() -> Histogram:
    """Three x bins and two y bins, with one fill in (1,1), two in (2,1) and one in (2,2)."""
    made = Histogram.book("p", (3, 0.0, 3.0), (2, 0.0, 2.0))
    for x, y in ((0.5, 0.5), (1.5, 0.5), (1.5, 1.5)):
        made.fill(x, y)
    return made


def test_a_whole_number_is_a_bin_counted_from_zero_and_a_negative_one_from_the_end():
    """UHI's rule: ``h[1]`` is the **second** bin and ``h[-1]`` the last, and a locator names the
    bin a coordinate falls in."""
    filled = _filled()
    assert (filled[1], filled[-1], filled[loc(1.7)]) == (2.0, 1.0, 2.0)


def test_a_locator_names_a_flow_bin_the_bins_themselves_do_not():
    """``underflow`` and ``overflow`` are locators, so the fills that fell off the axis are
    reachable - which is what a whole number cannot do."""
    filled = _filled()
    assert (filled[underflow], filled[overflow]) == (1.0, 1.0)


def test_a_slice_cuts_the_axis_and_pours_what_it_cut_into_the_flow():
    """``h[1:3]`` leaves two bins whose contents are the second and third, and the bins it cut
    away arrive in the new flow bins - so nothing is lost and ``GetEntries`` is unchanged."""
    cut = _filled()[1:3]
    assert cut.values().tolist() == [2.0, 0.0]
    assert cut.values(flow=True).tolist() == [2.0, 2.0, 0.0, 2.0]  # 1 + 1 below, 0 + 1 above
    assert cut.entries == 6.0  # ROOT's Rebin keeps the entries: every fill is still somewhere


def test_a_rebin_step_merges_bins_and_the_ones_a_group_leaves_over_go_to_the_overflow():
    """``h[::rebin(2)]`` pairs the six bins into three; a group is only made whole, so whatever a
    group leaves over joins the overflow rather than making a short last bin."""
    merged = _filled()[::rebin(2)]
    assert merged.values().tolist() == [3.0, 0.0, 1.0]  # 1+2, 0+0, 0+1
    assert merged.values(flow=True).tolist() == [1.0, 3.0, 0.0, 1.0, 1.0]
    assert merged.entries == 6.0


def test_a_sum_step_sums_an_axis_away_flow_and_all_unless_the_slice_says_where():
    """``h[sum]`` is the whole histogram including its flow, and ``h[0:6:sum]`` is the bins
    without it - the difference being the two fills that fell off the axis."""
    filled = _filled()
    assert filled[sum] == 6.0
    assert filled[0:6:sum] == 4.0


def test_a_bin_picked_on_one_axis_of_a_plane_leaves_the_other_axis_s_projection():
    """Picking a row of a 2-D histogram with ``h[1]`` gives the projection along the remaining
    axis, which is ROOT's ``ProjectionY`` over that row's range."""
    row = _plane()[1]
    assert row.values().tolist() == [1.0, 1.0]  # the second x bin, summed over y
    assert row.values(flow=True).tolist() == [0.0, 1.0, 1.0, 0.0]


def test_a_plane_s_axes_are_indexed_one_at_a_time_with_a_tuple_and_a_dictionary():
    """``h[1, :]``, ``h[:, sum]`` and ``h[{0: slice(0, 2), 1: sum}]`` each say which axis every
    part of the index is for, and a summed axis leaves the other axis's contents."""
    plane = _plane()
    assert plane[1, :].values().tolist() == [1.0, 1.0]
    assert plane[:, sum].values().tolist() == [1.0, 2.0, 0.0]  # every y summed into its x
    assert plane[{0: slice(0, 2), 1: sum}].values().tolist() == [1.0, 2.0]


def test_setting_every_bin_at_once_makes_the_entries_the_total_weight():
    """``h[...] = counts`` is 6.40's ``SetSliceContent``: the bins are written and the statistics
    reset, so the entries become what was written - six for 1, 2 and 3."""
    made = Histogram.book("g", (3, 0.0, 3.0))
    made[...] = np.array([1.0, 2.0, 3.0])
    assert made.values().tolist() == [1.0, 2.0, 3.0]
    assert made.entries == 6.0


def test_a_profile_s_summed_axis_is_refused_because_a_sum_of_means_is_not_a_thing():
    """A profile's bins are means and ROOT has no sum of them; the refusal names the histogram and
    says to pick a bin or to slice without summing, while picking one bin still works."""
    profile = Profile.book("pr", (4, 0.0, 4.0))
    profile.fill(0.5, 2.0)
    assert profile[0] == pytest.approx(2.0)  # a bin of a profile is its mean
    with pytest.raises(UnsupportedFeatureError, match="a sum of means is not a thing"):
        profile[sum]
