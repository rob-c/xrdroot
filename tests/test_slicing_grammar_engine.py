"""The index a histogram takes: its locators, its steps, and the refusals around them.

Round 148 covered what UHI indexing *reads* - a bin, a slice, a summed axis, a rebinned one.
This file covers the parts of that grammar a reader meets when they write an index the kit cannot
take: ``loc`` and its two flow locators, ``rebin``, ``len`` as an end, the ellipsis, a dictionary
by axis, and every refusal in a sentence that says what the index may hold instead.

It also covers the arms of a cut that only a particular histogram reaches - an irregular axis
keeps the edges it was booked with, a profile keeps its per-bin weights beside its means, a
weighted histogram keeps its squares - and the two ways of *setting* bins: a slice, which resets
the statistics, and one bin, which counts an entry.

Everything is booked in code, so no file is opened and the tests stay inside the ``hist``
component's own isolation scope.
"""

from __future__ import annotations

import pytest

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.hist import Histogram
from xrdroot.profile import Profile
from xrdroot.slicing import loc, overflow, rebin, underflow


def _filled() -> Histogram:
    """Three bins of 0..3 holding 2, 1 and 1 - the shape every index below is applied to."""
    made = Histogram.book("h", (3, 0.0, 3.0))
    made.fill([0.5, 0.5, 1.5, 2.5])
    return made


def test_a_locator_is_a_bin_plus_a_whole_number_of_bins_and_prints_itself():
    """``loc(1.5) + 1`` is the bin after the one 1.5 falls in, and the offset is kept as written
    so a reader can see where it came from: ``loc(1.5) + 1``, not a bare number."""
    axis = _filled().axes[0]
    assert loc(1.5)(axis) == 1
    assert (loc(1.5) + 1)(axis) == 2
    assert (loc(1.5) - 2)(axis) == -1  # the underflow, which is a bin UHI numbers
    assert repr(loc(1.5)) == "loc(1.5)"
    assert repr(loc(1.5) + 1) == "loc(1.5) + 1"


def test_the_two_flow_locators_are_the_bins_below_and_above_the_axis():
    """``underflow`` and ``overflow`` are locators like any other - ``-1`` and the number of bins
    - and they print the name a reader wrote."""
    axis = _filled().axes[0]
    assert underflow(axis) == -1
    assert overflow(axis) == len(axis) == 3
    assert (repr(underflow), repr(overflow)) == ("underflow", "overflow")


def test_rebin_takes_a_whole_factor_of_one_or_more_and_nothing_else():
    """``rebin(n)`` merges n bins into one, so it wants a whole number of at least one - and
    ``True`` is not one, because a bool is not a count."""
    assert rebin(2).factor == 2
    assert repr(rebin(2)) == "rebin(2)"
    for refused in (0, 1.5, True):
        with pytest.raises(ValueError, match="does not merge bins"):
            rebin(refused)


def test_an_index_holds_at_most_one_ellipsis():
    """``...`` stands for every axis the index does not name, so two of them have nothing to
    stand for."""
    with pytest.raises(IndexError, match="one \\.\\.\\. and this one holds more"):
        _filled()[..., ...]


def test_an_index_with_more_items_than_the_histogram_has_axes_is_refused():
    """A histogram of one axis takes one item; two are refused by count, naming both."""
    with pytest.raises(IndexError, match="has 1 axes, and an index of 2 items"):
        _filled()[0, 1]


def test_a_dictionary_index_names_the_axes_and_refuses_one_the_histogram_has_not():
    """UHI's other spelling of an index names the axis each item is for, so a number that is no
    axis - or a word where a number belongs - is refused by name."""
    found = _filled()
    assert found[{0: 1}] == 1.0  # the second bin
    with pytest.raises(IndexError, match="2 is not an axis of 'h', which has 1"):
        found[{2: 1}]
    with pytest.raises(IndexError, match="'x' is not an axis of 'h'"):
        found[{"x": 0}]  # type: ignore[index]


def test_a_bin_a_locator_past_the_flow_and_a_type_that_picks_nothing_are_each_refused():
    """One bin is picked by a whole number inside the axis, or by a locator from ``-1`` to the
    number of bins; a number outside that, a locator outside it, and a type that is no locator at
    all each answer for themselves."""
    found = _filled()
    with pytest.raises(IndexError, match="bin 5 is not on an axis of 3 bins"):
        found[5]
    with pytest.raises(IndexError, match="locates bin 4, and an axis of 3 bins has them from -1"):
        found[lambda axis: axis.nbins + 1]
    with pytest.raises(TypeError, match="a str does not pick a bin"):
        found["x"]  # type: ignore[index]


def test_a_slice_s_end_may_be_len_or_a_locator_and_a_word_is_no_end():
    """``len`` is the whole axis, which is how a slice says "up to the end" without knowing how
    many bins there are; a locator is a coordinate's bin; anything else is refused."""
    found = _filled()
    assert found[1:len].sum() == 2.0  # bins 1 and 2, which hold 1 and 1
    assert found[0:loc(2.0)].sum() == 3.0  # 2.0 is in the third bin, so the first two
    with pytest.raises(TypeError, match="a str does not end a slice"):
        found[0:"x"]  # type: ignore[slice-item]


def test_a_step_that_is_not_a_rebinning_is_refused():
    """A slice's step is how bins are merged, so a bare number - Python's own stride - is no
    step here: ``rebin(n)`` says n bins become one and nothing else does."""
    with pytest.raises(TypeError, match="is not a step a histogram's slice takes"):
        _filled()[::2]


def test_summing_a_range_that_stops_before_it_starts_is_refused():
    """``h[3:1:sum]`` asks to sum a block of no bins, which is no sum at all."""
    with pytest.raises(ValueError, match="sums no bins of 'xaxis'"):
        _filled()[3:1:sum]


def test_a_rebinning_that_leaves_no_group_of_bins_is_refused_with_both_counts():
    """Merging two bins needs two of them: a slice of one bin says how many it had and how many
    the group wanted."""
    with pytest.raises(ValueError, match="1 bins from 0 make no group of 2"):
        _filled()[0:1:rebin(2)]


def test_an_irregular_axis_keeps_the_edges_a_cut_and_a_rebinning_leave():
    """An axis booked with its own edges keeps them through a cut, which an evenly binned one
    cannot: ``[0, 1, 3, 6, 10]`` cut to its middle two bins is ``[1, 3, 6]``, and merged two at a
    time it is ``[0, 3, 10]``."""
    uneven = Histogram.book("v", [0.0, 1.0, 3.0, 6.0, 10.0])
    uneven.fill([0.5, 2.0, 5.0, 8.0])
    assert uneven[1:3].edges().tolist() == [1.0, 3.0, 6.0]
    assert uneven[0:4:rebin(2)].edges().tolist() == [0.0, 3.0, 10.0]
    assert uneven[0:4:rebin(2)].values().tolist() == [2.0, 2.0]
    even = Histogram.book("e", (4, 0.0, 4.0))
    even.fill([0.5, 1.5, 2.5, 3.5])
    assert even[1:3].edges().tolist() == [1.0, 2.0, 3.0]
    assert even[0:4:rebin(2)].edges().tolist() == [0.0, 2.0, 4.0]


def test_a_cut_carries_every_per_bin_array_the_histogram_keeps():
    """A cut is a new histogram of every per-bin array, not only of the contents: a weighted
    histogram keeps its squares and a profile keeps the weights behind its means."""
    weighted = Histogram.book("w", (3, 0.0, 3.0))
    weighted.fill([0.5, 0.5, 1.5], weight=[2.0, 3.0, 1.0])
    cut = weighted[0:2]
    assert cut.values().tolist() == [5.0, 1.0]
    assert cut.variances().tolist() == [13.0, 1.0]  # 2**2 + 3**2, and 1**2
    profile = Profile.book("p", (2, 0.0, 2.0))
    profile.fill([0.5, 0.5, 1.5], [1.0, 3.0, 4.0])
    kept = profile[0:1]
    assert isinstance(kept, Profile)
    assert kept.values().tolist() == [2.0]  # the mean of 1 and 3
    assert kept.bin_entries().tolist() == [2.0]  # and the weight behind it


def test_setting_bins_takes_bins_and_slices_so_sum_and_rebin_are_both_refused():
    """``h[index] = value`` writes contents, which a summed axis and a rebinned one have no
    single value for."""
    found = _filled()
    with pytest.raises(ValueError, match="sum sets none"):
        found[0:2:sum] = 1.0
    with pytest.raises(ValueError, match="rebin sets none"):
        found[::rebin(2)] = 1.0


def test_a_profile_s_bins_cannot_be_set_because_a_mean_has_no_value_of_its_own():
    """A profile's bin is the mean of what was filled into it, so writing a number there would
    lose what it was the mean of; the refusal says to fill it instead."""
    profile = Profile.book("p", (2, 0.0, 2.0))
    profile.fill([0.5, 0.5, 1.5], [1.0, 3.0, 4.0])
    with pytest.raises(UnsupportedFeatureError, match="a mean cannot be set without saying what"):
        profile[0] = 1.0


def test_values_that_do_not_fit_the_bins_are_refused_with_both_shapes():
    """NumPy's own error says only that the shapes differ; the sentence says which shape was
    given and which block it was set into."""
    with pytest.raises(ValueError, match=r"values of shape \(3,\) do not fit the \(2,\) bins"):
        _filled()[0:2] = [1.0, 2.0, 3.0]


def test_setting_a_whole_axis_takes_its_flow_bins_when_the_values_have_two_more():
    """UHI's rule for the flow: a plain slice over a whole axis whose value has exactly two more
    entries than the axis has bins sets the flow bins too, rather than refusing."""
    fresh = Histogram.book("g", (3, 0.0, 3.0))
    fresh[:] = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert fresh.values(flow=True).tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    fresh[:] = [7.0, 8.0, 9.0]  # three values are the three bins, and the flow keeps its own
    assert fresh.values(flow=True).tolist() == [1.0, 7.0, 8.0, 9.0, 5.0]
    scalar = Histogram.book("s", (3, 0.0, 3.0))
    scalar[:] = 5.0  # one number for every bin, which is a value of no rows at all
    assert scalar.values(flow=True).tolist() == [0.0, 5.0, 5.0, 5.0, 0.0]


def test_setting_one_bin_counts_an_entry_and_forgets_the_running_sums():
    """``SetBinContent`` of one bin is an entry as far as ROOT is concerned - ``fEntries`` grows
    by one - and the sums of the moments it no longer describes are cleared."""
    found = _filled()
    before = found.entries
    found[1] = 5.0
    assert found.entries == before + 1
    assert found.values(flow=True).tolist() == [0.0, 2.0, 5.0, 1.0, 0.0]
    assert found.members["TH1"]["fTsumw"] == 0.0  # the running sums start again
