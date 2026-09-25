"""A histogram read as a distribution: ``GetCumulative``, ``GetQuantiles``, ``Smooth``, ``Sumw2``.

The cumulative histograms are ROOT's own tests of them - ``test_TH1.cxx``'s
``GetCumulative1D``, ``2D``, ``3D`` and ``Errors``, with their hand-checked
anchors and their brute-force sums - and the quantiles the cases
``TH1::GetQuantiles``'s documentation describes: empty bins skipped at zero,
an empty stretch's middle when a probability lands on its edge.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from xrdroot import Histogram, Profile, UnsupportedFeatureError, stats


def brute(values, at, forward):
    """Every bin at or before ``at`` along every axis - or at or after it - added up."""
    region = tuple(slice(0, i + 1) if forward else slice(i, None) for i in at)
    return float(np.asarray(values)[region].sum())


# -- GetCumulative ----------------------------------------------------------


def test_a_cumulative_of_one_axis_is_the_running_sum_both_ways_as_roots_test_has_it():
    h = Histogram.new("h1", np.arange(6.0), [1.0, 2.0, 3.0, 4.0, 5.0])
    forward = h.cumulative()
    assert forward.values().tolist() == [1.0, 3.0, 6.0, 10.0, 15.0]
    assert h.cumulative(False).values().tolist() == [15.0, 14.0, 12.0, 9.0, 5.0]
    assert forward.name == "h1_cumulative" and h.cumulative(suffix="_c").name == "h1_c"
    assert forward.values(flow=True)[[0, -1]].tolist() == [0.0, 0.0]


def test_a_cumulative_is_set_bin_by_bin_so_its_entries_are_its_bins():
    h = Histogram.book("h", (4, 0.0, 4.0))
    h.fill([0.5, 1.5, 1.5, 3.5, 9.0])
    made = h.cumulative()
    assert made.entries == 4
    assert made.mean() == pytest.approx((0.5 * 1 + 1.5 * 3 + 2.5 * 3 + 3.5 * 4) / 11)


def test_a_cumulative_of_floats_adds_in_floats_as_a_th1f_stores_each_bin():
    h = Histogram.book("f", (50, 0.0, 1.0), kind="F")
    h.fill(np.linspace(0.005, 0.995, 50), weight=0.1)
    made = h.cumulative()
    expected = np.add.accumulate(np.full(50, np.float32(0.1)))
    assert made._bins[1:-1].tolist() == expected.tolist()
    assert made.variances() == pytest.approx(np.cumsum(np.full(50, 0.1**2)))
    whole = Histogram.book("i", (3, 0.0, 3.0), kind="I")
    whole.fill([0.5, 1.5, 1.5, 2.5])
    assert whole.cumulative(False).values().tolist() == [4, 3, 1]


def test_a_cumulative_of_two_axes_is_everything_below_and_to_the_left():
    values = np.arange(1.0, 10.0).reshape(3, 3).T  # h(ix, iy) = ix + 3 (iy - 1), from one
    h = Histogram.new("h2", (np.arange(4.0), np.arange(4.0)), values)
    forward = h.cumulative()
    assert (forward.values()[0, 0], forward.values()[1, 1], forward.values()[2, 2]) == (
        1.0,
        12.0,
        45.0,
    )
    for direction in (True, False):
        made = h.cumulative(direction).values()
        for at in itertools.product(range(3), range(3)):
            assert made[at] == brute(values, at, direction)
    assert forward.entries == 9


def test_a_cumulative_of_three_axes_agrees_with_the_brute_force_sum():
    values = np.arange(1.0, 13.0).reshape(2, 3, 2, order="F")
    h = Histogram.new("h3", (np.arange(3.0), np.arange(4.0), np.arange(3.0)), values)
    for direction in (True, False):
        made = h.cumulative(direction).values()
        for at in itertools.product(range(2), range(3), range(2)):
            assert made[at] == brute(values, at, direction)


def test_the_squares_of_the_weights_accumulate_over_the_same_bins_as_the_contents():
    values = np.arange(1.0, 10.0).reshape(3, 3).T
    h = Histogram.new("e", (np.arange(4.0), np.arange(4.0)), values, errors=0.5 * values)
    for direction in (True, False):
        made = h.cumulative(direction)
        for at in itertools.product(range(3), range(3)):
            assert made.variances()[at] == pytest.approx(brute((0.5 * values) ** 2, at, direction))


def test_a_profile_has_no_cumulative():
    with pytest.raises(UnsupportedFeatureError, match="a cumulative of 'p' is not offered"):
        Profile.book("p", (2, 0, 2)).cumulative()


# -- GetQuantiles -----------------------------------------------------------


def test_a_quantile_is_on_the_line_across_the_bin_the_probability_falls_in():
    h = Histogram.new("q", np.arange(6.0), [1.0, 2.0, 3.0, 4.0, 5.0])
    # The distribution is 0, 1/15, 3/15, 6/15, 10/15, 1: a tenth is a quarter
    # of the way across the second bin.
    assert h.quantiles([0.1, 0.5, 1.0]).tolist() == pytest.approx([1.25, 3.375, 5.0])
    assert h.quantiles(0.5).tolist() == pytest.approx([3.375])


def test_without_probabilities_the_quantiles_are_at_the_distributions_own_steps():
    h = Histogram.new("q", np.arange(6.0), [1.0, 2.0, 3.0, 4.0, 5.0])
    assert h.quantiles().tolist() == [0.0, 0.5, 1.5, 2.5, 3.5, 5.0]


def test_empty_bins_at_the_start_are_skipped_at_zero_as_roots_documentation_says():
    # "If the CDF is [0., 0., 0.1, ...], the quantiles would be [3., 3., 3., ...],
    # with the third bin starting at 3."
    h = Histogram.book("q", (5, 1.0, 6.0))
    h[...] = [0.0, 0.0, 1.0, 4.0, 5.0]
    assert h.quantiles().tolist() == [3.0, 3.0, 3.0, 3.5, 4.5, 6.0]


def test_a_probability_on_the_edge_of_an_empty_stretch_is_its_middle():
    h = Histogram.new("q", np.arange(5.0), [1.0, 0.0, 0.0, 1.0])
    assert h.quantiles([0.5]).tolist() == [2.0]
    ended = Histogram.new("q", [0.0, 1.0, 2.0, 4.0], [1.0, 1.0, 0.0])
    assert ended.quantiles([1.0]).tolist() == [2.0]  # the last bin with anything in it
    assert ended.quantiles([0.75]).tolist() == [1.5]


def test_what_has_no_quantiles_is_refused_by_name():
    h = Histogram.new("q", np.arange(3.0), [1.0, 1.0])
    with pytest.raises(ValueError, match="a probability from 0 to 1"):
        h.quantiles([1.5])
    with pytest.raises(ValueError, match="has nothing in its bins"):
        Histogram.book("e", (2, 0, 1)).quantiles()
    with pytest.raises(ValueError, match="of 2 axes is not offered"):
        Histogram.book("g", (2, 0, 1), (2, 0, 1)).quantiles()


# -- Smooth -----------------------------------------------------------------


def test_smoothing_a_histogram_replaces_its_contents_and_nothing_else():
    h = Histogram.book("s", (12, 0.0, 12.0))
    bumpy = [3, 7, 2, 9, 4, 4, 4, 1, 8, 6, 5, 0]
    h.fill(np.repeat(np.arange(12) + 0.5, bumpy))
    h.sumw2()  # without the squares kept, an error is the root of whatever is in the bin
    before = (h.entries, h.mean(), h.std(), h.errors().tolist())
    h.smooth()
    assert h.values().tolist() == stats.smooth_array(bumpy).tolist()
    assert (h.entries, h.mean(), h.std(), h.errors().tolist()) == before
    again = Histogram.new("s", np.arange(13.0), bumpy)
    again.smooth(3)
    assert again.values().tolist() == stats.smooth_array(bumpy, 3).tolist()


def test_a_th1f_is_smoothed_into_floats():
    h = Histogram.new("s", np.arange(13.0), [3, 7, 2, 9, 4, 4, 4, 1, 8, 6, 5, 0])
    floats = Histogram.book("f", (12, 0.0, 12.0), kind="F")
    floats[...] = h.values()
    floats.smooth()
    assert floats._bins.dtype == np.float32
    assert floats.values().tolist() == stats.smooth_array(h.values()).astype(np.float32).tolist()


def test_what_cannot_be_smoothed_is_refused_by_name():
    with pytest.raises(ValueError, match="at least 3 bins, and 's' has 2"):
        Histogram.book("s", (2, 0, 1)).smooth()
    with pytest.raises(ValueError, match="smoothing a histogram of 2 axes"):
        Histogram.book("g", (3, 0, 1), (3, 0, 1)).smooth()
    with pytest.raises(UnsupportedFeatureError, match="smoothing 'p' is not offered"):
        Profile.book("p", (3, 0, 3)).smooth()


# -- Sumw2 --------------------------------------------------------------------


def test_sumw2_starts_keeping_the_squares_from_the_counts_and_can_stop():
    h = Histogram.book("w", (2, 0.0, 2.0))
    h.fill([0.5, 1.5, 1.5])
    assert not h.weighted
    h.sumw2()
    assert h.weighted and h.variances().tolist() == [1.0, 2.0]
    h.fill(0.5, weight=3.0)
    assert h.variances().tolist() == [10.0, 2.0]
    h.sumw2(False)
    assert not h.weighted and h.variances().tolist() == [4.0, 2.0]


def test_a_profile_forgets_its_squared_weights_but_never_its_squared_values():
    p = Profile.book("p", (2, 0.0, 2.0))
    p.fill([0.5, 0.5], [1.0, 3.0], weight=2.0)
    assert p.weighted
    spread = p.spread().tolist()
    p.sumw2(False)
    assert not p.weighted and p.spread().tolist() == spread
    p.sumw2()
    assert p.weighted
