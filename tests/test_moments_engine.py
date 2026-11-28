"""``GetMean``, ``GetStdDev``, ``Integral``, ``FindBin``: ROOT's statistics from a histogram in
code. ROOT does not work a histogram's mean out of its bin centres: every fill adds to running
sums, and the mean, the spread and the effective entries are made from those. The moments are
ROOT's own: the skewness is the third central moment over ``sigma**3``, and the kurtosis the
fourth over ``sigma**4`` **less three**, so an evenly filled histogram reads -1.3 and a
symmetric one reads 0. Every number below was measured from this kit's own histogram before it
was asserted. No file is opened, which is what lets this test sit inside the ``hist``
component's own scope."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot.hist import Histogram
from xrdroot.moments import (
    central_moment,
    effective_entries,
    extreme,
    extreme_index,
    find_bin,
    integral,
    interpolate,
    interpolate_grid,
    mean,
    mean_error,
    std,
    std_error,
)


def _even() -> Histogram:
    """Two fills at each centre of a 5-bin axis of 0..5: the mean is 2.5 and the variance 2."""
    made = Histogram.book("h", (5, 0.0, 5.0))
    for _ in range(2):
        for value in (0.5, 1.5, 2.5, 3.5, 4.5):
            made.fill(value)
    return made


def test_the_mean_and_the_spread_come_from_the_running_sums():
    """``GetMean`` is the mean of what was **filled** and ``GetStdDev`` the spread about it: two
    fills at each centre from 0.5 to 4.5 are exactly 2.5 and sqrt(2)."""
    even = _even()
    assert (mean(even, 0), std(even, 0), even.entries) == (2.5, math.sqrt(2.0), 10.0)


def test_an_empty_histogram_has_no_moments_rather_than_a_division_by_zero():
    """ROOT answers 0 for the mean, the spread and their errors of nothing, and NaN for a higher
    moment, which is a division by a zero spread."""
    empty = Histogram.book("empty", (5, 0.0, 5.0))
    assert (mean(empty, 0), std(empty, 0), effective_entries(empty)) == (0.0, 0.0, 0.0)
    assert (mean_error(empty, 0), std_error(empty, 0)) == (0.0, 0.0)
    assert math.isnan(central_moment(empty, 0, 3))


def test_a_weighted_histogram_counts_its_weights_and_its_effective_entries():
    """``GetEffectiveEntries`` is ``(sum w)**2 / sum w**2``, and the errors divide the spread by it:
    three fills of weight two are three entries, whatever their weight adds up to."""
    weighted = Histogram.book("w", (5, 0.0, 5.0))
    for value in (0.5, 2.5, 4.5):
        weighted.fill(value, weight=2.0)
    assert (weighted.entries, effective_entries(weighted)) == (3.0, 3.0)
    assert mean(weighted, 0) == pytest.approx(2.5)
    assert std(weighted, 0) == pytest.approx(math.sqrt(8.0 / 3.0))  # 0.5, 2.5 and 4.5 about 2.5
    assert mean_error(weighted, 0) == pytest.approx(std(weighted, 0) / math.sqrt(3.0))
    assert std_error(weighted, 0) == pytest.approx(math.sqrt(8.0 / 3.0 / 6.0))  # sigma/sqrt(2n)


def test_the_skewness_and_the_kurtosis_are_the_third_and_fourth_central_moments():
    """``GetSkewness`` is the third central moment over ``sigma**3`` and ``GetKurtosis`` the fourth
    over ``sigma**4`` less three: evenly filled bins read 0 and -1.3, and three fills low with one
    high read positive."""
    assert central_moment(_even(), 0, 3) == pytest.approx(0.0, abs=1e-12)
    assert central_moment(_even(), 0, 4) == pytest.approx(-1.3, abs=1e-12)
    skewed = Histogram.book("skew", (5, 0.0, 5.0))
    for value in (0.5, 0.5, 0.5, 4.5):
        skewed.fill(value)
    assert central_moment(skewed, 0, 3) == pytest.approx(2.0 / math.sqrt(3.0), abs=1e-12)


def test_an_axis_that_is_not_there_is_refused_by_name():
    """A 1-D histogram has one axis, so ``axis=1`` is refused saying which histogram it is."""
    with pytest.raises(ValueError, match="is not an axis of 'h', which has 1"):
        mean(_even(), 1)
    with pytest.raises(ValueError, match="is not a binned axis of 'h', which has 1"):
        central_moment(_even(), 1, 3)


def test_the_integral_sums_the_bins_in_a_range_and_none_of_the_flow():
    """``Integral`` takes ROOT's bin numbers - flow at 0 and ``nbins + 1`` - and by default sums
    every bin and neither flow bin."""
    filled = _even()
    filled.fill(-1.0)  # the underflow
    filled.fill(9.0)  # and the overflow
    assert integral(filled, None, None, False)[0] == pytest.approx(10.0)
    assert integral(filled, 1, 3, False)[0] == pytest.approx(6.0)  # bins 1..3, two fills each
    assert integral(filled, 0, 6, False)[0] == pytest.approx(12.0)  # the flow bins too
    assert integral(filled, 1, 3, True)[0] == pytest.approx(6.0)  # a unit width changes nothing


def test_a_range_beyond_the_axis_is_clamped_the_way_root_clamps_it():
    """A last bin past the overflow is taken as it, and one **before** the first is taken as the
    whole rest of the axis - the clamping ``TH1::DoIntegral`` does, not a refusal."""
    filled = _even()
    filled.fill(-1.0)
    filled.fill(9.0)
    assert integral(filled, 1, 99, False)[0] == pytest.approx(11.0)  # the bins and the overflow
    assert integral(filled, 3, 1, False)[0] == pytest.approx(7.0)  # from bin 3 up, as it reads


def test_the_integral_s_error_is_the_root_of_the_variances_and_a_width_scales_both():
    """``IntegralAndError`` adds the bin variances and takes their root; ``width`` multiplies each
    bin by its width first, and its variance by that width squared."""
    wide = Histogram.book("wide", (2, 0.0, 10.0))
    wide.fill(2.5)
    wide.fill(7.5)
    wide.fill(7.5)
    total, error = integral(wide, None, None, False)
    assert (total, error) == pytest.approx((3.0, math.sqrt(3.0)))
    area, area_error = integral(wide, None, None, True)
    assert (area, area_error) == pytest.approx((15.0, 5.0 * math.sqrt(3.0)))  # five wide


def test_find_bin_is_root_s_global_number_with_x_running_fastest():
    """``FindBin`` counts the flow bins, x fastest: a 4-by-3 histogram is six cells in x and five in
    y, so its first in-range bin is 7 and one y bin up is six further along."""
    plane = Histogram.book("p", (4, 0.0, 4.0), (3, 0.0, 3.0))
    assert find_bin(plane, (0.5, 0.5)) == 7  # x bin 1 of y bin 1
    assert find_bin(plane, (0.5, 1.5)) == 13  # one y bin up is six x bins along
    assert find_bin(plane, (-1.0, 0.5)) == 6  # underflow in x is the end of the row below
    numbers = find_bin(plane, (np.array([0.5, 3.5]), np.array([0.5, 2.5])))
    assert numbers.tolist() == [7, 22]  # an array of coordinates gives an array of bins


def test_find_bin_wants_one_coordinate_per_axis():
    """A 2-D histogram cannot be found from one number, and the refusal says how many it needs."""
    plane = Histogram.book("p", (4, 0.0, 4.0), (3, 0.0, 3.0))
    with pytest.raises(ValueError, match="a bin is found from one coordinate per axis, not 1"):
        find_bin(plane, [0.5])


def test_interpolate_draws_a_line_between_the_nearest_centres_and_holds_flat_outside():
    """``TH1::Interpolate`` is linear between two centres and holds the outermost bin's content
    beyond them, so contents 2 and 1 read 2 below the first centre, 1.5 halfway and 1 above the
    last."""
    line = Histogram.book("line", (2, 0.0, 2.0))
    line.fill(0.5)
    line.fill(0.5)
    line.fill(1.5)  # contents two and one
    assert [interpolate(line, x) for x in (0.25, 0.5, 0.75, 1.0, 1.75)] == pytest.approx(
        [2.0, 2.0, 1.75, 1.5, 1.0]
    )
    assert math.isnan(interpolate(line, float("nan")))


def test_interpolate_needs_one_axis_and_a_grid_takes_one_point_per_axis():
    """``interpolate`` is ``TH1``'s and refuses a plane; ``interpolate_grid`` is ``TH2``'s and reads
    each axis's nearest centres, so a cell's content comes back at that cell's centre."""
    plane = Histogram.book("p", (2, 0.0, 2.0), (2, 0.0, 2.0))
    with pytest.raises(ValueError, match="needs one axis, and it has 2"):
        interpolate(plane, 0.5)
    plane.fill(0.5, 0.5)
    plane.fill(0.5, 0.5)
    plane.fill(1.5, 0.5)  # two in the first cell of the first row, one in the second
    assert interpolate_grid(plane, (0.5, 0.5)) == pytest.approx(2.0)
    assert interpolate_grid(plane, (1.5, 0.5)) == pytest.approx(1.0)


def test_the_extremes_are_the_bins_and_a_limit_set_on_the_frame_answers_instead():
    """``GetMaximum`` is the largest bin's content, and the index is the first bin ROOT's loop would
    meet, x running fastest; a limit set with ``SetMaximum`` answers instead of the bins."""
    filled = Histogram.book("h", (5, 0.0, 5.0))
    for value in (0.5, 1.5, 2.5, 3.5, 4.5):
        filled.fill(value)
    filled.fill(2.5)  # the third bin now holds two
    assert (extreme(filled, True), extreme(filled, False)) == (2.0, 1.0)
    assert (extreme_index(filled, True), extreme_index(filled, False)) == (2, 0)
    filled._core["fMaximum"] = 99.0  # what SetMaximum leaves on the frame
    assert extreme(filled, True) == 99.0
