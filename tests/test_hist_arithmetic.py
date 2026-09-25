"""Adding, scaling, multiplying, dividing and merging histograms, as ROOT's TH1 does.

Each expected number is worked out by hand from the ROOT method it is named
for - ``TH1::Add``, ``Scale``, ``Multiply``, ``Divide`` with and without
option ``"B"``, and ``Merge`` - including where ROOT's own arithmetic makes a
square out of a square root and so is not quite the number on paper: those
are written the way ROOT computes them, so the comparison is exact.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot import Histogram, Profile, UnsupportedFeatureError


def counted():
    """Two, then one: filled without weights."""
    h = Histogram.book("one", (2, 0, 2))
    h.fill([0.5, 0.5, 1.5])
    return h


def weighted():
    """Two, then three, from weights of 2, 1, 1 and 1."""
    h = Histogram.book("two", (2, 0, 2))
    h.fill([0.5, 1.5, 1.5, 1.5], weight=[2.0, 1.0, 1.0, 1.0])
    return h


def moments(h):
    core = h.members["TH1"]
    return [core[name] for name in ("fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2")]


def test_adding_adds_the_errors_in_quadrature_and_the_running_sums_as_they_are():
    total = counted() + weighted()
    assert total.values().tolist() == [4, 4]
    # The counted one starts keeping squares; the other's are its errors squared.
    assert total.variances().tolist() == [2 + 4, 1 + math.sqrt(3) * math.sqrt(3)]
    assert moments(total) == [8, 10, 8, 10]
    assert total.entries == 7 and total.mean() == 1.0


def test_subtracting_makes_the_running_sums_again_from_the_bins():
    difference = counted() - weighted()
    assert difference.values().tolist() == [0, -2]
    assert moments(difference)[0] == -2 and moments(difference)[2] == -3
    assert difference.mean() == 1.5
    assert difference.entries == 2  # |sum of weights|, a negative total having no effective count


def test_a_normalisation_set_on_the_histogram_added_is_honoured():
    other = weighted()
    other.members["TH1"]["fNormFactor"] = 10.0  # ten, over its five: every bin counts twice
    made = counted()
    made.add(other)
    assert made.values().tolist() == [2 + 4, 1 + 6]


def test_scaling_scales_the_bins_errors_and_sums_but_not_the_entries():
    h = counted()
    h.scale(2.0)
    assert h.values().tolist() == [4, 2]
    assert h.variances().tolist() == [8, 4]
    assert moments(h) == [6, 12, 5, 5.5]
    assert h.entries == 3 and h.mean() == counted().mean()
    same = counted()
    same.scale(1.0)
    assert not same.weighted  # scaling by one leaves the errors the roots of the counts


def test_scaling_by_width_divides_each_bin_by_its_size_and_rebuilds_the_sums():
    h = Histogram.book("w", [0.0, 1.0, 4.0])
    h.fill([0.5, 2.0, 2.0])
    h.scale(1.0, width=True)
    assert h.values().tolist() == [1.0, 2 / 3]
    assert h.entries == 1 + 2 / 3  # made again from the bins: their total weight
    heavy = Histogram.book("w", [0.0, 1.0, 4.0])
    heavy.fill([0.5, 2.0, 2.0])
    heavy.scale(3.0, width=True)
    assert heavy.values().tolist() == [3.0, 6 / 3]
    error = math.sqrt(2) / 3
    assert heavy.variances().tolist() == [9 * 1.0, 9 * error * error]


def test_multiplying_adds_the_relative_errors_in_quadrature():
    product = counted() * weighted()
    assert product.values().tolist() == [4, 3]
    assert product.variances().tolist() == [2 * 2 * 2 + 4 * 2 * 2, 1 * 3 * 3 + 3 * 1 * 1]
    assert product.entries == pytest.approx(7 * 7 / 36, rel=1e-15)


def test_dividing_propagates_the_errors_of_a_ratio_and_gives_zero_over_zero():
    ratio = counted() / weighted()
    assert ratio.values().tolist() == [1, 1 / 3]
    assert ratio.variances().tolist() == [(2 * 4 + 4 * 4) / 16, (1 * 9 + 3 * 1) / 81]
    empty = Histogram.book("e", (2, 0, 2))
    empty.fill(0.5)
    over = counted() / empty
    assert over.values().tolist() == [2, 0] and over.errors()[1] == 0


def test_a_binomial_division_takes_the_error_of_a_fraction_that_passed():
    passed, total = Histogram.book("p", (3, 0, 3)), Histogram.book("t", (3, 0, 3))
    total.fill([0.5, 0.5, 1.5, 1.5, 1.5, 2.5])
    passed.fill([0.5, 1.5, 2.5])
    passed.divide(total, binomial=True)
    assert passed.values().tolist() == [0.5, 1 / 3, 1.0]
    assert passed.variances().tolist() == [
        abs(((1 - 2 * 1 / 2) * 1 + 1 * 2 / 4) / 4),  # p(1-p)/n, 1/8
        abs(((1 - 2 * 1 / 3) * 1 + 1 * 3 / 9) / 9),  # 2/27
        0.0,  # everything passed: ROOT gives no error at all
    ]
    assert passed.entries == total.entries


def test_the_operators_make_new_histograms_and_their_in_place_forms_change_this_one():
    h = counted()
    assert (h * 2).values().tolist() == (2 * h).values().tolist() == [4, 2]
    assert (h / 2).values().tolist() == [1, 0.5]
    assert h.values().tolist() == [2, 1]  # untouched by any of that
    same = h
    h += weighted()
    h -= weighted()
    h *= 3
    h /= weighted()
    assert h is same and h.values().tolist() == [3, 1]
    np.testing.assert_array_equal((counted() * weighted()).values(), [4, 3])


def test_what_is_not_a_histogram_or_a_number_is_refused_by_the_operators():
    h = counted()
    with pytest.raises(TypeError, match="adding a int and a histogram is not a thing ROOT does"):
        h + 1
    with pytest.raises(TypeError, match="subtracting a str"):
        h - "x"
    with pytest.raises(TypeError, match="multiplying a histogram by a bool"):
        h * True
    with pytest.raises(TypeError, match="dividing a histogram by a str"):
        h / "x"
    with pytest.raises(TypeError):
        1 + h


def test_sum_adds_histograms_up_from_nothing():
    total = sum([counted(), weighted()])
    assert total.values().tolist() == [4, 4] and total.entries == 7


def test_merging_adds_everything_up_as_hadd_does():
    merged = Histogram.merge([counted(), weighted()])
    assert merged.values().tolist() == [4, 4]
    assert merged.variances().tolist() == [6, 4]  # the squares themselves, added
    assert moments(merged) == [8, 10, 8, 10]
    assert merged.entries == 7
    assert Histogram.merge([counted()]).values().tolist() == [2, 1]


def test_merging_refuses_nothing_mixed_kinds_and_different_bins():
    with pytest.raises(ValueError, match="at least one histogram"):
        Histogram.merge([])
    with pytest.raises(TypeError, match="a histogram and a profile"):
        Histogram.merge([counted(), Profile.book("p", (2, 0, 2))])
    with pytest.raises(ValueError, match="'one' and 'wide' are binned differently"):
        Histogram.merge([counted(), Histogram.book("wide", (3, 0, 2))])


def test_profiles_merge_every_sum_they_keep():
    light, heavy = Profile.book("p", (1, 0, 1)), Profile.book("p", (1, 0, 1))
    light.fill(0.5, 1.0)
    heavy.fill(0.5, 3.0, weight=2.0)
    merged = Profile.merge([light, heavy])
    assert merged.bin_entries().tolist() == [3]
    assert merged.members["fBinSumw2"].tolist() == [0, 1 + 4, 0]
    assert merged.values().tolist() == [7 / 3]
    assert merged.entries == 2
    both = Profile.merge([heavy, light])
    assert both.members["fBinSumw2"].tolist() == [0, 4 + 1, 0]


def test_arithmetic_on_a_profile_is_refused_with_the_way_out():
    p = Profile.book("p", (2, 0, 2))
    with pytest.raises(UnsupportedFeatureError, match=r"Profile\.merge adds profiles up"):
        counted() + p
    with pytest.raises(UnsupportedFeatureError, match="whose bins are means"):
        p.scale(2.0)


def test_histograms_binned_differently_are_refused_by_name():
    with pytest.raises(ValueError, match=r"'one' and 'two' are binned differently - 2 bins"):
        counted().add(Histogram.book("two", (2, 0, 3)))


def test_a_reset_empties_everything_but_keeps_the_binning():
    h = weighted()
    h.reset()
    assert h.values(flow=True).tolist() == [0] * 4
    assert h.variances(flow=True).tolist() == [0] * 4 and h.weighted
    assert moments(h) == [0, 0, 0, 0] and h.entries == 0
    p = Profile.book("p", (1, 0, 1))
    p.fill(0.5, 2.0, weight=3.0)
    p.reset()
    assert p.bin_entries(flow=True).tolist() == [0, 0, 0] and p.members["fTsumwy"] == 0


def test_normalising_makes_the_bins_add_to_one_or_their_areas():
    assert counted().normalized().values().tolist() == [2 / 3, 1 / 3]
    h = Histogram.book("w", [0.0, 1.0, 4.0])
    h.fill([0.5, 2.0, 2.0])
    density = h.normalized(width=True)
    assert density.values().tolist() == [1 / 3 * 1 / 1, 1 / 3 * 2 / 3]
    with pytest.raises(ValueError, match="nothing on its axes to normalise"):
        Histogram.book("e", (1, 0, 1)).normalized()


def test_adding_to_integer_bins_saturates():
    small = Histogram.book("c", (1, 0, 1), kind="C")
    small.fill(np.full(100, 0.5))
    small.add(small.copy())
    assert small.values().tolist() == [127]
    small.add(small.copy(), -3.0)
    assert small.values().tolist() == [-127]


def test_a_copy_shares_nothing_and_takes_a_new_name():
    h = counted()
    twin = h.copy("twin")
    twin.fill(0.5)
    assert twin.name == "twin" and h.name == "one"
    assert h.values().tolist() == [2, 1] and twin.values().tolist() == [3, 1]


def test_histograms_without_squares_multiply_and_merge_without_gaining_any():
    product = counted() * counted()
    assert product.values().tolist() == [4, 1] and not product.weighted
    merged = Histogram.merge([counted(), counted()])
    assert merged.values().tolist() == [4, 2] and not merged.weighted
    kept = Histogram.merge([weighted(), counted()])
    assert kept.variances().tolist() == [4 + 2, 3 + 1]
    both = weighted() * counted()
    assert both.variances().tolist() == [4 * 2 * 2 + 2 * 2 * 2, 3 * 1 * 1 + 1 * 3 * 3]
    light = Profile.book("p", (1, 0, 1))
    light.fill(0.5, 1.0)
    assert not Profile.merge([light, light]).weighted


def test_scaling_integer_bins_keeps_whole_numbers():
    small = Histogram.book("c", (2, 0, 2), kind="C")
    small.fill([0.5, 0.5, 1.5])
    small.scale(2.6)
    assert small.values().tolist() == [5, 2]  # 5.2 and 2.6, cut towards zero
