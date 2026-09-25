"""Comparing two histograms: ROOT's ``Chi2Test``, ``Chi2TestX`` and ``KolmogorovTest``.

The chi-square test is checked against the example ROOT's own documentation
works through - ``tutorials/math/chi2test.C``, an unweighted histogram of 200
events against a weighted one of 500, whose chi-square ROOT quotes as 21.09
with a p-value of 0.33, and 32.33 with 0.029 once 17 events are added to one
bin - and against cases small enough to work by hand, go-hep's among them.
The corners are ROOT's code rather than its documentation: which bins are
skipped and what that does to the degrees of freedom, and the count nudged
up by one where the weighted comparison would divide zero by zero.

The Kolmogorov test is checked the same way: distances worked out by hand,
and the probability they make through ``TMath::KolmogorovProb``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot import Histogram, Profile, TRandom3, UnsupportedFeatureError, stats

#: ``chi2test.C``'s unweighted histogram, before the 17 events are added.
COUNTS = [0, 1, 0, 1, 1, 6, 7, 2, 22, 30, 27, 20, 13, 9, 9, 13, 19, 11, 9, 0]

#: Its weighted histogram, and the error on each bin.
WEIGHTS = [
    2.20173025, 3.30143857, 2.5892849, 2.99990201, 4.92877054, 8.33036995, 6.95084763,
    15.206357, 23.9236012, 44.3848114, 49.4465599, 25.1868858, 16.3129692, 13.0289612,
    16.7857609, 22.9914703, 30.5279255, 12.5252123, 16.4104557, 7.86067867,
]  # fmt: skip
ERRORS = [
    0.38974303, 0.536510944, 0.529702604, 0.642001867, 0.969341516, 1.47611344, 1.69797957,
    3.28577447, 5.40784931, 9.10106468, 9.73541737, 5.55019951, 3.57914758, 2.77877331,
    3.23697519, 4.3608489, 5.77172089, 3.38666105, 2.98861837, 1.58402085,
]  # fmt: skip


def tutorial(extra: float = 0.0):
    counts = list(COUNTS)
    counts[14] += extra
    edges = np.linspace(4, 16, 21)
    unweighted = Histogram.new("h1", edges, counts, entries=217)
    weighted = Histogram.new("h2", edges, WEIGHTS, errors=ERRORS, entries=500)
    return unweighted, weighted


def counts(*values, name="h"):
    return Histogram.new(name, np.arange(len(values) + 1.0), list(values))


def weighted(values, squares, name="w"):
    return Histogram.new(name, np.arange(len(values) + 1.0), values, variances=squares)


# -- Chi2Test ---------------------------------------------------------------


def test_a_count_against_a_weighted_histogram_is_the_chi_square_roots_documentation_quotes():
    found = tutorial()[0].chi2_test_full(tutorial()[1], "UW")
    assert round(found.chi2, 2) == 21.09 and round(found.p, 2) == 0.33
    assert found.ndf == 19 and found.igood == 1  # bins of h1 with no events in them
    assert found.chi2 == pytest.approx(21.085123538246044, rel=1e-12)
    added = tutorial(17)[0].chi2_test_full(tutorial(17)[1], "UW")
    assert round(added.chi2, 2) == 32.33 and round(added.p, 3) == 0.029
    assert len(added.residuals) == 20
    assert np.argmax(added.residuals) == 14  # the bin the 17 events went into stands out


def test_option_p_prints_roots_summary_line(capsys):
    one, other = tutorial()
    assert one.chi2_test(other, "UW P") == pytest.approx(0.3321164977503373, rel=1e-12)
    assert capsys.readouterr().out == "Chi2 = 21.085124, Prob = 0.332116, NDF = 19, igood = 1\n"


def test_options_chi2_and_chi2_over_ndf_give_the_chi_square_instead():
    one, other = tutorial()
    chi2 = one.chi2_test(other, "UW CHI2")
    assert chi2 == one.chi2_test_full(other, "UW").chi2
    assert one.chi2_test(other, "uw chi2/ndf") == chi2 / 19
    lone = counts(5.0)
    assert lone.chi2_test(counts(7.0), "CHI2/NDF") == 0.0  # one bin leaves no freedom


def test_two_counts_worked_by_hand_as_go_hep_works_them():
    one, other = counts(10.0, 20.0), counts(20.0, 10.0)
    found = one.chi2_test_full(other)
    # Each total is 30; each bin differs by 30 * 10 - 30 * 20 = -+300, and
    # 300**2 / 30 twice over, over 30 * 30, is 20 / 3.
    assert found.chi2 == pytest.approx(20 / 3, rel=1e-15) and found.ndf == 1
    assert found.p == stats.prob(found.chi2, 1) and found.igood == 0
    # Haberman's residual: (10 - 15) / sqrt(15), over sqrt((1 - 1/2)(1 - 1/2)).
    assert found.residuals.tolist() == pytest.approx([-10 / math.sqrt(15), 10 / math.sqrt(15)])
    same = one.chi2_test_full(one)
    assert (same.chi2, same.p) == (0.0, 1.0)


def test_a_bin_empty_in_both_counts_for_nothing_and_takes_a_degree_of_freedom():
    one, other = counts(5.0, 0.0, 5.0, 0.0), counts(5.0, 0.0, 5.0, 0.0)
    found = one.chi2_test_full(other)
    assert found.ndf == 1 and len(found.residuals) == 2
    thin = counts(0.0, 4.0, 2.0).chi2_test_full(counts(3.0, 0.0, 2.0))
    assert thin.igood == 3 and thin.ndf == 2  # an empty bin on each side


def test_a_weighted_histogram_asked_for_as_counts_is_warned_about_as_root_warns():
    scaled = weighted([10.0, 20.0], [5.0, 8.0])
    with pytest.warns(RuntimeWarning, match="both histograms are not unweighted"):
        scaled.chi2_test(counts(20.0, 10.0), "UU")
    with pytest.warns(RuntimeWarning, match="the first histogram is weighted"):
        scaled.chi2_test(weighted([20.0, 10.0], [10.0, 4.0]), "UW")


def test_asked_for_no_test_the_histograms_own_sums_say_which():
    plain, other = counts(10.0, 20.0), counts(20.0, 10.0)
    heavy, heavier = weighted([10.0, 20.0], [5.0, 8.0]), weighted([20.0, 10.0], [10.0, 4.0])
    assert plain.chi2_test(other, "") == plain.chi2_test(other, "UU")
    assert plain.chi2_test(heavier, "") == plain.chi2_test(heavier, "UW")
    assert heavy.chi2_test(heavier, "") == heavy.chi2_test(heavier, "WW")


def test_option_norm_takes_scaled_counts_back_to_the_entries_behind_them():
    one, other = counts(10.0, 20.0), counts(20.0, 10.0)
    big, small = one.copy(), other.copy()
    big.scale(2.0)
    small.scale(0.5)
    assert big.chi2_test(small, "UU NORM") == one.chi2_test(other, "UU")
    with pytest.raises(ValueError, match="errors of zero throughout"):
        Histogram.new("f", [0, 1, 2], [1.0, 2.0], errors=[0.0, 0.0]).chi2_test(one, "UU NORM")


def test_the_flow_bins_are_compared_only_when_asked_for():
    one, other = Histogram.book("a", (2, 0, 2)), Histogram.book("b", (2, 0, 2))
    one.fill([-1, 0.5, 1.5, 1.5, 5])
    other.fill([-1, -1, 0.5, 1.5, 5])
    assert one.chi2_test_full(other).ndf == 1
    assert one.chi2_test_full(other, "UU UF OF").ndf == 3
    assert one.chi2_test_full(other, "UU UF").ndf == 2


def test_two_weighted_histograms_worked_by_hand():
    one, other = weighted([10.0, 20.0], [5.0, 8.0]), weighted([20.0, 10.0], [10.0, 4.0])
    found = one.chi2_test_full(other, "WW")
    # (W1 w2 - W2 w1)**2 / (W1**2 s2**2 + W2**2 s1**2): 90000 / 13500 + 90000 / 10800.
    assert found.chi2 == pytest.approx(15.0, rel=1e-15) and found.ndf == 1
    # The residual is taken on the side of the larger error: the second in
    # the first bin, the first in the second.
    assert found.residuals.tolist() == pytest.approx(
        [-(20 - 30 * 4 / 9) / math.sqrt(10 * 2 / 3), (20 - 30 * 4 / 9) / math.sqrt(8 * 2 / 3)]
    )
    assert found.igood == 0
    thin = weighted([2.0, 20.0], [1.0, 8.0]).chi2_test_full(weighted([2.0, 20.0], [1.0, 4.0]), "WW")
    assert thin.igood == 3


def test_two_weighted_histograms_with_no_errors_between_them_are_refused():
    none = Histogram.new("f", [0, 1, 2], [1.0, 2.0], errors=[0.0, 0.0])
    with pytest.raises(ValueError, match="errors of zero throughout, and so no weights"):
        none.chi2_test(none.copy(), "WW")
    one = weighted([1.0, 2.0], [1.0, 0.0])
    with pytest.raises(ValueError, match=r"an error of zero in bin \(2\)"):
        one.chi2_test(weighted([1.0, 3.0], [1.0, 0.0]), "WW")
    grid = Histogram.new("g", ([0, 1, 2], [0, 1, 2]), [[1.0, 2.0], [3.0, 4.0]])
    grid.sumw2()
    grid._sumw2()[:] = 0.0
    grid._sumw2()[5] = 1.0
    with pytest.raises(ValueError, match=r"in bin \(1, 2\)"):
        grid.chi2_test(grid.copy(), "WW")


def _uw_bin(sum1, sum2, cnt1, cnt2, e2sq):
    """One bin of the count-against-weighted chi-square, from Gagunashvili's formula."""
    var1 = sum2 * cnt2 - sum1 * e2sq
    var2 = math.sqrt(var1 * var1 + 4 * sum2 * sum2 * cnt1 * e2sq)
    p = (var1 + var2) / (2 * sum2 * sum2)
    return (cnt1 - p * sum1) ** 2 / (p * sum1) + (cnt2 - p * sum2) ** 2 / e2sq


def test_a_count_of_nothing_is_nudged_up_by_one_as_root_nudges_it_and_so_is_its_total():
    # Nothing in the count's first bin, and one weighted entry against it
    # with totals equal: the estimate is zero over zero, and ROOT adds an
    # entry to the bin and the total - for every bin after it too.
    found = counts(0.0, 5.0, 5.0).chi2_test_full(weighted([1.0, 4.0, 5.0], [1.0, 4.0, 5.0]), "UW")
    expected = _uw_bin(11, 10, 1, 1, 1) + _uw_bin(11, 10, 5, 4, 4) + _uw_bin(11, 10, 5, 5, 5)
    assert found.chi2 == pytest.approx(expected, rel=1e-14)
    assert found.igood == 3


def test_a_count_of_nothing_whose_estimate_would_be_zero_is_nudged_by_roots_second_loop():
    found = counts(0.0, 6.0, 6.0).chi2_test_full(weighted([1.0, 5.0, 4.0], [1.0, 5.0, 4.0]), "UW")
    expected = _uw_bin(13, 10, 1, 1, 1) + _uw_bin(13, 10, 6, 5, 5) + _uw_bin(13, 10, 6, 4, 4)
    assert found.chi2 == pytest.approx(expected, rel=1e-14)


def test_a_count_against_an_exact_prediction_is_pearsons_chi_square():
    theory = Histogram.new("f", [0, 1, 2], [15.0, 15.0], errors=[0.0, 0.0])
    found = counts(10.0, 20.0).chi2_test_full(theory, "UW")
    assert found.chi2 == pytest.approx(25 / 15 * 2, rel=1e-15)
    assert found.residuals.tolist() == pytest.approx([-5 / math.sqrt(15), 5 / math.sqrt(15)])


def test_an_empty_weighted_bin_borrows_the_average_error_or_is_refused_without_one():
    data = counts(4.0, 5.0, 6.0)
    borrowed = data.chi2_test_full(weighted([0.0, 5.0, 6.0], [0.0, 5.0, 6.0]), "UW")
    assert borrowed.chi2 == pytest.approx(
        _uw_bin(15, 11, 4, 0, 1) + _uw_bin(15, 11, 5, 5, 5) + _uw_bin(15, 11, 6, 6, 6)
    )
    theory = Histogram.new("f", [0, 1, 2, 3], [0.0, 5.0, 6.0], errors=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"nothing and no error in bin \(1\)"):
        data.chi2_test(theory, "UW")
    counted = counts(0.0, 5.0, 6.0).chi2_test_full(weighted([0.0, 5.0, 6.0], [0.0, 5.0, 6.0]), "UW")
    assert counted.ndf == 1


def test_a_chi_square_of_two_axes_visits_the_bins_x_outermost_as_root_does():
    grid = Histogram.new("g", ([0, 1, 2], [0, 1, 2]), [[5.0, 5.0], [5.0, 5.0]])
    other = Histogram.new("o", ([0, 1, 2], [0, 1, 2]), [[5.0, 9.0], [5.0, 5.0]])
    found = grid.chi2_test_full(other)
    assert found.ndf == 3 and np.argmin(found.residuals) == 1  # bin (1, 2), second visited


def test_what_the_chi_square_test_cannot_compare_it_refuses_by_name():
    one = counts(1.0, 2.0)
    with pytest.raises(ValueError, match="has 1 axes and 'g' 2"):
        one.chi2_test(Histogram.new("g", ([0, 1, 2], [0, 1]), [[1.0], [2.0]]))
    with pytest.raises(ValueError, match="2 x bins and 'h' 3"):
        one.chi2_test(counts(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="'e' has nothing in the bins compared"):
        one.chi2_test(counts(0.0, 0.0, name="e"))
    with pytest.raises(UnsupportedFeatureError, match="is a profile"):
        Profile.book("p", (2, 0, 2)).chi2_test(one)


# -- KolmogorovTest ---------------------------------------------------------


def test_a_histogram_has_its_own_shape():
    h = Histogram.book("h", (10, 0, 10))
    h.fill(np.repeat(np.arange(10) + 0.5, np.arange(1, 11)))
    assert h.kolmogorov_test(h) == 1.0
    assert h.kolmogorov_test(h, "M") == 0.0


def test_the_largest_distance_is_worked_out_by_hand():
    rising, falling = counts(1.0, 2.0, 3.0, 4.0), counts(4.0, 3.0, 2.0, 1.0)
    # The cumulative distributions are .1 .3 .6 1 and .4 .7 .9 1: 0.4 apart at most.
    dmax = rising.kolmogorov_test(falling, "M")
    assert dmax == pytest.approx(0.4, abs=1e-15)
    # Ten entries each, so z is the distance times sqrt(10 * 10 / 20).
    expected = stats.kolmogorov_prob(dmax * math.sqrt(10 * 10 / 20))
    assert rising.kolmogorov_test(falling) == expected


def test_everything_on_one_side_against_everything_on_the_other_is_no_match():
    left, right = Histogram.book("l", (2, 0, 2)), Histogram.book("r", (2, 0, 2))
    left.fill(np.full(100, 0.5))
    right.fill(np.full(100, 1.5))
    assert left.kolmogorov_test(right, "M") == 1.0
    assert left.kolmogorov_test(right) == 0.0  # z of 7.07 is past where PROBKL stops


def test_what_a_histogram_is_worth_is_its_effective_entries_not_what_it_holds():
    left, right = Histogram.book("l", (2, 0, 2)), Histogram.book("r", (2, 0, 2))
    left.fill(0.5, weight=100.0)
    right.fill(1.5, weight=100.0)
    # One entry each: z is sqrt(1 / 2), and two entries tell nothing apart.
    assert left.kolmogorov_test(right) == stats.kolmogorov_prob(math.sqrt(0.5))
    assert left.kolmogorov_test(right) > 0.5


def test_the_flow_bins_are_compared_only_when_asked_for_by_u_and_o():
    one, other = Histogram.book("a", (2, 0, 2)), Histogram.book("b", (2, 0, 2))
    one.fill([-1, -1, 0.5, 1.5])
    other.fill([0.5, 1.5, 5, 5])
    assert one.kolmogorov_test(other, "M") == 0.0
    assert one.kolmogorov_test(other, "UO M") == pytest.approx(0.5)
    assert one.kolmogorov_test(other, "U M") == pytest.approx(0.5)
    assert one.kolmogorov_test(other, "O M") == pytest.approx(0.5)


def test_option_n_combines_the_shape_with_how_far_the_totals_are_apart():
    one, other = counts(10.0, 20.0, 30.0), counts(12.0, 20.0, 40.0)
    shape = one.kolmogorov_test(other)
    totals = stats.prob((60.0 - 72.0) ** 2 / (60.0 + 72.0), 1)
    assert one.kolmogorov_test(other, "N") == pytest.approx(
        shape * totals * (1 - math.log(shape * totals)), rel=1e-15
    )
    lopsided = counts(10000.0, 10000.0, 10000.0)
    assert lopsided.kolmogorov_test(counts(1.0, 0.0, 0.0), "N") == 0.0


def test_a_histogram_with_no_errors_is_compared_as_a_function_would_be():
    exact = Histogram.new("f", [0, 1, 2, 3], [1.0, 1.0, 1.0], errors=[0.0, 0.0, 0.0])
    data = counts(10.0, 14.0, 6.0)
    dmax = data.kolmogorov_test(exact, "M")
    assert data.kolmogorov_test(exact) == stats.kolmogorov_prob(dmax * math.sqrt(30.0))
    assert exact.kolmogorov_test(data) == data.kolmogorov_test(exact)
    assert exact.kolmogorov_test(data, "N") == exact.kolmogorov_test(data)  # N needs two counts
    with pytest.raises(ValueError, match="both have errors of zero"):
        exact.kolmogorov_test(exact.copy())


def test_option_d_prints_roots_debug_lines(capsys):
    one, other = counts(1.0, 2.0, 3.0, 4.0, name="up"), counts(4.0, 3.0, 2.0, 1.0, name="down")
    one.kolmogorov_test(other, "D")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == " Kolmo Prob  h1 = up, sum bin content =10  effective entries =10"
    assert lines[1] == " Kolmo Prob  h2 = down, sum bin content =10  effective entries =10"
    assert lines[2].startswith(" Kolmo Prob     = 0.") and lines[2].endswith("Max Dist = 0.4")
    one.kolmogorov_test(other, "D N X=10")
    lines = capsys.readouterr().out.splitlines()
    assert lines[3].endswith("for normalisation alone") and "shape alone" in lines[3]
    assert lines[4].endswith("with 10 pseudo-experiments")


def test_option_x_counts_the_pseudo_experiments_that_stray_further_repeatably():
    rng = np.random.default_rng(1)
    one, other = Histogram.book("a", (20, -3, 3)), Histogram.book("b", (20, -3, 3))
    one.fill(rng.normal(size=60))
    other.fill(rng.normal(0.3, size=80))
    found = one.kolmogorov_test(other, "X=200", rng=TRandom3(4357))
    assert 0 < found < 1 and found * 200 == round(found * 200)
    assert one.kolmogorov_test(other, "X=200", rng=TRandom3(4357)) == found  # the same draws
    assert 0 < one.kolmogorov_test(other, "X=200", rng=TRandom3(9)) < 1
    assert 0 < one.kolmogorov_test(other, "X=200") < 1  # gRandom's, moving it on
    with pytest.warns(RuntimeWarning, match="0 is not a number of pseudo-experiments"):
        assert 0 <= one.kolmogorov_test(other, "X=0", rng=TRandom3(5)) <= 1


def test_option_x_draws_many_entries_per_bin_by_poisson_and_corrects_the_total():
    # More than ten entries a bin: Poisson numbers, then entries added or taken
    # away one at a time - from a bin that has some - until the count is right.
    small, large = Histogram.book("s", (2, 0, 2)), Histogram.book("l", (2, 0, 2))
    small.fill(np.repeat([0.5, 1.5], [1, 100]))
    large.fill(np.repeat([0.5, 1.5], [3, 120]))
    found = small.kolmogorov_test(large, "X=2000", rng=TRandom3(3))
    assert 0 < found < 1


def test_option_x_against_an_exact_histogram_draws_only_the_other_side():
    exact = Histogram.new("f", [0, 1, 2, 3], [1.0, 1.0, 1.0], errors=[0.0, 0.0, 0.0])
    data = counts(10.0, 14.0, 6.0)
    assert 0 <= exact.kolmogorov_test(data, "X=50") <= 1
    assert 0 <= data.kolmogorov_test(exact, "X=50") <= 1
    thousand = data.kolmogorov_test(exact, "X")  # a thousand of them unless told otherwise
    assert thousand * 1000 == round(thousand * 1000)


def test_option_x_with_nothing_to_draw_finds_no_distance_at_all():
    # The parent - the side of more effective entries - is negative throughout,
    # and emptied of its negative bins it has nothing to draw from.
    negative = Histogram.new("n", [0, 1, 2], [-2.0, -2.0])
    assert negative.kolmogorov_test(counts(1.0, 2.0), "X=20") == 0.0
    # A side worth less than one entry draws none.
    faint = Histogram.new("f", [0, 1, 2], [0.5, 0.5], errors=[1.0, 1.0])
    assert faint.kolmogorov_test(counts(1.0, 2.0), "X=20") == 0.0


def test_what_the_kolmogorov_test_cannot_compare_it_refuses_by_name():
    one = counts(1.0, 2.0)
    grid = Histogram.new("g", ([0, 1, 2], [0, 1]), [[1.0], [2.0]])
    with pytest.raises(ValueError, match="compares histograms of one axis"):
        one.kolmogorov_test(grid)
    with pytest.raises(ValueError, match="'h' has 2 bins and 'h' 3"):
        one.kolmogorov_test(counts(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="different bin edges"):
        one.kolmogorov_test(Histogram.new("s", [0, 1, 3], [1.0, 2.0]))
    near = Histogram.new("s", [0, 1, 2 + 1e-16], [1.0, 2.0])
    assert one.kolmogorov_test(near) == 1.0  # the same edge, to AreEqualRel's 1e-15
    with pytest.raises(ValueError, match="'e' has nothing in the bins compared"):
        one.kolmogorov_test(counts(0.0, 0.0, name="e"))
    with pytest.raises(UnsupportedFeatureError, match="is a profile"):
        one.kolmogorov_test(Profile.book("p", (2, 0, 2)))
