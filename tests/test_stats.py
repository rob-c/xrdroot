"""The probabilities ROOT's tests end in, and the smoother ``TH1::Smooth`` uses.

``TMath::KolmogorovProb`` is checked against the table every text prints -
the distances that leave a tenth, a twentieth, a hundredth and a thousandth -
and against go-hep's independent port of CERNLIB's ``PROBKL``, which it
matches to the bit; ``TMath::Prob`` against the closed forms one and two
degrees of freedom have, and against gonum's chi-square survival function.
``TH1::SmoothArray`` is checked against ROOT's own cross-check - a straight
line and a flat one come back as they went - and against go-hep's port of
the same ``hsmoof`` routine, which it matches to the last bit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot import stats


def test_kolmogorovs_probability_is_the_table_every_text_prints():
    for z, expected, tolerance in [
        (0.0, 1.0, 0.0),
        (0.1, 1.0, 0.0),
        (1.0, 0.27, 5e-3),
        (1.22, 0.10, 3e-3),
        (1.36, 0.05, 1e-3),
        (1.63, 0.01, 2e-4),
        (1.95, 0.001, 1e-5),
        (-1.36, 0.05, 1e-3),  # the sign of a distance does not matter
    ]:
        assert abs(stats.kolmogorov_prob(z) - expected) <= tolerance


def test_kolmogorovs_probability_is_go_heps_port_of_probkl_to_the_bit():
    # go-hep's hbook.KolmogorovProb, a separate translation of the same CERNLIB routine.
    assert stats.kolmogorov_prob(0.5) == 0.96394524373148627
    assert stats.kolmogorov_prob(0.7) == 0.71123519556318282
    assert stats.kolmogorov_prob(1.0) == 0.26999967167737987
    assert stats.kolmogorov_prob(1.5) == 0.022217962616525123
    assert stats.kolmogorov_prob(2.5) == 7.4533063441573419e-06
    assert stats.kolmogorov_prob(5.0) == 3.8574996959278356e-22
    assert stats.kolmogorov_prob(2.0) == pytest.approx(2 * math.exp(-8), abs=1e-12)
    assert stats.kolmogorov_prob(6.9) == 0.0  # past 6.8116 it is none at all


def test_kolmogorovs_probability_falls_as_the_distance_grows_and_stays_a_probability():
    found = [stats.kolmogorov_prob(z) for z in np.arange(0, 8, 0.05)]
    assert all(0 <= p <= 1 for p in found)
    assert all(later <= earlier + 1e-12 for earlier, later in zip(found, found[1:]))


def test_nint_rounds_a_half_to_the_even_number_as_tmath_does():
    assert [stats.nint(x) for x in (0.4, 1.5, 2.5, 3.5, 2.6)] == [0, 2, 2, 4, 3]
    assert [stats.nint(x) for x in (-0.4, -1.5, -2.5, -3.5, -2.6)] == [0, -2, -2, -4, -3]


def test_the_chi_square_probability_is_the_closed_form_for_one_and_two_degrees():
    for chi2 in (0.5, 1.0, 3.0, 10.0, 50.0):
        assert stats.prob(chi2, 1) == pytest.approx(math.erfc(math.sqrt(chi2 / 2)), rel=1e-14)
        assert stats.prob(chi2, 2) == pytest.approx(math.exp(-chi2 / 2), rel=1e-14)
        assert stats.prob(chi2, 4) == pytest.approx(math.exp(-chi2 / 2) * (1 + chi2 / 2), rel=1e-14)


def test_the_chi_square_probability_is_gonums_survival_function():
    for chi2, ndf, expected in [
        (21.085123538246044, 19, 0.33211649775033736),
        (3.5, 1, 0.061368829139402302),
        (12.0, 7, 0.10055886850835877),
        (200.0, 4, 3.7572767357810602e-42),
    ]:
        assert stats.prob(chi2, ndf) == pytest.approx(expected, rel=1e-13)
    assert stats.prob(0.3, 30) == 1.0  # all but certain, to a double


def test_the_chi_square_probability_follows_roots_rules_at_the_edges():
    assert stats.prob(0.0, 3) == 1.0
    assert stats.prob(-1.0, 3) == 0.0
    assert stats.prob(1.0, 0) == 0.0 and stats.prob(1.0, -2) == 0.0
    assert stats.prob(2.0, 2.9) == stats.prob(2.0, 2)  # an Int_t, cut to a whole number


def test_the_incomplete_gamma_functions_are_cephes_and_add_to_one():
    for a, x, lower in [
        (2.5, 1.0, 0.15085496391539038),
        (0.5, 3.0, 0.98569412156457037),
        (10.0, 3.0, 0.0011024881301154759),
    ]:
        assert stats.incomplete_gamma(a, x) == pytest.approx(lower, rel=1e-13)
        assert stats.incomplete_gamma(a, x) + stats.incomplete_gamma_c(a, x) == pytest.approx(1)
    assert (stats.incomplete_gamma(0.0, 1.0), stats.incomplete_gamma_c(0.0, 1.0)) == (1.0, 0.0)
    assert (stats.incomplete_gamma(1.0, 0.0), stats.incomplete_gamma_c(1.0, 0.0)) == (0.0, 1.0)
    # So far out that x**a * exp(-x) / Gamma(a) is below a double: Cephes says zero.
    assert stats.incomplete_gamma(500.0, 1e-10) == 0.0
    assert stats.incomplete_gamma_c(1.0, 800.0) == 0.0


def test_the_log_gamma_function_is_cephes_in_every_range_it_has():
    for x in (0.5, 1.5, 2.0, 2.5, 5.0, 12.7, 13.0, 20.0, 150.0, 1500.0, 2e8):
        assert stats.log_gamma(x) == pytest.approx(math.lgamma(x), rel=1e-14)
    assert stats.log_gamma(3.0) == math.log(2.0)  # exactly two, shifted down to two
    assert stats.log_gamma(3e305) == math.inf
    with pytest.raises(ValueError, match="positive number, not 0"):
        stats.log_gamma(0.0)


def test_smoothing_leaves_a_straight_line_and_a_flat_one_as_they_were():
    # ROOT's TH1.SmoothArrayCrossCheck, and the same smoothed five times over.
    for line in ([0.0, 1, 2, 3, 4], [1.0, 1, 1, 1, 1], [0.0, 2, 4, 6, 8, 10, 12, 14]):
        assert stats.smooth_array(line).tolist() == line
        assert stats.smooth_array(line, 5) == pytest.approx(line, abs=1e-6)


def test_smoothing_is_go_heps_port_of_hsmoof_to_the_bit():
    bumpy = [3, 7, 2, 9, 4, 4, 4, 1, 8, 6, 5, 0]
    assert stats.smooth_array(bumpy).tolist() == [
        3, 3.25, 3.75, 4, 4, 4, 4, 4.453125, 5.4670138888888893, 5.8975694444444446,
        4.9461805555555554, 2.9583333333333321,
    ]  # fmt: skip
    assert stats.smooth_array(bumpy, 3).tolist() == [
        3, 3.3345197512779707, 3.6767363371672457, 3.9163196705005787, 3.9815539450311856,
        3.9742743095743309, 4.1437902882265947, 4.6469039603025344, 5.2133493096413108,
        5.3495050961425115, 4.8390721869239748, 3.8436977559156347,
    ]  # fmt: skip
    peaked = [1, 4, 9, 9, 9, 4, 1, 0, 2, 2, 2, 5, 6]
    assert stats.smooth_array(peaked, 2).tolist() == [
        1, 4.6059027777777777, 7.6613859953703702, 8.8746383101851833, 7.5806568287037033,
        4.8096788194444438, 2.42578125, 1.21484375, 1.203125, 1.98828125, 3.05859375,
        4.453125, 6,
    ]  # fmt: skip
    assert stats.smooth_array([5, 1, 7]).tolist() == [5, 5, 5]


def test_smoothing_something_negative_may_come_back_negative():
    assert stats.smooth_array([-2, 5, -1, 3, 8, -4, 2, 2, 7]).tolist() == [
        -2, -0.57291666666666652, 1.3819444444444446, 2.7847222222222223, 2.8819444444444446,
        2.3020833333333335, 2, 2, 2,
    ]  # fmt: skip


def test_smoothing_steps_over_a_spike_rather_than_smearing_it():
    spiked = np.full(21, 10.0)
    spiked[10] = 1000.0
    assert stats.smooth_array(spiked).tolist() == [10.0] * 21


def test_a_quadratic_gives_a_plateau_the_medians_left_its_shape_back_on_either_side():
    # A plateau of three whose neighbours two away are both below it, one
    # nearer than the other: the patch leans to whichever side is nearer.
    left = stats.smooth_array([0, 3, 5, 5, 5, 4, 3], 1)
    right = stats.smooth_array([3, 4, 5, 5, 5, 3, 0], 1)
    assert left.tolist() == right.tolist()[::-1]
    assert left[3] != 5.0


def test_smoothing_needs_three_values_and_smooths_nothing_no_times():
    with pytest.raises(ValueError, match="at least 3 values, and 2 were given"):
        stats.smooth_array([1.0, 2.0])
    assert stats.smooth_array([1.0, 9.0, 1.0], 0).tolist() == [1.0, 9.0, 1.0]
