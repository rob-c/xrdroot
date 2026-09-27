"""RooFit's closed-form densities and integrals, ``RooFit/Detail/MathFuncs.h``, term for term.

Each expected number came from ROOT 6.40's ``RooFit::Detail::MathFuncs``
called from PyROOT with the same arguments. The cases were chosen to reach
every branch: both tails and the middle of a Gaussian, a zero exponential
slope, a Crystal Ball's core, tail, both, and ``n == 1``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot.roofit import mathfuncs as mf

COEFS = [0.5, -0.25, 0.125]


def test_erf_erfc_and_lgamma_take_numbers_and_arrays_alike() -> None:
    """Densities call these on single points and on whole columns, so both must work."""
    assert mf.erf(0.5) == math.erf(0.5)
    assert mf.erfc(0.5) == math.erfc(0.5)
    assert mf.lgamma(4.5) == math.lgamma(4.5)
    np.testing.assert_array_equal(mf.erf(np.array([0.5, 1.0])), [math.erf(0.5), math.erf(1.0)])
    np.testing.assert_array_equal(mf.erfc(np.array([0.5])), [math.erfc(0.5)])
    np.testing.assert_array_equal(mf.lgamma(np.array([4.5])), [math.lgamma(4.5)])


def test_approx_erf_is_exactly_one_beyond_five_widths() -> None:
    """ROOT clips ``erf`` beyond five, which a Crystal Ball's integral inherits."""
    assert float(mf.approx_erf(6.0)) == 1.0
    assert float(mf.approx_erf(-6.0)) == -1.0
    assert float(mf.approx_erf(0.3)) == pytest.approx(0.32862675945912734, rel=1e-15)


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (-1.0, 2.0, 2.566873175676446),
        (-5.0, -1.0, 0.5960726475843944),
        (1.0, 4.0, 1.3521730932777316),
    ],
)
def test_a_gaussian_integral_straddling_or_in_either_tail_matches_root(
    low: float, high: float, expected: float
) -> None:
    """Each case is mapped to the upper tail differently; all three must agree with ROOT."""
    assert float(mf.gaussian_integral(low, high, 0.5, 1.5)) == pytest.approx(expected, rel=1e-14)


def test_an_exponential_integral_with_a_zero_slope_is_the_width() -> None:
    """At ``c == 0`` the closed form divides by zero, and ROOT's answer is the range's width."""
    assert float(mf.exponential_integral(0.0, 2.0, -0.5)) == pytest.approx(
        1.2642411176571153, rel=1e-15
    )
    assert float(mf.exponential_integral(0.0, 2.0, 0.0)) == 2.0


def test_a_polynomial_by_horner_matches_root_with_and_without_a_lowest_order() -> None:
    """A density's polynomial adds one when its terms start above the constant."""
    assert mf.polynomial(COEFS, 0, 1.5, False) == 0.40625
    assert mf.polynomial(COEFS, 2, 1.5, False) == 0.9140625
    assert mf.polynomial(COEFS, 2, 1.5, True) == 1.9140625
    assert mf.polynomial([], 0, 1.5, False) == 0.0


def test_a_polynomial_integral_matches_root() -> None:
    """The integral a fit divides by at every step must be ROOT's to the last bit."""
    assert mf.polynomial_integral(COEFS, 0, -1.0, 2.0, False) == 1.5
    assert mf.polynomial_integral(COEFS, 1, -1.0, 2.0, True) == 3.46875


def test_a_polynomial_integral_without_coefficients_is_the_implied_constant_or_nothing() -> None:
    """A density with no coefficients above order zero is flat; a function without them is zero."""
    assert mf.polynomial_integral([], 1, -1.0, 2.0, True) == 3.0
    assert mf.polynomial_integral([], 0, -1.0, 2.0, True) == 0.0
    assert mf.polynomial_integral([], 1, -1.0, 2.0, False) == 0.0


def test_a_chebychev_series_and_its_integral_match_root() -> None:
    """Each number of coefficients - none, one, several - takes its own path."""
    assert mf.chebychev(COEFS, 0.3, -1.0, 2.0) == pytest.approx(1.2232592592592593, rel=1e-15)
    assert mf.chebychev([], 0.3, -1.0, 2.0) == 1.0
    assert mf.chebychev_integral(COEFS, -1.0, 2.0, -0.5, 1.5) == pytest.approx(
        2.3518518518518516, rel=1e-15
    )
    assert mf.chebychev_integral([0.5], -1.0, 2.0, -0.5, 1.5) == pytest.approx(2.0, rel=1e-15)
    assert mf.chebychev_integral([], -1.0, 2.0, -0.5, 1.5) == pytest.approx(2.0, rel=1e-15)


def test_a_crystal_ball_is_gaussian_in_its_core_and_a_power_law_in_its_tail() -> None:
    """A negative ``alpha`` puts the tail on the other side, mirrored."""
    m = np.array([0.5, -3.0])
    found = mf.cb_shape(m, 0.0, 1.0, np.float64(1.5), 2.0)
    np.testing.assert_allclose(found, [0.8824969025845955, 0.07189535609319857], rtol=1e-14)
    mirrored = mf.cb_shape(np.array([3.0]), 0.0, 1.0, np.float64(-1.5), 2.0)
    np.testing.assert_allclose(mirrored, [0.07189535609319857], rtol=1e-14)


@pytest.mark.parametrize(
    ("low", "high", "alpha", "n", "expected"),
    [
        (-1.0, 2.0, 1.5, 2.0, 2.0519124051726916),
        (-5.0, -2.0, 1.5, 2.0, 0.1954021120150569),
        (-5.0, 2.0, 1.5, 2.0, 2.595598885636807),
        (-5.0, 2.0, 1.5, 1.0, 2.6787760601733632),
        (-5.0, -2.0, 1.5, 1.0, 0.2755142983434287),
        (-2.0, 5.0, -1.5, 2.0, 2.595598885636807),
    ],
)
def test_a_crystal_ball_integral_over_core_tail_or_both_matches_root(
    low: float, high: float, alpha: float, n: float, expected: float
) -> None:
    """``n == 1`` is a logarithm rather than a power, and ROOT treats it apart."""
    found = mf.cb_shape_integral(low, high, 0.0, 1.0, alpha, n)
    assert found == pytest.approx(expected, rel=1e-9)


def test_a_crystal_ball_integral_that_underflows_is_a_tiny_positive_number() -> None:
    """ROOT returns ``1e-300`` rather than zero, so that a normalisation never divides by zero."""
    assert mf.cb_shape_integral(30.0, 40.0, 0.0, 1.0, 1.5, 2.0) == 1e-300


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (-3.0, -1.0, 0.39430605284939874),
        (1.0, 3.0, 1.2118561974122646),
        (-1.0, 3.0, 3.0273310272241813),
    ],
)
def test_a_bifurcated_gaussian_integral_on_either_side_or_across_matches_root(
    low: float, high: float, expected: float
) -> None:
    """Each side has its own width, so the integral is split where it crosses the mean."""
    assert mf.bifurgauss_integral(low, high, 0.0, 1.0, 2.0) == pytest.approx(expected, rel=1e-14)
