"""``ROOT::Math::Cephes::ndtri`` and the two tails it serves, to MathCore's own last bit.

MathCore does not call the standard library's inverse normal: it carries Cephes's rational
approximations, and ``normal_quantile_c``/``normal_cdf_c`` are built on them. **Every number
below was read off the ROOT 6.40.04 oracle** - ``ROOT.Math.normal_quantile`` and
``normal_cdf_c`` - printed with ``%.17g``, because the tails are where a reimplementation
rounds differently.

The argument is ``normal_cdf_c(x, sigma, x0)``'s ``z = (x - x0) / (sigma * sqrt(2))``, so
``x = 0.5`` is ``z = 0.354`` and the routine's ``z > 1`` switch is at ``x = sqrt(2)``.
"""

from __future__ import annotations

import math

import pytest

from xrdroot.function.ndtri import ndtri, normal_cdf_c, normal_quantile_c


def test_the_median_and_the_two_sigma_points_are_the_inverse_normal_s_everyone_knows():
    """``ndtri(0.5)`` is zero and ``ndtri(0.975)`` is 1.96..., from the central expansion Cephes
    uses either side of the median."""
    assert ndtri(0.5) == pytest.approx(0.0, abs=1e-12)
    assert ndtri(0.975) == pytest.approx(1.959963984540054, abs=1e-15)
    assert ndtri(0.025) == pytest.approx(-1.959963984540054, abs=1e-15)
    assert ndtri(0.1) == pytest.approx(-1.2815515655446004, abs=1e-15)  # the upper flag's side


def test_the_tails_are_reached_by_their_own_expansion():
    """Past one less ``exp(-2)`` the routine switches to the tail's expansion, and past eight
    sigma to the second set of coefficients; both answer ROOT's own value."""
    assert ndtri(1e-10) == pytest.approx(-6.3613409024040557, abs=1e-14)  # the tail's expansion
    assert ndtri(1e-30) == pytest.approx(-11.464024688443613, abs=1e-12)  # x >= 8, so P2/Q2


def test_a_chance_of_none_or_all_is_an_infinite_quantile():
    """ROOT answers the ends of the real line rather than raising: no chance is below -inf, a
    certainty below +inf - and ``1 - 1e-30`` *is* a certainty in double, which ROOT reads as inf."""
    assert ndtri(0.0) == -math.inf
    assert ndtri(-1.0) == -math.inf
    assert ndtri(1.0) == math.inf
    assert ndtri(2.0) == math.inf
    assert ndtri(1.0 - 1e-30) == math.inf


def test_the_upper_quantile_is_the_lower_one_negated_and_scaled():
    """``normal_quantile_c`` is ``-sigma * ndtri(z)``: the chance above the answer, not below."""
    assert normal_quantile_c(0.025) == pytest.approx(1.9599639845400545, abs=1e-15)
    assert normal_quantile_c(0.025, 2.0) == pytest.approx(3.919927969080109, abs=1e-15)


def test_the_upper_tail_is_erfc_past_one_sigma_and_one_minus_erf_inside():
    """``normal_cdf_c`` changes form at ``z = 1``: ``erfc`` where it is larger and ``1 - erf``
    where it is not, the two agreeing at the join."""
    assert normal_cdf_c(0.0) == pytest.approx(0.5, abs=1e-15)
    assert normal_cdf_c(0.5) == pytest.approx(0.30853753872598694, abs=1e-15)  # 1 - erf, z < 1
    assert normal_cdf_c(1.0) == pytest.approx(0.15865525393145707, abs=1e-15)  # likewise
    assert normal_cdf_c(2.0) == pytest.approx(0.022750131948179219, abs=1e-15)  # erfc, z > 1
    assert normal_cdf_c(2.0 * math.sqrt(2.0)) == pytest.approx(0.0023388674905236331, abs=1e-15)
    # a mean and a sigma move and widen it: x0 = 10, sigma = 2, x = 12 is one sigma out
    assert normal_cdf_c(10.0, 2.0, 10.0) == pytest.approx(0.5, abs=1e-15)
    assert normal_cdf_c(12.0, 2.0, 10.0) == pytest.approx(0.15865525393145707, abs=1e-15)
