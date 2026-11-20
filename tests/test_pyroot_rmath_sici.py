"""``ROOT.Math``'s sine and cosine integrals, against ROOT's own answers and the tables.

ROOT computes ``sinint`` and ``cosint`` with Cephes's two pieces, a power series for
arguments under 1.5 and a continued fraction past it; xrdroot splits them at 5, where
each is cheapest. Both are exact to a unit or two in the last place either way, so the
assertions are on the value ROOT 6.40.04 printed for the same call - and on the
deviations from the handbook tables that ``mathcore-testSpecFuncSiCi`` itself asks for,
which is what says the split is in a place that does not cost a digit.
"""

from __future__ import annotations

import cmath
import math

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.core.rmath import _ci_series, _e1_of_ix, _si_series

M = ROOT.Math

#: What ROOT 6.40.04 answers for ``ROOT::Math::sinint`` and ``ROOT::Math::cosint`` at each
#: argument, printed by the oracle beside this checkout: a point in each of the two pieces
#: and one on either side of the split, plus the limits.
ROOTS = {
    -10.0: (-1.6583475942188739, -0.045456433004455378),
    -3.4: (-1.8419139833261433, -0.0045180779307421514),
    -1.2: (-1.1080471990137186, 0.42045918289423967),
    -0.5: (-0.49310741804306668, -0.17778407880661337),
    -1e-4: (-9.9999999944444442e-05, -8.6331247095746484),
    0.5: (0.49310741804306668, -0.17778407880661337),
    1.0: (0.94608307036718309, 0.33740392290096755),
    4.9: (1.5695589381006521, -0.19477980602623737),
    5.0: (1.5499312449446738, -0.19002974965664376),
    5.1: (1.5312532047129219, -0.18347626315929869),
    10.0: (1.6583475942188739, -0.045456433004455378),
    100.0: (1.5622254668890563, -0.005148825142610493),
    10000.0: (1.5708915453859618, -3.0551916724485215e-05),
}


@pytest.mark.parametrize(("x", "roots"), sorted(ROOTS.items()))
def test_the_integrals_are_roots_answers(x, roots):
    sine, cosine = roots
    assert M.sinint(x) == pytest.approx(sine, rel=1e-14)
    assert M.cosint(x) == pytest.approx(cosine, rel=1e-14)


def test_the_sine_integral_is_odd_and_the_cosine_even():
    assert M.sinint(-7.0) == -M.sinint(7.0)
    assert M.cosint(-7.0) == M.cosint(7.0)


def test_the_sine_integral_oscillates_around_half_pi_closer_and_closer():
    """``Si(x) = pi/2 - f cos x - g sin x`` with both tails under ``1/x``, so it swings
    across ``pi/2`` while closing on it."""
    for x in (40.0, 1e6):
        assert abs(M.sinint(x) - math.pi / 2) <= 1.0 / x
    assert M.sinint(40.0) > math.pi / 2 > M.sinint(1e6)


def test_the_cosine_integral_starts_at_minus_infinity_and_falls_away():
    assert M.cosint(0.0) == -math.inf
    assert M.cosint(1e-300) == pytest.approx(0.5772156649015329 - 690.7755278982137, rel=1e-15)


def test_zero_is_where_the_sine_integral_starts():
    assert M.sinint(0.0) == 0.0
    assert M.sinint(1e-300) == 1e-300


def test_a_non_finite_argument_is_the_nan_root_answers():
    assert math.isnan(M.sinint(math.inf))
    assert math.isnan(M.sinint(-math.inf))
    assert math.isnan(M.sinint(math.nan))
    assert math.isnan(M.cosint(math.inf))
    assert math.isnan(M.cosint(math.nan))


def test_each_sum_is_bounded_by_its_term_count():
    """``sinint`` and ``cosint`` hand each sum an argument inside its own range, where it
    converges in thirty terms; one outside that range stops at the count rather than
    running until a double stops changing, so reaching a helper directly cannot hang it -
    nor is what comes back accurate there, which is why the range is the caller's."""
    for stopped in (_si_series(50.0), _ci_series(50.0)):
        assert math.isfinite(stopped)
    assert cmath.isfinite(_e1_of_ix(0.01))


def test_both_integrals_meet_the_handbook_tables_the_way_the_suite_asks():
    """The deviations ``mathcore-testSpecFuncSiCi`` measures against Mathematica's values."""
    #: (x, Si(x)) from that test's table, and its threshold in the last place of a double.
    sine_table = [
        (-9.8, -1.6675696168513867),
        (0.0, 0.0),
        (1.0, 0.946083070367183),
        (4.9, 1.569558938100652),
        (9.8, 1.6675696168513867),
    ]
    cosine_table = [
        (-9.6, -0.007707036058533522),
        (-3.4, -0.004518077930741954),
        (-0.5, -0.1777840788066129),
        (0.5, -0.1777840788066129),
        (3.4, -0.004518077930741954),
        (9.6, -0.007707036058533521),
    ]
    for x, expected in sine_table:
        assert abs(M.sinint(x) - expected) <= 1.25e-15, x
    for x, expected in cosine_table:
        assert abs(M.cosint(x) - expected) <= 1.25e-15, x
        assert abs(M.cosint(x) / expected - 1) <= 1e-13, x
