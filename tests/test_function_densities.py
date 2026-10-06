"""The densities and Bessel functions a ``TFormula`` may call, element by element.

The Bessel functions are checked against their closed forms at half-integer
orders and against tabulated values (Abramowitz and Stegun, table 9.8) to
1e-12, the Gauss-Legendre sums' accuracy over these arguments; the densities
against their definitions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.function import bessel, densities
from xrdroot.function.library import CALLS

#: How close the integrals come to the closed forms over the arguments plots use.
CLOSE = 1e-12


def test_bessel_functions_of_half_a_whole_order_meet_their_closed_forms() -> None:
    sine = math.sqrt(2 / (math.pi * 2.0)) * math.sin(2.0)
    assert bessel.cyl_bessel_j(0.5, 2.0) == pytest.approx(sine, abs=CLOSE)
    i_half = math.sqrt(2 / (math.pi * 3.0)) * (math.cosh(3.0) - math.sinh(3.0) / 3.0)
    assert bessel.cyl_bessel_i(1.5, 3.0) == pytest.approx(i_half, rel=CLOSE)
    sines = math.sin(1.0) - math.cos(1.0)
    assert bessel.sph_bessel(1, 1.0) == pytest.approx(sines, abs=CLOSE)


def test_bessel_functions_of_whole_orders_meet_the_tables() -> None:
    assert bessel.cyl_bessel_j(0, 1.0) == pytest.approx(0.7651976865579666, abs=CLOSE)
    assert bessel.cyl_bessel_k(0, 1.0) == pytest.approx(0.42102443824070834, rel=CLOSE)
    assert bessel.cyl_bessel_k(2, 0.0) == math.inf
    assert bessel.cyl_bessel_j(0.5, 0.0) == pytest.approx(0.0, abs=CLOSE)
    assert list(bessel.sph_bessel([0, 1], [0.0, 0.0])) == [1.0, 0.0]


def test_the_laplace_gamma_and_lognormal_densities_are_their_definitions() -> None:
    tails = [0.5 / math.e, 1 - 0.5 / math.e]
    assert list(densities.laplace_dist_i([-1.0, 1.0])) == pytest.approx(tails)
    assert densities.laplace_dist(0.0, 0.0, 2.0) == 0.25
    assert densities.gamma_dist(2.0, 2, 0, 1) == pytest.approx(2 * math.exp(-2))
    assert densities.log_normal(1.0, 0.5) == pytest.approx(1 / (0.5 * math.sqrt(2 * math.pi)))
    peak = densities.breit_wigner_relativistic(91.0, 91.0, 2.5)
    assert peak > densities.breit_wigner_relativistic(95.0, 91.0, 2.5)


@pytest.mark.parametrize("alpha", [1.2, -1.2])
def test_the_crystal_balls_chances_below_and_above_add_to_one(alpha: float) -> None:
    below = densities.crystalball_cdf(0.3, alpha, 2.0, 1.0, 0.5)
    above = densities.crystalball_cdf_c(0.3, alpha, 2.0, 1.0, 0.5)
    assert below + above == pytest.approx(1.0)


def test_a_crystal_ball_with_no_finite_integral_has_no_chances() -> None:
    assert densities.crystalball_cdf(0, 1, 1, 1) == 0
    assert densities.crystalball_cdf_c(0, 1, 1, 1) == 0
    assert densities.crystalball_cdf(-50.0, 1.2, 3.0, 1.0) < 1e-3


def test_a_formula_names_them_as_root_does() -> None:
    xs = np.array([0.25, 0.5])
    assert CALLS["TMath::Student"].apply(xs).tolist() == densities.student(xs, 1.0).tolist()
    assert CALLS["TMath::StudentI"].apply(0.0) == pytest.approx(0.5)
    f = ROOT.TF1("lap", "TMath::LaplaceDist(x, [0], [1])", -5, 5)
    f.SetParameters(0.0, 1.0)
    assert f.Eval(0.0) == pytest.approx(0.5)


def test_tmath_and_root_math_name_them_as_root_does() -> None:
    assert ROOT.TMath.LaplaceDist(0.0) == 0.5 and ROOT.TMath.LaplaceDistI(0.0) == 0.5
    peak = float(densities.breit_wigner_relativistic(91.0, 91.0, 2.5))
    assert ROOT.TMath.BreitWignerRelativistic(91.0, 91.0, 2.5) == pytest.approx(peak)
    assert ROOT.Math.cyl_bessel_j(0, 1.0) == pytest.approx(0.7651976865579666, abs=CLOSE)
    assert ROOT.Math.cyl_bessel_i(0, 0.0) == pytest.approx(1.0)
    assert ROOT.Math.cyl_bessel_k(0, 1.0) == pytest.approx(0.42102443824070834, rel=CLOSE)
    assert ROOT.Math.sph_bessel(0, 0.0) == 1.0
    total = ROOT.Math.crystalball_cdf(0.3, 1.2, 2.0, 1.0) + ROOT.Math.crystalball_cdf_c(
        0.3, 1.2, 2.0, 1.0)  # fmt: skip
    assert total == pytest.approx(1.0)
