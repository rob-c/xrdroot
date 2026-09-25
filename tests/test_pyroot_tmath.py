"""``TMath``'s edges: what ROOT answers outside a function's domain, and the array helpers."""

from __future__ import annotations

import array
import math

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.core import distributions as dist

T = ROOT.TMath


def test_the_constants_are_functions_as_in_root():
    assert T.Pi() == math.pi
    assert T.TwoPi() == 2 * math.pi
    assert T.E() == math.e
    assert T.C() == 2.99792458e8
    assert T.RadToDeg() * T.DegToRad() == pytest.approx(1)
    assert T.Na() == 6.02214076e23
    assert T.Qe() == 1.602176634e-19


def test_numbers_one_at_a_time_have_roots_answers_off_their_domains():
    assert T.Abs(-2) == 2
    assert T.Sign(3.0, -1.0) == -3.0
    assert T.Sign(3, -1) == -3
    assert T.Sign(3, 2) == 3
    assert T.Min(3, 1) == 1
    assert T.Max(3, 1) == 3
    assert T.Min(3, [5, 2, 9]) == 2
    assert T.Max(2, [5, 2]) == 5
    assert T.MinElement([4, 2]) == 2
    assert T.MaxElement(1, [4, 9]) == 4
    assert T.LocMin(0, []) == -1
    assert T.LocMax(0, []) == -1
    assert T.LocMin(3, [3, 1, 1]) == 1
    assert math.isnan(T.Sqrt(-1))
    assert T.Sqrt(4) == 2
    assert T.Sq(3) == 9
    assert T.Power(2, 10) == 1024
    assert math.isnan(T.Power(-8, 1 / 3))
    assert T.Power(10, 400) == math.inf
    assert T.Exp(1000) == math.inf
    assert T.Exp(0) == 1
    assert T.Log(0) == -math.inf
    assert math.isnan(T.Log(-1))
    assert T.Log(math.e) == 1
    assert T.Log10(0) == -math.inf
    assert math.isnan(T.Log10(-1))
    assert T.Log10(100) == 2
    assert T.Log2(0) == -math.inf
    assert math.isnan(T.Log2(-1))
    assert T.Log2(8) == 3
    assert math.isnan(T.ASin(2))
    assert T.Sin(0) == 0
    assert T.CosH(0) == 1
    assert T.ATanH(0) == 0
    assert T.ATan2(1, 1) == pytest.approx(math.pi / 4)
    assert T.Hypot(3, 4) == 5
    assert (T.Floor(1.5), T.Ceil(1.5), T.FloorNint(1.5), T.CeilNint(1.5)) == (1.0, 2.0, 1, 2)
    assert (T.Nint(2.5), T.Nint(3.5), T.Nint(-2.5), T.Nint(2.4), T.Nint(2.6)) == (2, 4, -2, 2, 3)
    assert T.Even(4)
    assert T.Odd(3)
    assert T.IsNaN(T.QuietNaN())
    assert T.IsNaN(T.SignalingNaN())
    assert not T.Finite(T.Infinity())
    assert T.Finite(1.0)
    assert T.AreEqualAbs(1.0, 1.05, 0.1)
    assert not T.AreEqualAbs(1.0, 2.0, 0.1)
    assert T.AreEqualRel(1.0, 1.0 + 1e-9, 1e-6)
    assert T.AreEqualRel(math.inf, math.inf, 0)
    assert not T.AreEqualRel(1.0, 2.0, 1e-6)


def test_counting_and_special_functions_off_their_domains():
    assert T.Factorial(5) == 120
    assert T.Factorial(-1) == 0
    assert T.Binomial(5, 2) == 10
    assert T.Binomial(2, 5) == 0
    assert T.Binomial(-1, 0) == 0
    assert T.BinomialI(0.5, 4, 0) == 1
    assert T.BinomialI(0.5, 4, 5) == 0
    assert T.Gamma(-2.0) == 0
    assert T.Gamma(-0.5) == pytest.approx(math.gamma(-0.5))
    assert T.LnGamma(-1) == 0
    assert T.ErfInverse(1.0) == 0
    assert T.ErfInverse(0.0) == 0
    assert T.NormQuantile(0) == 0
    assert T.NormQuantile(1) == 0
    assert T.Poisson(-1, 2) == 0
    assert T.Poisson(0, 2) == pytest.approx(math.exp(-2))


def test_student_quantile_follows_every_branch_of_hills_algorithm(capsys):
    assert T.StudentQuantile(0.75, 1) == pytest.approx(1.0)
    assert T.StudentQuantile(0.75, 2) == pytest.approx(0.816496580927726)
    assert T.StudentQuantile(0.9999, 3) == pytest.approx(22.2037, rel=1e-4)
    assert T.StudentQuantile(0.5001, 40) == pytest.approx(0.000252, rel=1e-2)
    assert T.StudentQuantile(0.1, 10) == pytest.approx(-T.StudentQuantile(0.9, 10))
    assert T.StudentQuantile(0.1, 10, False) == pytest.approx(T.StudentQuantile(0.9, 10))
    assert T.StudentQuantile(0.9, 10, False) == pytest.approx(T.StudentQuantile(0.1, 10))
    assert T.StudentQuantile(0.2, 4.2) == pytest.approx(-0.94, rel=1e-2)
    assert T.StudentQuantile(1.5, 3) == 0
    assert "illegal parameter values" in capsys.readouterr().err


def test_array_statistics_are_roots():
    values = array.array("d", [3, 1, 4, 1, 5])
    assert T.Mean(5, values) == 2.8
    assert T.Mean(0, values) == 0.0
    assert T.StdDev(5, values) == T.RMS(5, values)
    assert T.RMS(1, values) == 0.0
    assert T.Median(0, values) == 0.0
    assert T.KOrdStat(5, values, 1) == 1
    order = array.array("i", [0] * 5)
    T.Sort(5, values, order)
    assert list(order) == [4, 2, 0, 1, 3]
    T.Sort(5, values, order, False)
    assert list(order) == [1, 3, 0, 2, 4]
    assert T.BinarySearch(5, [1, 2, 3, 4, 5], 0.5) == -1


def test_three_vectors_are_normalised_and_crossed_in_place():
    v = array.array("d", [3, 0, 4])
    assert T.Normalize(v) == 5
    assert list(v) == [0.6, 0.0, 0.8]
    zero = [0.0, 0.0, 0.0]
    assert T.Normalize(zero) == 0
    out = np.zeros(3)
    T.Cross([1, 0, 0], [0, 1, 0], out)
    assert list(out) == [0, 0, 1]


def test_quantiles_are_hyndman_and_fans():
    found = np.zeros(3)
    T.Quantiles(5, 3, [5, 1, 4, 2, 3], found, [0.25, 0.5, 0.75])
    assert list(found) == [2, 3, 4]
    T.Quantiles(5, 1, [5, 1, 4, 2, 3], found, [0.5], True, None, 1)
    assert found[0] == 3


def test_kolmogorov_test_of_two_samples(capsys):
    one, other = [1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 5.0]
    chance = T.KolmogorovTest(4, one, 4, other, "D")
    assert 0 < chance <= 1
    assert "Kolmogorov Probability" in capsys.readouterr().out
    assert T.KolmogorovTest(4, one, 4, other, "M") == pytest.approx(0.25)
    assert T.KolmogorovTest(2, one, 4, other) == -1
    assert "Sets must have more than 2 points" in capsys.readouterr().err


def test_permute_steps_through_every_ordering():
    values = [1, 2, 3]
    seen = [list(values)]
    while T.Permute(3, values):
        seen.append(list(values))
    assert len(seen) == 6
    assert seen[-1] == [3, 2, 1]


def test_the_distributions_edges():
    assert dist.normal_quantile(0) == -math.inf
    assert dist.normal_quantile(1) == math.inf
    assert dist.gamma_pdf(-1, 2, 1) == 0
    assert dist.gamma_pdf(0, 1, 2) == 0.5
    assert dist.gamma_pdf(0, 2, 1) == 0
    assert dist.chi2_quantile(0, 3) == 0
    assert dist.chi2_quantile(0.999999, 1) > 20
    assert dist.student_quantile(0, 3) == -math.inf
    assert dist.student_quantile(1, 3) == math.inf
    assert dist.student_quantile(0.5, 3) == 0
    assert dist.student_quantile(0.1, 3) < 0
    assert dist.f_pdf(-1, 3, 4) == 0
    assert dist.f_pdf(0, 3, 4) == 0
    assert dist.f_pdf(0, 2, 4) == 1
    assert dist.f_pdf(0, 1, 4) == math.inf
    assert dist.f_cdf(0, 3, 4) == 0
    assert dist.f_quantile(0, 3, 4) == 0
    assert dist.f_quantile(1, 3, 4) == math.inf
    assert dist.beta_pdf(-0.1, 2, 2) == 0
    assert dist.beta_pdf(0, 2, 2) == 0
    assert dist.beta_pdf(0, 0.5, 2) == math.inf
    assert dist.beta_pdf(0, 1, 3) == 3
    assert dist.beta_pdf(1, 3, 1) == 3
    assert dist.poisson_pdf(-1, 2) == 0
    assert dist.poisson_pdf(0, 2) == pytest.approx(math.exp(-2))
    assert dist.poisson_cdf(-1, 2) == 0
    assert ROOT.Math.poisson_cdf_c(-1, 2) == 1
    assert dist.binomial_pdf(-1, 0.3, 5) == 0
    assert dist.binomial_pdf(0, 0.0, 5) == 1
    assert dist.binomial_pdf(5, 1.0, 5) == 1
    assert dist.binomial_cdf(5, 0.3, 5) == 1
    assert dist.binomial_cdf(-1, 0.3, 5) == 0
    assert dist.lognormal_pdf(-1, 0, 1) == 0
    assert dist.lognormal_cdf(-1, 0, 1) == 0
    assert dist.exponential_cdf(-1, 2) == 0
    assert ROOT.Math.exponential_pdf(-1, 2) == 0
    assert ROOT.Math.exponential_cdf_c(-1, 2) == 1
    assert ROOT.Math.uniform_pdf(5, 0, 1) == 0
    assert ROOT.Math.uniform_cdf_c(0.25, 0, 1) == 0.75
    assert ROOT.Math.Pi() == math.pi
    assert ROOT.Math.beta(2, 3) == pytest.approx(1 / 12)
    assert ROOT.Math.erfc(0) == 1
    assert ROOT.Math.gaussian_quantile(0.5, 1) == 0
    assert ROOT.Math.gaussian_quantile_c(0.5, 1) == 0


def test_the_minimizer_options_are_set_and_read_back():
    options = ROOT.Math.MinimizerOptions
    options.SetDefaultMinimizer("Minuit", "Simplex")
    assert options.DefaultMinimizerType() == "Minuit"
    assert options.DefaultMinimizerAlgo() == "Simplex"
    options.SetDefaultMinimizer("Minuit2")
    options.SetDefaultTolerance(0.001)
    options.SetDefaultPrintLevel(1)
    assert options.DefaultTolerance() == 0.001
    assert options.DefaultPrintLevel() == 1
    options.SetDefaultMinimizer("Minuit2", "Migrad")
    options.SetDefaultTolerance(0.01)
    options.SetDefaultPrintLevel(0)
