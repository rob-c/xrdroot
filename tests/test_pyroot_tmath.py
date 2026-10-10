"""``TMath``'s edges: what ROOT answers outside a function's domain, and the array helpers."""

from __future__ import annotations

import array
import math

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from xrdroot.function import distributions as dist

T = ROOT.TMath


def test_the_constants_are_functions_as_in_root():
    expect(
        (T.Pi(), math.pi),
        (T.TwoPi(), 2 * math.pi),
        (T.E(), math.e),
        (T.C(), 2.99792458e8),
        (T.RadToDeg() * T.DegToRad(), pytest.approx(1)),
        (T.Na(), 6.02214076e23),
        (T.Qe(), 1.602176634e-19),
    )


def test_numbers_one_at_a_time_have_roots_answers_off_their_domains():
    expect(
        (T.Abs(-2), 2),
        (T.Sign(3.0, -1.0), -3.0),
        (T.Sign(3, -1), -3),
        (T.Sign(3, 2), 3),
        (T.Min(3, 1), 1),
        (T.Max(3, 1), 3),
        (T.Min(3, [5, 2, 9]), 2),
        (T.Max(2, [5, 2]), 5),
        (T.MinElement([4, 2]), 2),
        (T.MaxElement(1, [4, 9]), 4),
        (T.LocMin(0, []), -1),
        (T.LocMax(0, []), -1),
        (T.LocMin(3, [3, 1, 1]), 1),
        (bool(math.isnan(T.Sqrt(-1))), True),
        (T.Sqrt(4), 2),
        (T.Sq(3), 9),
        (T.Power(2, 10), 1024),
        (bool(math.isnan(T.Power(-8, 1 / 3))), True),
        (T.Power(10, 400), math.inf),
        (T.Exp(1000), math.inf),
        (T.Exp(0), 1),
        (T.Log(0), -math.inf),
        (bool(math.isnan(T.Log(-1))), True),
        (T.Log(math.e), 1),
        (T.Log10(0), -math.inf),
        (bool(math.isnan(T.Log10(-1))), True),
        (T.Log10(100), 2),
        (T.Log2(0), -math.inf),
        (bool(math.isnan(T.Log2(-1))), True),
        (T.Log2(8), 3),
        (bool(math.isnan(T.ASin(2))), True),
        (T.Sin(0), 0),
        (T.CosH(0), 1),
        (T.ATanH(0), 0),
        (T.ATan2(1, 1), pytest.approx(math.pi / 4)),
        (T.Hypot(3, 4), 5),
        ((T.Floor(1.5), T.Ceil(1.5), T.FloorNint(1.5), T.CeilNint(1.5)), (1.0, 2.0, 1, 2)),
        ((T.Nint(2.5), T.Nint(3.5), T.Nint(-2.5), T.Nint(2.4), T.Nint(2.6)), (2, 4, -2, 2, 3)),
        (bool(T.Even(4)), True),
        (bool(T.Odd(3)), True),
        (bool(T.IsNaN(T.QuietNaN())), True),
        (bool(T.IsNaN(T.SignalingNaN())), True),
        (bool(not T.Finite(T.Infinity())), True),
        (bool(T.Finite(1.0)), True),
        (bool(T.AreEqualAbs(1.0, 1.05, 0.1)), True),
        (bool(not T.AreEqualAbs(1.0, 2.0, 0.1)), True),
        (bool(T.AreEqualRel(1.0, 1.0 + 1e-9, 1e-6)), True),
        (bool(T.AreEqualRel(math.inf, math.inf, 0)), True),
        (bool(not T.AreEqualRel(1.0, 2.0, 1e-6)), True),
    )


def test_counting_and_special_functions_off_their_domains():
    expect(
        (T.Factorial(5), 120),
        (T.Factorial(-1), 0),
        (T.Binomial(5, 2), 10),
        (T.Binomial(2, 5), 0),
        (T.Binomial(-1, 0), 0),
        (T.BinomialI(0.5, 4, 0), 1),
        (T.BinomialI(0.5, 4, 5), 0),
        (T.Gamma(-2.0), 0),
        (T.Gamma(-0.5), pytest.approx(math.gamma(-0.5))),
        (T.LnGamma(-1), 0),
        (T.ErfInverse(1.0), 0),
        (T.ErfInverse(0.0), 0),
        (T.NormQuantile(0), 0),
        (T.NormQuantile(1), 0),
        (T.Poisson(-1, 2), 0),
        (T.Poisson(0, 2), pytest.approx(math.exp(-2))),
    )


def test_student_quantile_follows_every_branch_of_hills_algorithm(capsys):
    expect(
        (T.StudentQuantile(0.75, 1), pytest.approx(1.0)),
        (T.StudentQuantile(0.75, 2), pytest.approx(0.816496580927726)),
        (T.StudentQuantile(0.9999, 3), pytest.approx(22.2037, rel=1e-4)),
        (T.StudentQuantile(0.5001, 40), pytest.approx(0.000252, rel=1e-2)),
        (T.StudentQuantile(0.1, 10), pytest.approx(-T.StudentQuantile(0.9, 10))),
        (T.StudentQuantile(0.1, 10, False), pytest.approx(T.StudentQuantile(0.9, 10))),
        (T.StudentQuantile(0.9, 10, False), pytest.approx(T.StudentQuantile(0.1, 10))),
        (T.StudentQuantile(0.2, 4.2), pytest.approx(-0.94, rel=1e-2)),
        (T.StudentQuantile(1.5, 3), 0),
        (bool("illegal parameter values" in capsys.readouterr().err), True),
    )


def test_array_statistics_are_roots():
    values = array.array("d", [3, 1, 4, 1, 5])
    expect(
        (T.Mean(5, values), 2.8),
        (T.Mean(0, values), 0.0),
        (T.StdDev(5, values), T.RMS(5, values)),
        (T.RMS(1, values), 0.0),
        (T.Median(0, values), 0.0),
        (T.KOrdStat(5, values, 1), 1),
    )
    order = array.array("i", [0] * 5)
    T.Sort(5, values, order)
    assert list(order) == [4, 2, 0, 1, 3]
    T.Sort(5, values, order, False)
    expect(
        (list(order), [1, 3, 0, 2, 4]),
        (T.BinarySearch(5, [1, 2, 3, 4, 5], 0.5), -1),
    )


def test_three_vectors_are_normalised_and_crossed_in_place():
    v = array.array("d", [3, 0, 4])
    expect(
        (T.Normalize(v), 5),
        (list(v), [0.6, 0.0, 0.8]),
    )
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
    expect(
        (bool(0 < chance <= 1), True),
        (bool("Kolmogorov Probability" in capsys.readouterr().out), True),
        (T.KolmogorovTest(4, one, 4, other, "M"), pytest.approx(0.25)),
        (T.KolmogorovTest(2, one, 4, other), -1),
        (bool("Sets must have more than 2 points" in capsys.readouterr().err), True),
    )


def test_permute_steps_through_every_ordering():
    values = [1, 2, 3]
    seen = [list(values)]
    while T.Permute(3, values):
        seen.append(list(values))
    expect(
        (len(seen), 6),
        (seen[-1], [3, 2, 1]),
    )


def test_the_distributions_edges():
    expect(
        (dist.normal_quantile(0), -math.inf),
        (dist.normal_quantile(1), math.inf),
        (dist.gamma_pdf(-1, 2, 1), 0),
        (dist.gamma_pdf(0, 1, 2), 0.5),
        (dist.gamma_pdf(0, 2, 1), 0),
        (dist.chi2_quantile(0, 3), 0),
        (bool(dist.chi2_quantile(0.999999, 1) > 20), True),
        (dist.student_quantile(0, 3), -math.inf),
        (dist.student_quantile(1, 3), math.inf),
        (dist.student_quantile(0.5, 3), 0),
        (bool(dist.student_quantile(0.1, 3) < 0), True),
        (dist.f_pdf(-1, 3, 4), 0),
        (dist.f_pdf(0, 3, 4), 0),
        (dist.f_pdf(0, 2, 4), 1),
        (dist.f_pdf(0, 1, 4), math.inf),
        (dist.f_cdf(0, 3, 4), 0),
        (dist.f_quantile(0, 3, 4), 0),
        (dist.f_quantile(1, 3, 4), math.inf),
        (dist.beta_pdf(-0.1, 2, 2), 0),
        (dist.beta_pdf(0, 2, 2), 0),
        (dist.beta_pdf(0, 0.5, 2), math.inf),
        (dist.beta_pdf(0, 1, 3), 3),
        (dist.beta_pdf(1, 3, 1), 3),
        (dist.poisson_pdf(-1, 2), 0),
        (dist.poisson_pdf(0, 2), pytest.approx(math.exp(-2))),
        (dist.poisson_cdf(-1, 2), 0),
        (ROOT.Math.poisson_cdf_c(-1, 2), 1),
        (dist.binomial_pdf(-1, 0.3, 5), 0),
        (dist.binomial_pdf(0, 0.0, 5), 1),
        (dist.binomial_pdf(5, 1.0, 5), 1),
        (dist.binomial_cdf(5, 0.3, 5), 1),
        (dist.binomial_cdf(-1, 0.3, 5), 0),
        (dist.lognormal_pdf(-1, 0, 1), 0),
        (dist.lognormal_cdf(-1, 0, 1), 0),
        (dist.exponential_cdf(-1, 2), 0),
        (ROOT.Math.exponential_pdf(-1, 2), 0),
        (ROOT.Math.exponential_cdf_c(-1, 2), 1),
        (ROOT.Math.uniform_pdf(5, 0, 1), 0),
        (ROOT.Math.uniform_cdf_c(0.25, 0, 1), 0.75),
        (ROOT.Math.Pi(), math.pi),
        (ROOT.Math.beta(2, 3), pytest.approx(1 / 12)),
        (ROOT.Math.erfc(0), 1),
        (ROOT.Math.gaussian_quantile(0.5, 1), 0),
        (ROOT.Math.gaussian_quantile_c(0.5, 1), 0),
    )


def test_the_minimizer_options_are_set_and_read_back():
    options = ROOT.Math.MinimizerOptions
    options.SetDefaultMinimizer("Minuit", "Simplex")
    expect(
        (options.DefaultMinimizerType(), "Minuit"),
        (options.DefaultMinimizerAlgo(), "Simplex"),
    )
    options.SetDefaultMinimizer("Minuit2")
    options.SetDefaultTolerance(0.001)
    options.SetDefaultPrintLevel(1)
    expect(
        (options.DefaultTolerance(), 0.001),
        (options.DefaultPrintLevel(), 1),
    )
    options.SetDefaultMinimizer("Minuit2", "Migrad")
    options.SetDefaultTolerance(0.01)
    options.SetDefaultPrintLevel(0)


WIDTHS = [(0.3, 1.7, False), (1.0, 0.01, True), (2.0, 0.0, False), (-1.0, -2.0, True)]


@pytest.mark.parametrize(("mean", "sigma", "norm"), WIDTHS)
def test_gaus_of_one_number_is_the_arrays_gaus_to_the_bit(mean, sigma, norm):
    from xrdroot.function import special

    xs = np.linspace(-50.0, 50.0, 2001)
    one = np.array([T.Gaus(x, mean, sigma, norm) for x in xs.tolist()])
    assert np.array_equal(one, special.gaus(xs, mean, sigma, norm))


def test_gaus_of_an_array_is_the_arrays_gaussian() -> None:
    xs = np.array([-1.0, 0.0, 2.0, 50.0])
    assert ROOT.TMath.Gaus(xs, 1.0, 2.0).tolist() == [ROOT.TMath.Gaus(x, 1.0, 2.0) for x in xs]
    assert ROOT.TMath.Gaus(xs, 0.0, 0.0).tolist() == [ROOT.TMath.Gaus(x, 0.0, 0.0) for x in xs]
