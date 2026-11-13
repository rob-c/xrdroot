"""``RooIntegralMorph``: the density between two densities as alpha moves from 0 to 1."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT


@pytest.fixture
def morph():
    x = ROOT.RooRealVar("x", "x", -10.0, 10.0)
    first_mean = ROOT.RooRealVar("m1", "m1", 0.0)
    first_sigma = ROOT.RooRealVar("s1", "s1", 1.0, 0.1, 5.0)
    second_mean = ROOT.RooRealVar("m2", "m2", 2.0)
    second_sigma = ROOT.RooRealVar("s2", "s2", 0.5, 0.1, 5.0)
    first = ROOT.RooGaussian("g1", "g1", x, first_mean, first_sigma)
    second = ROOT.RooGaussian("g2", "g2", x, second_mean, second_sigma)
    alpha = ROOT.RooRealVar("alpha", "alpha", 0.0, 1.0)
    alpha.setBins(50, "cache")
    morphed = ROOT.RooIntegralMorph("lmorph", "lmorph", first, second, x, alpha)
    return x, first, second, alpha, morphed


def test_alpha_at_zero_is_the_first_density_and_at_one_the_second(morph):
    x, first, second, alpha, morphed = morph
    x.setVal(0.5)
    alpha.setVal(0.0)
    assert morphed.compute({"x": 0.5}) == pytest.approx(first.compute({"x": 0.5}))
    alpha.setVal(1.0)
    assert morphed.compute({"x": 0.5}) == pytest.approx(second.compute({"x": 0.5}))


def test_between_the_ends_it_is_the_mixture_of_the_two(morph):
    x, first, second, alpha, morphed = morph
    x.setVal(0.5)
    alpha.setVal(0.25)
    expected = 0.75 * first.compute({"x": 0.5}) + 0.25 * second.compute({"x": 0.5})
    assert morphed.compute({"x": 0.5}) == pytest.approx(expected)
    assert "g1/g2" in repr(morphed)
