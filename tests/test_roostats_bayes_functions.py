"""The integrands of RooStats' Bayesian calculator: ``RooFunctor``, the likelihood, the posterior
and the cumulative posterior, over RooFit formulae chosen for what they make the integrals do.

A polynomial likelihood integrates exactly - by QAGS in one dimension, by
the Genz-Malik cubature in two - so its values are known; a prior that
turns negative pushes the cumulative posterior past one, a divergent one
makes it infinite, and one whose integral cancels leaves an error larger
than the integral: each of which RooStats reports as here.
"""

from __future__ import annotations

import math

import pytest

from xrdroot.roofit.functions import RooFormulaVar
from xrdroot.roofit.variables import RooRealVar
from xrdroot.roostats.bayesfuncs import CdfFunction, Functor, Likelihood, PosteriorFunction


def unit(name: str) -> RooRealVar:
    return RooRealVar(name, name, 0.5, 0.0, 1.0)


def test_a_functor_sets_its_parameters_and_warns_of_those_its_function_ignores(capfd) -> None:
    """``RooFunctor`` sets each parameter before it asks, counting the calls, and says which
    parameters its function does not depend on."""
    x, y = unit("x"), unit("y")
    square = RooFormulaVar("square", "@0*@0", [x])
    functor = Functor(square, [x, y])
    assert (functor([0.25, 0.75]), y.getVal(), functor.calls) == (0.0625, 0.75, 1)
    assert ("RooRealBinding: The function square does not depend on the parameter y. Note that "
            "passing copies of the parameters is not supported.") in capfd.readouterr().out


def test_the_likelihood_is_exp_of_minus_the_nll_less_the_offset_times_the_prior() -> None:
    """``LikelihoodFunction``: ``exp(-(nll - offset))``, with or without a prior, its largest
    value remembered."""
    x = unit("x")
    nll = Functor(RooFormulaVar("nll", "2*@0", [x]), [x])
    prior = Functor(RooFormulaVar("prior", "@0", [x]), [x])
    alone, times = Likelihood(nll, None, 1.0), Likelihood(nll, prior, 1.0)
    assert alone([0.5]) == 1.0
    assert times([0.25]) == pytest.approx(0.25 * math.exp(0.5), rel=1e-15)
    assert alone([0.0]) == pytest.approx(math.e, rel=1e-15)
    assert alone.max == pytest.approx(math.e, rel=1e-15)


def test_a_posterior_integrates_its_nuisance_parameters_in_one_or_two_dimensions() -> None:
    """With no nuisance parameter the posterior is the likelihood; with one it is QAGS', with
    two the cubature's integral - over a likelihood ``1 + x + y`` all exact."""
    poi, x, y = unit("poi"), unit("x"), unit("y")
    nll = RooFormulaVar("nll", "-log(1+@0+@1+@2)", [poi, x, y])
    bare = PosteriorFunction(nll, poi, [], None, 2.0, 0.0)
    assert bare(0.5) == pytest.approx(2.5 / 2.0, rel=1e-15)
    assert bare.error == 0.0
    one = PosteriorFunction(nll, poi, [x], None, 1.0, 0.0)
    assert one(0.0) == pytest.approx(1.0 + 0.5 + 0.5, rel=1e-14)  # y still 0.5
    two = PosteriorFunction(nll, poi, [x, y], None, 1.0, 0.0, 500)
    assert two(1.0) == pytest.approx(3.0, rel=1e-14)
    assert two.error < 1e-12


def test_a_posterior_integrated_too_coarsely_is_warned_of(capfd) -> None:
    """A narrow peak integrated with too few calls has an error above 20%, which RooStats
    reports with the point."""
    poi, x, y = unit("poi"), unit("x"), unit("y")
    nll = RooFormulaVar("nll", "400*((@0-0.1)^2+(@1-0.9)^2)", [x, y, poi])
    found = PosteriorFunction(nll, poi, [x, y], None, 1.0, 0.0, 1)
    found(0.5)
    assert found.error > 0.2 * found(0.5)
    assert ("PosteriorFunction::DoEval - Error from integration in 2 Dim is larger than 20 % "
            "x = 0.5 p(x) = ") in capfd.readouterr().out


def test_the_cdf_is_normalised_and_built_on_its_cached_values(capfd) -> None:
    """``PosteriorCdfFunction``: the integral up to ``x`` over the whole, less the offset; the
    ends are exact, and each value starts from the nearest one below it already found."""
    x = unit("x")
    nll = RooFormulaVar("nll", "-log(1+@0)", [x])
    cdf = CdfFunction(nll, [x], None, 0.0)
    assert cdf.norm == pytest.approx(1.5, rel=1e-15)
    assert "PosteriorCdfFunction - integral of posterior = 1.5 +/- " in capfd.readouterr().out
    assert cdf(0.5) == pytest.approx(0.625 / 1.5, rel=1e-14)
    assert cdf(0.75) == pytest.approx((0.75 + 0.75**2 / 2) / 1.5, rel=1e-14)
    assert cdf.lows[0] == 0.5
    cdf.SetOffset(0.25)
    assert (cdf(1.0), cdf(0.25)) == (0.75, -0.25)


def test_a_negative_prior_takes_the_cdf_past_one(capfd) -> None:
    """A prior negative above 0.8 makes the cdf at 0.9 larger than one, which is said."""
    x = unit("x")
    zero = RooFormulaVar("zero", "0*@0", [x])
    prior = RooFormulaVar("prior", "1-2*(@0>0.8)", [x])
    cdf = CdfFunction(zero, [x], prior, 0.0)
    assert cdf(0.9) == pytest.approx(0.7 / 0.6, rel=1e-14)
    assert ("PosteriorCdfFunction: normalized cdf values is larger than 1 x = 0.9 normcdf(x) = "
            "1.16667 +/- ") in capfd.readouterr().out


def test_a_cancelling_prior_leaves_an_error_larger_than_the_integral(capfd) -> None:
    """``cos(pi x)`` integrates to nothing over [0, 1]; the error is larger, and said."""
    x = unit("x")
    zero = RooFormulaVar("zero", "0*@0", [x])
    prior = RooFormulaVar("prior", "cos(3.141592653589793*@0)", [x])
    cdf = CdfFunction(zero, [x], prior, 0.0)
    assert abs(cdf.norm) < 1e-15 < cdf.norm_error
    assert ("PosteriorCdfFunction: integration error  is larger than 20 %   x0 = 0 x = 1 "
            "cdf(x) = ") in capfd.readouterr().out


def test_a_divergent_posterior_is_an_error_of_the_normalisation(capfd) -> None:
    """A likelihood ``exp(1000 x)`` overflows to infinity, as C's ``exp`` does: the huge value
    is warned of, the cdf is in error, and the integral and the normalisation said to fail."""
    x = unit("x")
    nll = RooFormulaVar("nll", "-1000*@0", [x])
    cdf = CdfFunction(nll, [x], None, 0.0)
    assert cdf.error
    assert cdf.like.max == math.inf
    out = capfd.readouterr().out
    expected = "LikelihoodFunction::()  WARNING - Huge likelihood value found for  parameters "
    assert expected + " x[0 ] = " in out
    assert "PosteriorFunction::Error computing integral - cdf = inf" in out
    assert "PosteriorFunction::Error computing normalization - norm = inf" in out
