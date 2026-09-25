"""Unbinned maximum-likelihood fits to arrays, in one variable and in two.

The model is normalised over the range at every step, so a density's
parameters come out without its scale; the answers are checked against
what the likelihood's own maximum is in closed form - the mean and the
standard deviation of the points, for a Gaussian - and against the number of
points, for an extended fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot import Function, TRandom3
from xrdroot.fit import unbinned


def normal_points(n=4000, seed=1):
    return TRandom3(seed).gaus(0.4, 1.3, n=n)


def test_an_unbinned_gaussian_fit_finds_the_points_mean_and_spread():
    x = normal_points()
    f = Function("g", "gaus", range=(-20, 20), parameters=[1.0, 0.0, 1.0])
    f.fix(0)
    result = unbinned(x, f)
    assert result.parameter("Mean") == pytest.approx(x.mean(), abs=2e-4)
    assert result.parameter("Sigma") == pytest.approx(x.std(), rel=2e-4)
    assert result.error("Mean") == pytest.approx(x.std() / np.sqrt(len(x)), rel=0.01)
    assert result.npoints == 4000 and result.ndf == 3998 and result.function is f
    assert f.fit_result["npfits"] == 4000


def test_a_python_model_and_a_formula_take_their_parameters_and_the_range():
    x = normal_points(2000, 2)
    code = unbinned(
        x,
        lambda t, p: np.exp(-0.5 * ((t - p[0]) / p[1]) ** 2),
        parameters=[0.0, 1.0],
        range=(-10, 10),
    )
    formula = unbinned(x, "exp(-0.5*((x-[0])/[1])^2)", parameters=[0.0, 1.0], range=(-10, 10))
    np.testing.assert_allclose(code.parameters, formula.parameters, rtol=1e-4)
    inside = unbinned(x, "exp(-0.5*((x-[0])/[1])^2)", parameters=[0.0, 1.0])  # the points' span
    assert inside.parameter(0) == pytest.approx(formula.parameter(0), abs=0.01)
    with pytest.raises(ValueError, match="needs its parameters="):
        unbinned(x, "gaus")


def test_an_extended_fit_finds_the_number_of_points_as_the_models_integral():
    x = normal_points(1500, 3)
    f = Function("g", "gausn", range=(-15, 15), parameters=[1000.0, 0.0, 1.0])
    result = unbinned(x, f, extended=True, parameters=[1000.0, 0.0, 1.0])
    assert result.parameter("Constant") == pytest.approx(1500, rel=1e-3)
    assert result.error("Constant") == pytest.approx(np.sqrt(1500), rel=0.02)


def test_points_outside_the_range_are_left_out_and_limits_are_kept():
    x = np.concatenate([normal_points(1000, 4), [50.0, -50.0]])
    result = unbinned(
        x,
        "[0]*exp(-0.5*((x-[1])/[2])^2)",
        parameters=[1.0, 0.0, 1.0],
        range=(-8, 8),
        fixed=[True, False, False],
        limits=[None, None, (0.1, 5.0)],
    )
    assert result.npoints == 1000 and result.bounded == (False, False, True)


def test_a_model_of_two_variables_is_normalised_over_its_area():
    rng = TRandom3(6)
    points = np.stack([rng.gaus(0.5, 0.8, n=3000), rng.gaus(-0.3, 1.2, n=3000)], axis=1)
    result = unbinned(
        points,
        lambda xy, p: np.exp(
            -0.5 * ((xy[:, 0] - p[0]) / p[1]) ** 2 - 0.5 * ((xy[:, 1] - p[2]) / p[3]) ** 2
        ),
        parameters=[0.0, 1.0, 0.0, 1.0],
        range=[(-6, 6), (-7, 7)],
    )
    np.testing.assert_allclose(result.parameters[[0, 2]], points.mean(axis=0), atol=2e-3)
    np.testing.assert_allclose(result.parameters[[1, 3]], points.std(axis=0), rtol=2e-3)
    with pytest.raises(ValueError, match="points of 2"):
        unbinned(points, Function("g", "gaus", parameters=[1, 0, 1]))
