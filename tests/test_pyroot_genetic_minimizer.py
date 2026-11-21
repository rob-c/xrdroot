"""``ROOT::Math::GeneticMinimizer``: TMVA's genetic search, as ROOT's minimiser.

The search is ROOT's - a population bred over the ranges the caller gave - and so is
what it can and cannot answer: the least value and where it is, and no error at all,
since nothing there is a covariance. Its own test program (``testGAMinimizer``) is
run here on the three functions it uses, to the tolerances it asks for.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT

Math = ROOT.Math


class Function:
    """A macro's own ``IMultiGenFunction``: a Python function and how many coordinates."""

    def __init__(self, fn, ndim):
        self._fn, self._ndim = fn, ndim

    def NDim(self):
        return self._ndim

    def __call__(self, x):
        return float(self._fn(x))


def parabole(x):
    return x[0] * x[0]


def rosenbrock(x):
    return 100 * (x[1] - x[0] * x[0]) ** 2 + (1 - x[0]) ** 2


def multimin(x):
    return 0.6 * x[0] ** 4 + 0.1 * x[0] ** 3 - 2 * x[0] ** 2 + 1


def searched(fn, ndim, ranges, seed=11):
    minimizer = Math.GeneticMinimizer(ndim)
    minimizer.SetFunction(Function(fn, ndim))
    for at, (low, high) in enumerate(ranges):
        minimizer.SetLimitedVariable(at, f"x{at}", 0.0, 0.1, low, high)
    minimizer.SetRandomSeed(seed)
    return minimizer


def test_a_parabola_is_minimised_to_the_tolerance_root_asks_of_it():
    minimizer = searched(parabole, 1, [(-5.0, 5.0)])
    assert minimizer.Minimize() is True
    assert abs(minimizer.MinValue()) < 1e-3
    assert abs(minimizer.X()[0]) < 1e-3


def test_a_rosenbrock_function_is_minimised_to_the_tolerance_root_asks_of_it():
    minimizer = searched(rosenbrock, 2, [(-5.0, 5.0), (-5.0, 5.0)], seed=111)
    minimizer.SetMaxIterations(500)
    assert minimizer.Minimize() is True
    assert abs(minimizer.MinValue()) < 5e-2
    assert minimizer.NCalls() > 0


def test_a_function_with_two_minima_is_taken_to_the_lower_one():
    # a gradient search from x = 0 finds the shoulder at x = 1.23 rather than the
    # minimum at x = -1.35, which is what a population spread over the range is for
    minimizer = searched(multimin, 1, [(-5.0, 5.0)], seed=3)
    assert minimizer.Minimize() is True
    assert abs(minimizer.MinValue() + 0.8982) < 1e-3
    assert minimizer.X()[0] == pytest.approx(-1.3545, abs=1e-2)


def test_a_search_that_runs_out_of_generations_says_so():
    minimizer = searched(parabole, 1, [(-5.0, 5.0)])
    minimizer.SetMaxIterations(2)
    assert minimizer.Minimize() is True
    assert minimizer.Status() == 1


def test_a_search_that_ran_to_its_limit_says_so():
    # ROOT asks for as many generations as it is willing to run, and stops on that limit
    # before it asks whether the value has settled - so a search reports 1, as ROOT's does
    minimizer = Math.GeneticMinimizer()
    parameters = Math.GeneticMinimizerParameters()
    parameters.fNsteps, parameters.fPopSize = 4, 20
    minimizer.SetParameters(parameters)
    minimizer.SetFunction(Function(parabole, 1))
    minimizer.SetLimitedVariable(0, "x", 0.0, 0.1, -1.0, 1.0)
    minimizer.SetRandomSeed(8)
    assert minimizer.Minimize() is True
    assert minimizer.Status() == 1
    assert minimizer.NCalls() == 5 * 20  # the generations it ran, and this many points each


def test_a_fixed_coordinate_is_filled_in_at_every_point_the_search_tries():
    seen = []

    def at(x):
        seen.append(list(x))
        return (x[0] - 2.0) ** 2

    minimizer = Math.GeneticMinimizer()
    minimizer.SetFunction(Function(at, 2))
    minimizer.SetLimitedVariable(0, "x", 0.0, 0.1, -5.0, 5.0)
    minimizer.SetFixedVariable(1, "fixed", 7.0)
    minimizer.SetRandomSeed(4)
    minimizer.SetMaxIterations(3)
    assert minimizer.Minimize() is True
    assert (minimizer.NDim(), minimizer.NFree()) == (2, 1)
    assert all(point[1] == 7.0 for point in seen)
    assert minimizer.X()[1] == 7.0


def test_a_coordinate_without_limits_is_given_a_range_fifty_steps_either_side(capsys):
    minimizer = Math.GeneticMinimizer()
    minimizer.SetFunction(Function(parabole, 1))
    assert minimizer.SetVariable(0, "x", 1.0, 0.1) is True
    minimizer.SetRandomSeed(2)
    minimizer.SetMaxIterations(2)
    minimizer.Minimize()
    reported = capsys.readouterr().err
    assert "set automatic range to 50 times step size for x : [-4.000000, 6.000000]" in reported
    assert -4.0 <= minimizer.X()[0] <= 6.0


def test_a_minimiser_with_no_function_refuses_to_search(capsys):
    minimizer = Math.GeneticMinimizer()
    assert minimizer.Minimize() is False
    assert "Fitness function has not been set" in capsys.readouterr().err
    # and what it has found is nothing
    assert (minimizer.MinValue(), list(minimizer.X())) == (0.0, [])


def test_what_a_genetic_search_cannot_answer_is_said_by_what_it_answers():
    minimizer = searched(parabole, 1, [(-5.0, 5.0)])
    minimizer.Minimize()
    assert (
        minimizer.ProvidesError(),
        minimizer.Errors(),
        minimizer.Edm(),
        minimizer.CovMatrix(0, 0),
        minimizer.MinGradient(),
    ) == (False, [], 0.0, 0.0, [])


def test_the_settings_the_search_runs_with_are_roots_own_defaults():
    parameters = Math.GeneticMinimizerParameters()
    assert (
        parameters.fPopSize,
        parameters.fNsteps,
        parameters.fCycles,
        parameters.fSC_steps,
        parameters.fSC_rate,
        parameters.fSC_factor,
        parameters.fConvCrit,
        parameters.fSeed,
    ) == (300, 100, 3, 10, 5, 0.95, 0.1, 0)


def test_settings_given_are_what_the_search_runs_with():
    minimizer = Math.GeneticMinimizer()
    parameters = Math.GeneticMinimizerParameters()
    parameters.fNsteps, parameters.fPopSize, parameters.fConvCrit = 4, 20, 0.05
    minimizer.SetParameters(parameters)
    minimizer.SetFunction(Function(parabole, 1))
    minimizer.SetLimitedVariable(0, "x", 0.0, 0.1, -5.0, 5.0)
    assert (minimizer.MinimizerParameters() is parameters, minimizer.MaxIterations()) == (True, 4)
    assert minimizer.Tolerance() == pytest.approx(0.005)
    minimizer.Minimize()
    assert minimizer.NCalls() >= 20  # the population was bred at least once


def test_a_search_at_a_print_level_says_what_it_is_doing(capsys):
    minimizer = searched(parabole, 1, [(-5.0, 5.0)])
    minimizer.SetPrintLevel(1)
    minimizer.SetMaxIterations(1)
    minimizer.Minimize()
    out = capsys.readouterr().out
    assert "GeneticMinimizer::Minimize  - Start iterating - max iterations = 1" in out
    assert "Finished Iteration" in out
    assert " Parameter-0\t=\t" in out


def test_clearing_forgets_the_function_and_the_coordinates():
    minimizer = searched(parabole, 1, [(-5.0, 5.0)])
    minimizer.Minimize()
    minimizer.Clear()
    assert (minimizer.NDim(), minimizer.NFree(), minimizer.NCalls()) == (0, 0, 0)
    assert minimizer.Minimize() is False


def test_the_factory_makes_the_genetic_search_by_name():
    made = Math.Factory.CreateMinimizer("Genetic", "")
    assert isinstance(made, Math.GeneticMinimizer)
    assert isinstance(Math.Factory.CreateMinimizer("Minuit2", ""), Math.Minimizer)


def test_the_result_of_a_search_is_an_array_of_the_coordinates():
    minimizer = searched(parabole, 1, [(-5.0, 5.0)])
    minimizer.Minimize()
    assert isinstance(minimizer.X(), np.ndarray)


def test_a_minimiser_with_no_limits_and_no_tolerance_still_searches():
    # neither the generation count nor the tolerance the base class keeps needs setting,
    # and a search of a generation or two is enough to say it ran
    minimizer = Math.GeneticMinimizer()
    minimizer.SetFunction(Function(parabole, 1))
    minimizer.SetLimitedVariable(0, "x", 0.0, 0.1, -1.0, 1.0)
    minimizer._settings["iterations"] = 0  # ROOT's SetMaxIterations never clears the limit
    minimizer.SetTolerance(0.0)
    assert (minimizer.MaxIterations(), minimizer.Tolerance()) == (0, 0.0)
    assert minimizer.Minimize() is True
    assert minimizer.MinValue() >= 0.0


def test_a_search_whose_convergence_count_is_none_settles_at_once():
    # with no generation limit and no quiet generations wanted, the first generation is
    # the answer, which is the only way ROOT's own loop reaches its convergence test first
    minimizer = Math.GeneticMinimizer()
    parameters = Math.GeneticMinimizerParameters()
    parameters.fNsteps, parameters.fPopSize = 0, 20
    minimizer.SetParameters(parameters)
    minimizer._settings["iterations"] = 0
    minimizer.SetFunction(Function(parabole, 1))
    minimizer.SetLimitedVariable(0, "x", 0.0, 0.1, -1.0, 1.0)
    minimizer.SetRandomSeed(6)
    assert minimizer.Minimize() is True
    assert minimizer.Status() == 0
    assert minimizer.NCalls() == 20
