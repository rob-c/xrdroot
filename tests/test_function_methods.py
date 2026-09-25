"""A function to evaluate, differentiate, integrate and fit, the way ``TF1`` is one.

The numbers are checked against closed forms: the gradient of a Gaussian
against its derivatives written out, integrals against the areas they are
known to have, extrema and roots where they are known to be. The gradient
is also checked against ROOT's numerical recipe, which is what a formula
calling a function with no known derivative falls back on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot import Function, UnsupportedFeatureError
from xrdroot.function import numeric


def gaussian():
    return Function("g", "gaus", range=(-5, 5), parameters=[3.0, 0.5, 1.5])


# -- what it is --------------------------------------------------------------------------


def test_a_function_is_the_class_its_variables_make_it():
    assert Function("f", "x").classname == "TF1"
    assert Function("f", "x*y").classname == "TF2"
    assert Function("f", "x*y*z").classname == "TF3"
    assert Function("f", "[0]", range=((0, 1), (0, 2))).dimensions == 2
    f = Function("f", "x*y*z", range=((0, 1), (0, 2), (0, 3)))
    assert f.range == ((0.0, 1.0), (0.0, 2.0), (0.0, 3.0))
    assert f(2, 3, 4) == 24


def test_a_function_of_four_variables_has_no_class():
    with pytest.raises(UnsupportedFeatureError, match="TF1, TF2 and TF3"):
        Function("f", "x", range=((0, 1),) * 4)


def test_a_function_starts_as_tf1s_constructor_starts_one():
    f = Function("line", "[0] + [1]*x")
    assert (f.name, f.title, f.range) == ("line", "[0] + [1]*x", (0.0, 1.0))
    assert list(f.parameters) == [0.0, 0.0]
    assert f.fit_result is None
    assert f.fixed == (False, False)
    assert f.parameter_limits == (None, None)
    assert Function("line", "x", title="a line").title == "a line"
    assert repr(f) == "<TF1 'line': [p0]+[p1]*x, 2 parameters>"


def test_the_wrong_number_of_parameters_or_names_is_refused():
    with pytest.raises(ValueError, match="2 parameters, and 3 values"):
        Function("f", "[0]+[1]*x", parameters=[1, 2, 3])
    with pytest.raises(ValueError, match="2 parameters, and 1 names"):
        Function("f", "[0]+[1]*x", parameter_names=["a"])
    with pytest.raises(ValueError, match="1 parameters, and 2 names"):
        Function.from_callable("f", lambda x, p: x, 1, parameter_names=["a", "b"])


def test_parameters_can_be_named_when_the_function_is_made():
    f = Function("f", "[0]*exp(-x/[1])", parameter_names=["N", "tau"], parameters=[2, 1])
    assert f.formula == "[N]*exp(-x/[tau])"
    assert f.members["fFormula"]["fParams"] == {"N": 0, "tau": 1}
    assert f(1.0) == pytest.approx(2 / math.e)


# -- the parameters ------------------------------------------------------------------------


def test_parameters_are_set_in_order_by_name_or_all_at_once():
    f = gaussian()
    f.set_parameters(1.0)
    f.set_parameters(Sigma=2.0)
    assert list(f.parameters) == [1.0, 0.5, 2.0]
    f.parameters = [4, 5, 6]
    assert list(f.members["fFormula"]["fClingParameters"]) == [4, 5, 6]
    f.parameters[0] = 7.0  # the array is the members'
    assert f.members["fFormula"]["fClingParameters"][0] == 7.0


def test_a_parameter_that_is_not_there_is_refused_by_what_there_is():
    f = gaussian()
    with pytest.raises(KeyError, match="Constant, Mean, Sigma"):
        f.set_parameters(Width=1.0)
    with pytest.raises(IndexError, match="it has 3"):
        f.fix(3)
    with pytest.raises(ValueError, match="3 parameters, and 4 were given"):
        f.set_parameters(1, 2, 3, 4)
    with pytest.raises(KeyError, match="none"):
        Function("f", "x").release("a")


def test_renaming_the_parameters_renames_them_in_the_formula():
    f = gaussian()
    f.parameter_names = ["A", "mu", "s"]
    assert f.formula == "[A]*exp(-0.5*((x-[mu])/[s])*((x-[mu])/[s]))"
    assert f.members["fFormula"]["fParams"] == {"A": 0, "mu": 1, "s": 2}
    assert f(0.5) == 3.0
    with pytest.raises(ValueError, match="3 parameters, and 1 names"):
        f.parameter_names = ["A"]
    code = Function.from_callable("c", lambda x, p: p[0] * x, 1)
    code.parameter_names = ["slope"]
    assert code.parameter_names == ("slope",)


def test_errors_and_limits_are_kept_where_tf1_keeps_them():
    f = gaussian()
    f.parameter_errors = [0.1, 0.2, 0.3]
    assert list(f.members["fParErrors"]) == [0.1, 0.2, 0.3]
    f.parameter_limits = [None, (-1, 1), None]
    assert f.parameter_limits == (None, (-1.0, 1.0), None)
    assert list(f.members["fParMin"]) == [0, -1, 0]


def test_a_fixed_parameter_is_marked_as_fixparameter_marks_it():
    f = gaussian()
    f.fix("Mean", 0.0)
    assert f.fixed == (False, True, False)
    assert (f.members["fParMin"][1], f.members["fParMax"][1]) == (1.0, 1.0)  # zero cannot be told
    assert f.parameters[1] == 0.0
    f.fix(0)
    assert (f.members["fParMin"][0], f.members["fParMax"][0]) == (3.0, 3.0)
    f.fixed = [False, False, True]
    assert f.fixed == (False, False, True)
    f.fixed = [False, False, True]  # nothing to release, nothing changed
    assert f.fixed == (False, False, True)
    f.release(2)
    assert f.fixed == (False, False, False)


def test_a_fit_result_is_what_the_tf1_keeps_of_its_fit():
    f = gaussian()
    f.fit_result = {"chi2": 12.5, "ndf": 17, "npfits": 20, "errors": [1, 2, 3]}
    result = f.fit_result
    assert (result["chi2"], result["ndf"], result["npfits"]) == (12.5, 17, 20)
    assert list(result["errors"]) == [1, 2, 3]
    assert list(result["parameters"]) == [3.0, 0.5, 1.5]
    f.fit_result = None
    assert f.fit_result is None


# -- evaluation ------------------------------------------------------------------------------


def test_a_number_gives_a_number_and_an_array_an_array_of_its_shape():
    f = gaussian()
    assert isinstance(f(0.5), float)
    assert f(np.zeros((2, 3))).shape == (2, 3)
    g = Function("g", "x*y")
    assert g(np.array([1.0, 2.0]), 3.0).tolist() == [3.0, 6.0]
    with pytest.raises(ValueError, match="2 variables, and was given 1"):
        g(1.0)


def test_evaluate_takes_points_and_parameters_of_its_own():
    f = gaussian()
    assert f.evaluate([0.5], [1, 0.5, 1]).tolist() == [1.0]
    g = Function("g", "[0]*x+y")
    assert g.evaluate(np.array([[1.0, 2.0], [3.0, 4.0]]), [10]).tolist() == [12.0, 34.0]
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        g.evaluate([1.0, 2.0])
    with pytest.raises(ValueError, match="takes 1 parameters, and was given 2"):
        g.evaluate([[1.0, 2.0]], [1, 2])
    assert Function("c", "[0]", parameters=[4]).evaluate([1, 2, 3]).tolist() == [4, 4, 4]


def test_a_python_model_is_a_function_like_any_other():
    def model(x, p):
        return p[0] * np.exp(-x / p[1])

    f = Function.from_callable("decay", model, 2, range=(0, 10), parameters=[5, 2])
    assert f.formula is None
    assert f.parameter_names == ("p0", "p1")
    assert f(2.0) == pytest.approx(5 / math.e)
    assert f.integral(0, math.inf) == pytest.approx(10.0, rel=1e-9)
    plane = Function.from_callable(
        "plane", lambda xy, p: p[0] * xy[:, 0] + xy[:, 1], 1, dimensions=2, parameters=[2]
    )
    assert plane(1.0, 1.0) == 3.0
    constant = Function.from_callable("k", lambda x, p: p[0], 1, parameters=[3])
    assert constant(np.array([1.0, 2.0])).tolist() == [3.0, 3.0]
    assert repr(constant) == "<TF1 'k': compiled code, 1 parameters>"


def test_a_normalised_function_is_divided_by_its_integral_as_its_parameters_change():
    f = Function("f", "[0]*x", range=(0, 2), parameters=[3])
    f.normalized = True
    assert f.members["fNormIntegral"] == pytest.approx(6.0)
    assert f(1.0) == pytest.approx(0.5)
    f.set_parameters(5)
    assert f(1.0) == pytest.approx(0.5)
    f.range = (0, 1)
    assert f(1.0) == pytest.approx(2.0)
    f.normalized = False
    assert f(1.0) == 5.0


# -- the gradient ----------------------------------------------------------------------------


def test_the_gradient_of_a_gaussian_is_its_derivatives_exactly():
    f = gaussian()
    x = np.linspace(-3, 3, 13)
    a, m, s = f.parameters
    u = (x - m) / s
    shape = np.exp(-0.5 * u * u)
    expected = np.stack([shape, a * shape * u / s, a * shape * u * u / s], axis=1)
    np.testing.assert_allclose(f.gradient(x), expected, rtol=1e-13, atol=1e-15)


def test_the_gradient_of_a_polynomial_is_the_powers():
    f = Function("f", "pol3", parameters=[1, 2, 3, 4])
    x = np.array([-2.0, 0.5, 3.0])
    np.testing.assert_array_equal(f.gradient(x), np.stack([x**0, x, x**2, x**3], axis=1))


def test_a_parameter_inside_a_function_with_no_derivative_is_differentiated_by_roots_recipe():
    f = Function("f", "[0]*TMath::Landau(x,[1],[2])", parameters=[2, 1, 0.5])
    x = np.linspace(-1, 4, 7)
    gradient = f.gradient(x)
    np.testing.assert_array_equal(gradient[:, 0], f.evaluate(x) / 2)  # exact: linear in [0]
    for k in (1, 2):
        h = 0.01

        def moved(step, k=k):
            params = np.array(f.parameters)
            params[k] += step
            return f.evaluate(x, params)

        recipe = (8 * (moved(h / 2) - moved(-h / 2)) - (moved(h) - moved(-h))) / (6 * h)
        np.testing.assert_allclose(gradient[:, k], recipe, rtol=1e-12)


def test_a_parameter_with_an_error_is_stepped_by_a_hundredth_of_it():
    f = Function.from_callable("c", lambda x, p: p[0] ** 3 * x, 1, parameters=[2])
    f.parameter_errors = [10.0]  # a step of 0.1, and Richardson is exact for a cubic
    assert f.gradient([1.0])[0, 0] == pytest.approx(12.0, rel=1e-12)


@pytest.mark.parametrize(
    ("formula", "params", "x", "expected"),
    [
        ("x^[0]", [2.0], 3.0, 9 * math.log(3)),
        ("pow(x,[0])", [2.0], 3.0, 9 * math.log(3)),
        ("[0]^2", [3.0], 1.0, 6.0),
        ("x/[0]", [2.0], 3.0, -0.75),
        ("[0]/x", [2.0], 4.0, 0.25),
        ("-[0]*x", [2.0], 3.0, -3.0),
        ("+[0]", [2.0], 3.0, 1.0),
        ("x-[0]", [2.0], 3.0, -1.0),
        ("fmod(x,[0])+x%[0]", [2.0], 5.0, None),
        ("([0]>1)*x+!x", [2.0], 3.0, 0.0),
        ("x>1 ? [0]*x : [0]", [2.0], 3.0, 3.0),
        ("[0]+(x>1 ? x : 2)", [2.0], 3.0, 1.0),
        ("x>1 ? [0] : TMath::Landau(x,[0])", [2.0], 3.0, None),
        ("sqrt([0])+log([0])+sin([0])+cos([0])+sq([0])+abs(-[0])+exp([0])", [2.0], 1.0, None),
    ],
)
def test_the_chain_rule_passes_through_every_operator(formula, params, x, expected):
    f = Function("f", formula, parameters=params)
    gradient = float(f.gradient([x])[0, 0])
    if expected is None:  # no closed form written here: ROOT's recipe, near enough
        h = 1e-6
        expected = (f.evaluate([x], [params[0] + h])[0] - f.evaluate([x], [params[0] - h])[0]) / (
            2 * h
        )
        assert gradient == pytest.approx(expected, rel=1e-6)
        return
    assert gradient == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_a_fixed_parameter_has_no_gradient():
    f = gaussian()
    f.fix("Mean")
    assert not f.gradient([0.0, 1.0])[:, 1].any()


# -- TF1's numerical methods ---------------------------------------------------------------


def test_the_integral_of_a_gaussian_is_its_area():
    f = gaussian()
    area = 3.0 * 1.5 * math.sqrt(2 * math.pi)
    assert f.integral(-math.inf, math.inf) == pytest.approx(area, rel=1e-12)
    assert f.integral(0.5, math.inf) == pytest.approx(area / 2, rel=1e-12)
    assert f.integral(-math.inf, 0.5) == pytest.approx(area / 2, rel=1e-12)
    assert f.integral(5, -5) == pytest.approx(-f.integral(-5, 5))
    assert f.integral(1, 1) == 0.0


def test_an_integral_that_cannot_converge_is_taken_as_far_as_it_goes():
    f = Function("f", "1/sqrt(x)")
    assert f.integral(0, 1) == pytest.approx(2.0, rel=1e-3)
    narrow = numeric.integrate(lambda t: np.where(t > 0, 1.0, 0.0), -1e-300, 5e-324)
    assert narrow >= 0


def test_the_derivative_is_richardsons():
    f = Function("f", "pol2", range=(0, 10), parameters=[1, 2, 3])
    assert f.derivative(2.0) == pytest.approx(2 + 6 * 2.0, rel=1e-12)
    assert f.derivative(np.array([[0.0, 1.0]])).shape == (1, 2)
    point = Function("f", "x*x", range=(1, 1))
    assert point.derivative(3.0) == pytest.approx(6.0, rel=1e-9)


def test_extrema_and_roots_are_found_as_brents_method_finds_them():
    f = gaussian()
    assert f.maximum_x() == pytest.approx(0.5, abs=1e-6)  # as flat as a peak is
    assert f.maximum() == pytest.approx(3.0, rel=1e-12)
    assert f.minimum() == pytest.approx(f(-5.0), rel=1e-9)
    assert f.minimum_x(0, 5) == pytest.approx(5.0, abs=1e-8)
    assert f.x_at(3 * math.exp(-0.5), 0.5, 5) == pytest.approx(2.0, abs=1e-8)
    assert f.maximum(3, 1) == pytest.approx(3.0)  # a backwards range is the function's own
    bowl = Function("bowl", "(x-[0])^2+[1]", range=(-10, 10), parameters=[1.25, -2])
    assert bowl.minimum_x() == pytest.approx(1.25, abs=1e-8)
    assert bowl.minimum() == pytest.approx(-2.0, abs=1e-12)


def test_a_panel_as_narrow_as_the_doubles_allow_is_not_cut_again():
    after_one = float(np.nextafter(1.0, 2.0))
    tiny = numeric.integrate(lambda t: t * 0 + 1, 1.0, after_one, epsabs=-1.0, epsrel=-1.0)
    assert tiny == pytest.approx(after_one - 1.0, rel=1e-12)


def test_a_search_that_runs_out_of_steps_tries_again_from_a_new_bracket():
    found = numeric.minimise(lambda t: (t - 0.3) ** 2, -1.0, 1.0, npx=1, iterations=2)
    assert -1.0 <= found <= 1.0


def test_the_one_dimensional_methods_refuse_a_function_of_more():
    g = Function("g", "x*y")
    with pytest.raises(UnsupportedFeatureError, match="TF2 of 2"):
        g.integral(0, 1)
    with pytest.raises(UnsupportedFeatureError, match="maximum_x is TF1's"):
        g.maximum()
    with pytest.raises(UnsupportedFeatureError, match="derivative is TF1's"):
        g.derivative(0.0)


# -- copying and writing ----------------------------------------------------------------------


def test_a_copy_shares_nothing_and_may_be_renamed():
    f = gaussian()
    twin = f.copy("twin")
    twin.set_parameters(9)
    assert (twin.name, twin.members["fFormula"]["TNamed"]["fName"]) == ("twin", "twin")
    assert f.parameters[0] == 3.0
    assert f.copy().name == "g"
    code = Function.from_callable("c", lambda x, p: p[0] * x, 1, parameters=[2])
    assert code.copy()(3.0) == 6.0


def test_a_python_model_is_written_as_the_values_root_would_sample():
    f = Function.from_callable("sq", lambda x, p: p[0] * x * x, 1, range=(0, 2), parameters=[3])
    written = f.written_members()
    assert len(written["fSave"]) == 100 + 3
    assert written["fSave"][-2:].tolist() == [0.0, 2.0]
    assert written["fSave"][50] == pytest.approx(3.0)
    assert len(f.members["fSave"]) == 0  # sampled for the file, not kept
    formula = gaussian()
    assert formula.written_members() is formula.members
