"""ROOT's formula language, as ``TFormula`` parses it, and the shapes it predefines.

The shapes are checked two ways: by the text each expands into, which is
ROOT's own - from ``test_TFormula.cxx`` as go-hep's ``root_formula_test.go``
ports it, and from the formulas ROOT wrote into ``tformula.root`` - and by
the value that text takes, worked out by hand. The arithmetic is go-hep's
``feval_test.go`` table, and the special functions are checked at the
points where their values are known in closed form.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from xrdroot import FormulaError, Function, UnsupportedFeatureError, open_root
from xrdroot.function import special
from xrdroot.function.language import parse
from xrdroot.function.shapes import expand, param_order

DATA = pathlib.Path(__file__).parent / "data"

#: The peak of the Landau density, and its height there.
LANDAU_PEAK = -0.22278298
LANDAU_TOP = 0.18065564


def value(formula, params=(), *coordinates):
    f = Function("f", formula, parameters=list(params) or None)
    return f(*(coordinates or (0.0,) * f.dimensions))


# -- ROOT's own tests of the shapes, from test_TFormula.cxx --------------------------


@pytest.mark.parametrize(
    ("formula", "expanded", "params", "at", "expected"),
    [
        ("pol1", "([p0]+[p1]*x)", (1, 2), (3,), 1 + 2 * 3),
        ("pol2", "([p0]+[p1]*x+[p2]*TMath::Sq(x))", (1, 2, 3), (2,), 1 + 2 * 2 + 3 * 4),
        ("pol1(y,0)", "([p0]+[p1]*y)", (1, 2), (0, 3), 1 + 2 * 3),
        ("pol2(z,0)", "([p0]+[p1]*z+[p2]*TMath::Sq(z))", (1, 2, 3), (0, 0, 2), 17),
        ("pol1(x,[A], [B])", "([A]+[B]*x)", (1, 2), (3,), 7),
        ("pol2(x, [A], [B], [C])", "([A]+[B]*x+[C]*TMath::Sq(x))", (1, 2, 3), (2,), 17),
        ("pol1(x,0) + pol1(y,2)", "([p0]+[p1]*x)+([p2]+[p3]*y)", (1, 2, 3, 4), (5, 6), 11 + 27),
        ("xyexpo", "exp([p0]+[p1]*x+[p2]*y)", (0.5, 2, 3), (1, 2), math.exp(8.5)),
    ],
)
def test_each_shape_expands_to_what_root_expands_it_to(formula, expanded, params, at, expected):
    f = Function("f", formula, parameters=params)
    assert f.formula == expanded
    assert f(*at) == pytest.approx(expected, rel=1e-15)


def test_the_two_dimensional_landaus_are_roots_and_the_normalised_one_has_no_amplitude():
    plain = Function("f", "xylandau", parameters=[2, 0, 1, 0, 1])
    assert plain.formula == (
        "[p0]*TMath::Landau(x,[p1],[p2],false)*TMath::Landau(y,[p3],[p4],false)"
    )
    assert plain(LANDAU_PEAK, LANDAU_PEAK) == pytest.approx(2 * LANDAU_TOP**2, rel=1e-6)
    normalised = Function("f", "xylandaun", parameters=[0, 2, 0, 2])
    assert normalised.formula == "TMath::Landau(x,[p0],[p1],true)*TMath::Landau(y,[p2],[p3],true)"
    peak = 2 * LANDAU_PEAK
    assert normalised(peak, peak) == pytest.approx((LANDAU_TOP / 2) ** 2, rel=1e-6)


def test_an_unbalanced_bracket_is_not_a_formula():
    for broken in ("[", "(", "gaus(0", "[0"):
        with pytest.raises(FormulaError):
            Function("f", broken)


def test_a_formula_that_is_one_shape_gets_roots_names_for_its_parameters():
    with open_root(str(DATA / "tformula.root")) as root:
        convolved = root["fconv"]
        gaussian, exponential = convolved["fFunction2"], convolved["fFunction1"]
        theirs = (
            gaussian.members["fFormula"]["fFormula"],
            exponential.members["fFormula"]["fFormula"],
        )
    gaus, expo = Function("gaus", "gaus"), Function("expo", "expo")
    assert (gaus.formula, expo.formula) == theirs
    assert gaus.parameter_names == ("Constant", "Mean", "Sigma")
    assert expo.parameter_names == ("Constant", "Slope")


def test_every_shape_root_names_the_parameters_of_is_named_as_root_names_them():
    assert Function("f", "xygaus").parameter_names == (
        "Constant",
        "MeanX",
        "SigmaX",
        "MeanY",
        "SigmaY",
    )
    assert Function("f", "crystalball").parameter_names[-2:] == ("Alpha", "N")
    assert Function("f", "breitwigner").parameter_names == ("Constant", "Mean", "Gamma")
    assert Function("f", "landau").parameter_names == ("Constant", "MPV", "Sigma")
    assert Function("f", "bigaus").parameter_names[-1] == "Rho"
    assert Function("f", "xyzgaus").parameter_names == tuple(f"p{i}" for i in range(7))
    assert Function("f", "gaus+expo").parameter_names == ("p0", "p1", "p2")


def test_a_shape_inside_a_longer_formula_is_bracketed_so_it_stays_whole():
    f = Function("f", "x/gaus", parameters=[2, 0, 1])
    assert f.formula.startswith("x/([p0]*exp(")
    assert f(1.0) == pytest.approx(1 / (2 * math.exp(-0.5)), rel=1e-15)


def test_shapes_start_their_parameters_where_they_are_told():
    f = Function("f", "gaus(0)+pol1(3)", parameters=[1, 0, 1, 10, 2])
    assert f.npar == 5
    assert f(0.0) == 1 + 10
    assert Function("f", "[0]*gaus(1)", parameters=[3, 2, 0, 1])(0.0) == 6


def test_the_chebyshev_series_are_the_polynomials_they_are_defined_by():
    polynomials = [
        lambda x: 1,
        lambda x: x,
        lambda x: 2 * x * x - 1,
        lambda x: 4 * x**3 - 3 * x,
        lambda x: 8 * x**4 - 8 * x * x + 1,
    ]
    points = np.array([-1, -0.4, 0, 0.25, 1, 2])
    for degree in range(5):
        coefficients = [index + 0.5 for index in range(degree + 1)]
        f = Function("f", f"cheb{degree}", parameters=coefficients)
        expected = sum(
            c * np.array([polynomials[i](x) for x in points]) for i, c in enumerate(coefficients)
        )
        np.testing.assert_allclose(f(points), expected, rtol=1e-12, atol=1e-12)
    assert Function("f", "cheb10").npar == 11


def test_a_chebyshev_series_past_the_tenth_is_refused_by_name():
    with pytest.raises(UnsupportedFeatureError, match="cheb10, the highest"):
        Function("f", "cheb11")


def test_a_shape_given_arguments_it_does_not_take_is_refused_with_the_forms_it_does():
    with pytest.raises(FormulaError, match="TMath::Gaus"):
        Function("f", "gaus(x,0,1)")
    with pytest.raises(FormulaError, match="3 parameters, and 1 names"):
        Function("f", "gaus(x,[A])")
    with pytest.raises(FormulaError, match="2 parameters, and 1 names"):
        Function("f", "pol1(x,[A])")


def test_the_old_spellings_name_the_variables_of_the_shape():
    f = Function("f", "ygaus", parameters=[1, 0, 1], range=((0, 1), (0, 1)))
    assert f.dimensions == 2
    assert f(5.0, 0.0) == 1.0
    assert Function("f", "zexpo", parameters=[0, 1]).dimensions == 3
    assert Function("f", "xexpo(2)", parameters=[0, 0, 0, 1])(1.0) == math.e


def test_parameters_are_numbered_by_their_number_and_named_ones_take_the_next_free():
    assert Function("f", "[0]+[A]*x").parameter_names == ("p0", "A")
    assert Function("f", "[slope]*x+[offset]").parameter_names == ("slope", "offset")
    assert Function("f", "[1]*x").parameter_names == ("p0", "p1")
    assert Function("f", "[A]*x+[A]").npar == 1
    assert Function("f", "[0]+[p0]*x").parameter_names == ("p0",)


def test_parameter_names_sort_as_tformula_param_order_sorts_them():
    names = ["p10", "b", "p2", "3", "a"]
    assert sorted(names, key=param_order) == ["p2", "3", "p10", "a", "b"]


def test_a_formula_read_keeps_the_names_it_came_with():
    written = expand("[Constant]*exp(-x/[Tau])+[2]", ("Constant", "Tau", "Base"))
    assert written.text == "[Constant]*exp(-x/[Tau])+[Base]"
    assert written.names == ("Constant", "Tau", "Base")
    assert not written.predefined


# -- the arithmetic, from go-hep's feval_test.go -------------------------------------


@pytest.mark.parametrize(
    ("formula", "params", "at", "expected"),
    [
        ("1+2*3", (), (), 7),
        ("(1+2)*3", (), (), 9),
        ("2^3^2", (), (), 512),
        ("-2^2", (), (), -4),
        ("2**3", (), (), 8),
        ("7/2", (), (), 3.5),
        ("7%3", (), (), 1),
        ("1e-2", (), (), 0.01),
        ("1.5e3", (), (), 1500),
        (".5", (), (), 0.5),
        ("x", (), (3,), 3),
        ("x*x", (), (4,), 16),
        ("x+y+z", (), (1, 2, 3), 6),
        ("x[0]+x[1]", (), (5, 6), 11),
        ("[0]+[1]*x", (2, 3), (4,), 14),
        ("[2]", (0, 0, 9), (), 9),
        ("sqrt(16)", (), (), 4),
        ("abs(-3)", (), (), 3),
        ("exp(0)", (), (), 1),
        ("log(exp(2))", (), (), 2),
        ("pow(2,10)", (), (), 1024),
        ("max(1,5,3)", (), (), 5),
        ("min(1,5,3)", (), (), 1),
        ("atan2(1,1)", (), (), math.pi / 4),
        ("TMath::Sqrt(25)", (), (), 5),
        ("TMath::Abs(-7)", (), (), 7),
        ("pi", (), (), math.pi),
        ("pi()", (), (), math.pi),
        ("TMath::Pi()", (), (), math.pi),
        ("sin(pi/2)", (), (), 1),
        ("sign(-2)", (), (), -1),
        ("TMath::Sign(3,-1)", (), (), -3),
        ("binomial(5,2)", (), (), 10),
        ("TMath::Binomial(2,5)", (), (), 0),
        ("1<2", (), (), 1),
        ("2<1", (), (), 0),
        ("1<=1", (), (), 1),
        ("2>=3", (), (), 0),
        ("1==1", (), (), 1),
        ("1!=1", (), (), 0),
        ("1<2 && 3>2", (), (), 1),
        ("1>2 || 3>2", (), (), 1),
        ("!0", (), (), 1),
        ("+x", (), (2,), 2),
        ("x>0 ? 1 : -1", (), (5,), 1),
        ("x>0 ? 1 : -1", (), (-5,), -1),
        ("e", (), (), math.e),
        ("ln10+loge+sqrt2+eg+c*0+infinity*0+true-false", (), (), None),
        ("gaus", (2, 0, 1), (0,), 2),
        ("expo", (0, 1), (2,), math.exp(2)),
        ("pol0", (7,), (99,), 7),
        ("landau", (2, 0, 1), (LANDAU_PEAK,), 2 * LANDAU_TOP),
        ("TMath::Landau(x)", (), (LANDAU_PEAK,), LANDAU_TOP),
        ("TMath::Landau(x,0,2)", (), (2 * LANDAU_PEAK,), LANDAU_TOP),
        ("TMath::Landau(x,0,2,1)", (), (2 * LANDAU_PEAK,), LANDAU_TOP / 2),
        ("landaun", (2, 0, 2), (2 * LANDAU_PEAK,), LANDAU_TOP),
    ],
)
def test_a_formula_evaluates_to_what_root_evaluates_it_to(formula, params, at, expected):
    f = Function("f", formula, parameters=list(params) or None)
    got = f(*(at or (0.0,) * f.dimensions))
    if expected is None:  # the constants are their TMath values, except infinity * 0
        assert math.isnan(got)
        return
    assert got == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("formula", "complaint"),
    [
        ("1+", "a number, a parameter"),
        ("(1+2", "was expected at character 4"),
        ("sqrt(", "a number"),
        ("sqrt(1,2)", "takes 1 arguments"),
        ("nosuchfunc(1)", "not a function TFormula knows"),
        ("x $ 2", "not part of any formula"),
        ("y[0]", "is not a variable"),
        ("x[7]", "is not a variable"),
        ("x>0 ? 1", "':' was expected"),
        ("mystery", "not a variable, a constant or a function"),
        ("", "empty formula"),
        ("1)", "an operator or the end"),
    ],
)
def test_a_formula_that_is_not_one_says_what_is_wrong_with_it(formula, complaint):
    with pytest.raises(FormulaError, match=complaint):
        Function("f", formula)


def test_the_physical_constants_and_a_fourth_variable_are_refused_by_name():
    with pytest.raises(UnsupportedFeatureError, match="physical constants"):
        Function("f", "h*x")
    with pytest.raises(UnsupportedFeatureError, match="no TF4"):
        Function("f", "x*t")


def test_the_parser_reads_a_numbered_parameter_and_steps_over_spaces():
    from xrdroot.function.nodes import Env

    tree = parse("[0] * 2 + [a]", {"a": 1})
    assert tree.evaluate(Env([np.zeros(1)], np.array([3.0, 1.0]))) == 7.0


def test_a_parameter_name_the_formula_was_not_built_with_is_refused():
    with pytest.raises(FormulaError, match=r"\[mean\] .* it has \[a\]"):
        parse("[a]*[mean]", {"a": 0})
    with pytest.raises(FormulaError, match="none by name"):
        parse("[mean]", {})


# -- the special functions --------------------------------------------------------------


def test_tmath_gaus_has_roots_edges_a_width_of_zero_and_a_tail_that_stops():
    assert special.gaus(3.0, 3.0, 0.0) == 1e30
    assert special.gaus(40.0, 0.0, 1.0) == 0.0
    assert special.gaus(1.0, 0.0, 1.0, 1) == pytest.approx(math.exp(-0.5) / math.sqrt(2 * math.pi))


def test_the_landau_density_peaks_where_it_should_and_needs_a_positive_width():
    assert special.landau(LANDAU_PEAK) == pytest.approx(LANDAU_TOP, rel=1e-7)
    assert special.landau(1.0, 0.0, 0.0) == 0.0
    assert special.landau_pdf(1.0, 0.0) == 0.0
    # Every region of DENLAN, against the density Landau defined by an integral.
    for point, density in ((-6.0, 1.7051e-64), (-8.0, 0.0), (6.0, 0.029558712), (1000.0, 1.012e-6)):
        assert special.landau_pdf(point) == pytest.approx(density, rel=1e-3, abs=1e-300)


def test_the_normalised_densities_integrate_to_one():
    for formula, params in (
        ("gausn", [1, 0.5, 2]),
        ("[0]*ROOT::Math::breitwigner_pdf(x,[1],[2])", [1, 2, 0.5]),
        ("crystalballn", [1, 0, 1, 1.5, 3]),
        ("ROOT::Math::gaussian_pdf(x,[0],[1])", [2, 1]),
    ):
        f = Function("f", formula, parameters=params, range=(-math.inf, math.inf))
        assert f.integral(-math.inf, math.inf, epsrel=1e-9) == pytest.approx(1.0, rel=1e-6), formula


def test_the_crystal_ball_tail_joins_its_core_and_takes_either_side():
    left = Function("f", "crystalball", parameters=[1, 0, 1, 1.5, 3])
    assert left(-1.5) == pytest.approx(math.exp(-0.5 * 1.5**2), rel=1e-12)
    assert left(-3.0) > math.exp(-4.5)  # the tail falls slower than the Gaussian
    right = Function("f", "crystalball", parameters=[1, 0, 1, -1.5, 3])
    assert right(3.0) == pytest.approx(left(-3.0), rel=1e-12)
    assert special.crystalball_function(0.0, 1.0, 2.0, -1.0) == 0.0
    assert math.isnan(special.crystalball_pdf(0.0, 1.0, 1.0, 1.0))
    assert special.crystalball_pdf(0.0, 1.0, 2.0, -1.0) == 0.0


def test_the_breit_wigners_are_roots_cauchy_densities():
    gamma = 2.0
    assert special.breit_wigner(1.0, 1.0, gamma) == pytest.approx(2 / (math.pi * gamma))
    assert Function("f", "breitwigner", parameters=[1, 1, gamma])(1.0) == pytest.approx(
        2 / (math.pi * gamma)
    )


def test_the_correlated_gaussian_peaks_at_its_normalisation():
    f = Function("f", "bigaus", parameters=[1, 1, 2, -1, 3, 0.5])
    assert f(1.0, -1.0) == pytest.approx(1 / (2 * math.pi * 2 * 3 * math.sqrt(0.75)))
