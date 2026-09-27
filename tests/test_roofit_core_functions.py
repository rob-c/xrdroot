"""RooFit's functions that are not densities: ``RooFormulaVar``, ``RooPolyVar``, products, sums.

Every value came from ROOT 6.40 on the same variables: ``x = 1.5`` in
``[-2, 3]``, ``a = 2``, ``b = -0.5``. Where xrdroot's printing or a default
still differs from ROOT's, the test says what ROOT prints and is marked as
an expected failure until the engine catches up.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from xrdroot.roofit.functions import RooAddition, RooFormulaVar, RooPolyVar, RooProduct
from xrdroot.roofit.variables import RooConstVar, RooRealVar


def variables() -> tuple[RooRealVar, RooRealVar, RooRealVar]:
    x = RooRealVar("x", "x", 1.5, -2, 3)
    a = RooRealVar("a", "a", 2.0)
    b = RooRealVar("b", "b", -0.5)
    return x, a, b


def test_a_formula_var_evaluates_its_formula_as_root_does() -> None:
    """``RooFormulaVar("f", "f", "a*x+b", [a, x, b])`` is 2.5 at these values in ROOT."""
    x, a, b = variables()
    f = RooFormulaVar("f", "f", "a*x+b", [a, x, b])
    assert f.getVal() == 2.5
    assert f.expression() == "a*x+b"
    assert [one.GetName() for one in f.dependents_list()] == ["a", "x", "b"]
    np.testing.assert_array_equal(f.compute({"x": np.array([0.0, 1.0])}), [-0.5, 1.5])


def test_a_formula_var_without_a_formula_takes_its_title_as_the_formula() -> None:
    """``RooFormulaVar("f2", "x*@0", [x])`` is ROOT's shorter form, and is 2.25."""
    x, _, _ = variables()
    assert RooFormulaVar("f2", "x*@0", [x]).getVal() == 2.25
    assert RooFormulaVar("f4", "2*3").getVal() == 6.0


def test_a_formula_var_serves_only_the_variables_it_uses(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT's ``f3`` over ``(a, x)`` using only ``a`` prints and depends on ``a`` alone."""
    x, a, _ = variables()
    f3 = RooFormulaVar("f3", "f3", "a*a", [a, x])
    assert f3.getVal() == 4.0
    f3.Print("t")
    out = re.sub(r"0x[0-9a-f]+", "@", capsys.readouterr().out)
    assert out == "@ RooFormulaVar::f3 = 4 [Auto,Clean] \n  @/V- RooRealVar::a = 2\n"


@pytest.mark.xfail(strict=True, reason="ROOT prints the formula as it handed it to TFormula")
def test_a_formula_var_prints_its_formula_in_tformulas_numbered_form(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT 6.40 prints ``formula="x[0]*x[1]+x[2]"`` for ``a*x+b``."""
    x, a, b = variables()
    RooFormulaVar("f", "f", "a*x+b", [a, x, b]).Print()
    assert capsys.readouterr().out == (
        'RooFormulaVar::f[ actualVars=(a,x,b) formula="x[0]*x[1]+x[2]" ] = 2.5\n'
    )


def test_a_poly_var_and_its_integral_match_root(capsys: pytest.CaptureFixture[str]) -> None:
    """``a x + b x^2 + 0.25 x^3`` from order one: 2.71875 at 1.5, as ROOT gives."""
    x, a, b = variables()
    p = RooPolyVar("p", "p", x, [a, b, RooConstVar("0.25", "0.25", 0.25)], 1)
    assert p.getVal() == 2.71875
    p.Print()
    assert capsys.readouterr().out == "RooPolyVar::p[ x=x coefList=(a,b,0.25) ] = 2.71875\n"
    assert p.analytic_names(frozenset(["x", "a"]), None) == frozenset(["x"])
    assert p.analytic(frozenset(["x"]), {}, None) == pytest.approx(3.229166666666667, rel=1e-15)
    x.setRange("r", -1.0, 2.0)
    assert p.analytic(frozenset(["x"]), {}, "r") == pytest.approx(2.4375, rel=1e-15)


def test_a_poly_var_without_coefficients_is_zero() -> None:
    """ROOT's ``RooPolyVar`` with an empty list is 0, not one as a density would be."""
    x, _, _ = variables()
    assert RooPolyVar("p0", "p0", x).getVal() == 0.0
    assert RooPolyVar("pn", "pn", x, [2.0], -3).getVal() == 2.0


def test_a_product_multiplies_its_terms() -> None:
    """``RooProduct(a, x, b)`` is -1.5 in ROOT."""
    x, a, b = variables()
    assert RooProduct("pr", "pr", [a, x, b]).getVal() == -1.5


def test_an_addition_sums_its_terms_or_the_products_of_two_lists() -> None:
    """``RooAddition(a, x, b)`` is 3; ``RooAddition((a, x), (b, a))`` is ``ab + xa`` = 2."""
    x, a, b = variables()
    assert RooAddition("ad", "ad", [a, x, b]).getVal() == 3.0
    assert RooAddition("ad2", "ad2", [a, x], [b, a]).getVal() == 2.0
    assert RooAddition("none", "none", []).getVal() == 0.0
    column = RooAddition("col", "col", [a, x]).compute({"x": np.array([0.0, 1.0])})
    np.testing.assert_array_equal(column, [2.0, 3.0])


@pytest.mark.xfail(strict=True, reason="ROOT's level is 1.0 unless the sum holds a RooNLLVar")
def test_an_addition_without_a_likelihood_has_an_error_level_of_one() -> None:
    """ROOT 6.40 answers 1.0, and logs that the sum holds neither an NLL nor a chi2."""
    x, a, b = variables()
    assert RooAddition("ad", "ad", [a, x, b]).defaultErrorLevel() == 1.0


@pytest.mark.xfail(strict=True, reason="ROOT prints a product's and a sum's terms with operators")
def test_a_product_and_a_sum_print_their_terms_joined_by_their_operator(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT 6.40 prints ``RooProduct::pr[ a * x * b ]`` and ``RooAddition::ad[ a + x + b ]``."""
    x, a, b = variables()
    RooProduct("pr", "pr", [a, x, b]).Print()
    RooAddition("ad", "ad", [a, x, b]).Print()
    assert capsys.readouterr().out == (
        "RooProduct::pr[ a * x * b ] = -1.5\nRooAddition::ad[ a + x + b ] = 3\n"
    )
