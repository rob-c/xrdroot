"""Literal squares retain TFormula's multiplication order and IEEE values."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from xrdroot import Function
from xrdroot.function import library


def assert_same_bits(found, expected):
    np.testing.assert_array_equal(
        np.asarray(found, dtype=np.float64).view(np.uint64),
        np.asarray(expected, dtype=np.float64).view(np.uint64),
    )


@pytest.mark.parametrize("operator", ["^", "**"])
def test_a_literal_square_preserves_the_parentheses_of_its_surrounding_product(operator):
    # TFormula::HandleExponentiation emits TMath::Sq(x), which is x*x.
    # Keeping a separate outer multiplication matters to its final rounding.
    x = np.linspace(-17.3, 9.7, 4099)
    function = Function("square", f"[0]*x{operator}2+[1]*x+[2]", parameters=[0.17, -0.31, 0.63])
    expected = 0.17 * (x * x) + (-0.31 * x) + 0.63
    assert_same_bits(function.evaluate(x), expected)


@pytest.mark.parametrize("operator", ["^", "**"])
def test_a_literal_square_has_multiplications_signed_zero_and_overflow(operator):
    x = np.array([-sys.float_info.max, -1.5, -0.0, 0.0, 1.5, sys.float_info.max])
    function = Function("square", f"x{operator}2")
    with np.errstate(over="ignore"):
        expected = x * x
    assert_same_bits(function.evaluate(x), expected)
    assert_same_bits(Function("negative_square", f"-x{operator}2").evaluate(x), -expected)


@pytest.mark.parametrize("formula", ["(x+[0])^2", "(x+[0])**2"])
def test_a_parenthesized_base_is_squared_as_a_whole(formula):
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    function = Function("shifted_square", formula, parameters=[0.5])
    base = x + 0.5
    assert_same_bits(function.evaluate(x), base * base)
    np.testing.assert_array_equal(function.gradient(x)[:, 0], 2.0 * base)


def test_the_gradient_of_a_parameterized_square_contains_both_inner_derivatives():
    # Dyadic values make the analytic derivatives exact on every supported libm.
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    function = Function("affine_square", "([0]*x+[1])^2", parameters=[0.5, 0.25])
    base = 0.5 * x + 0.25
    expected = np.column_stack([x * (2.0 * base), 2.0 * base])
    np.testing.assert_array_equal(function.gradient(x), expected)
    function.fix(0)
    expected[:, 0] = 0.0
    np.testing.assert_array_equal(function.gradient(x), expected)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [("x^2^3", 256.0), ("(x^2)^3", 64.0), ("x^(2^3)", 256.0)],
)
def test_a_square_inside_a_chain_of_powers_keeps_roots_right_association(formula, expected):
    function = Function("powers", formula)
    assert function(2.0) == expected


def test_squaring_a_function_result_evaluates_the_base_once_for_each_point(monkeypatch):
    # A C++ TMath::Sq(f(x)) evaluates f(x) once, unlike the textual rewrite f(x)*f(x).
    # Register an ordinary function whose side effect records the evaluated points.
    seen = []

    def record(x):
        seen.extend(np.atleast_1d(x).tolist())
        return np.asarray(x) + 0.5

    monkeypatch.setitem(library.CALLS, "record_square_base", library.Call(1, 1, record))
    function = Function("one_evaluation", "record_square_base(x)^2")
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    base = x + 0.5
    assert_same_bits(function.evaluate(x), base * base)
    assert seen == x.tolist()
