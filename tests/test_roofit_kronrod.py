"""ROOT 6.40.04's embedded rules and adaptive subdivision, including error estimates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from refmachine import roots
from xrdroot.random import libm
from xrdroot.roofit.kronrod import integrate, rule


def integrand(kind):
    return (
        lambda x: np.ones_like(x),
        lambda x: x * x,
        lambda x: libm.exp(-100 * x * x),
        lambda x: np.abs(x - 0.123),
        lambda x: libm.sin(100 * x),
        libm.exp,
    )[kind]


CASES = json.loads((Path(__file__).parent / "data/roofit-kronrod-6.40.json").read_text())


@pytest.mark.parametrize("case", CASES)
def test_each_embedded_rule_and_its_adaptive_integral_match_root(case):
    fn = integrand(case["kind"])
    estimate = rule(fn, case["a"], case["b"], case["points"])
    result = integrate(fn, case["a"], case["b"], 1e-9, 1e-9, 100, case["points"])
    # Other libm implementations round exp/sin/pow differently. Error
    # estimators magnify cancellation between the Gauss and Kronrod sums.
    assert estimate == roots(tuple(case["rule"]), rel=2e-8, abs=2e-13)
    assert (result.value, result.error) == roots(tuple(case["integral"][:2]), rel=2e-8, abs=2e-13)
    assert result.status == "success"
    assert result.evaluations >= case["points"]


def test_invalid_accuracy_is_reported_before_evaluating_the_function():
    result = integrate(lambda x: pytest.fail("the integrand must not run"), 0.0, 1.0, 0.0, 0.0)
    assert result == (0.0, 0.0, "tolerance", 0)


def test_an_empty_workspace_is_rejected():
    with pytest.raises(ValueError, match="at least one integration segment"):
        integrate(lambda x: x, 0.0, 1.0, limit=0)


def test_roundoff_prevents_accuracy_below_machine_precision():
    result = integrate(lambda x: np.ones_like(x), 0.0, 1.0, 1e-30, 0.0)
    assert result.value == roots(1.0, rel=1e-15)
    assert result.status == "roundoff" and result.evaluations == 21


@pytest.mark.parametrize("limit", (1, 2, 3, 4, 5))
def test_a_subdivision_limit_retains_the_best_estimate(limit):
    result = integrate(lambda x: np.abs(x - 0.123), -1.0, 1.0, 1e-20, 1e-20, limit)
    assert result.status == "limit"
    assert result.value == pytest.approx(1.015129, rel=0.02)
    assert result.evaluations == 21 * (2 * limit - 1)


def test_a_reversed_interval_reverses_the_value_but_keeps_errors_positive():
    forward = rule(libm.exp, -1.0, 2.0)
    backward = rule(libm.exp, 2.0, -1.0)
    assert backward.value == -forward.value
    assert backward.error == forward.error
    assert backward.absolute == forward.absolute
    assert backward.deviation == forward.deviation


def test_a_zero_integrand_has_no_error():
    assert integrate(lambda x: np.zeros_like(x), -1.0, 1.0) == (0.0, 0.0, "success", 21)


def test_roundoff_after_subdivision_keeps_the_converged_cusp_integral():
    result = integrate(lambda x: np.abs(x - 0.123), -1.0, 1.0, 1e-30, 0.0)
    assert result.value == roots(1.015129, rel=1e-14)
    assert result.status == "roundoff" and result.evaluations > 21


def test_an_interval_too_small_to_resolve_a_jump_reports_the_bad_integrand():
    high = 1.0 + 10 * np.finfo(float).eps
    middle = 0.5 * (1.0 + high)
    result = integrate(lambda x: np.where(x < middle, 0.0, 1.0), 1.0, high, 1e-40, 0.0)
    assert result.status == "bad-integrand" and result.evaluations == 63
    assert result.value == 0.5 * (high - 1.0)
