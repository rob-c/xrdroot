"""RooFit's numerical integrals: Romberg, the open-ended transform and the choice between them.

The expected values are what ROOT 6.40.04 printed for ``createIntegral(...)
.getVal()`` on a ``RooFormulaVar`` or ``RooGaussian`` built the same way,
with ``repr``. Most agree to the last bit; where the order ROOT adds its
terms in differs in the last place, the comparison allows 1e-14.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from xrdroot.roofit import integration
from xrdroot.roofit.collections import RooArgList, RooArgSet
from xrdroot.roofit.functions import RooFormulaVar
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.variables import RooRealVar


def open_line() -> tuple[RooRealVar, RooFormulaVar]:
    """``exp(-x*x)`` over a variable open at both ends, with half-open ranges beside."""
    x = RooRealVar("x", "x", 0.0, -math.inf, math.inf)
    x.setRange("lo1", -math.inf, 1.0)
    x.setRange("lom", -math.inf, -1.0)
    x.setRange("hi", 0.5, math.inf)
    x.setRange("hin", -2.0, math.inf)
    return x, RooFormulaVar("f", "f", "exp(-x*x)", RooArgList(x))


@pytest.mark.parametrize(
    ("rng", "expected"),
    [
        (None, 1.772453854622262),
        ("lo1", 1.6330510604062254),
        ("lom", 0.13940279421603663),
        ("hi", 0.4249464477663606),
        ("hin", 1.7683082906184777),
    ],
)
def test_an_open_ended_integral_maps_each_open_end_as_roots_improper_integrator(
    rng: Any, expected: float
) -> None:
    """Each of the four ways a range can be open is split at +-1 and inverted as ROOT does."""
    _, f = open_line()
    found = f.integrate(frozenset({"x"}), {}, rng)
    assert float(found) == pytest.approx(expected, rel=1e-14)


def test_an_open_ended_integral_is_announced_as_the_improper_integrator(capsys: Any) -> None:
    """RooFit names the integrator it picked when an integral is made."""
    x, f = open_line()
    capsys.readouterr()
    f.createIntegral(RooArgSet(x), "hi")
    assert capsys.readouterr().out == (
        "[#1] INFO:NumericIntegration -- RooRealIntegral::init(f_Int[x|hi]) using numeric "
        "integrator RooImproperIntegrator1D to calculate Int(x)\n"
    )


def test_a_closed_range_is_integrated_by_romberg() -> None:
    """A step and a cubic together: the trapezoid rule, extrapolated, gives ROOT's number."""
    y = RooRealVar("y", "y", 1.0, -5.0, 5.0)
    step = RooFormulaVar("st", "st", "(y<0.3)*1.0+(y*y*y)", RooArgList(y))
    assert float(step.integrate(frozenset({"y"}), {})) == pytest.approx(
        5.300105683670066, rel=1e-14
    )


def test_a_range_of_zero_width_integrates_to_zero() -> None:
    """ROOT returns 0 for a range whose ends meet, without evaluating a refinement."""
    y = RooRealVar("y", "y", 1.0, -5.0, 5.0)
    y.setRange("z", 1.0, 1.0)
    step = RooFormulaVar("st", "st", "(y<0.3)*1.0+(y*y*y)", RooArgList(y))
    assert float(step.integrate(frozenset({"y"}), {}, "z")) == 0.0
    assert float(integration.romberg(lambda p: p * p, 2.0, 2.0)) == 0.0


def test_a_range_of_several_parts_adds_the_integral_over_each() -> None:
    """``Range("a,b")`` is the sum over ``a`` and over ``b``, as ROOT's 6.6666666666666661."""
    y = RooRealVar("y", "y", 1.0, -5.0, 5.0)
    y.setRange("a", -1.0, 0.0)
    y.setRange("b", 2.0, 3.0)
    square = RooFormulaVar("cube", "cube", "y*y", RooArgList(y))
    assert float(square.integrate(frozenset({"y"}), {}, "a,b")) == 6.666666666666666


def test_what_has_no_closed_form_is_integrated_numerically_over_the_closed_form() -> None:
    """A Gaussian over ``y`` and ``sigma``: analytic in ``y``, Romberg in ``sigma``."""
    y = RooRealVar("y", "y", 1.0, -5.0, 5.0)
    m = RooRealVar("m", "m", 0.0)
    s = RooRealVar("s", "s", 1.0, 0.5, 2.0)
    g = RooGaussian("g", "g", y, m, s)
    assert integration.numeric_names(g, frozenset({"y", "s"})) == ["s"]
    assert float(g.integrate(frozenset({"y", "s"}), {})) == pytest.approx(
        4.687903024730919, rel=1e-14
    )
    assert float(g.integrate(frozenset({"y"}), {})) == 2.50662683757313


def test_an_integral_over_nothing_the_function_depends_on_is_its_value() -> None:
    """With no variable left to integrate, the integral is the function itself."""
    y = RooRealVar("y", "y", 1.0, -5.0, 5.0)
    square = RooFormulaVar("sq", "sq", "y*y", RooArgList(y))
    assert float(square.integrate(frozenset({"z"}), {})) == 1.0


def three_variables() -> RooFormulaVar:
    q = RooRealVar("q", "q", 1.0, 0.0, 3.0)
    y = RooRealVar("y", "y", 1.0, -5.0, 5.0)
    z = RooRealVar("z", "z", 0.5, 0.0, 2.0)
    return RooFormulaVar("three", "three", "q*z*z+y*y", RooArgList(q, y, z))


def test_two_closed_dimensions_are_integrated_by_cubature_once_per_event() -> None:
    """ROOT's ``RooAdaptiveIntegratorND``, for ``q`` = 1 and 2.5: 193.33 and 233.33."""
    three = three_variables()
    assert float(three.integrate(frozenset({"y", "z"}), {})) == 193.3333333333333
    found = three.integrate(frozenset({"y", "z"}), {"q": np.array([1.0, 2.5])})
    assert found.tolist() == [193.3333333333333, 233.33333333333331]


def test_two_closed_dimensions_are_announced_as_the_adaptive_integrator(capsys: Any) -> None:
    """The line names ``RooAdaptiveIntegratorND`` and both variables, in the function's order."""
    three = three_variables()
    capsys.readouterr()
    integration.announce(three, frozenset({"z", "y"}))
    assert capsys.readouterr().out == (
        "[#1] INFO:NumericIntegration -- RooRealIntegral::init(three_Int[y,z]) using numeric "
        "integrator RooAdaptiveIntegratorND to calculate Int(y,z)\n"
    )


def test_an_integral_in_closed_form_is_not_announced(capsys: Any) -> None:
    """Only numerical integrals get a line: a Gaussian over its observable is analytic."""
    y = RooRealVar("y", "y", 1.0, -5.0, 5.0)
    g = RooGaussian("g", "g", y, RooRealVar("m", "m", 0.0), RooRealVar("s", "s", 1.0))
    capsys.readouterr()
    integration.announce(g, frozenset({"y"}))
    assert capsys.readouterr().out == ""


def test_an_open_dimension_among_several_is_integrated_one_variable_inside_another() -> None:
    """ROOT has no integrator for an open 2D integral; here it is nested 1D ones: 4 sqrt(pi)."""
    x = RooRealVar("x", "x", 0.0, -math.inf, math.inf)
    z = RooRealVar("z", "z", 0.5, 0.0, 2.0)
    fz = RooFormulaVar("fz", "fz", "exp(-x*x)*(1+z)", RooArgList(x, z))
    found = fz.integrate(frozenset({"x", "z"}), {"q": np.array([1.0, 2.5])})
    assert float(found) == 7.089815418489046


def spike(points: Any) -> Any:
    """``1/sqrt(|y-0.3|)``: integrable, but too sharp for twenty refinements to settle."""
    return 1.0 / np.sqrt(np.abs(points - 0.3))


def test_an_integral_that_does_not_settle_warns_and_gives_the_last_refinement(
    capsys: Any,
) -> None:
    """ROOT warns after twenty refinements and returns the last one, 2.7671608447712166."""
    found = integration.romberg(spike, 0.0, 1.0, name="spike")
    assert float(found) == pytest.approx(2.7671608447712166, rel=1e-14)
    assert capsys.readouterr().out.splitlines()[0] == (
        "[#0] WARNING:Integration -- RooRombergIntegrator::integral: integral of spike over "
        "range (0,1) did not converge after 20 steps"
    )


@pytest.mark.xfail(strict=True, reason="the warning does not list each refinement as ROOT's does")
def test_an_integral_that_does_not_settle_lists_every_refinement_as_root_does(
    capsys: Any,
) -> None:
    """ROOT follows the warning with ``[j] h = ... , s = ...`` for each of the twenty steps."""
    integration.romberg(spike, 0.0, 1.0, name="spike")
    lines = capsys.readouterr().out.splitlines()
    assert lines[1:3] == ["   [1] h = 1 , s = 1.51049", "   [2] h = 0.25 , s = 1.87328"]
    assert lines[-1] == "   [20] h = 3.63798e-12 , s = 2.76716"
