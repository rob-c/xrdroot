"""``createIntegral`` and ``createCdf``: an integral as a function, named and printed as ROOT's.

Every name, value and message below is what ROOT 6.40.04 printed for the
same Gaussian ``g(y; m=0, s)`` with ``y`` in [-5, 5], through PyROOT, with
a stream added at ``DEBUG`` for the ``Integration`` topic (ROOT's ``DEBUG``
lines about shape and value dependence are not made here, so only the
``INFO`` lines are compared).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.collections import RooArgList, RooArgSet
from xrdroot.roofit.functions import RooFormulaVar
from xrdroot.roofit.messages import DEBUG, SERVICE, TOPICS
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.variables import RooRealVar


@pytest.fixture
def integration_stream() -> Iterator[None]:
    """A fourth stream, ``[#3]``, taking every ``Integration`` message, as the oracle had."""
    SERVICE.saveState()
    SERVICE.addStream(DEBUG, Topic=TOPICS["Integration"])
    yield
    SERVICE.restoreState()


def gaussian() -> tuple[RooRealVar, RooGaussian]:
    y = RooRealVar("y", "y", 1.0, -5.0, 5.0)
    y.setRange("r", -1.0, 2.0)
    m = RooRealVar("m", "m", 0.0)
    s = RooRealVar("s", "s", 1.0, 0.5, 2.0)
    return y, RooGaussian("g", "g", y, m, s)


def test_an_integral_in_closed_form_says_how_it_was_made(
    integration_stream: None, capsys: Any
) -> None:
    """The four ``INFO`` lines of ``RooRealIntegral``'s constructor, word for word."""
    y, g = gaussian()
    capsys.readouterr()
    integral = g.createIntegral(RooArgSet(y))
    assert capsys.readouterr().out == (
        "[#3] INFO:Integration -- RooRealIntegral::ctor(g_Int[y]) Constructing integral of "
        "function g over observables(y) with normalization () with range identifier <none>\n"
        "[#3] INFO:Integration -- g: Observable y is suitable for analytical integration "
        "(if supported by p.d.f)\n"
        "[#3] INFO:Integration -- g: Function integrated observables (y) internally with code 1\n"
        "[#3] INFO:Integration -- g: Observables (y) are analytically integrated with code 1\n"
    )
    assert integral.GetName() == "g_Int[y]"
    assert integral.GetTitle() == "Integral of g"
    assert integral.getVal() == 2.50662683757313
    integral.Print()
    assert capsys.readouterr().out == "RooRealIntegral::g_Int[y][ Int gd[Ana](y) ] = 2.50663\n"


def test_a_normalised_integral_over_a_range_is_named_for_both() -> None:
    """``NormSet`` and ``Range`` as commands, or as a set and a string: ``g_Int[y|r]_Norm[y]``."""
    y, g = gaussian()
    commands = g.createIntegral(
        RooArgSet(y), RooCmdArg("NormSet", RooArgSet(y)), RooCmdArg("Range", "r")
    )
    positional = g.createIntegral(RooArgSet(y), RooArgSet(y), "r")
    keywords = g.createIntegral(RooArgSet(y), NormSet=RooArgSet(y), Range="r")
    for integral in (commands, positional, keywords):
        assert integral.GetName() == "g_Int[y|r]_Norm[y]"
        assert integral.getVal() == 0.8185950834234984


@pytest.mark.xfail(strict=True, reason="the normalisation set is not printed as ROOT's _Norm(y)")
def test_a_normalised_integral_prints_its_normalisation_set(capsys: Any) -> None:
    """ROOT prints ``Int g_Norm(y) d[Ana](y)`` for an integral normalised over ``y``."""
    y, g = gaussian()
    integral = g.createIntegral(RooArgSet(y), RooArgSet(y), "r")
    capsys.readouterr()
    integral.Print()
    assert capsys.readouterr().out == (
        "RooRealIntegral::g_Int[y|r]_Norm[y][ Int g_Norm(y) d[Ana](y) ] = 0.818595\n"
    )


def test_a_numerical_integral_is_marked_num_and_names_its_integrator(capsys: Any) -> None:
    """``y*y + m`` has no closed form here, so the integral is numerical: 83.33 over [-5, 5]."""
    y, g = gaussian()
    m = g.mean
    f = RooFormulaVar("f", "f", "y*y+m", RooArgList(y, m))
    capsys.readouterr()
    integral = f.createIntegral(RooArgSet(y))
    assert capsys.readouterr().out == (
        "[#1] INFO:NumericIntegration -- RooRealIntegral::init(f_Int[y]) using numeric "
        "integrator RooIntegrator1D to calculate Int(y)\n"
    )
    assert integral.getVal() == 83.33333333333333
    assert integral.printArgs().endswith("d[Num](y) ]")


@pytest.mark.xfail(strict=True, reason="ROOT puts a space between the name and d[Num]")
def test_a_numerical_integral_prints_as_roots(capsys: Any) -> None:
    """ROOT prints ``Int f d[Num](y)``: ``" d[Num]"`` starts with a space in its source."""
    y, g = gaussian()
    f = RooFormulaVar("f", "f", "y*y+m", RooArgList(y, g.mean))
    integral = f.createIntegral(RooArgSet(y))
    capsys.readouterr()
    integral.Print()
    assert capsys.readouterr().out == "RooRealIntegral::f_Int[y][ Int f d[Num](y) ] = 83.3333\n"


@pytest.mark.xfail(strict=True, reason="a variable the function does not use is not integrated")
def test_an_integral_over_a_variable_the_function_does_not_use_is_times_its_range() -> None:
    """ROOT factorises such a variable out: ``g`` at y = 1 times the width 2 of ``z``."""
    _, g = gaussian()
    z = RooRealVar("z", "z", 0.5, 0.0, 2.0)
    assert g.createIntegral(RooArgSet(z)).getVal() == 1.2130613194252668


def test_a_cdf_is_the_normalised_integral_up_to_the_value() -> None:
    """``createCdf``: ROOT's 0.1587, 0.5 and 0.9773 at y = -1, 0 and 2."""
    y, g = gaussian()
    cdf = g.createCdf(RooArgSet(y))
    expected = [0.15865505823732884, 0.5, 0.9772501416608274]
    for value, want in zip((-1.0, 0.0, 2.0), expected):
        y.setVal(value)
        assert cdf.getVal() == pytest.approx(want, rel=1e-15)
    found = cdf.compute({"y": np.array([-1.0, 0.0, 2.0])})
    assert found.tolist() == pytest.approx(expected, rel=1e-15)
    assert cdf.printValue() == "0.97725"
    assert "_cdf_" not in y._shared


@pytest.mark.xfail(strict=True, reason="ROOT names a cdf g_cdf_Int[y_prime|CDF]_Norm[y_prime]")
def test_a_cdf_is_named_as_roots() -> None:
    """ROOT's cdf is an integral of a clone ``g_cdf`` over ``y_prime`` in the range ``CDF``."""
    y, g = gaussian()
    assert g.createCdf(RooArgSet(y)).GetName() == "g_cdf_Int[y_prime|CDF]_Norm[y_prime]"
