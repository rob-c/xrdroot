"""``RooPowerSum``: the power law the binned-fit tutorial fits."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT


def test_a_single_term_is_the_power_law_the_tutorial_asks_for():
    x = ROOT.RooRealVar("x", "x", 1.0, 10.0)
    a = ROOT.RooRealVar("a", "a", -0.3, -5.0, 5.0)
    powerlaw = ROOT.RooPowerSum("powerlaw", "powerlaw", x, ROOT.RooFit.RooConst(1.0), a)
    x.setVal(2.0)
    assert powerlaw.getVal() == pytest.approx(2.0**-0.3)
    a.setVal(2.0)
    assert powerlaw.getVal() == pytest.approx(4.0)


def test_several_terms_are_summed_with_their_own_coefficients():
    x = ROOT.RooRealVar("x", "x", 1.0, 10.0)
    coefficients = ROOT.RooArgList(ROOT.RooFit.RooConst(2.0), ROOT.RooFit.RooConst(3.0))
    exponents = ROOT.RooArgList(ROOT.RooFit.RooConst(1.0), ROOT.RooFit.RooConst(0.0))
    added = ROOT.RooPowerSum("added", "2x + 3", x, coefficients, exponents)
    x.setVal(2.0)
    assert added.getVal() == pytest.approx(7.0)


def test_a_term_without_an_exponent_says_so_rather_than_guessing():
    x = ROOT.RooRealVar("x", "x", 1.0, 10.0)
    with pytest.raises(ValueError, match="one exponent for every coefficient"):
        ROOT.RooPowerSum(
            "bad", "bad", x,
            ROOT.RooArgList(ROOT.RooFit.RooConst(1.0), ROOT.RooFit.RooConst(2.0)),
            ROOT.RooFit.RooConst(1.0),
        )
