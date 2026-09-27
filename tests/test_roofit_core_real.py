"""``RooAbsReal``: a node with a value - its units, labels, integrals, histograms and curves.

The printed lines, the histogram's contents and the integral are ROOT
6.40.04's, for the same ``RooFormulaVar("f", "the f", "x*x", x)`` with ``x``
from 0 to 4 in ``cm``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.functions import RooFormulaVar
from xrdroot.roofit.integral import RooCdf
from xrdroot.roofit.real import RooAbsReal, names_in, value_of
from xrdroot.roofit.variables import RooRealVar


def square() -> tuple[RooRealVar, RooFormulaVar]:
    x = RooRealVar("x", "x", 1, 0, 4, "cm")
    return x, RooFormulaVar("f", "the f", "x*x", [x])


def test_a_computed_value_is_read_as_its_first_element() -> None:
    """A value computed for one point may come back as an array: ``value_of`` makes it a float."""
    assert value_of(np.array([[2.5, 3.0]])) == 2.5
    assert value_of(4) == 4.0


def test_a_normalisation_set_is_read_as_its_names_or_none() -> None:
    """``getVal(nset)`` normalises over the names in ``nset``; with none given, over nothing."""
    x, _ = square()
    assert names_in(None) is None
    assert names_in([x]) == frozenset({"x"})


def test_a_bare_real_has_no_value_and_no_closed_form_integral() -> None:
    """A class that says neither how to compute itself nor its integral must say so loudly."""
    bare = RooAbsReal("bare", "bare")
    with pytest.raises(NotImplementedError):
        bare.compute({})
    with pytest.raises(NotImplementedError):
        bare.analytic(frozenset({"x"}), {}, None)
    assert bare.analytic_names(frozenset({"x"}), None) == frozenset()
    assert bare.integral_code(frozenset({"x"})) == 1
    assert bare.isValueDirty()


def test_a_function_prints_its_value_as_root_does(capsys: pytest.CaptureFixture[str]) -> None:
    """``Print()`` and ``Print("v")`` of a formula: ROOT's one line, and its ``RooAbsReal`` part."""
    _, f = square()
    f.Print()
    f.Print("v")
    assert capsys.readouterr().out == (
        'RooFormulaVar::f[ actualVars=(x) formula="x*x" ] = 1\n'
        '--- RooAbsReal ---\n\n  Plot label is "f"\n'
    )
    assert f.printValue() == "1"
    assert float(f) == 1.0


def test_a_title_shows_the_unit_only_when_asked_and_only_if_there_is_one() -> None:
    """``getTitle(true)`` appends ``(unit)``: an axis title is made this way."""
    _, f = square()
    assert f.getTitle(True) == "the f"
    assert f.getUnit() == ""
    f.setUnit("cm^2")
    assert f.getTitle() == "the f"
    assert f.getTitle(True) == "the f (cm^2)"


def test_a_plot_label_is_the_name_until_one_is_set() -> None:
    """The legend of a curve reads the plot label: the name, unless given another."""
    _, f = square()
    assert f.getPlotLabel() == "f"
    f.setPlotLabel("f(x)")
    assert f.getPlotLabel() == "f(x)"


def test_a_function_knows_the_range_of_each_variable_under_it() -> None:
    """``bounds`` finds a variable by name and reads the ends of the range asked for."""
    x, f = square()
    x.setRange("low", 0.0, 1.5)
    assert f.variable("x") is x
    assert f.bounds("x", None) == (0.0, 4.0)
    assert f.bounds("x", "low") == (0.0, 1.5)


def test_a_function_integrated_over_its_variable_is_roots_number() -> None:
    """The integral of ``x*x`` from 0 to 4 is 64/3, by ``createIntegral`` or ``integrate``."""
    x, f = square()
    assert f.createIntegral([x]).getVal() == pytest.approx(21.3333333333, abs=1e-10)
    assert float(np.asarray(f.integrate({"x"}, {}))) == pytest.approx(64 / 3, rel=1e-12)


def test_a_function_histogram_holds_its_value_at_each_bin_centre() -> None:
    """``createHistogram("h", x, Binning(4))`` of ``x*x``: ROOT's 0.25, 2.25, 6.25, 12.25."""
    x, f = square()
    made: Any = f.createHistogram("h", x, RooCmdArg("Binning", 4))
    assert list(made.values()) == [0.25, 2.25, 6.25, 12.25]


def test_a_function_can_make_its_cumulative_distribution() -> None:
    """``createCdf`` hands back the integral up to each variable's value, over those asked for."""
    x, f = square()
    cdf = f.createCdf([x])
    assert isinstance(cdf, RooCdf)


def test_a_function_plotted_on_a_frame_adds_one_curve() -> None:
    """``plotOn`` puts the function's curve on the frame, as ROOT's one item."""
    x, f = square()
    frame = x.frame()
    f.plotOn(frame)
    assert frame.numItems() == 1
