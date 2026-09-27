"""``RooFormula`` and its compiled closures: the C++ expressions RooFit's formulas are written in.

The values came from ROOT 6.40's ``RooFormulaVar::getVal()`` on the same
formulas and variables. A formula is translated into the language
:mod:`xrdroot.formula` parses - each variable under an alias - and made
once into closures, so these tests check the translation, the closures,
and that single points and whole columns both evaluate.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.compiled import Compiled
from xrdroot.roofit.formula import RooFormula, translate
from xrdroot.roofit.variables import RooRealVar


def variables() -> tuple[RooRealVar, RooRealVar, RooRealVar]:
    x = RooRealVar("x", "x", 1.5, -2, 3)
    a = RooRealVar("a", "a", 2.0)
    b = RooRealVar("b", "b", -0.5)
    return x, a, b


def test_each_kind_of_node_compiles_to_a_closure_over_named_values() -> None:
    """Numbers, names, operators, the ternary, calls and casts are all a formula needs."""
    values = {"x": np.float64(1.5), "y": np.float64(-2.0)}
    assert float(Compiled("2.5", ["x"])(values)) == 2.5
    assert float(Compiled("x", ["x"])(values)) == 1.5
    assert float(Compiled("-x", ["x"])(values)) == -1.5
    assert float(Compiled("x*y+1", ["x", "y"])(values)) == -2.0
    assert float(Compiled("x>0 ? 10 : 20", ["x"])(values)) == 10.0
    assert float(Compiled("y>0 ? 10 : 20", ["y"])(values)) == 20.0
    assert float(Compiled("TMath::Abs(y)", ["y"])(values)) == 2.0
    assert float(Compiled("(int)(x*3)", ["x"])(values)) == 4.0


def test_a_compiled_formula_evaluates_whole_columns_too() -> None:
    """A dataset's column goes through the same closures, one value per event."""
    found = Compiled("x>0 ? x : -x", ["x"])({"x": np.array([-1.0, 2.0])})
    np.testing.assert_array_equal(found, [1.0, 2.0])


def test_tree_draw_only_names_are_refused_in_a_roofit_formula() -> None:
    """``Entry$`` is TTree::Draw's, not a function of RooFit's variables, and says so."""
    with pytest.raises(UnsupportedFeatureError, match="TTree::Draw's language"):
        Compiled("x + Entry$", ["x"])


def test_variables_are_found_by_name_by_at_number_and_by_x_index() -> None:
    """``@0``, ``x[0]`` and the name are three ways to say the first variable."""
    x, a, b = variables()
    text, used = translate("a*x+b + @0 - x[1]", [a, x, b])
    assert text == "_rf0_*_rf1_+_rf2_ + _rf0_ - _rf1_"
    assert used == [0, 1, 2]


def test_a_longer_name_is_replaced_before_a_shorter_one_it_contains() -> None:
    """``xx`` must not be read as ``x`` followed by ``x``."""
    x = RooRealVar("x", "x", 1.0)
    xx = RooRealVar("xx", "xx", 2.0)
    text, used = translate("xx + x", [x, xx])
    assert text == "_rf1_ + _rf0_"
    assert used == [0, 1]


def test_a_category_label_is_its_index_and_other_paths_are_left_alone() -> None:
    """``c::Minus`` is the category's number, while ``TMath::Pi`` stays a function."""
    x, _, _ = variables()
    c = RooCategory("c", "c")
    c.defineType("Plus", 1)
    c.defineType("Minus", -1)
    c.setLabel("Minus")
    formula = RooFormula("(c==c::Minus)*x + TMath::Pi()", [c, x])
    assert formula.text == "(_rf0_==-1)*_rf1_ + TMath::Pi()"
    assert formula.evaluate({}) == pytest.approx(4.641592653589793, rel=1e-15)


def test_a_formula_knows_which_of_its_variables_it_uses() -> None:
    """``RooFormulaVar`` serves only the variables its formula really names, as ROOT's does."""
    x, a, _ = variables()
    formula = RooFormula("a*a", [a, x])
    assert [one.GetName() for one in formula.actual()] == ["a"]
    assert formula.GetTitle() == "a*a"
    assert formula.evaluate({}) == 4.0


def test_a_formula_at_one_point_is_a_float_and_over_a_column_an_array() -> None:
    """``getVal`` wants a number; a likelihood wants every event's value at once."""
    x, a, b = variables()
    formula = RooFormula("a*x+b", [a, x, b])
    point = formula.evaluate({})
    assert isinstance(point, float)
    assert point == 2.5
    column = formula.evaluate({"x": np.array([0.0, 1.0])})
    np.testing.assert_array_equal(column, [-0.5, 1.5])
