"""A function read from a file: the members a ``TF1`` holds, and what it answers from them.

``TF1`` keeps its formula in a ``TFormula`` member, its parameter values beside it, and - for
a function whose code the reading session does not have - the values it was sampled into,
``fSave``. :meth:`Function.from_members` is the way in, so everything here is a plain
dictionary and no file is opened, which keeps ``xrdroot.function.function`` inside the
``hist`` component's own isolation scope.

Two ROOT behaviours run through the file. A **layout ROOT 5 wrote** - a ``TF1`` that *was* a
``TFormula`` rather than holding one - stays the dictionary it was read as, so a histogram
hanging off it still reads; and a formula or shape this kit cannot work out is **kept, not
refused**, so the function still reads and writes back as it came, and only evaluating it
says why not.
"""

from __future__ import annotations

import array
import copy
import math

import numpy as np
import pytest

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.function import numeric
from xrdroot.function.analytic import integral as closed_form
from xrdroot.function.function import Function, dress
from xrdroot.random import TRandom3

#: A ``TF1`` made in code, which is what a file of one holds once it has been written.
DECAY = Function("decay", "[N]*exp(-x/[tau])", range=(0, 10), parameters=[100, 2])


def _a_file_row() -> dict[str, object]:
    """The members a ``TF1`` of ``[N]*exp(-x/[tau])`` is written with, as a file gives them."""
    return copy.deepcopy(DECAY.members)  # type: ignore[return-value]


def test_a_tf1_is_read_back_from_the_members_a_file_holds():
    """The formula, its parameter values and its range are all in the members, so a function
    read from them answers what the one that was written answered."""
    found = Function.from_members("TF1", _a_file_row())
    assert isinstance(found, Function)
    assert found.classname == "TF1"
    assert found.formula == "[N]*exp(-x/[tau])"
    assert found.parameter_names == ("N", "tau")
    assert round(float(found(2.0)), 6) == 36.787944  # the same number ROOT prints


def test_a_layout_root_five_wrote_stays_the_dictionary_it_was_read_as():
    """A ``TF1`` from ROOT 5 was a ``TFormula`` itself and holds none, and a ``TFormula`` with
    no ``fClingParameters`` is no formula this kit can make: both stay dictionaries, so the
    histogram they hang off still reads."""
    old = _a_file_row()
    old.pop("fFormula")
    assert isinstance(Function.from_members("TF1", old), dict)
    formula = copy.deepcopy(_a_file_row()["fFormula"])  # type: ignore[index]
    formula.pop("fClingParameters")
    assert isinstance(Function.from_members("TFormula", formula), dict)


def test_a_tformula_read_from_a_file_is_a_function_of_its_own():
    """A ``TFormula`` is a function in its own right - ``TF1``'s formula member is one - and its
    parameters, names and value come from the members it was written with."""
    formula = copy.deepcopy(_a_file_row()["fFormula"])  # type: ignore[index]
    found = Function.from_members("TFormula", formula)
    assert isinstance(found, Function)
    assert found.classname == "TFormula"
    assert found.formula == "[N]*exp(-x/[tau])"
    assert found.parameter_names == ("N", "tau")
    assert round(float(found(2.0)), 6) == 36.787944
    assert found.range == (0.0, 1.0)  # a TFormula has no range, so it gets ROOT's default one


def test_a_tformula_takes_the_parameter_values_a_file_gave_it_even_as_a_plain_array():
    """A file's array of doubles may be an ``array.array`` rather than NumPy's, and asking for
    the parameters makes one array of doubles that is the member - so a change to it changes
    the function."""
    formula = copy.deepcopy(_a_file_row()["fFormula"])  # type: ignore[index]
    formula["fClingParameters"] = array.array("d", [100.0, 2.0])
    found = Function.from_members("TFormula", formula)
    assert found.npar == 2
    assert found.parameters.tolist() == [100.0, 2.0]
    found.parameters[0] = 200.0
    assert round(float(found(2.0)), 6) == 73.575888


def test_a_formula_makes_a_tformula_and_a_function_of_code_cannot():
    """``TF1::GetFormula`` is the formula a function computes, which is what ROOT writes for a
    ``TFormula`` made by name; a function of code has none to give."""
    made = DECAY.as_formula()
    assert made.classname == "TFormula"
    assert made.formula == DECAY.formula
    assert round(float(made(2.0)), 6) == 36.787944
    code = Function.from_callable("code", lambda x, params: params[0] * x, 1, parameters=[2.0])
    with pytest.raises(UnsupportedFeatureError, match="computed by code rather than by a formula"):
        code.as_formula()


def test_a_formula_this_kit_cannot_parse_still_reads_and_says_why_it_cannot_be_evaluated():
    """A file may hold a formula calling something this kit does not have. The function still
    reads - its members are the file's - and evaluating it refuses in a sentence quoting the
    reason, rather than failing at the read."""
    row = _a_file_row()
    formula = copy.deepcopy(row["fFormula"])  # type: ignore[assignment]
    formula["fFormula"] = "someCode(x)"
    formula["fParams"] = {}
    row["fFormula"] = formula
    found = Function.from_members("TF1", row)
    assert isinstance(found, Function)
    assert found.formula == "someCode(x)"
    assert "is not a function TFormula knows here" in found._problem
    with pytest.raises(UnsupportedFeatureError, match="holds no saved values to fall back on"):
        found(1.0)


def test_a_function_of_compiled_code_reads_back_from_the_values_that_were_saved():
    """ROOT writes a function it has code for as its values at ``fNpx + 1`` points, and a
    function of Python code goes the same way - so it reads back here, and in ROOT, as those."""
    code = Function.from_callable(
        "line", lambda x, params: params[0] * x, 1, range=(0, 4), parameters=[2.0]
    )
    written = code.written_members()
    # a value at each of fNpx + 1 points, then the two ends they span
    assert len(written["fSave"]) == code._npx() + 3
    assert written["fSave"][-2:].tolist() == [0.0, 4.0]  # the ends, as ROOT writes them
    found = Function.from_members("TF1", copy.deepcopy(written))
    assert isinstance(found, Function)
    assert found.formula is None  # there is no formula, only the values
    assert round(float(found(3.0)), 6) == 6.0


def test_saved_values_of_two_variables_are_a_grid_this_reader_does_not_interpolate():
    """One variable's saved values are a line, which interpolates; a ``TF2``'s are a grid, which
    this reader does not, and the refusal says so rather than reading the first row."""
    plane = Function.from_callable(
        "plane",
        lambda points, params: params[0] * points[:, 0],
        1,
        dimensions=2,
        range=((0, 1), (0, 1)),
        parameters=[2.0],
    )
    row = copy.deepcopy(plane.members)
    row["fFormula"] = None  # a function of code: no formula, only what was saved
    row["fSave"] = np.array([1.0, 2.0])
    found = Function.from_members("TF2", row)
    with pytest.raises(UnsupportedFeatureError, match="a grid this reader does not interpolate"):
        found([0.25], [0.5])  # two coordinates, each one point


def test_a_named_parameter_the_formula_has_no_index_for_is_passed_over():
    """A formula's ``fParams`` maps a name to the parameter's index; a row whose index is past
    the parameters it has leaves that one unnamed rather than failing the read."""
    formula = copy.deepcopy(_a_file_row()["fFormula"])  # type: ignore[index]
    formula["fParams"] = {"N": 0, "tau": 5}  # the formula has two parameters, not six
    found = Function.from_members("TFormula", formula)
    assert found.parameter_names == ("N", "p1")


def test_the_arrays_of_a_value_per_parameter_are_made_from_what_a_file_held():
    """``fParErrors`` and its kin are one value per parameter, and a file's may be a plain list:
    asking for one makes an array of doubles of the length the function has."""
    row = _a_file_row()
    row["fParErrors"] = [1.5, 2.5]
    found = Function.from_members("TF1", row)
    assert isinstance(found, Function)
    assert found.parameter_errors.tolist() == [1.5, 2.5]
    found.parameter_errors = [3.0, 4.0]
    assert found.parameter_errors.tolist() == [3.0, 4.0]


def test_a_row_with_no_parameter_record_has_no_parameters():
    """A function of code keeps its values in a ``TF1Parameters`` record; a row without one
    makes an empty record rather than failing, so what it holds is no parameters at all."""
    code = Function.from_callable("code", lambda x, params: params[0] * x, 1, parameters=[2.0])
    row = code.written_members()
    row["fParams"] = None
    found = Function.from_members("TF1", row)
    assert isinstance(found, Function)
    assert found.npar == 0
    assert found.parameters.tolist() == []


def test_a_shape_with_no_closed_form_is_integrated_numerically():
    """ROOT's ``TF1::Integral`` uses a predefined shape's closed form where it has one and
    integrates numerically where it has not - and a Gaussian of no width has none, so the two
    roads must agree."""
    found = Function("g", "gaus", range=(0, 5), parameters=[1.0, 2.0, 0.0])
    assert math.isnan(closed_form(1, [1.0, 2.0, 0.0], 0.0, 5.0, False))  # no closed form
    assert found.integral(0, 5) == numeric.integrate(found.evaluate, 0.0, 5.0, 1e-9, 1e-9)
    normalised = Function("n", "[0]*x", range=(0, 2), parameters=[3.0])
    normalised.normalized = True
    assert round(normalised.integral(0, 2), 12) == 1.0  # a normalised function integrates to one


def test_the_table_a_formula_draws_from_is_kept_until_what_it_comes_from_changes():
    """``GetRandom`` tabulates the integral once and keeps it, as ROOT's ``TF1::Update``
    decides - so two draws from the same generator are the same numbers, and a parameter that
    changes the shape makes them different."""
    found = Function("r", "[0]*x+[1]", range=(0, 2), parameters=[1.0, 0.0])
    first = found.get_random(3, rng=TRandom3(7)).tolist()
    assert found.get_random(3, rng=TRandom3(7)).tolist() == first  # the table was kept
    found.set_parameters(1.0, 10.0)
    assert found.get_random(3, rng=TRandom3(7)).tolist() != first


def test_a_copy_is_the_same_function_under_another_name():
    """``Clone`` shares nothing with the original: the members are copied, the name is changed
    where a name is given, and a ``TF1``'s formula member is renamed beside it - a
    ``TFormula``'s *is* the formula, so it needs no second rename."""
    other = DECAY.copy("other")
    assert other.name == "other"
    assert other._formula["TNamed"]["fName"] == "other"  # type: ignore[index]
    assert round(float(other(2.0)), 6) == 36.787944
    other.set_parameters(1.0, 2.0)
    assert round(float(DECAY(2.0)), 6) == 36.787944  # the original did not move
    formula = DECAY.as_formula()
    renamed = formula.copy("tf2")
    assert renamed.name == "tf2"
    assert renamed.formula == formula.formula
    code = Function.from_callable("code", lambda x, params: params[0] * x, 1, parameters=[2.0])
    assert code.copy("c2").name == "c2"


def test_a_function_a_file_holds_in_a_wrapper_is_unwrapped():
    """A reader may hand the formula over as an object rather than as its members, which
    happens when the formula was made a key of its own: the members are taken out of it, and
    the name is the way in for whatever ``kinds`` is asked to make."""
    row = _a_file_row()
    row["fFormula"] = Function.from_members("TFormula", copy.deepcopy(row["fFormula"]))  # type: ignore[arg-type]
    found = Function.from_members("TF1", row)
    assert isinstance(found, Function)
    assert found.formula == "[N]*exp(-x/[tau])"
    assert round(float(found(2.0)), 6) == 36.787944
    assert isinstance(dress("TF1", copy.deepcopy(_a_file_row())), Function)
    assert dress("TF1", {"TObject": {}}) == {"TObject": {}}  # nothing to make a function of
