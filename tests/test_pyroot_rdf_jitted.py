"""A frame's string that is more C++ than an expression, translated and called once per entry."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.rdf import jitted


def _frame():
    return ROOT.RDF.FromNumpy({"x": np.arange(4.0), "n": np.arange(4, dtype=np.int32)})


def test_a_string_calling_groots_generator_draws_once_per_entry_in_order() -> None:
    ROOT.gRandom.SetSeed(5)
    drawn = _frame().Define("g", "gRandom->Gaus(x, 1)").AsNumpy(["g"])["g"]
    ROOT.gRandom.SetSeed(5)
    assert drawn.tolist() == [ROOT.gRandom.Gaus(x, 1) for x in range(4)]
    ROOT.gRandom.SetSeed(5)
    alone = _frame().Define("u", "gRandom->Uniform(0, 16)").AsNumpy(["u"])["u"]
    ROOT.gRandom.SetSeed(5)
    assert alone.tolist() == [ROOT.gRandom.Uniform(0, 16) for _ in range(4)]


def test_statements_ending_in_a_return_are_a_functions_body() -> None:
    frame = _frame().Define("y", "auto twice = 2 * x; return twice + n;")
    assert frame.AsNumpy(["y"])["y"].tolist() == [0.0, 3.0, 6.0, 9.0]
    redefined = frame.Redefine("y", "return 1./(x+1)")
    assert redefined.AsNumpy(["y"])["y"].tolist() == [1.0, 0.5, 1 / 3, 0.25]
    kept = frame.Filter("int k = n; return k % 2 == 0;", "even")
    assert kept.Count().GetValue() == 2 and kept.GetFilterNames() == ["even"]
    assert frame.Filter("return x > 1.5").Count().GetValue() == 2


def test_a_string_neither_way_can_run_is_refused_with_both_reasons() -> None:
    with pytest.raises(UnsupportedFeatureError, match="as a C\\+\\+ function of its columns it"):
        _frame().Define("bad", "x +* ;")


def test_a_frames_own_error_is_raised_as_it_is() -> None:
    with pytest.raises(Exception, match="x"):
        _frame().Define("x", "1")  # a column of that name is there: no translation helps


def test_only_a_strings_refusal_by_the_evaluator_is_retried() -> None:
    frame = ROOT.RDF.FromNumpy({"x": np.arange(2.0)})._inner
    refusal = UnsupportedFeatureError("no")
    assert jitted.retried("Display", frame, ["x"], refusal) is None
    assert jitted.retried("Define", frame, ["y", len], refusal) is None
    assert jitted.retried("Define", frame, ["y"], refusal) is None
    assert jitted.retried("Define", frame, ["y", "x"], KeyError("x")) is None
    assert jitted._body("x") == "return x;" and jitted._body("{ return x; }") == "{ return x; }"


def test_a_function_of_code_giving_ragged_collections_makes_a_column_of_objects() -> None:
    from xrdroot.pyroot.rdf.entrywise import _column

    ragged = [[[1], [2, 3]], [[4]]]
    assert _column(ragged) is ragged
