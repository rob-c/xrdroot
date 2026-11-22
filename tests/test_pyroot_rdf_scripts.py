"""A PyROOT script's Python functions given to a frame: called once per entry, as ROOT's are."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from xrdroot.pyroot.core import messages


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(messages, "PYROOT_SCRIPT", True)


def test_in_a_script_a_function_of_a_collection_is_given_one_entrys_collection(script):
    df = ROOT.RDataFrame(3).Define("x", "rdfentry_").Define("v", "ROOT::RVecD{1., 2.}")
    sums = df.Define("s", lambda v: v.sum(), ["v"]).Define("a", lambda x: np.array([x, x]))
    assert list(sums.AsNumpy(["s"])["s"]) == [3.0, 3.0, 3.0]
    assert [list(each) for each in sums.AsNumpy(["a"])["a"]] == [[0, 0], [1, 1], [2, 2]]


def test_in_a_script_the_columns_left_out_are_the_functions_parameters(script):
    df = ROOT.RDataFrame(3).Define("x", "rdfentry_")
    assert list(df.Define("y", lambda x: x * x).AsNumpy(["y"])["y"]) == [0, 1, 4]
    assert df.Filter(lambda x: x > 0).Count().GetValue() == 2


def test_extra_arguments_are_given_to_the_function_by_name_and_its_others_read_columns():
    def scaled(x, y):
        return x * y

    df = ROOT.RDataFrame(3).Define("x", "rdfentry_")
    assert list(df.Define("xy", scaled, extra_args={"y": 0.5}).AsNumpy(["xy"])["xy"]) == [
        0.0, 0.5, 1.0]  # fmt: skip
    named = df.Define("xz", scaled, ["x"], extra_args={"y": 2})
    assert list(named.AsNumpy(["xz"])["xz"]) == [0, 2, 4]


def test_a_macros_overloaded_function_is_called_once_per_entry_with_the_overload_that_fits():
    ROOT.gInterpreter.Declare(
        "ULong64_t script_overload(ULong64_t l) { return l; }\n"
        "ULong64_t script_overload(ULong64_t l, ULong64_t m) { return l * m; }\n"
    )
    df = ROOT.RDataFrame(3)
    one = df.Define("one", ROOT.script_overload, ["rdfentry_"])
    two = one.Define("two", ROOT.script_overload, ["rdfentry_", "rdfentry_"])
    assert list(two.AsNumpy(["one", "two"])["two"]) == [0, 1, 4]


def test_an_array_on_the_left_of_an_rvec_gives_an_array_as_pyroot_does():
    found = np.array([1.0, 2.0], dtype=np.float32) == ROOT.RVecF([1.0, 3.0])
    assert type(found) is np.ndarray and list(found) == [True, False]
    assert type(ROOT.RVecF([1.0]) + np.array([1.0])) is not np.ndarray


def test_take_gives_a_std_vector_of_numbers_which_compares_with_a_list():
    df = ROOT.RDataFrame(3).Define("x", "(Long64_t)rdfentry_").Define("s", 'std::string("a")')
    taken = df.Take["Long64_t"]("x").GetValue()
    assert isinstance(taken, ROOT.std.vector["Long64_t"]) and taken == [0, 1, 2]
    assert list(df.Take["std::string"]("s").GetValue()) == ["a", "a", "a"]


def test_take_of_a_column_of_vectors_is_a_vector_of_vectors_as_roots_is():
    """``Take<std::vector<bool>>`` gives ``std::vector<std::vector<bool>>``: ROOT's value, whose
    two levels both answer ``size()`` - a row of a ``Jagged`` is a NumPy array, whose ``size``
    is a number and not a call."""
    frame = ROOT.RDataFrame(1).Define("x", "std::vector<bool>({true, false})")
    taken = frame.Take["std::vector<bool>"]("x").GetValue()
    expect(
        (taken.size(), 1),
        (taken[0].size(), 2),
        (taken[0][0], True),
        (taken[0][1], False),
    )
