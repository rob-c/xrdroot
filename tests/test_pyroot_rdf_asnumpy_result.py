"""PyROOT's ``AsNumpy``: arrays that know their result, and the lazy result, cloned."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pyroot._pythonization._rdataframe import _clone_asnumpyresult, ndarray


def test_a_column_is_roots_ndarray_whose_result_is_the_memory_it_views():
    column = ROOT.RDataFrame(3).Define("x", "(double)rdfentry_").AsNumpy(["x"])["x"]
    assert isinstance(column, ndarray) and isinstance(column, np.ndarray)
    read = column.result_ptr.GetValue()
    column[0] = 42
    assert read[0] == 42
    part = column[:2]
    assert part.result_ptr is column.result_ptr and not part.flags["OWNDATA"]
    assert list(pickle.loads(pickle.dumps(column))) == [42.0, 1.0, 2.0]


def test_a_lazy_read_waits_for_its_value_and_a_clone_reads_again():
    frame = ROOT.RDataFrame(4).Define("x", "(int)rdfentry_")
    lazy = frame.AsNumpy(["x"], lazy=True)
    assert lazy.GetValue() is lazy.GetValue()
    assert lazy.GetValue()["x"].tolist() == [0, 1, 2, 3]
    again = _clone_asnumpyresult(lazy)
    assert again is not lazy and again.GetValue()["x"].tolist() == [0, 1, 2, 3]


def test_a_numpy_column_of_objects_other_than_strings_is_refused_as_root_refuses_it():
    strings = ROOT.RDF.FromNumpy({"s": np.array(["a", "b"], dtype=object)})
    assert list(strings.AsNumpy()["s"]) == ["a", "b"]
    for refused in (np.array([], dtype=object), np.array([1, "a"], dtype=object)):
        with pytest.raises(RuntimeError, match="Failure in creating column 'x' for RDataFrame"):
            ROOT.RDF.FromNumpy({"x": refused})
