"""``ROOT.VecOps.AsRVec``: an ``RVec`` over a NumPy array's own memory."""

from __future__ import annotations

import sys

import numpy as np
import pytest

import xrdroot.pyroot as ROOT


def test_an_adopted_array_is_written_through_and_kept_alive_by_the_rvec():
    values = np.array([1.0, 2.0], dtype="float32")
    rvec = ROOT.VecOps.AsRVec(values)
    values[0] = 7.0
    rvec[1] = 9.0
    assert rvec[0] == 7.0 and values[1] == 9.0 and rvec.size() == 2
    assert rvec.__adopted__ is values and rvec.__python_owns__ is True
    assert isinstance(rvec, ROOT.RVec["float"])
    before = sys.getrefcount(values)
    del rvec
    assert sys.getrefcount(values) == before - 1


def test_a_grid_is_adopted_flattened_and_remembered_whole():
    grid = np.array([[1, 2], [3, 4]], dtype="int64")
    rvec = ROOT.VecOps.AsRVec(grid)
    rvec[3] = 40
    assert list(rvec) == [1, 2, 3, 40] and grid[1, 1] == 40
    assert rvec.__adopted__ is grid


def test_an_rvec_adopted_again_shares_the_first_ones_elements():
    first = ROOT.VecOps.RVec("float")(1)
    second = ROOT.VecOps.AsRVec(first)
    second[0] = 43
    assert first[0] == 43 and second.__adopted__ is first


def test_strings_are_copied_into_an_rvec_of_strings_of_one_type_only():
    for kind in (object, np.str_):
        rvec = ROOT.VecOps.AsRVec(np.array(["a", "bc"], dtype=kind))
        assert rvec.size() == 2 and rvec[1] == "bc"
    assert ROOT.VecOps.AsRVec(np.array([], dtype=object)).size() == 0
    numbers = ROOT.VecOps.AsRVec(np.array([1, 2], dtype=object))
    with pytest.raises(TypeError, match="All elements in the numpy array must be of the same"):
        ROOT.VecOps.AsRVec(np.array(["a", {}], dtype=object))
    assert list(numbers) == [1, 2]


def test_one_class_is_adopted_once_however_many_arrays_come_through_it():
    """``_adopting`` caches the class it makes, so every ``AsRVec`` of one element type is that
    one class - which is what ``isinstance`` and a tree's check of what it was handed need."""
    from xrdroot.pyroot.rdf.rvec import _adopting

    class Held:
        __slots__ = ()
        __python_owns__ = False

    first = _adopting(Held)
    assert _adopting(Held) is first and first.__name__ == "Held"
    assert first.__python_owns__ is True and "_adopted" in first.__slots__
