"""``TVectorT``, the template of ``TVectorD``, and the bounds a vector's index keeps to."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError


def test_the_template_of_double_or_pythons_float_is_tvectord():
    assert ROOT.TVectorT[float] is ROOT.TVectorT("double") is ROOT.TVectorD
    assert repr(ROOT.TVectorT) == "<template TVectorT>"
    with pytest.raises(UnsupportedFeatureError, match="TVectorT<int> is not here"):
        ROOT.TVectorT["int"]


def test_an_index_outside_the_vectors_bounds_is_an_index_error_both_ways():
    vector = ROOT.TVectorD(2, 4)
    vector[4] = 7.0
    assert vector[4] == 7.0 and list(vector) == [0.0, 0.0, 7.0]
    for outside in (1, 5, -1):
        with pytest.raises(IndexError, match=r"outside this TVectorD of 2\.\.4"):
            vector[outside]
    with pytest.raises(IndexError):
        vector[5] = 1.0


def test_a_three_vector_is_three_long():
    assert len(ROOT.TVector3(1, 2, 3)) == 3


def test_an_index_past_either_end_of_an_array_is_an_index_error():
    numbers = ROOT.TArrayI(3)
    numbers[2] = 4
    assert numbers[2] == 4 and list(numbers) == [0, 0, 4]
    for outside in (-1, 3):
        with pytest.raises(IndexError, match="outside this TArrayI of 3"):
            numbers[outside]
    with pytest.raises(IndexError):
        numbers[3] = 1


def test_a_context_given_two_directories_goes_back_to_the_first():
    first, second = ROOT.TDirectory("context_a", "a"), ROOT.TDirectory("context_b", "b")
    with ROOT.TDirectory.TContext(first, second):
        assert ROOT.gDirectory == second
    assert ROOT.gDirectory == first
    ROOT.gROOT.cd()
