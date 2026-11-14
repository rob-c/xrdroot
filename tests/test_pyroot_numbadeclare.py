"""``ROOT.Numba.Declare``: a Python function compiled with numba and called from C++ by name."""

from __future__ import annotations

import builtins
import importlib.util
import sys

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot import numbadeclare
from xrdroot.pyroot.core import troot


@pytest.fixture(autouse=True)
def _fresh_line_processor():
    """``ProcessLine`` finds the translator afresh after each test, as it would in a new run."""
    yield
    troot.set_line_processor(None)


def test_the_types_of_a_signature_are_read_as_cpp_spells_them():
    rvec = numbadeclare.CxxType("const ROOT::VecOps::RVec<float>&")
    assert (rvec.container, rvec.element, rvec.dtype) == ("RVec", "float", np.float32)
    assert numbadeclare.CxxType("RVecUL").element == "unsigned long"
    assert numbadeclare.CxxType("std::array<int, 3>").container == "array"
    assert numbadeclare.CxxType("unsigned int").container is None


def test_a_type_numba_is_not_given_here_is_refused_by_name():
    with pytest.raises(UnsupportedFeatureError, match="ROOT::Math::PtEtaPhiMVector"):
        numbadeclare.CxxType("ROOT::Math::PtEtaPhiMVector")


def test_without_numba_declaring_is_refused_with_the_install_that_fixes_it(monkeypatch):
    real = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "numba":
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(UnsupportedFeatureError, match="pip install numba"):
        ROOT.Numba.Declare(["float"], "float")(lambda x: x)


#: The tests that compile with numba, skipped where it is not installed.
needs_numba = pytest.mark.skipif(
    importlib.util.find_spec("numba") is None, reason="numba is not installed"
)


@needs_numba
def test_a_declared_function_is_called_with_cpps_types_and_answers_in_them():
    @ROOT.Numba.Declare(["float"], "int")
    def truncated(x):
        return int(x)

    assert ROOT.Numba.truncated(2.7) == 2 and type(ROOT.Numba.truncated(2.7)) is int
    assert truncated(2.7) == 2 and hasattr(truncated, "numba_func")
    assert "namespace Numba" in truncated.__cpp_wrapper__
    assert truncated.__py_wrapper__.startswith("def truncated_py(x_0):")


@needs_numba
def test_a_bool_comes_back_as_pythons_and_a_name_can_be_given():
    @ROOT.Numba.Declare(["float"], "bool", name="positive_x")
    def positive(x):
        return x > 0

    assert ROOT.Numba.positive_x(1.0) is True and not hasattr(ROOT.Numba, "positive")


@needs_numba
def test_containers_go_in_as_arrays_and_come_back_as_the_container_declared():
    @ROOT.Numba.Declare(["RVec<float>"], "RVec<float>")
    def reversed_rvec(x):
        return x[::-1]

    @ROOT.Numba.Declare(["std::vector<int>"], "std::vector<int>")
    def squares(x):
        return x * x

    @ROOT.Numba.Declare(["std::array<int, 2>"], "std::array<int, 2>")
    def doubled(x):
        return x + x

    assert list(ROOT.Numba.reversed_rvec(ROOT.RVecF([1, 2]))) == [2.0, 1.0]
    assert list(ROOT.Numba.squares(ROOT.std.vector["int"]([1, 2]))) == [1, 4]
    assert list(ROOT.Numba.doubled([1, 2])) == [2, 4]


@needs_numba
def test_a_return_type_left_out_is_numbas_and_cpp_calls_the_function_by_name():
    @ROOT.Numba.Declare(["double"])
    def halved(x):
        return x / 2

    assert ROOT.Numba.halved(3.0) == 1.5
    ROOT.gInterpreter.ProcessLine("numba_halved = Numba::halved(5.0);")
    assert ROOT.numba_halved == 2.5
