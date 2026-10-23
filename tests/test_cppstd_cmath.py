"""C math keeps its scalar bits and array promotion behind a supplied result constructor."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from cppstdsupport import isolated


def test_cmath_exports_and_generated_wrappers_keep_legacy_identity() -> None:
    from xrdroot.cint.runtime import cmath as legacy
    from xrdroot.cppstd import cmath

    assert legacy.__all__ is cmath.__all__
    for name in cmath.__all__:
        assert getattr(legacy, name) is getattr(cmath, name), name
    assert cmath.fast_exp is cmath.exp
    assert cmath.fast_log is cmath.log
    assert cmath.VECTOR_RESULT is legacy._rvec


@pytest.mark.parametrize(("name", "args", "expected"), [
    ("sqrt", (-0.0,), -0.0), ("log", (0.0,), -math.inf),
    ("exp", (1000.0,), math.inf), ("pow", (0.0, -1.0), math.inf),
    ("cround", (-0.25,), -0.0), ("cround", (-0.5,), -1.0),
    ("cround", (0.5,), 1.0), ("cbrt", (-8.0,), -2.0),
])
def test_scalar_cmath_retains_ieee_signs_and_edge_answers(name, args, expected) -> None:
    from xrdroot.cppstd import cmath

    assert struct.pack("d", getattr(cmath, name)(*args)) == struct.pack("d", expected)


def test_cmath_result_constructor_receives_the_original_promoted_arrays(monkeypatch) -> None:
    from xrdroot.cppstd import cmath

    class Vector:
        __cpp_name__ = "vector<float>"

        def data(self):
            return np.array([1, 4, 9], dtype=np.float32)

    observed = []
    token = object()

    def construct(values):
        observed.append(values)
        return token

    monkeypatch.setattr(cmath, "VECTOR_RESULT", cmath.VECTOR_RESULT)
    cmath.set_vector_result(construct)
    assert cmath.sqrt(Vector()) is token
    assert len(observed) == 1
    assert observed[0].dtype == np.float32
    assert observed[0].tolist() == [1.0, 2.0, 3.0]
    plain = cmath.pow(np.array([2, 3], dtype=np.int32), 2)
    assert plain.dtype == np.float64
    assert plain.tolist() == [4.0, 9.0]
    assert len(observed) == 1


def test_the_runtime_adapter_keeps_actual_rvec_results_and_promoted_types() -> None:
    from xrdroot.cint.runtime import cmath
    from xrdroot.pyroot.rdf.rvec import RVec

    values = RVec["float"]([1, 4, 9])
    result = cmath.sqrt(values)
    assert type(result) is RVec["float"]
    assert result.data().tolist() == [1.0, 2.0, 3.0]
    result = cmath.pow(RVec["int"]([2, 3]), 2)
    assert type(result) is RVec["double"]
    assert result.data().tolist() == [4.0, 9.0]


def test_scalar_and_array_cmath_run_without_root_or_the_interpreter() -> None:
    isolated("""
import math
import numpy as np
from xrdroot.cppstd import cmath
assert cmath.VECTOR_RESULT is np.asarray
assert math.isnan(cmath.sqrt(-1))
assert math.isnan(cmath.pow(-1, 0.5))
assert cmath.log(0) == -math.inf
assert cmath.exp(1000) == math.inf
assert cmath.cabs(-3) == 3 and type(cmath.cabs(-3)) is int
result = cmath.sqrt(np.array([[1, 4], [9, 16]], dtype=np.float32))
assert result.dtype == np.float32
assert result.tolist() == [[1.0, 2.0], [3.0, 4.0]]
class Vector:
    __cpp_name__ = 'vector<float>'
    def data(self):
        return np.array([1, 4], dtype=np.float32)
result = cmath.sqrt(Vector())
assert type(result) is np.ndarray
assert result.dtype == np.float32
assert result.tolist() == [1.0, 2.0]
""")
