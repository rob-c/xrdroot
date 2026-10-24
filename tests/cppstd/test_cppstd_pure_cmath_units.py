"""C math retains IEEE edges and promotion; standard duration literals retain their units."""

from __future__ import annotations

import datetime
import math
import struct

import numpy as np
import pytest

from xrdroot.cppstd import cmath, containers, istreams, units


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("sqrt", (-0.0,), -0.0),
        ("log", (0.0,), -math.inf),
        ("exp", (1000.0,), math.inf),
        ("pow", (0.0, -1.0), math.inf),
        ("cround", (-0.25,), -0.0),
        ("cround", (-0.5,), -1.0),
        ("cround", (0.5,), 1.0),
        ("cbrt", (-8.0,), -2.0),
    ],
)
def test_scalar_c_math_preserves_ieee_signs_and_defined_edge_results(name, arguments, expected):
    # Exact powers, zeros and infinities avoid platform-dependent transcendental last bits.
    assert struct.pack("d", getattr(cmath, name)(*arguments)) == struct.pack("d", expected)


def test_c_math_domain_errors_and_plain_arrays_keep_generic_result_types() -> None:
    assert math.isnan(cmath.sqrt(-1)) and math.isnan(cmath.pow(-1, 0.5))
    assert cmath.cabs(-3) == 3 and type(cmath.cabs(-3)) is int
    result = cmath.sqrt(np.array([[1, 4], [9, 16]], dtype=np.float32))
    assert type(result) is np.ndarray and result.dtype == np.float32
    assert result.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    promoted = cmath.pow(np.array([2, 3], dtype=np.int32), 2)
    assert promoted.dtype == np.float64 and promoted.tolist() == [4.0, 9.0]


def test_explicit_math_constructor_receives_a_standard_vectors_promoted_values() -> None:
    observed = []
    token = object()

    def construct(values):
        observed.append(values)
        return token

    cmath.set_vector_result(construct)
    values = containers.std.vector["float"]([1, 4, 9])
    assert cmath.sqrt(values) is token
    assert len(observed) == 1 and observed[0].dtype == np.float32
    assert observed[0].tolist() == [1.0, 2.0, 3.0]
    plain = cmath.sqrt(np.array([4, 9], dtype=np.float32))
    assert type(plain) is np.ndarray
    assert plain.dtype == np.float32
    assert plain.shape == (2,)
    assert plain.tolist() == [2.0, 3.0]
    _assert_plain_singleton_array_follows_builtin_float(observed)
    assert len(observed) == 1


@pytest.mark.parametrize(
    ("suffix", "value", "expected"),
    [
        ("ns", 1500, datetime.timedelta(microseconds=2)),
        ("us", 25, datetime.timedelta(microseconds=25)),
        ("ms", -2.5, datetime.timedelta(milliseconds=-2.5)),
        ("s", 1.25, datetime.timedelta(seconds=1.25)),
        ("min", 2, datetime.timedelta(minutes=2)),
        ("h", 0.5, datetime.timedelta(minutes=30)),
    ],
)
def test_standard_duration_suffixes_keep_their_units_and_timedelta_rounding(
    suffix, value, expected
):
    assert units.duration_literal(suffix, value) == expected


def test_unknown_duration_suffixes_remain_available_to_the_callers_adapter() -> None:
    assert set(units.DURATIONS) == {"ns", "us", "ms", "s", "min", "h"}
    assert units.duration_literal("_normal", 0.25) is None


def test_state_restoration_preserves_math_constructor_and_input_table_identity(
    cppstd_state_context,
):
    constructor, kinds, patterns = cmath.VECTOR_RESULT, istreams.KINDS, istreams.PATTERNS
    saved_kinds, saved_patterns = dict(kinds), dict(patterns)
    with cppstd_state_context():
        cmath.set_vector_result(lambda values: values)
        kinds["CppstdPureInputFixture"] = "integral"
        patterns["CppstdPurePatternFixture"] = patterns["integral"]
        stream = istreams.istringstream("17")
        assert stream.extract("CppstdPureInputFixture") == 17
    assert cmath.VECTOR_RESULT is constructor
    assert istreams.KINDS is kinds and kinds == saved_kinds
    assert istreams.PATTERNS is patterns and patterns == saved_patterns


def _assert_plain_singleton_array_follows_builtin_float(observed) -> None:
    # NumPy 1 accepts this deprecated coercion; newer NumPy can reject it.
    # The original scalar-first C-math wrapper follows that builtin capability.
    singleton = np.array([4], dtype=np.float32)
    try:
        scalar = float(singleton)
    except TypeError:
        result = cmath.sqrt(singleton)
        _assert_plain_singleton_array_result(result)
    else:
        assert scalar == 4.0
        result = cmath.sqrt(singleton)
        assert type(result) is float
        assert result == 2.0
    assert len(observed) == 1


def _assert_plain_singleton_array_result(result) -> None:
    assert type(result) is np.ndarray
    assert result.dtype == np.float32
    assert result.shape == (1,)
    assert result.tolist() == [2.0]
