"""Scalar arithmetic and fixed-size bits follow their C++ value contracts."""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.cppstd import arith, bitset, chrono, strings, threads


@pytest.mark.parametrize(
    ("left", "right", "quotient", "remainder"),
    [
        (-7, 3, -2, -1),
        (7, -3, -2, 1),
        (-7, -3, 2, -1),
        (7, 3, 2, 1),
    ],
)
def test_integer_division_and_remainder_truncate_toward_zero(left, right, quotient, remainder):
    assert arith.idiv(left, right) == quotient
    assert arith.imod(left, right) == remainder


def test_integer_narrowing_and_float_stores_keep_their_declared_widths() -> None:
    assert arith.u8(256) == 0 and arith.i8(128) == -128
    assert arith.u16(-1) == 65535 and arith.i16(65535) == -1
    stored = arith.f32(16777217)
    assert stored == 16777216 and isinstance(stored, float)
    array = arith.f32(np.array([16777217], dtype="float64"))
    assert array.dtype == np.dtype("float64") and array.tolist() == [16777216]


def test_string_algorithms_preserve_npos_and_write_only_the_requested_range() -> None:
    assert strings.find_first_of("abc", "xyz") == strings.npos
    assert strings.find_last_of("a/b/c", "/", 2) == 1
    assert strings.resize("ab", 4) == "ab\0\0"
    source, target = [1, 2], [0, 0, 0]
    assert strings.transformed(source, 0, 2, target, 1, lambda x: x * 10) is target
    assert target == [0, 10, 20] and source == [1, 2]


def test_bitsets_mask_shifts_and_mutate_bits_in_place() -> None:
    value = bitset.bitset[4]("1010")
    assert value.count() == 2 and value.to_string() == "1010"
    assert value.flip(0) is value and value.to_ulong() == 11
    value <<= 1
    assert value.to_string() == "0110"
    value >>= 2
    assert value.to_string() == "0001" and value.any() and not value.none()


def test_duration_units_and_atomic_updates_keep_their_value_semantics() -> None:
    duration = chrono.chrono.milliseconds(1500)
    assert duration.count() == 1500
    assert chrono.chrono.duration_cast["std::chrono::seconds"](duration).count() == 1
    value = threads.atomic(3)
    assert value.fetch_add(2) == 3 and value.load() == 5
    assert value.fetch_sub(1) == 5 and value.load() == 4
    value.store(9)
    assert value.load() == 9
