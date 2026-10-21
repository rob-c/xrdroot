"""Same-kind numeric values copy logical storage without Python scalar inference."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xrdroot.pyroot import stl, vectorvalue
from xrdroot.pyroot.rdf.rvec import RVec, _rvec_class


@pytest.mark.parametrize("maker", [stl.std.vector, RVec], ids=["std", "rvec"])
@pytest.mark.parametrize(("name", "word", "dtype", "bits"), [
    ("unsigned long", "uint64", "uint64", [0, 2**64 - 1, 2**63, 1, 2**63 + 1, 2**64 - 2]),
    ("float", "uint32", "float32", [0x80000000, 1, 0x7FC00042, 0x7F800043]),
    ("double", "uint64", "float64", [0x8000000000000000, 1,
                                     0x7FF8000000000042, 0x7FF0000000000043]),
    ("bool", "uint8", "bool", [0, 1, 1, 0]),
])
def test_same_kind_constructor_and_assign_preserve_all_logical_integer_and_ieee_bits(
    maker: Any, name: str, word: str, dtype: str, bits: list[int],
) -> None:
    words = np.array(bits, dtype=word)
    kind = maker[name]
    source = kind(words.view(dtype))
    source.reserve(19)
    source._data.view(word)[len(words):] = np.resize(words, source.capacity() - len(words))
    before = source._data.tobytes()
    constructed = kind(source)
    assigned = kind()
    assigned.assign(source)
    for value in [constructed, assigned]:
        assert type(value) is kind and value.dtype == np.dtype(dtype)
        assert value.size() == len(words) and value.capacity() == len(words)
        assert value.data().view(word).tobytes() == words.tobytes()
        assert not np.shares_memory(value._data, source._data)
    assert not np.shares_memory(constructed._data, assigned._data)
    constructed._data.view(word)[0] = words[-1]
    assert assigned.data().view(word).tobytes() == words.tobytes()
    assert source._data.tobytes() == before


@pytest.mark.parametrize("maker", [stl.std.vector, RVec], ids=["std", "rvec"])
@pytest.mark.parametrize(("name", "word", "dtype", "bits"), [
    ("unsigned long", "uint64", "uint64", [0, 2**64 - 1, 2**63, 1]),
    ("float", "uint32", "float32", [0x80000000, 0x7F800043, 0x7FC00042, 1]),
])
def test_numeric_self_assign_replaces_its_buffer_with_an_independent_bit_exact_copy(
    maker: Any, name: str, word: str, dtype: str, bits: list[int],
) -> None:
    words = np.array(bits, dtype=word)
    value = maker[name](words.view(dtype))
    previous = value._data
    value.assign(value)
    assert value._data is not previous and not np.shares_memory(value._data, previous)
    assert value.data().view(word).tobytes() == previous.view(word).tobytes() == words.tobytes()
    value._data.view(word)[0] = words[-1]
    assert previous.view(word).tobytes() == words.tobytes()


def test_foreign_iterables_keep_iteration_length_and_raising_conversion_order() -> None:
    calls: list[str] = []

    class Given:
        def __iter__(self) -> Any:
            calls.append("iter")
            return iter([3, 4])

        def __len__(self) -> int:
            calls.append("len")
            return 2

    value = stl.std.vector["int"]([1, 2])
    value.assign(Given())
    assert calls == ["iter", "len"] and list(value) == [3, 4]
    previous = value._data

    class Broken(Given):
        def __iter__(self) -> Any:
            calls.append("broken iter")
            raise ValueError("The iterable is unavailable.")

    calls.clear()
    with pytest.raises(ValueError, match="The iterable is unavailable"):
        value.assign(Broken())
    assert calls == ["broken iter"] and value._data is previous


def test_foreign_list_and_array_inputs_keep_the_original_conversion_and_target_dtype() -> None:
    given = [[1, 2], np.array([1, 2], dtype="int16"), np.array([1, 2], dtype="float64")]
    for one in given:
        expected = np.asarray(one if isinstance(one, np.ndarray) else list(one)).astype("int32")
        value = stl.std.vector["int"]()
        value.assign(one)
        assert value.dtype == np.dtype("int32") and value.data().tobytes() == expected.tobytes()


def _fresh_kind(rvec: bool) -> type:
    return _rvec_class(np.dtype("int16")) if rvec else stl._vector_class(np.dtype("int16"))


def _ordinary(values: Any, dtype: Any) -> np.ndarray[Any, Any]:
    given = np.asarray(values if isinstance(values, np.ndarray) else list(values))
    return given.astype(dtype, copy=True).reshape(-1)


@pytest.mark.parametrize("rvec", [False, True], ids=["std", "rvec"])
@pytest.mark.parametrize("method", ["__iter__", "__len__", "data", "__getattribute__"])
def test_same_kind_public_reading_overrides_keep_the_original_callback_trace(
    rvec: bool, method: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    kind = _fresh_kind(rvec)
    source, target = kind([1, 2]), kind()
    original = getattr(kind, method)

    def replaced(self: Any, *args: Any) -> Any:
        if self is source:
            calls.append(method)
        return original(self, *args)

    monkeypatch.setattr(kind, method, replaced)
    expected = _ordinary(source, kind.dtype)
    expected_calls = list(calls)
    calls.clear()
    target.assign(source)
    assert calls == expected_calls and calls
    assert target.data().tobytes() == expected.tobytes()


@pytest.mark.parametrize("rvec", [False, True], ids=["std", "rvec"])
def test_subclass_and_rebased_native_sources_keep_their_iterable_behavior(rvec: bool) -> None:
    calls: list[str] = []
    kind = _fresh_kind(rvec)

    class Custom(kind):
        def __iter__(self) -> Any:
            calls.append("custom iter")
            return iter([8, 9])

    target = kind()
    target.assign(Custom([1, 2]))
    assert calls == ["custom iter"] and list(target) == [8, 9]
    source = kind([1, 2])

    class Rebased(stl._Vector):
        __slots__ = ()

        def __iter__(self) -> Any:
            calls.append("rebased iter")
            return iter([5, 6])

    original = kind.__bases__
    try:
        kind.__bases__ = (Rebased,)
        calls.clear()
        target.assign(source)
        assert not vectorvalue.native(kind)
        assert calls == ["rebased iter"] and target.data().tolist() == [5, 6]
    finally:
        kind.__bases__ = original
    assert kind.__bases__ == original and vectorvalue.native(kind)


@pytest.mark.parametrize("rvec", [False, True], ids=["std", "rvec"])
def test_a_mutated_factory_dtype_falls_through_the_original_conversion(
    rvec: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = _fresh_kind(rvec)
    source, target = kind([1, 2]), kind()
    monkeypatch.setattr(kind, "dtype", np.dtype("int32"))
    expected = _ordinary(source, kind.dtype)
    target.assign(source)
    assert not vectorvalue.native(kind)
    assert target._data.dtype == np.dtype("int32") and target.data().tobytes() == expected.tobytes()


@pytest.mark.parametrize("state", [
    "subclass", "foreign", "shape", "dtype", "size-type", "size-range",
])
def test_nonstandard_backing_storage_falls_through_without_a_speculative_data_call(
    state: str,
) -> None:
    calls: list[str] = []
    kind = _fresh_kind(False)
    source, target = kind([1, 2]), kind()

    class Array(np.ndarray):
        def __getitem__(self, index: Any) -> Any:
            calls.append("slice")
            return super().__getitem__(index)

    class Foreign:
        def __getitem__(self, index: Any) -> Any:
            calls.append("foreign slice")
            return np.array([3, 4], dtype="int16")

    states = {
        "subclass": np.array([1, 2], dtype="int16").view(Array),
        "foreign": Foreign(), "shape": np.array([[1, 2], [3, 4]], dtype="int16"),
        "dtype": np.array([1, 2], dtype="int32"),
        "size-type": np.array([1, 2], dtype="int16"),
        "size-range": np.array([1, 2], dtype="int16"),
    }
    source._data = states[state]
    if state == "size-type":
        source._size = np.int64(2)
    if state == "size-range":
        source._size = 9
    expected = _ordinary(source, kind.dtype)
    expected_calls = list(calls)
    calls.clear()
    target.assign(source)
    assert calls == expected_calls
    assert target.data().tobytes() == expected.tobytes()


@pytest.mark.parametrize("rvec", [False, True], ids=["std", "rvec"])
@pytest.mark.parametrize("method", ["__iter__", "data"])
def test_numeric_base_reading_overrides_keep_the_original_callback_trace(
    rvec: bool, method: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    kind = _fresh_kind(rvec)
    source, target = kind([1, 2]), kind()
    original = getattr(stl._Vector, method)

    def replaced(self: Any) -> Any:
        if self is source:
            calls.append(method)
        return original(self)

    monkeypatch.setattr(stl._Vector, method, replaced)
    expected = _ordinary(source, kind.dtype)
    expected_calls = list(calls)
    calls.clear()
    target.assign(source)
    assert calls == expected_calls and calls
    assert target.data().tobytes() == expected.tobytes()


@pytest.mark.parametrize("member", ["_data", "_size"])
def test_a_replaced_storage_descriptor_keeps_its_original_reads_without_speculation(
    member: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    kind = _fresh_kind(False)
    source, target = kind([1, 2]), kind()
    descriptor = vars(stl._Vector)[member]

    def get(self: Any) -> Any:
        if self is source:
            calls.append(member)
        return descriptor.__get__(self, kind)

    def set_value(self: Any, value: Any) -> None:
        descriptor.__set__(self, value)

    monkeypatch.setattr(kind, member, property(get, set_value))
    expected = _ordinary(source, kind.dtype)
    expected_calls = list(calls)
    calls.clear()
    target.assign(source)
    assert calls == expected_calls and calls
    assert target.data().tobytes() == expected.tobytes()


def test_the_numeric_input_guard_returns_a_real_nested_factory_value_without_reading_its_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf = stl.std.vector["int"]
    kind = stl.std.vector[leaf]
    source = kind([leaf([1, 2])])
    calls: list[str] = []

    def data(self: Any) -> Any:
        calls.append("data")
        raise AssertionError("The numeric input guard must not read object-vector storage.")

    monkeypatch.setattr(kind, "data", data)
    assert vectorvalue.native(kind)
    assert vectorvalue.numeric_input(kind, source, stl._NUMERIC_READERS, stl._ARRAY_TYPE) is source
    assert calls == []


@pytest.mark.parametrize("maker", [stl.std.vector, RVec], ids=["std", "rvec"])
@pytest.mark.parametrize("name", ["float", "unsigned long"])
def test_an_empty_reserved_numeric_source_copies_no_spare_elements(maker: Any, name: str) -> None:
    kind = maker[name]
    source = kind()
    source.reserve(7)
    source._data[:] = 1
    before = source._data.tobytes()
    copied = kind(source)
    assigned = kind([1])
    assigned.assign(source)
    assert copied.size() == assigned.size() == copied.capacity() == assigned.capacity() == 0
    assert copied._data is not assigned._data and copied._data is not source._data
    assert source.size() == 0 and source.capacity() == 7 and source._data.tobytes() == before


def test_a_removed_numeric_base_data_method_keeps_the_original_error_and_target_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = _fresh_kind(False)
    source, target = kind([1, 2]), kind([3])
    previous = target._data
    monkeypatch.delattr(stl._Vector, "data")
    with pytest.raises(AttributeError, match="data"):
        target.assign(source)
    assert target._data is previous


def test_a_negative_foreign_logical_size_keeps_the_original_length_error() -> None:
    kind = _fresh_kind(False)
    source, target = kind([1, 2]), kind([3])
    previous = target._data
    source._size = -1
    with pytest.raises(ValueError, match="__len__"):
        target.assign(source)
    assert target._data is previous
