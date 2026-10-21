"""Whole numeric operator= preserves its destination; public assign keeps its own API."""
from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from xrdroot.cint.runtime import assign_into, assign_vector, store_through, value_copy
from xrdroot.pyroot import stl, vectorassign
from xrdroot.pyroot.rdf.rvec import RVec


@pytest.mark.parametrize("maker", [stl.std.vector, RVec], ids=["std", "rvec"])
@pytest.mark.parametrize(("name", "word", "dtype", "bits"), [
    ("unsigned long", "uint64", "uint64", [0, 2**64 - 1, 2**63, 1]),
    ("float", "uint32", "float32", [0x80000000, 0x7F800043, 0x7FC00042, 1]),
    ("double", "uint64", "float64", [0x8000000000000000, 0x7FF0000000000043,
                                     0x7FF8000000000042, 1]),
])
def test_whole_assignment_preserves_logical_words_destination_and_reserved_storage(
    maker: Any, name: str, word: str, dtype: str, bits: list[int],
) -> None:
    words = np.array(bits, dtype=word)
    kind = maker[name]
    source = kind(words.view(dtype))
    source.reserve(19)
    target = kind([1, 2])
    target.reserve(16)
    storage, capacity, source_bytes = target._data, target.capacity(), source._data.tobytes()
    assert assign_into(target, source) is target
    assert target._data is storage and target.capacity() == capacity and target.size() == 4
    assert target.data().view(word).tobytes() == words.tobytes()
    assert source._data.tobytes() == source_bytes
    assert not np.shares_memory(target.data(), source.data())
    target._data.view(word)[0] = words[-1]
    assert source.data().view(word).tobytes() == words.tobytes()


@pytest.mark.parametrize("maker,family", [(stl.std.vector, "std"), (RVec, "rvec")])
@pytest.mark.parametrize("count", [0, 2, 4, 9, 25])
def test_copy_assignment_capacity_is_relative_to_the_current_backing(
    maker: Any, family: str, count: int,
) -> None:
    # Constructor/initial RVec inline capacities are a separate existing parity gap.
    kind = maker["unsigned long"]
    target, source = kind([1, 2, 3, 4]), kind(np.arange(count, dtype="uint64"))
    target.reserve(8)
    storage, capacity = target._data, target.capacity()
    assert assign_into(target, source) is target
    grown = count if family == "std" else max(count, 2 * capacity + 1)
    expected = capacity if count <= capacity else grown
    assert target.capacity() == expected and target.size() == count
    assert target._data is storage if count <= capacity else target._data is not storage
    assert target.data().tobytes() == source.data().tobytes()
    assert not np.shares_memory(target.data(), source.data())


@pytest.mark.parametrize("maker", [stl.std.vector, RVec], ids=["std", "rvec"])
def test_self_copy_assignment_never_copies_or_replaces_the_payload(
    maker: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = maker["double"]
    value = kind(np.array([0x8000000000000000, 0x7FF0000000000043], dtype="uint64").view("float64"))
    value.reserve(13)
    storage, capacity, bits = value._data, value.capacity(), value._data.tobytes()

    def refuse_copy(self: Any, memo: Any) -> Any:
        raise AssertionError("Self assignment must not copy its RHS.")

    monkeypatch.setattr(kind, "__deepcopy__", refuse_copy, raising=False)
    assert assign_vector(value, value, True) is value
    assert assign_into(value, value) is value
    assert value._data is storage and value.capacity() == capacity and value._data.tobytes() == bits


def test_custom_assignment_keeps_its_priority_and_pointer_rhs_copy_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = stl.std.vector["double"]
    source, target = kind([1, 2]), kind([3, 4])
    calls: list[tuple[str, bool]] = []

    def copied(self: Any, memo: Any) -> Any:
        calls.append(("copy", self is source))
        return kind(self.data())

    def assigned(self: Any, value: Any) -> None:
        calls.append(("assign", value is source))
        assert self is target

    monkeypatch.setattr(kind, "__deepcopy__", copied, raising=False)
    monkeypatch.setattr(kind, "_assign", assigned, raising=False)
    assert assign_into(target, source) is source and calls == [("assign", True)]
    calls.clear()
    result = assign_vector(target, source, True)
    assert result is not source and calls == [("copy", True), ("assign", False)]
    calls.clear()
    assert assign_vector(target, source, True, False) is source
    assert calls == [("assign", True)] and list(target) == [3, 4]


def _outcome(action: Callable[[], Any]) -> tuple[type[BaseException] | None, str]:
    try:
        action()
    except Exception as error:
        return type(error), str(error)
    return None, ""


def test_a_present_none_getattr_keeps_the_original_pointer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = stl.std.vector["double"]
    source, target = kind([1, 2]), kind([3, 4])
    storage, bits = target._data, target._data.tobytes()
    monkeypatch.setattr(kind, "__getattr__", None, raising=False)
    expected = _outcome(lambda: store_through(target, value_copy(source)))
    actual = _outcome(lambda: assign_vector(target, source, True))
    assert expected[0] is TypeError and actual == expected
    assert target._data is storage and target._data.tobytes() == bits


@pytest.mark.parametrize("name", ["data", "__iter__", "__len__"])
def test_replaced_public_readers_are_not_speculatively_called(
    name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = stl.std.vector["double"]
    source, target = kind([1, 2]), kind([3, 4])
    calls: list[str] = []
    storage, bits = target._data, target._data.tobytes()

    def foreign(self: Any) -> Any:
        calls.append(name)
        raise RuntimeError("The foreign reader was called.")

    monkeypatch.setattr(kind, name, foreign)
    expected = _outcome(lambda: store_through(target, value_copy(source)))
    expected_calls = list(calls)
    calls.clear()
    actual = _outcome(lambda: assign_vector(target, source, True))
    assert actual == expected and calls == expected_calls == []
    assert target._data is storage and target._data.tobytes() == bits


@pytest.mark.parametrize("slot", ["_data", "_size"])
def test_replaced_native_slot_descriptors_keep_the_original_callback_sequence(
    slot: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = stl.std.vector["double"]
    source, target = kind([1, 2]), kind([3, 4])
    descriptor = getattr(stl._Vector, slot)
    storage, bits = target._data, target._data.tobytes()
    calls: list[bool] = []

    def read(self: Any) -> Any:
        calls.append(self is source)
        return descriptor.__get__(self, kind)

    monkeypatch.setattr(kind, slot, property(read), raising=False)
    expected = _outcome(lambda: store_through(target, value_copy(source)))
    expected_calls = list(calls)
    calls.clear()
    actual = _outcome(lambda: assign_vector(target, source, True))
    assert actual == expected and calls == expected_calls
    assert storage.tobytes() == bits


def test_a_custom_setter_keeps_rhs_reconstruction_before_the_pointer_target_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = stl.std.vector["double"]
    source, target = kind([1, 2]), kind([3, 4])
    calls: list[str] = []
    storage, bits = target._data, target._data.tobytes()

    def write(self: Any, name: str, value: Any) -> None:
        calls.append(name)
        object.__setattr__(self, name, value)

    monkeypatch.setattr(kind, "__setattr__", write)
    expected = _outcome(lambda: store_through(target, value_copy(source)))
    expected_calls = list(calls)
    calls.clear()
    actual = _outcome(lambda: assign_vector(target, source, True))
    assert actual == expected and calls == expected_calls
    assert target._data is storage and target._data.tobytes() == bits


def _malformed(source: Any, state: str) -> None:
    changes = {
        "list": lambda: setattr(source, "_data", [1, 2]),
        "shape": lambda: setattr(source, "_data", np.ones((2, 2))),
        "dtype": lambda: setattr(source, "_data", np.arange(2, dtype="int64")),
        "strided": lambda: setattr(source, "_data", np.arange(4, dtype="float64")[::2]),
        "unaligned": lambda: setattr(source, "_data", np.ndarray(
            2, dtype="float64", buffer=bytearray(17), offset=1,
        )),
        "readonly": lambda: source._data.setflags(write=False),
        "negative": lambda: setattr(source, "_size", -1),
        "oversized": lambda: setattr(source, "_size", 20),
        "boolean-size": lambda: setattr(source, "_size", True),
        "missing-data": lambda: delattr(source, "_data"),
        "missing-size": lambda: delattr(source, "_size"),
    }
    changes[state]()


@pytest.mark.parametrize("state", [
    "list", "shape", "dtype", "strided", "unaligned", "readonly", "negative", "oversized",
    "boolean-size", "missing-data", "missing-size",
])
def test_malformed_native_payloads_decline_without_mutating_the_destination(state: str) -> None:
    kind = stl.std.vector["double"]
    source, target = kind([1, 2]), kind([3, 4])
    storage, bits = target._data, target._data.tobytes()
    _malformed(source, state)
    with pytest.raises(TypeError, match="returned by value cannot be assigned to in place"):
        assign_into(target, source)
    assert target._data is storage and target._data.tobytes() == bits


def test_a_readonly_destination_and_overlapping_foreign_storage_are_not_written() -> None:
    kind = stl.std.vector["double"]
    source, target = kind([1, 2]), kind([3, 4])
    target._data.setflags(write=False)
    with pytest.raises(TypeError, match="returned by value"):
        assign_into(target, source)
    assert target.data().tobytes() == np.array([3., 4.]).tobytes()
    target._data.setflags(write=True)
    source._data, source._size = target._data[1:], 1
    before = target._data.tobytes()
    with pytest.raises(TypeError, match="returned by value"):
        assign_into(target, source)
    assert target._data.tobytes() == before


def test_bool_nested_and_subclass_values_keep_their_existing_unsupported_whole_operator() -> None:
    numeric = stl.std.vector["double"]

    class Derived(numeric):
        __slots__ = ()

    for kind in (stl.std.vector["bool"], stl.std.vector[numeric], Derived):
        left, right = kind(), kind()
        with pytest.raises(TypeError, match="returned by value"):
            assign_into(left, right)


def test_foreign_metaclass_hashes_are_not_entered_by_native_dispatch() -> None:
    calls: list[str] = []

    class Meta(type):
        def __hash__(self) -> int:
            calls.append("hash")
            raise AssertionError("Foreign class hashing is not native dispatch.")

    class Foreign(metaclass=Meta):
        pass

    value = Foreign()
    assert vectorassign.assigned(value, value) is False and calls == []



def test_foreign_sources_and_array_subclasses_decline_without_payload_callbacks() -> None:
    kind = stl.std.vector["double"]
    target = kind([3, 4])
    calls: list[str] = []

    class Foreign:
        def __iter__(self) -> Any:
            calls.append("iter")
            raise AssertionError("An unsupported source must not be iterated.")

    class Array(np.ndarray[Any, Any]):
        def __getattribute__(self, name: str) -> Any:
            calls.append(name)
            return super().__getattribute__(name)

    with pytest.raises(TypeError, match="returned by value"):
        assign_into(target, Foreign())
    source = kind([1, 2])
    source._data = source._data.view(Array)
    calls.clear()
    with pytest.raises(TypeError, match="returned by value"):
        assign_into(target, source)
    assert calls == [] and list(target) == [3, 4]


def test_changed_factory_dtype_does_not_enter_the_raw_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = stl.std.vector["double"]
    source, target = kind([1, 2]), kind([3, 4])
    storage, bits = target._data, target._data.tobytes()
    monkeypatch.setattr(kind, "dtype", np.dtype("int32"))
    with pytest.raises(TypeError, match="returned by value"):
        assign_into(target, source)
    assert target._data is storage and target._data.tobytes() == bits


def test_foreign_assignment_does_not_import_an_absent_native_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "xrdroot.pyroot.vectorassign")
    left, right = [1], [2, 3]
    assert assign_into(left, right) is right and left == [2, 3]
    old, new = {"old": 1}, {"new": 2}
    assert assign_into(old, new) is new and old == new
    assert "xrdroot.pyroot.vectorassign" not in sys.modules


def test_custom_attribute_lookup_keeps_the_original_pointer_callback_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = stl.std.vector["double"]
    source, target = kind([1, 2]), kind([3, 4])
    storage, bits = target._data, target._data.tobytes()
    calls: list[tuple[bool, str]] = []

    def read(self: Any, name: str) -> Any:
        calls.append((self is source, name))
        return object.__getattribute__(self, name)

    monkeypatch.setattr(kind, "__getattribute__", read)
    expected = _outcome(lambda: store_through(target, value_copy(source)))
    expected_calls = list(calls)
    calls.clear()
    actual = _outcome(lambda: assign_vector(target, source, True))
    assert actual == expected and calls == expected_calls
    assert object.__getattribute__(target, "_data") is storage and storage.tobytes() == bits
