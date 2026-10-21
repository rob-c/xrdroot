"""Native vector values copy per occurrence while stored elements remain references."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xrdroot.pyroot import stl, vectorvalue
from xrdroot.pyroot.rdf.rvec import RVec


@pytest.mark.parametrize("outer", [stl.std.vector, RVec], ids=["std", "rvec"])
@pytest.mark.parametrize("leaf", [stl.std.vector["int"], RVec["int"]], ids=["std", "rvec"])
def test_construction_push_emplace_assign_and_fill_copy_each_native_leaf(
    outer: Any, leaf: Any,
) -> None:
    source = leaf([1, 2])
    kind = outer[leaf]
    values = kind([source, source])
    values.push_back(source)
    values.emplace_back(source)
    values.resize(6, source)
    assert len(values) == 6 and all(type(item) is leaf for item in values)
    assert len({id(item) for item in values}) == 6
    assert all(item is not source and list(item) == [1, 2] for item in values)
    source[0] = 9
    assert [list(item) for item in values] == [[1, 2]] * 6
    values[0][1] = 7
    assert list(values[1]) == [1, 2]


@pytest.mark.parametrize("outer", [stl.std.vector, RVec], ids=["std", "rvec"])
@pytest.mark.parametrize("leaf", [stl.std.vector["int"], RVec["int"]], ids=["std", "rvec"])
def test_assign_copies_the_native_leaf_values_before_replacing_the_outer_sequence(
    outer: Any, leaf: Any,
) -> None:
    values = outer[leaf]([leaf([1, 2]), leaf([3, 4])])
    assigned = outer[leaf]()
    assigned.assign(values)
    assert all(new is not old for new, old in zip(assigned, values, strict=True))
    assert [list(item) for item in assigned] == [list(item) for item in values]
    assigned[0][0] = 7
    assert list(values[0]) == [1, 2]


@pytest.mark.parametrize("outer", [stl.std.vector, RVec], ids=["std", "rvec"])
@pytest.mark.parametrize("leaf", [stl.std.vector["double"], RVec["double"]], ids=["std", "rvec"])
def test_assignment_updates_the_existing_element_and_self_assignment_keeps_its_storage(
    outer: Any, leaf: Any,
) -> None:
    values = outer[leaf]([leaf([1., 2.]), leaf([3., 4.])])
    held = values[0]
    sibling = values[1]
    source = leaf([8., 9.])
    values[0] = source
    assert values[0] is held and values.at(0) is held and values.front() is held
    assert values.begin().__deref__() is held and values.back() is sibling
    assert list(held) == [8., 9.] and held is not source
    source[0] = 10.
    assert list(held) == [8., 9.] and list(sibling) == [3., 4.]
    storage = held._data
    values[0] = held
    assert held._data is storage and values[0] is held and list(held) == [8., 9.]
    values[-1] = source
    assert values[-1] is sibling and list(sibling) == [10., 9.]


@pytest.mark.parametrize("outer", [stl.std.vector, RVec], ids=["std", "rvec"])
def test_nested_assignment_keeps_common_element_references_while_growing_and_shrinking(
    outer: Any,
) -> None:
    leaf = RVec["unsigned long"]
    middle = outer[leaf]
    top = outer[middle]
    values = top([middle([leaf([1]), leaf([2])])])
    held = values[0]
    common = held[0]
    replacement = middle([leaf([7]), leaf([8]), leaf([9])])
    values[0] = replacement
    assert values[0] is held and held[0] is common
    assert [list(item) for item in held] == [[7], [8], [9]]
    assert all(new is not old for new, old in zip(held, replacement, strict=True))
    replacement[0][0] = 99
    assert list(common) == [7]
    values[0] = middle([leaf([4])])
    assert values[0] is held and held[0] is common and len(held) == 1
    assert list(common) == [4]


@pytest.mark.parametrize(("floating", "integer", "bits"), [
    ("float", "uint32", [0x80000000, 0x7F800001, 0x7FC00042, 1, 0x7F800000]),
    ("double", "uint64", [0x8000000000000000, 0x7FF0000000000001,
                          0x7FF8000000000042, 1, 0x7FF0000000000000]),
])
@pytest.mark.parametrize("maker", [stl.std.vector, RVec], ids=["std", "rvec"])
def test_numeric_value_copy_keeps_signaling_nan_payload_signed_zero_and_subnormal_bits(
    floating: str, integer: str, bits: list[int], maker: Any,
) -> None:
    words = np.array(bits, dtype=integer)
    dtype = "float32" if floating == "float" else "float64"
    leaf = maker[floating]
    source = leaf(words.view(dtype))
    source.reserve(17)
    values = maker[leaf]([source, source])
    for item in values:
        assert type(item) is leaf and item is not source
        assert np.array_equal(item.data().view(integer), words)
        assert not np.shares_memory(item.data(), source.data())
    held = values[0]
    values[0] = source
    assert values[0] is held and np.array_equal(held.data().view(integer), words)
    assert np.array_equal(source.data().view(integer), words)


@pytest.mark.parametrize(("name", "given"), [
    ("bool", [True, False]), ("unsigned long", [0, 2**64 - 1]),
])
def test_bool_and_uint64_values_copy_without_object_dtype_or_integer_rounding(
    name: str, given: list[Any],
) -> None:
    leaf = stl.std.vector[name]
    source = leaf(np.array(given, dtype=leaf.dtype))
    values = RVec[leaf]([source])
    assert values[0] is not source and type(values[0]) is leaf
    assert values[0].dtype.kind in "bu"
    assert values[0].data().tobytes() == source.data().tobytes()


def test_user_vector_subclasses_keep_their_original_protocol_and_reference_behavior() -> None:
    calls: list[str] = []

    class Custom(stl.std.vector["int"]):
        __cpp_name__ = "ValueCopyCustomIntVector"

        def __init__(self) -> None:
            calls.append("construct")
            super().__init__([1])

        def __deepcopy__(self, memo: Any) -> Any:
            raise AssertionError("An excluded custom vector must not enter deepcopy.")

    source = Custom()
    values = stl.std.vector[Custom]([source])
    values.push_back(source)
    values.resize(3, source)
    values[0] = source
    assert all(item is source for item in values)
    assert calls == ["construct"] and not vectorvalue.native(Custom)


def test_pointer_string_pair_and_map_conversion_keep_their_existing_behavior() -> None:
    source = object()
    pointers = stl.std.vector["SomeUserClass*"]([source])
    pointers.push_back(source)
    pointers.resize(3, source)
    pointers[0] = source
    assert all(item is source for item in pointers)
    word = stl.std.string("text")
    words = stl.std.vector["string"]([word])
    assert str(words[0]) == "text" and words[0] is not word
    leaf = stl.std.vector["int"]([1])
    assert stl._converted(type(leaf), leaf) is leaf
    pair = stl.std.pair["int", type(leaf)](1, leaf)
    mapping = stl.std.map["int", type(leaf)]()
    mapping[1] = leaf
    assert pair.second is leaf and mapping[1] is leaf


def test_custom_indices_keep_one_index_conversion_after_value_conversion() -> None:
    calls: list[str] = []

    class Index:
        def __index__(self) -> int:
            calls.append("index")
            return 0

    leaf = stl.std.vector["int"]
    source = leaf([7])
    values = stl.std.vector[leaf]([leaf([1])])
    values[Index()] = source
    assert calls == ["index"] and list(values[0]) == [7] and values[0] is not source


def test_factory_metadata_changes_object_storage_and_unknown_kinds_decline_copy_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not vectorvalue.native(None) and not vectorvalue.native(object)
    leaf = stl.std.vector["short"]
    assert vectorvalue.native(leaf)
    monkeypatch.setattr(leaf, "dtype", np.dtype(object))
    assert not vectorvalue.native(leaf)
    outer = RVec[RVec["float"]]
    assert vectorvalue.native(outer)
    monkeypatch.setattr(outer, "element", stl.string)
    assert not vectorvalue.native(outer)


@pytest.mark.parametrize("raising", [False, True], ids=["normal", "raising"])
def test_foreign_value_conversion_runs_before_the_invalid_index_error(raising: bool) -> None:
    calls: list[str] = []

    class Values:
        def __iter__(self) -> Any:
            calls.append("iterate")
            if raising:
                raise ValueError("Foreign conversion failed first.")
            return iter([7])

    leaf = stl.std.vector["int"]
    values = stl.std.vector[leaf]([leaf([1])])
    error = ValueError if raising else IndexError
    message = "Foreign conversion failed first" if raising else "list assignment index out of range"
    with pytest.raises(error, match=message):
        values[9] = Values()
    assert calls == ["iterate"] and list(values[0]) == [1]


def test_a_foreign_value_and_custom_index_keep_rhs_conversion_then_one_index_callback() -> None:
    calls: list[str] = []

    class Values:
        def __iter__(self) -> Any:
            calls.append("iterate")
            return iter([7])

    class Index:
        def __index__(self) -> int:
            calls.append("index")
            return 9

    leaf = stl.std.vector["int"]
    values = stl.std.vector[leaf]([leaf([1])])
    with pytest.raises(IndexError, match="list assignment index out of range"):
        values[Index()] = Values()
    assert calls == ["iterate", "index"] and list(values[0]) == [1]


def test_an_exact_source_subclass_does_not_enter_native_assignment_or_conversion_hooks() -> None:
    calls: list[str] = []
    leaf = stl.std.vector["int"]

    class Custom(leaf):
        __cpp_name__ = "ValueCopySourceSubclass"

        def data(self) -> Any:
            calls.append("data")
            raise AssertionError("A source subclass must keep the ordinary conversion path.")

    source = Custom()
    values = stl.std.vector[leaf]([leaf([1])])
    with pytest.raises(IndexError, match="list assignment index out of range"):
        values[9] = source
    values[0] = source
    assert values[0] is source and calls == []


def test_factory_origin_rejects_a_rebased_native_class_without_entering_its_data() -> None:
    leaf = stl._vector_class(np.dtype("int16"))
    assert vectorvalue.native(leaf)

    class Rebased(stl._Vector):
        __slots__ = ()

    original = leaf.__bases__
    try:
        leaf.__bases__ = (Rebased,)
        assert not vectorvalue.native(leaf)
    finally:
        leaf.__bases__ = original
    assert leaf.__bases__ == original and vectorvalue.native(leaf)


def test_object_storage_subclasses_keep_public_factory_identity_and_callbacks() -> None:
    calls: list[str] = []
    marker = object()

    class Objects(stl._Vector):
        __slots__ = ()
        dtype = np.dtype(object)
        __cpp_name__ = "ObjectStorageValueCopyControl"

        def __init__(self) -> None:
            calls.append("construct")
            super().__init__(np.array([marker], dtype=object))

        def data(self) -> Any:
            calls.append("data")
            raise AssertionError("An excluded object-storage vector must not be copied.")

        def __deepcopy__(self, memo: Any) -> Any:
            raise AssertionError("An excluded object-storage vector must not enter deepcopy.")

    source = Objects()
    kind = stl.std.vector[Objects]
    values = kind([source])
    values.push_back(source)
    values.resize(3, source)
    values[0] = source
    assert not vectorvalue.native(Objects) and not vectorvalue.native(kind)
    assert all(item is source for item in values) and source._data[0] is marker
    assert calls == ["construct"]


def test_nested_assignment_keeps_foreign_subclass_values_on_the_ordinary_path() -> None:
    leaf = stl.std.vector["int"]

    class Custom(leaf):
        __cpp_name__ = "NestedValueCopySourceSubclass"

    source = Custom([7])
    middle = stl.std.vector[leaf]
    original = middle([leaf([1])])
    replacement = middle([source])
    values = stl.std.vector[middle]([original])
    held = values[0]
    values[0] = replacement
    assert values[0] is held and held[0] is source


@pytest.mark.parametrize("index", [1, -2, 9, -9])
@pytest.mark.parametrize("maker", [stl.std.vector, RVec], ids=["std", "rvec"])
def test_invalid_native_indices_keep_the_original_assignment_error_and_unchanged_values(
    index: int, maker: Any,
) -> None:
    leaf = maker["int"]
    source = leaf([7])
    values = maker[leaf]([leaf([1])])
    held = values[0]
    with pytest.raises(IndexError, match=r"^list assignment index out of range$"):
        values[index] = source
    assert values[0] is held and list(held) == [1] and list(source) == [7]
