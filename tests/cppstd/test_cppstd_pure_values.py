"""Reference stores and bit-preserving values stay within the generic runtime."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from xrdroot.cppstd import cells, containers, references, valuecopy


def test_value_copy_preserves_raw_floating_payloads_and_independent_storage() -> None:
    bits = np.array([0x80000000, 0x7F800043, 0x7FC00042], dtype="uint32")
    original = bits.view("float32")
    copied = valuecopy.value_copy(original)
    assert copied.view("uint32").tobytes() == bits.tobytes()
    assert not np.shares_memory(original, copied)


def test_numeric_vector_values_keep_nan_payloads_signed_zero_and_independent_storage() -> None:
    bits = np.array([0x80000000, 0x7F800001, 0x7FC00042, 1], dtype="uint32")
    leaf = containers.std.vector["float"]
    source = leaf(bits.view("float32"))
    copied = containers.std.vector[leaf]([source, source])
    assert copied[0] is not source and copied[0] is not copied[1]
    for item in copied:
        assert item.data().view("uint32").tobytes() == bits.tobytes()
        assert not np.shares_memory(item.data(), source.data())


def test_cell_item_and_attribute_references_write_the_storage_they_capture() -> None:
    number = cells.Cell(1, "int")
    assert references.store_through(number, 7) == 7 and number[0] == 7
    values = [2, 3, 4]
    item = cells.ItemRef(values, 1)
    assert references.store_through(item, 8) == 8 and values == [2, 8, 4]
    item[1] = 9
    assert values == [2, 8, 9]
    owner = SimpleNamespace(value=2)
    member = cells.AttrRef(owner, "value")
    assert references.store_through(member, 10) == 10 and owner.value == 10
    with pytest.raises(IndexError, match="only element 0"):
        number[1] = 4


def test_generic_assignment_updates_held_containers_and_keeps_the_return_contract() -> None:
    target, source = [1], [2, 3]
    assert references.assign_into(target, source) is source and target == [2, 3]
    mapping, entries = {"old": 1}, {"new": 2}
    assert references.assign_into(mapping, entries) is entries and mapping == {"new": 2}
    grid = np.zeros((2, 2))
    assert references.assign_call(grid, (0, 1), 7) == 7 and grid[0, 1] == 7
    with pytest.raises(TypeError, match="returned by value"):
        references.assign_into(3, 4)
