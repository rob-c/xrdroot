"""Generic arrays, overloads and algorithms work through their direct standard APIs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from xrdroot.cppstd import cells, objects


def test_numeric_arrays_pad_rows_and_copy_raw_values_into_independent_storage() -> None:
    values = objects.array("short", (2, 3), [[1, 2], [3]])
    assert values.dtype == np.int16
    assert values.tolist() == [[1, 2, 0], [3, 0, 0]]
    copied = objects.value_copy(values)
    values[0, 0] = 9
    assert copied.tolist() == [[1, 2, 0], [3, 0, 0]]
    assert not np.shares_memory(values, copied)


def test_object_arrays_construct_each_element_and_convert_explicit_initializers() -> None:
    def make(value=""):
        return SimpleNamespace(value=value)

    made = objects.array("Object", 2, ["a"], make)
    assert [item.value for item in made] == ["a", ""] and made[0] is not made[1]
    nested = objects.array("Object", (2, 2), make=make)
    nested[0][0].value = "changed"
    assert [[item.value for item in row] for row in nested] == [["changed", ""], ["", ""]]


def test_overload_sets_select_types_check_arity_and_bind_instance_methods() -> None:
    overload = objects.Overloaded(
        "value", (lambda value: "int", 1, 1, (int,)), (lambda value: "other", 1, 1, (None,))
    )
    assert overload(3) == "int" and overload("x") == "other"
    with pytest.raises(TypeError, match="no overload of value takes 0 arguments"):
        overload()

    class Holder:
        offset = 4
        value = objects.Overloaded(
            "value", (lambda self, value: self.offset + value, 2, 2, (None, int))
        )

    assert Holder().value(3) == 7 and Holder.value is vars(Holder)["value"]


def test_range_algorithms_change_only_the_requested_range_and_keep_value_pairs() -> None:
    values = [3, 1, 2, 9]
    objects.sort_range(values, 0, 3, lambda left, right: left > right)
    assert values == [3, 2, 1, 9]
    objects.reverse_range(values, 1, None)
    assert values == [3, 9, 1, 2]
    assert list(objects.iterate({"key": 7})) == [objects.Pair("key", 7)]
    assert tuple(objects.Pair(1, 2)) == (1, 2)
    assert objects.Pair(1, 2) < objects.Pair(2, 1)


def test_generic_stores_and_increments_preserve_return_values_and_original_references() -> None:
    value = cells.Cell(3, "int")
    assert objects.preinc(value, 2) == 5 and objects.postinc(value, -1) == 5
    assert value.value == 4
    items = [0]
    assert objects.set_item(items, 0, 8) == 8 and items == [8]
    assert objects.set_attr(value, "value", 6) == 6 and value.value == 6
    assert objects.dynamic_cast(list, items) is items
    assert objects.dynamic_cast(dict, items) is None
    assert objects.pointee(value, "get") is value
    assert objects.pointee(SimpleNamespace(get=lambda: value), "get") is value


def test_thrown_exceptions_preserve_the_original_value_and_exception_identity() -> None:
    exception = objects.CppException("failure")
    assert exception.what() == "failure"
    with pytest.raises(objects.CppException) as caught:
        objects.throw(exception)
    assert caught.value is exception
    with pytest.raises(objects.CppException) as caught:
        objects.throw(7)
    assert caught.value.value == 7 and caught.value.what() == "7"
