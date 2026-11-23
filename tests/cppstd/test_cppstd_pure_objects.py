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


def test_generate_puts_one_call_into_each_element_of_the_span_and_no_other() -> None:
    from xrdroot.cppstd import objects

    values = [0, 0, 9, 9]
    objects.generate_range(values, 0, 2, lambda: 7)
    assert values == [7, 7, 9, 9]
    objects.generate_range(values, 1, None, lambda: 1)
    assert values == [7, 1, 1, 1]


def test_shuffle_permutes_the_span_in_place_with_the_default_or_given_engine() -> None:
    from xrdroot.cppstd import objects, random

    values = list(range(20))
    objects.shuffle_range(values, 0, None, random.default_random_engine(3))
    assert sorted(values) == list(range(20)) and values != sorted(values)
    again = list(range(20))
    objects.shuffle_range(again, 0, 20, random.default_random_engine(3))
    assert again == values  # the same engine and seed draw the same swaps
    tail = [1, 2, 3]
    objects.shuffle_range(tail, 0, 1)
    assert tail == [1, 2, 3]  # a span of one has nowhere to swap


def test_next_permutation_walks_the_orders_and_rewinds_to_the_first() -> None:
    """Each call is one step of the lexicographic order, with a duplicate treated as C++ does:
    the last order rewinds to the first and answers false."""
    from xrdroot.cppstd import objects

    values = [1, 2, 3]
    assert objects.next_permutation_range(values, 0, None) is True
    assert values == [1, 3, 2]
    assert objects.next_permutation_range(values, 0, 3) is True
    assert values == [2, 1, 3]
    assert objects.next_permutation_range([3, 2, 1], 0, None) is False
    assert objects.next_permutation_range([1, 2, 2], 0, 3) is True
    # The suffix is descending, so its last element is its smallest and the successor is
    # often one back: ``[2, 3, 1]`` takes the 3, not the 1 it ends with.
    backwards = [2, 3, 1]
    assert objects.next_permutation_range(backwards, 0, 3) is True
    assert backwards == [3, 1, 2]


def test_next_permutation_rewinds_the_last_order_and_leaves_one_element_alone() -> None:
    from xrdroot.cppstd import objects

    rewind = [3, 2, 1]
    assert objects.next_permutation_range(rewind, 0, 3) is False
    assert rewind == [1, 2, 3]
    single = [5]
    assert objects.next_permutation_range(single, 0, None) is False
    assert single == [5]
    assert objects.next_permutation_range([1, 2, 2], 0, 3) is True


def test_the_last_permutation_rewinds_to_the_first_and_one_element_is_the_last() -> None:
    from xrdroot.cppstd import objects

    rewind = [3, 2, 1]
    assert objects.next_permutation_range(rewind, 0, None) is False
    assert rewind == [1, 2, 3]
    single = [5]
    assert objects.next_permutation_range(single, 0, 3) is False
    assert single == [5]
    assert objects.next_permutation_range([3, 2, 1], 0, 3) is False


def test_a_value_is_stored_where_a_c_pointer_would_put_it() -> None:
    """``store``/``load``: the two shapes an out-parameter has here - a cell's ``.value`` and an
    array's first element - and the refusal for something that is neither. A null pointer is
    nowhere, and writing through one does nothing."""
    cell = cells.Cell(0.0)
    cells.store(cell, 2.5)
    assert cells.load(cell) == 2.5
    array = np.zeros(2)
    cells.store(array, 3.5)
    assert array[0] == 3.5 and cells.load(array) == 3.5
    cells.store(None, 1)
    with pytest.raises(TypeError, match="not somewhere to put a number"):
        cells.store(object(), 1)
