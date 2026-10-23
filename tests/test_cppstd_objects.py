"""Generic objects keep array, overload and algorithm behavior through legacy imports."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from cppstdsupport import isolated


def test_generic_objects_keep_legacy_identity_and_old_pickle_references() -> None:
    from xrdroot.cint.runtime import objects as legacy
    from xrdroot.cppstd import objects
    from xrdroot.cppstd.valuecopy import value_copy

    for name in objects.__all__:
        assert getattr(legacy, name) is getattr(objects, name), name
    assert legacy.DTYPES is objects.DTYPES
    assert legacy.Candidate is objects.Candidate
    assert objects.value_copy is value_copy
    for name in ("Pair", "CppException", "Overloaded"):
        old_reference = f"cxrdroot.cint.runtime.objects\n{name}\n.".encode()
        assert pickle.loads(old_reference) is getattr(objects, name)


def test_root_object_adapters_keep_destruction_placement_and_dereference_behavior() -> None:
    from xrdroot.cint.runtime import objects

    events = []
    objects.delete(SimpleNamespace(_destruct=lambda: events.append("destruct"),
                                   Close=lambda: events.append("close"), IsOpen=lambda: True))
    objects.delete(SimpleNamespace(Close=lambda: events.append("close"), IsOpen=lambda: True))
    assert events == ["destruct", "close"]
    value = object()
    slots = [None]
    assert objects.construct_at(slots, 0, value) is value
    assert slots == [value]
    added = []
    assert objects.construct_at(SimpleNamespace(AddAt=lambda obj, at: added.append((obj, at))),
                                "2", value) is value
    assert added == [(value, 2)]
    assert objects.deref([value]) is value
    root_list = type("RootList", (list,), {"ClassName": lambda self: "TList"})([value])
    assert objects.deref(root_list) is root_list
    assert objects.deref(SimpleNamespace(value=value, _cint_cell=True)) is value


def test_generic_object_arrays_and_copying_preserve_widths_and_independence() -> None:
    from xrdroot.cppstd import objects

    values = objects.array("short", (2, 3), [[1, 2], [3]])
    assert values.dtype == np.int16
    assert values.tolist() == [[1, 2, 0], [3, 0, 0]]
    copied = objects.value_copy(values)
    values[0, 0] = 9
    assert copied[0, 0] == 1
    made = objects.array("Object", 2, ["a"], lambda value="": SimpleNamespace(value=value))
    assert [each.value for each in made] == ["a", ""]
    assert made[0] is not made[1]


def test_overloads_and_cpp_exceptions_keep_their_argument_and_value_contracts() -> None:
    from xrdroot.cppstd import objects

    overload = objects.Overloaded("value", (lambda value: "int", 1, 1, (int,)),
                                 (lambda value: "other", 1, 1, (None,)))
    assert overload(3) == "int"
    assert overload("x") == "other"
    with pytest.raises(TypeError, match="no overload of value takes 0 arguments"):
        overload()
    exception = objects.CppException("failure")
    assert exception.what() == "failure"
    with pytest.raises(objects.CppException) as caught:
        objects.throw(exception)
    assert caught.value is exception
    with pytest.raises(objects.CppException) as caught:
        objects.throw(7)
    assert caught.value.value == 7


def test_generic_algorithms_and_arrays_run_without_root_or_the_interpreter() -> None:
    isolated("""
from types import SimpleNamespace
from xrdroot.cppstd import objects
assert objects.array('int', 4, [3, 1, 2]).tolist() == [3, 1, 2, 0]
values = [3, 1, 2, 9]
objects.sort_range(values, 0, 3, lambda left, right: left > right)
assert values == [3, 2, 1, 9]
objects.reverse_range(values, 1, None)
assert values == [3, 9, 1, 2]
assert list(objects.iterate({'key': 7})) == [objects.Pair('key', 7)]
assert objects.Pair(1, 2) < objects.Pair(2, 1)
held = SimpleNamespace(value=3)
assert objects.preinc(held, 2) == 5
assert objects.postinc(held, -1) == 5
assert held.value == 4
assert objects.set_item(values, 0, 8) == 8
assert objects.set_attr(held, 'value', 6) == 6
assert held.value == 6
assert objects.dynamic_cast(dict, values) is None
assert objects.dynamic_cast(list, values) is values
assert objects.pointee(held, 'get') is held
assert objects.pointee(SimpleNamespace(get=lambda: held), 'get') is held
assert not {'delete', 'construct_at', 'deref'} & set(objects.__all__)
""")
