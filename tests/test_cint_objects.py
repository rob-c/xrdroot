"""The runtime's cells, arrays, overloads, input streams and object helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from xrdroot.cint import runtime as rt


def test_a_cell_is_read_and_written_as_value_and_as_element_zero() -> None:
    cell = rt.Cell(1.5, "double")
    cell[0] = 2.5
    assert (cell.value, cell[0], cell.ctype, repr(cell)) == (2.5, 2.5, "double", "Cell(2.5)")
    with pytest.raises(IndexError, match="has only element 0, not element 1"):
        cell[1]


def test_references_to_elements_and_members_read_and_write_through() -> None:
    values = [1, 2, 3]
    ref = rt.ItemRef(values, 1)
    ref.value = 20
    ref[1] = 30
    assert (values, ref.value, ref[1]) == ([1, 20, 30], 20, 30)
    holder = types.SimpleNamespace(x=1)
    member = rt.AttrRef(holder, "x")
    member.value = 5
    member[0] = member[0] + 1
    assert (holder.x, member.value) == (6, 6)


def test_arrays_are_numpy_of_the_declared_width_padded_as_c_pads() -> None:
    floats = rt.array("float", 4, [1, 2])
    assert floats.dtype == np.float32 and floats.tolist() == [1, 2, 0, 0]
    grid = rt.array("int", (2, 3), [[1, 2], [3]])
    assert grid.tolist() == [[1, 2, 0], [3, 0, 0]]
    assert rt.array("double", [2]).tolist() == [0.0, 0.0]
    assert rt.array("TH1F*", 2) == [None, None]
    assert rt.array("TString", (2, 2), make=str) == [["", ""], ["", ""]]
    assert rt.array("char*", 3, ["a", "b"]) == ["a", "b", None]


def test_an_overload_set_calls_the_candidate_the_arguments_fit() -> None:
    chosen = rt.Overloaded(
        "f",
        (lambda a: "int", 1, 1, (rt.INTEGRAL,)),
        (lambda a: "real", 1, 1, (rt.REAL,)),
        (lambda a, b=0: "two", 1, 2, (None, None)),
    )
    assert (chosen(1), chosen(1.5), chosen(1, 2), chosen("s")) == ("int", "real", "two", "two")
    with pytest.raises(TypeError, match="no overload of f takes 3 arguments"):
        chosen(1, 2, 3)

    class Holder:
        method = rt.Overloaded("method", (lambda self, a: a * 2, 2, 2, (None, None)))

    assert Holder().method(4) == 8
    assert Holder.method is Holder.__dict__["method"]


def test_delete_runs_a_destructor_or_closes_a_file() -> None:
    events = []
    rt.delete(types.SimpleNamespace(_destruct=lambda: events.append("destructed")))
    rt.delete(types.SimpleNamespace(Close=lambda: events.append("closed"), IsOpen=lambda: True))
    rt.delete(types.SimpleNamespace(Close=lambda: events.append("not a file")))
    rt.delete(None)
    assert events == ["destructed", "closed"]


def test_a_copy_of_an_object_is_a_new_object() -> None:
    original = types.SimpleNamespace(values=[1])
    copied = rt.value_copy(original)
    copied.values.append(2)
    assert original.values == [1]
    array = np.zeros(2)
    assert rt.value_copy(array) is not array
    assert rt.value_copy(3) == 3 and rt.value_copy(None) is None

    class Stubborn:
        def __deepcopy__(self, memo: object) -> None:
            raise TypeError("no")

    stubborn = Stubborn()
    assert isinstance(rt.value_copy(stubborn), Stubborn)


def test_dynamic_cast_gives_none_for_the_wrong_class() -> None:
    assert rt.dynamic_cast(int, 3) == 3
    assert rt.dynamic_cast(str, 3) is None
    assert rt.dynamic_cast("not a class", 3) == 3


def test_stores_inside_expressions_give_back_what_they_stored() -> None:
    values = [0]
    holder = types.SimpleNamespace(x=0)
    assert rt.set_item(values, 0, 5) == 5 and values == [5]
    assert rt.set_attr(holder, "x", 6) == 6 and holder.x == 6
    cell = rt.Cell(1)
    assert (rt.preinc(cell, 1), rt.postinc(cell, 1), cell.value) == (2, 2, 3)


def test_a_pair_has_first_and_second_and_unpacks() -> None:
    pair = rt.Pair(1, "a")
    first, second = pair
    assert (first, second, repr(pair)) == (1, "a", "Pair(1, 'a')")
    assert pair == rt.Pair(1, "a") and pair != (1, "a")
    assert sorted([rt.Pair(2, 0), rt.Pair(1, 5)])[0].first == 1


def test_a_range_for_over_a_map_visits_pairs() -> None:
    assert [tuple(p) for p in rt.iterate({"a": 1})] == [("a", 1)]
    assert list(rt.iterate([3, 4])) == [3, 4]
    assert list(rt.iterate(np.arange(2))) == [0, 1]


def test_sort_and_reverse_act_in_place_over_a_range() -> None:
    values = [3, 1, 2, 0]
    rt.sort_range(values, 0, 3)
    assert values == [1, 2, 3, 0]
    rt.sort_range(values, 0, None, lambda a, b: a > b)
    assert values == [3, 2, 1, 0]
    rt.sort_range(values, 0, None, lambda a, b: False)
    rt.reverse_range(values, 1, None)
    assert values == [3, 0, 1, 2]
    array = np.array([2.0, 1.0])
    rt.reverse_range(array, 0, 2)
    assert array.tolist() == [1.0, 2.0]


def test_a_throw_raises_what_was_thrown() -> None:
    with pytest.raises(rt.CppException) as caught:
        rt.throw(3)
    assert (caught.value.value, caught.value.what()) == (3, "3")
    with pytest.raises(ValueError, match="bad"):
        rt.throw(ValueError("bad"))
    wrapped = rt.CppException(types.SimpleNamespace(what=lambda: "why"))
    assert wrapped.what() == "why"


def test_dereferencing_an_unknown_pointer_reads_what_it_points_at() -> None:
    assert rt.deref(rt.Cell(4)) == 4
    assert rt.deref(np.array([5.0])) == 5.0
    assert rt.deref([6]) == 6
    thing = object()
    assert rt.deref(thing) is thing


def test_root_is_looked_up_late_in_whatever_was_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    bound = types.SimpleNamespace(TH1F="histogram")
    with rt.ROOT.bind(bound):
        assert rt.ROOT.TH1F == "histogram"
        with pytest.raises(AttributeError):
            rt.ROOT.__wrapped__  # noqa: B018
    fake = types.ModuleType("xrdroot.pyroot")
    fake.TH1D = "default"  # type: ignore[attr-defined]
    stl = types.ModuleType("xrdroot.pyroot.stl")
    stl.std = "the std"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xrdroot.pyroot", fake)
    monkeypatch.setitem(sys.modules, "xrdroot.pyroot.stl", stl)
    assert rt.ROOT.TH1D == "default"
    assert rt.ROOT.std == "the std"
    with pytest.raises(AttributeError):
        rt.ROOT.Missing  # noqa: B018
    assert "did not declare" in repr(rt.ROOT)


def test_an_input_stream_reads_numbers_words_and_characters(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("1 2.5 word x\nnext line\n")
    stream = rt.ifstream(str(path))
    got = [stream.extract(kind) for kind in ("int", "double", "std::string", "char")]
    assert got == [1, 2.5, "word", ord("x")]
    assert stream.getline() == "" and stream.getline() == "next line"
    assert stream.good() and stream.is_open() and bool(stream)
    assert stream.extract("int") == 0 and stream.fail() and stream.eof()
    assert stream.extract("std::string") == ""
    assert stream.getline() == ""
    stream.close()
    assert not stream.is_open()


def test_a_stream_on_a_missing_file_is_failed_and_not_open(tmp_path: Path) -> None:
    stream = rt.ifstream(str(tmp_path / "missing.txt"))
    assert stream.fail() and not stream.is_open()
    unopened = rt.ifstream()
    assert not unopened.is_open()


def test_a_string_stream_reads_its_string() -> None:
    stream = rt.istringstream("3 4")
    assert (stream.extract("int"), stream.extract("float")) == (3, 4.0)
    assert stream.str() == "3 4"
    assert stream.str("7") == "" and stream.extract("long") == 7
    assert not rt.istream("x").extract("int")


def test_arguments_of_no_candidates_kinds_go_to_the_first_that_takes_as_many() -> None:
    only = rt.Overloaded("g", (lambda a: f"got {a}", 1, 1, (rt.INTEGRAL,)))
    assert only("text") == "got text"


def test_a_macro_assigning_one_of_roots_globals_assigns_it_in_root() -> None:
    bound = types.SimpleNamespace(gErrorIgnoreLevel=0)
    with rt.ROOT.bind(bound):
        rt.ROOT.gErrorIgnoreLevel = 2000
    assert bound.gErrorIgnoreLevel == 2000
