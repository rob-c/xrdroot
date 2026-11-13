"""ROOT's collections with a Python list's methods, and ``TObjString`` compared as a string."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT


def listed(*names: str) -> ROOT.TList:
    made = ROOT.TList()
    for name in names:
        made.Add(ROOT.TObjString(name))
    return made


def names(collection: ROOT.TCollection) -> list[str]:
    return [str(item) for item in collection]


def test_a_collection_appends_extends_counts_and_removes_as_a_list_does():
    held = listed("a", "b")
    held.append(ROOT.TObjString("a"))
    held.extend(listed("c"))
    assert names(held) == ["a", "b", "a", "c"] and held.count("a") == 2
    first = held[0]
    held.remove(first)
    assert names(held) == ["b", "a", "c"]
    with pytest.raises(ValueError, match="not in list"):
        held.remove(first)


def test_collections_add_and_multiply_into_new_ones_or_in_place():
    left, right = listed("a"), listed("b")
    assert names(left + right) == ["a", "b"] and isinstance(left + right, ROOT.TList)
    assert names(left * 2) == names(2 * left) == ["a", "a"]
    left *= 3
    assert names(left) == ["a", "a", "a"]


def test_an_ordered_collection_reads_assigns_and_deletes_by_index_and_slice():
    held = listed("a", "b", "c")
    assert str(held[-1]) == "c" and names(held[::-1]) == ["c", "b", "a"]
    assert isinstance(held[:2], ROOT.TList)
    held[0] = ROOT.TObjString("z")
    held[1:3] = listed("y")
    assert names(held) == ["z", "y"]
    del held[0]
    del held[:]
    assert names(held) == []
    for wrong, error in ((5, IndexError), (1.0, TypeError), (True, TypeError)):
        with pytest.raises(error):
            held[wrong]
    with pytest.raises(ValueError):
        held[::0]


def test_an_ordered_collection_inserts_pops_reverses_sorts_and_finds_as_a_list_does():
    held = listed("b", "c", "a")
    held.insert(-10, ROOT.TObjString("d"))
    assert str(held.pop()) == "a" and str(held.pop(0)) == "d"
    held.reverse()
    assert names(held) == ["c", "b"]
    held.sort()
    assert names(held) == ["b", "c"]
    held.sort(key=str, reverse=True)
    assert names(held) == ["c", "b"] and held.index(held[1]) == 1
    with pytest.raises(ValueError, match="is not in list"):
        held.index(ROOT.TObjString("c"))
    with pytest.raises(IndexError):
        ROOT.TList().pop()


def test_a_hash_table_is_a_collection_that_counts_what_it_holds():
    table = ROOT.THashTable()
    held = ROOT.TObject()
    table.Add(held)
    assert len(table) == 1 and table[0] is held


def test_an_objstring_compares_and_measures_as_its_string_and_shows_as_one():
    word = ROOT.TObjString("test1")
    assert word == "test1" and word == ROOT.TString("test1") and word != 1
    assert word < ROOT.TObjString("test2") and word <= "test1"
    assert word > "test0" and word >= ROOT.TString("test1")
    assert len(word) == 5 and repr(word) == "'test1'"
    assert sorted([ROOT.TObjString("2"), ROOT.TObjString("1")]) == ["1", "2"]
