"""The containers ``roottest/root/io/bigevent``'s ``Event`` holds, read out of an entry.

That roottest writes ten ``Event`` objects into one unsplit branch and reads them all back, and
``Event.h`` fills its members with the shapes that are hardest to lay out: a map whose *keys* are
objects, a pointer to one, a multimap, a class that derives from a container, an array of
containers, a list of pointers to strings and a counted array of ``TString``.

The bytes below are the writer's - :mod:`xrdroot.wdobjects` writes exactly these, and the
``write`` half of that roottest is what it writes them for. A map is written as a container of
pairs, and the *pair* has a description of its own in the file - ``pair<TNamed*,int>`` is a class
like any other to ROOT's streamer information - so what a key and a value look like is read out
of that description and never out of the map's own spelling: a reader that guessed it would have
to know every class a key could be.
"""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.buffer import Buffer
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.interp import (
    STL_ARRAY,
    STL_POINTER,
    _container_step,
    _held_container,
    _loop_step,
    _object_step,
    _pair_map,
)
from xrdroot.rdobjects import _member
from xrdroot.streamers import Member


def _element(
    kind: str,
    name: str,
    stype: int,
    typename: str,
    length: int = 0,
    extras: tuple[Any, ...] = (),
) -> Member:
    """One streamer element as a reader takes it: only what these shapes read is spelled out."""
    return _member((kind, name, "", stype, 0, length, 0, (0, 0, 0, 0, 0), typename, extras))


#: ``pair<K,int>`` with a ``K`` held by value, which is how a map's pairs are described.
PAIR = {
    "first": _element("TStreamerObject", "first", 61, "K"),
    "second": _element("TStreamerBasicType", "second", 3, "int"),
}


class _File:
    """A source that describes the classes these tests need and nothing else."""

    def __init__(self, described: dict[str, dict[str, Member]]) -> None:
        self._described = described

    def streamers(self) -> dict[str, dict[str, Member]]:
        return self._described

    name = "fixture.root"


def _named(kind: str) -> tuple[dict[str, dict[str, Member]], _File]:
    return {"pair<K,int>": PAIR, kind: {}}, _File({"pair<K,int>": PAIR, kind: {}})


#: One pair of a ``map<K,int>``: ``K`` written empty, then the value 7.
ONE_PAIR = "4000 0016 400a 0000 00000000 00000001 4000 0002 0000 00000007"


def _step(buf_hex: str, described: Any) -> Any:
    source = _File(described)
    step = _pair_map("map<K,int>", source, ())
    assert step is not None
    return step(Buffer(bytes.fromhex(buf_hex)), {})


def test_a_map_whose_keys_are_objects_reads_its_pairs() -> None:
    """The keys are a column and then the values are, under the pair's own version - so the
    reading is the description of the pair and nothing about the map at all."""
    source = _File({"pair<K,int>": PAIR, "K": {}})
    step = _pair_map("map<K,int>", source, ())
    assert step is not None
    assert step(Buffer(bytes.fromhex(ONE_PAIR)), {}) == [({}, 7)]
    # the second entry of the same branch is read by the same plan, described the once
    assert step(Buffer(bytes.fromhex(ONE_PAIR)), {}) == [({}, 7)]


def test_a_map_nothing_was_put_in_is_read_without_its_pair_being_described() -> None:
    """An empty map is a count and no more, so a file that holds no pairs never describes them:
    a reader that insisted on the description would refuse a file with nothing in it to read."""
    assert _step("4000 000c 400a 0000 00000000 00000000", {}) == []


def test_a_pair_a_file_does_not_describe_is_refused_in_a_sentence() -> None:
    with pytest.raises(UnsupportedFeatureError, match="does not describe a pair"):
        _step(ONE_PAIR, {"K": {}})


def test_a_map_written_pair_by_pair_is_refused_in_a_sentence() -> None:
    """ROOT writes a map of objects a member at a time; one written the other way is not a
    shape this reader decodes, and saying so is better than reading it as something else."""
    with pytest.raises(UnsupportedFeatureError, match="field by field"):
        _step("4000 000a 000a 00000001", {"pair<K,int>": PAIR, "K": {}})


def test_a_class_deriving_from_a_container_names_it_in_the_elements_name() -> None:
    """``Event``'s ``UShortVector`` is a ``vector<UShort_t>``, and ROOT describes it with the
    container in the element's name and no type name beside it."""
    member = _element("TStreamerSTL", "vector<unsigned short>", 500, "")
    assert _held_container(member) == "vector<unsigned short>"


def test_a_class_deriving_from_a_container_reads_as_the_container_it_is() -> None:
    member = _element("TStreamerSTL", "vector<unsigned short>", 500, "")
    step = _container_step(member, _File({}), ())
    assert step is not None
    assert list(step(Buffer(bytes.fromhex("4000 0006 000a 00000000")), {})) == []


def test_an_array_of_containers_reads_one_container_per_position() -> None:
    """``vector<double> f[4]`` is one record for the whole array - ROOT gives no element a
    record of its own - so reading it as a single container leaves three behind."""
    member = _element("TStreamerSTL", "vector<double>", STL_ARRAY, "vector<double>", 2)
    step = _container_step(member, _File({}), ())
    assert step is not None
    found = step(
        Buffer(bytes.fromhex("4000 0012 000a 00000001 3ff0000000000000 00000000")), {}
    )
    assert [list(one) for one in found] == [[1.0], []]


def test_a_list_of_pointers_holds_objects_and_not_the_things_they_point_at() -> None:
    """``list<string*>`` is a list of *pointers*, so its items name the class they point at -
    a ``list<string>`` is the strings themselves, and reading one as the other loses the
    pointer that is in the bytes."""
    member = _element("TStreamerSTL", "list<string*>", 500, "list<string*>")
    step = _container_step(member, _File({}), ())
    assert step is not None
    # the count, then one pointer: a class name and nothing the file describes of it
    assert step(Buffer(bytes.fromhex("4000 0011 000a 00000001 40000009 ffffffff 6800")), {}) == [
        "h"
    ]


def test_a_counted_array_of_strings_reads_the_strings_themselves() -> None:
    """``TString *name //[n]`` is one record holding ``n`` strings and no object in it."""
    member = _element(
        "TStreamerLoop", "fTstrings", 501, "TString*", 0, (1, "fCount", "Event")
    )
    step = _loop_step(member, _File({}), (), {"fCount": ("fCount",)})
    assert step is not None
    assert step(Buffer(bytes.fromhex("4000 0004 000a 0161 026262")), {"fCount": 2}) == [
        "a",
        "bb",
    ]


def test_a_pointer_to_a_map_of_objects_reads_as_the_map_it_points_at() -> None:
    """``map<TAxis*,int> *fMapTAxisp`` is described with ``kSTLp``, and ROOT writes the map
    where the pointer stands with nothing to say a pointer was there."""
    member = _element("TStreamerSTL", "map<K,int>*", STL_POINTER, "map<K,int>*")
    assert _held_container(member) == "map<K,int>"
    step = _container_step(member, _File({"pair<K,int>": PAIR, "K": {}}), ())
    assert step is not None
    assert step(Buffer(bytes.fromhex(ONE_PAIR)), {}) == [({}, 7)]


def test_an_array_of_tstrings_reads_one_string_per_position() -> None:
    """``TString fNames[12]`` is described as the one string element with the array offset on
    its type, exactly as ``vector<double> f[4]`` is, and its twelve strings go where the
    declaration said. A ``TString`` is a length byte and its characters with no record round
    a run of them, so reading the single string the element looks like leaves the rest of the
    array - and every member after it - eleven bytes out of step.
    """
    member = _element("TStreamerSTLstring", "fNames", 85, "TString", 3)
    step = _container_step(member, _File({}), ())
    assert step is not None
    assert step(Buffer(bytes.fromhex("0161 00 026262")), {}) == ["a", "", "bb"]


def test_an_array_of_pointers_reads_one_pointer_per_position() -> None:
    """``TAxis *fVaxis[3]`` is described with the array's N as the pointer element's length
    and no array offset on its type, and ROOT writes a pointer per element. Read as the one
    pointer such an element otherwise is, two elements and everything after them are lost.
    """
    member = _element("TStreamerObjectPointer", "fVaxis", 64, "TAxis*", 3)
    step = _object_step(member, _File({}), ())
    found = step(Buffer(bytes.fromhex("40000006ffffffff4b00  00000000  40000006ffffffff4b00")), {})
    assert found == ["K", None, "K"]


def test_one_pointer_is_the_object_and_not_a_list_of_one() -> None:
    """A pointer a declaration gave no N to is the single object, and ROOT describes one with
    a length of one - its maximum index - so a list would put a container where every reader
    of the member expects the object itself."""
    member = _element("TStreamerObjectPointer", "fPaxis", 64, "TAxis*", 1)
    step = _object_step(member, _File({}), ())
    assert step(Buffer(bytes.fromhex("40000006ffffffff4b00")), {}) == "K"
