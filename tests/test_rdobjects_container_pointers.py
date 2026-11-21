"""Records of the classes a program declares, read back member by member by ``rdobjects``.

The bigevent roottest's ``Event`` is the class that names what this file covers: it holds a
``template2<template1<int> >`` by value, a ``TLine`` reached through a pointer, and - what the
reader refused until this change - a ``vector<TObject> *``, a container that is pointed at
rather than held.

ROOT writes a container the same bytes whether it is a member or what a member points at: a
record, a count and the items, with no tag in front to say a pointer stood there. So the
reader is given the container the pointer names, with the pointer taken off the type. An
*array* of such pointers - ``vector<TNamed> *f[6]`` - is written the other way about: one
record for the whole array, holding the count and the items of each element in turn, and no
word saying how many pointers there are, since the declaration gives that. These tests ask
the reader what it makes of each such member, and write and read the bytes back.
"""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.buffer import Buffer
from xrdroot.interp import STL_POINTERS, _Unreadable
from xrdroot.pycint.execute import run_source
from xrdroot.pyroot.core.declared import catalogue_of, find_class
from xrdroot.rdobjects import (
    STL_POINTER,
    STREAMER,
    DescribedSource,
    flat_reader_for,
    reader_for,
)
from xrdroot.streamers import Member
from xrdroot.wdobjects import DeclaredObject, Instance

#: A class of a member of every shape a container comes in: held, pointed at, and an array of
#: pointers - the streamer types ROOT describes them with, 300, 71 and 91 (71 with ROOT's
#: offset for a fixed-size array, 20).
CONTAINERS = r"""
class ContainerHolder {
public:
   Int_t fN;
   vector<int> fHeld;
   vector<int> *fPointed;
   deque<string> *fStrings;
   LINES
   ClassDef(ContainerHolder, 1)
};
"""


def _described(source_text: str, classname: str) -> tuple[Any, DescribedSource]:
    """The catalogue a macro's class reaches, and the layouts the reader asks it for."""
    run_source(source_text, "containers.C", call=False)
    catalogue = catalogue_of(find_class(classname))
    return catalogue, DescribedSource(catalogue)


def _members(source: DescribedSource, classname: str) -> dict[str, Member]:
    return source.streamers()[classname]


def test_a_member_that_is_a_pointer_to_a_container_is_read_as_the_container_it_points_at():
    """ROOT describes ``vector<TObject> *fVectorTobject`` as type 71, the container pointed
    at, and writes the container where the pointer stands: the reader is given the container
    the type names, which is the reading it already has for a container held by value."""
    _catalogue, source = _described(CONTAINERS.replace("LINES", ""), "ContainerHolder")
    pointed = _members(source, "ContainerHolder")["fPointed"]
    assert pointed.stype == STREAMER
    assert pointed.typename == "vector<int>"


def test_a_pointer_to_a_container_of_strings_is_a_container_of_strings_too():
    """The same rule for a ``deque<string> *``, which ROOT writes as the deque of strings and
    not as a pointer to one - ``Event::fListString`` is that member."""
    _catalogue, source = _described(CONTAINERS.replace("LINES", ""), "ContainerHolder")
    strings = _members(source, "ContainerHolder")["fStrings"]
    assert strings.stype == STREAMER
    assert strings.typename == "deque<string>"


def test_a_container_held_by_value_is_still_the_container_and_its_type_is_untouched():
    """The branch above must not take a star off a type that has none: a ``vector<int>``
    member reads by its own type, exactly as it did before pointers were known."""
    _catalogue, source = _described(CONTAINERS.replace("LINES", ""), "ContainerHolder")
    held = _members(source, "ContainerHolder")["fHeld"]
    assert held.stype == STREAMER
    assert held.typename == "vector<int>"


def test_a_pointer_member_that_is_not_a_container_keeps_its_star():
    """A ``TLine *`` is pointed at by name, which the reader follows, and the star is what
    says so: only a container has its pointer dropped."""
    _catalogue, source = _described(
        "class LineHolder { public: TLine *fLine; ClassDef(LineHolder, 1) };", "LineHolder"
    )
    line = _members(source, "LineHolder")["fLine"]
    assert line.typename == "TLine*"
    assert line.stype != STREAMER


def test_an_array_of_pointers_to_containers_leaves_the_type_as_the_file_describes_it():
    """``vector<TNamed> *f[2]`` - ``Event::fVectorTnamed`` - is type 91, the container pointed
    at *and* an array: the type keeps the star the file wrote, since the pointer is the
    array's and not each element's, and the array's length comes with it."""
    catalogue, source = _described(
        CONTAINERS.replace("LINES", "vector<TNamed> *fNamed[2];"), "ContainerHolder"
    )
    array = _members(source, "ContainerHolder")["fNamed"]
    assert array.stype == STL_POINTERS == STL_POINTER + 20
    assert array.typename == "vector<TNamed>*"
    assert array.length == 2
    assert reader_for("ContainerHolder", catalogue) is not None


def test_an_array_of_pointers_to_containers_reads_out_of_one_record_for_the_whole_array():
    """The member after an array of pointers to containers reads its own value, which is only
    possible if the array's elements shared one record that ended where its byte count said:
    one record per element would have left the second count where that member stood."""
    catalogue, _source = _described(
        CONTAINERS.replace("LINES", "vector<TNamed> *fNamed[2];\n   Int_t fAfter;"),
        "ContainerHolder",
    )
    named = Instance(
        "TNamed", {"TObject": {"fUniqueID": 0, "fBits": 0}, "fName": "a", "fTitle": "t"}
    )
    payload = DeclaredObject(
        Instance(
            "ContainerHolder",
            {"fN": 7, "fNamed": [[named], []], "fAfter": 9},
        ),
        catalogue,
    ).root_payload(0)[0]
    row = reader_for("ContainerHolder", catalogue)(Buffer(payload))
    assert row["fAfter"] == 9
    assert [len(one) for one in row["fNamed"]] == [1, 0]
    assert row["fNamed"][0][0]["fName"] == "a"


def test_an_array_of_pointers_to_containers_of_numbers_reads_the_runs_they_hold():
    """A container of numbers is written object-wise inside that one record - a count and the
    numbers - so each element comes back as the run of numbers a single container is."""
    catalogue, _source = _described(
        CONTAINERS.replace("LINES", "vector<int> *fNumbers[2];"), "ContainerHolder"
    )
    payload = DeclaredObject(
        Instance("ContainerHolder", {"fN": 1, "fNumbers": [[1, 2, 3], []]}), catalogue
    ).root_payload(0)[0]
    row = reader_for("ContainerHolder", catalogue)(Buffer(payload))
    assert [list(one) for one in row["fNumbers"]] == [[1, 2, 3], []]


def test_an_array_of_pointers_to_containers_of_pointers_reads_the_objects_they_name():
    """``vector<TNamed*> *f[2]``: what each element holds is a pointer, so it is written as
    the tag naming the class it points at, and read back as that class's object."""
    catalogue, _source = _described(
        CONTAINERS.replace("LINES", "vector<TNamed*> *fNamedp[2];"), "ContainerHolder"
    )
    named = Instance(
        "TNamed", {"TObject": {"fUniqueID": 0, "fBits": 0}, "fName": "b", "fTitle": "t"}
    )
    payload = DeclaredObject(
        Instance("ContainerHolder", {"fN": 1, "fNamedp": [[named], []]}), catalogue
    ).root_payload(0)[0]
    row = reader_for("ContainerHolder", catalogue)(Buffer(payload))
    assert [[one["fName"] for one in held] for held in row["fNamedp"]] == [["b"], []]


def test_an_array_of_pointers_to_containers_of_a_self_streaming_class_is_refused():
    """``vector<TArrayD> *f[2]`` holds a class that streams itself rather than being written
    a member at a time, and this reader has no reading for such a class inside a member-wise
    record: it names the member and says so rather than reading the record as another shape."""
    catalogue, _source = _described(
        CONTAINERS.replace("LINES", "vector<TArrayD> *fArray[2];"), "ContainerHolder"
    )
    with pytest.raises(_Unreadable, match=r"'fArray', an array of pointers to containers of"):
        reader_for("ContainerHolder", catalogue)


def test_an_array_of_pointers_to_maps_is_refused_by_the_container_it_does_not_read():
    """A ``map<string,int> *f[2]`` is an array of pointers to a container of more than one
    type, which is not the count-and-items shape this reading is for: the member is refused
    by the container it names."""
    catalogue, _source = _described(
        CONTAINERS.replace("LINES", "map<string,int> *fMaps[2];"), "ContainerHolder"
    )
    with pytest.raises(_Unreadable, match=r"'fMaps', an array of pointers to map<string,int>"):
        reader_for("ContainerHolder", catalogue)


def test_a_class_of_members_a_bulk_loop_cannot_read_is_read_member_by_member_instead():
    """The bulk loop of roottest's ``ReadBuffer`` reads a class of plain numbers as one
    struct, and asks first. A class holding a member the reader has no reading for is not one
    struct - and a class whose every member can be read is, so the two answers differ."""
    unreadable, _source = _described(
        CONTAINERS.replace("LINES", "vector<TNamed> *fNamed[2];"), "ContainerHolder"
    )
    assert flat_reader_for("ContainerHolder", unreadable) is None
    readable, _source = _described(
        "class Plain { public: Int_t fN; double fX; ClassDef(Plain, 1) };", "Plain"
    )
    assert flat_reader_for("Plain", readable) is not None


def test_a_container_written_through_a_pointer_reads_back_as_what_was_written():
    """The reading is checked against the writing, which is what the file the bigevent
    roottest's ``read`` test opens holds: a ``vector<int> *`` written by the writer and read
    back through the container the pointer names."""
    catalogue, _source = _described(
        "class Numbers { public: Int_t fN; vector<int> *fPointed; ClassDef(Numbers, 1) };",
        "Numbers",
    )
    payload = DeclaredObject(
        Instance("Numbers", {"fN": 7, "fPointed": [1, 2, 3]}), catalogue
    ).root_payload(0)[0]
    row = reader_for("Numbers", catalogue)(Buffer(payload))
    assert row["fN"] == 7
    assert list(row["fPointed"]) == [1, 2, 3]


def test_a_container_of_objects_written_through_a_pointer_reads_back_whole():
    """``vector<TNamed> *`` with an object in it: the container is written member by member
    and read back as the objects it holds."""
    catalogue, _source = _described(
        "class NamedHolder { public: vector<TNamed> *fNamed; ClassDef(NamedHolder, 1) };",
        "NamedHolder",
    )
    named = Instance(
        "TNamed", {"TObject": {"fUniqueID": 0, "fBits": 0}, "fName": "a", "fTitle": "t"}
    )
    payload = DeclaredObject(
        Instance("NamedHolder", {"fNamed": [named]}), catalogue
    ).root_payload(0)[0]
    row = reader_for("NamedHolder", catalogue)(Buffer(payload))
    assert row["fNamed"][0]["fName"] == "a"


#: The bigevent roottest's ``Event`` as far as the reader's own part of it goes: a ``TLine *``
#: member and a ``template2<template1<int> >`` held by value.
BIGEVENT = r"""
template <class T> struct template1 {};
template <class T> struct template2 {};
class EventLike {
public:
   Int_t fNtrack;
   TLine *fLine;
   template2< template1< int > > fTemplateMember;
   ClassDef(EventLike, 1)
};
"""


def test_an_empty_template_instance_is_read_by_the_layout_the_class_is_described_with():
    """``fTemplateMember`` is an empty ``template2<template1<int> >``: its record is a
    version, a byte count and its checksum, and the instance read back has no members - which
    is only possible because the layout is found under the name the type spells, spaces and
    all."""
    _catalogue, source = _described(BIGEVENT, "EventLike")
    member = _members(source, "EventLike")["fTemplateMember"]
    assert member.typename == "template2<template1<int> >"
    assert source.streamers()["template2<template1<int> >"] == {}


def test_the_line_a_member_points_at_is_read_as_the_object_it_names():
    """``TLine *fLine``: the object is written where the pointer stands, its class named in
    front of it, and read back as the ``TLine`` its row of attributes describes - beside a
    template instance that has nothing to read."""
    catalogue, _source = _described(BIGEVENT, "EventLike")
    line = Instance(
        "TLine",
        {
            "TObject": {"fUniqueID": 0, "fBits": 0},
            "TAttLine": {"fLineColor": 1, "fLineStyle": 1, "fLineWidth": 1},
            "TAttBBox2D": {},
            "fX1": 1.0,
            "fY1": 2.0,
            "fX2": 3.0,
            "fY2": 4.0,
        },
    )
    payload = DeclaredObject(
        Instance(
            "EventLike",
            {
                "fNtrack": 5,
                "fLine": line,
                "fTemplateMember": Instance("template2<template1<int> >", {}),
            },
        ),
        catalogue,
    ).root_payload(0)[0]
    row = reader_for("EventLike", catalogue)(Buffer(payload))
    assert row["fNtrack"] == 5
    assert row["fTemplateMember"] == {}  # an empty class: its record and nothing in it
    assert row["fLine"]["fX1"] == 1.0 and row["fLine"]["fY2"] == 4.0
