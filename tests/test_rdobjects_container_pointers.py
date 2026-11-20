"""Records of the classes a program declares, read back member by member by ``rdobjects``.

The bigevent roottest's ``Event`` is the class that names what this file covers: it holds a
``template2<template1<int> >`` by value, a ``TLine`` reached through a pointer, and - what the
reader refused until this change - a ``vector<TObject> *``, a container that is pointed at
rather than held.

ROOT writes a container the same bytes whether it is a member or what a member points at: a
record, a count and the items, with no tag in front to say a pointer stood there. So the
reader is given the container the pointer names, with the pointer taken off the type. These
tests ask the reader what it makes of such a member, and write and read the bytes back.
"""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.buffer import Buffer
from xrdroot.interp import _Unreadable
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
    at *and* an array, which ROOT writes into one record for the whole array rather than one
    record per element. Reading it would take the pointer off a type whose bytes the reader
    has no reading for, so the type is left as it is and the member refused rather than read
    wrongly."""
    catalogue, source = _described(
        CONTAINERS.replace("LINES", "vector<TNamed> *fNamed[2];"), "ContainerHolder"
    )
    array = _members(source, "ContainerHolder")["fNamed"]
    assert array.stype == STL_POINTER + 20
    assert array.typename == "vector<TNamed>*"
    with pytest.raises(_Unreadable, match=r"'fNamed', which is of a kind that this reader"):
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
