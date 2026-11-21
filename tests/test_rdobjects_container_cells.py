"""A member that is a container of objects, read back with the class of the things in it.

``roottest/root/io/bigevent``'s ``Event`` holds ``vector<TObject> *fVectorTobject`` (line 99)
and ``vector<TNamed> *fVectorTnamed[6]`` (line 100), and the bytes of such a member are a record, a
count and then one object's members after another - none of them saying which class they are,
because the declaration already did. The second is an array of them: ROOT writes the whole array
into one record, so what it reads as is one such list per position, and the class is named of
every item of every one of them. So what the reader hands back for the lot is a list of
dictionaries of members, and an item of it has no class to be asked anything on:
``event->GetVectorTobject()->at(0)->GetUniqueID()`` has nothing to be read off.

What the description says of the member is the class of every item of it, so that is what
``rdobjects`` now names - as a member held by value is already handed back as its members and
its class. A container of *pointers* is not one of these: an item names its own class in the
bytes, so it may well be of a subclass, and the declaration is not what says what it is.

The bytes here are the writer's own, read back by the reader the same class is read by.
"""

from __future__ import annotations

import pytest

from xrdroot.buffer import Buffer
from xrdroot.pycint.execute import run_source
from xrdroot.pyroot.core.declared import catalogue_of, find_class
from xrdroot.rdobjects import (
    ARRAY_OFFSET,
    STL_POINTER,
    STREAMER,
    DescribedSource,
    Members,
    _cell_members,
    _cell_of,
    _cells_of,
    reader_for,
)
from xrdroot.streamers import Member
from xrdroot.wdobjects import DeclaredObject, Instance


def _described(source_text: str, classname: str) -> tuple[DescribedSource, str]:
    """The layouts a macro's class is described with, and the text that declared it."""
    run_source(source_text, "cells.C", call=False)
    return DescribedSource(catalogue_of(find_class(classname))), source_text


def _read(source: DescribedSource, classname: str, value: dict[str, object]) -> dict[str, object]:
    """One object of a declared class, written and read back whole."""
    catalogue = source.catalogue
    payload = DeclaredObject(Instance(classname, value), catalogue).root_payload(0)[0]
    return reader_for(classname, catalogue)(Buffer(payload))


def _members(source: DescribedSource, classname: str) -> dict[str, Member]:
    """The members one class is described with, by name."""
    return source.streamers()[classname]


#: A ``TNamed`` as a macro makes one: the base every ROOT object has, and the two strings.
NAMED = Instance("TNamed", {"TObject": {"fUniqueID": 0, "fBits": 0}, "fName": "a", "fTitle": "t"})

#: A class with a container of objects reached through a pointer, and a number beside it.
NAMED_HOLDER = r"""
class NamedHolder {
public:
   Int_t fN;
   vector<TNamed> *fNamed;
   ClassDef(NamedHolder, 1)
};
"""


def test_an_item_of_a_container_of_objects_is_the_members_of_the_class_it_holds():
    """``vector<TNamed> *fNamed``: the container is read item by item and each item is the
    members of the ``TNamed`` the declaration says it holds, which is what a macro asks an item
    of a container for."""
    source, _text = _described(NAMED_HOLDER, "NamedHolder")
    row = _read(source, "NamedHolder", {"fN": 7, "fNamed": [NAMED]})
    assert row["fN"] == 7
    item = row["fNamed"][0]
    assert isinstance(item, Members)
    assert item.classname == "TNamed"
    assert item["fName"] == "a" and item["fTitle"] == "t"


def test_an_empty_container_holds_no_item_to_name():
    """A container with nothing in it is still a record and a count of none, and what it reads
    as is the empty list it is: there is no item to name, so nothing is made into a
    ``Members``."""
    source, _text = _described(NAMED_HOLDER, "NamedHolder")
    row = _read(source, "NamedHolder", {"fN": 7, "fNamed": None})
    assert row["fNamed"] == []


def test_the_class_a_container_member_holds_is_the_one_the_description_names():
    """Which members of a class hold a container of ROOT objects by value, and which class each
    of them holds - the one place a reader can learn it, since the bytes say nothing - and that
    such a member stands for one container and not for an array of them."""
    source, _text = _described(NAMED_HOLDER, "NamedHolder")
    assert _cells_of(source, "NamedHolder") == (("fNamed", "TNamed", False),)


def test_a_container_of_pointers_names_no_class_of_its_items():
    """``vector<TNamed*>`` writes each item as a pointer, which names the class in the bytes and
    may name a subclass: there the declaration is not what says what an item is, so it is not
    read off it."""
    source, _text = _described(
        "class Pointed { public: vector<TNamed*> fP; ClassDef(Pointed, 1) };", "Pointed"
    )
    assert _cells_of(source, "Pointed") == ()


def test_a_container_of_a_class_the_description_does_not_lay_out_names_nothing():
    """A macro's own class is described by its declaration and not by ROOT's catalogue, so a
    container of one holds nothing the class of an item can be named from here."""
    source, _text = _described(
        """
class Mine { public: Int_t fX; ClassDef(Mine, 1) };
class Mines { public: vector<Mine> fM; ClassDef(Mines, 1) };
""",
        "Mines",
    )
    assert _cells_of(source, "Mines") == ()


#: A class with an array of pointers to containers beside a plain one - the two shapes ROOT
#: gives the bigevent ``Event``'s ``fVectorTnamed`` and ``fVectorTobject``.
NAMED_ARRAY = r"""
class NamedArray {
public:
   Int_t fN;
   vector<TNamed> *fNamed[2];
   vector<TNamed> *fGrid[2][3];
   ClassDef(NamedArray, 1)
};
"""


@pytest.mark.parametrize("member", ["fNamed", "fGrid"])
def test_an_array_of_pointers_to_containers_names_the_class_of_every_item(member: str):
    """``vector<TNamed> *fVectorTnamed[6]`` - ``Event.h:100`` - is type 91, the container
    pointed at *and* a fixed-size array, and it names the same ``vector<TNamed>`` its neighbour
    ``fVectorTobject`` does: so the class of every item of every container of it is the one the
    description names, which is what the reader hands each of them back as."""
    source, _text = _described(NAMED_ARRAY, "NamedArray")
    cell = _members(source, "NamedArray")[member]
    assert cell.stype == STL_POINTER + ARRAY_OFFSET
    assert _cell_of(cell) == "TNamed"


def test_an_array_of_containers_is_the_one_shape_the_reader_hands_back_a_row_of():
    """The third thing a container member's description is read for: whether it is one container
    or one per position of an array, which decides how deep the name of the class goes - ROOT
    writes the whole array into a single record, so what comes back is a list of them."""
    source, _text = _described(NAMED_ARRAY, "NamedArray")
    assert _cells_of(source, "NamedArray") == (
        ("fNamed", "TNamed", True),
        ("fGrid", "TNamed", True),
    )


def test_the_array_a_member_of_containers_is_declared_as_is_the_length_it_is_read_by():
    """Nothing about such a member is dropped on the way to the reader: the type keeps its
    star, so the container is still spelled and the item's class is still named; the length is
    the dimension of the array, which is the count the one record begins with - a plain
    container member has none, which is why only an arrayed one is read as more than one - and
    a shape of more than one dimension is in ``dims``, the same as it is for a file's own
    description of the member."""
    source, _text = _described(NAMED_ARRAY, "NamedArray")
    members = _members(source, "NamedArray")
    assert (members["fNamed"].length, members["fNamed"].dims) == (2, ())
    assert (members["fGrid"].length, members["fGrid"].dims) == (6, (2, 3))


def test_each_container_of_an_array_of_them_comes_back_with_the_class_of_its_own_items():
    """What the reader makes of the value the array member reads as: one list of items per
    container, each item handed back as the members of the class the description names - where
    a member that is one container gives one list of them."""
    rows = [[{"fName": "a", "fTitle": ""}], [{"fName": "b", "fTitle": ""}]]
    arrayed = _cell_members(rows, "TNamed", True)
    assert [item["fName"] for held in arrayed for item in held] == ["a", "b"]
    assert all(isinstance(item, Members) for held in arrayed for item in held)
    plain = _cell_members(rows[1], "TNamed", False)
    assert [item["fName"] for item in plain] == ["b"]
    assert all(isinstance(item, Members) and item.classname == "TNamed" for item in plain)


@pytest.mark.parametrize(
    "member",
    [
        Member("fN", "", 1, "int", 0),
        Member("fV", "", STREAMER, "vector<int>", 0),
        Member("fM", "", STREAMER, "map<TAxis*,int>", 0),
        Member("fS", "", STREAMER, "string", 0),
        Member("fB", "", STREAMER, "vector", 0),
    ],
)
def test_a_member_that_is_no_container_of_objects_holds_no_class_to_name(member: Member):
    """A number, a container of numbers, a map - whose items are the pairs its two types make -
    a string, and a spelling that names no type at all are none of them a container of one
    class written where it stands, which is the one shape there is a class to name."""
    assert _cell_of(member) is None
