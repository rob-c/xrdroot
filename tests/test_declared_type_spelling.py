"""How a member's type is spelled in a file's description of the class that holds it.

Two spellings of one type decide two different things, and ROOT's own streamer information is
the arbiter of both. A member's *own* qualifiers say nothing about the class it names -
``const std::map<long long, std::string> &`` is a ``map<Long64_t,string>`` element. A template
*argument*'s ``const`` is counted: ROOT describes the member of a ``std::list<const TObject*>``
as ``list<const TObject*>``, and a class's checksum - the number a file's streamer record
carries and a reader matches the class by - is computed over that spelling.

The same question decides the bigevent roottest's ``Event::fTemplateMember``, a
``template2< template1< int > >``: the translation hands a member's type over as C++ writes it,
``template2<template1<int>>``, and the class a file describes is ``template2<template1<int> >``.
"""

from __future__ import annotations

from xrdroot.wdeclared import (
    Catalogue,
    Declaration,
    Field,
    class_name,
    normalised,
    parse_type,
)


def test_a_members_own_const_is_not_part_of_the_class_it_names():
    """``const`` and ``volatile`` on the member's own type are the member's: what the element
    names, and what is looked up, is the class without them."""
    assert normalised("const std::map<long long, std::string>&") == "map<Long64_t,string>"
    assert class_name(parse_type("const TObject *")) == "TObject"
    made = Catalogue({"Holder": Declaration(
        "Holder", 1, False, (), (Field("fP", "const TObject *", "", ()),))})
    element = made.info("Holder")[2][0]
    assert element[:4] == ("TStreamerObjectPointer", "fP", "", 64)
    assert element[8] == "TObject*"


def test_a_template_arguments_const_is_spelled_and_counted_as_root_spells_it():
    """The bigevent roottest's ``Event::fVectorTobjectp`` is a ``std::list<const TObject*>``,
    and ROOT describes its element ``list<const TObject*>``, stltype 2 - a list - and ctype 63,
    a pointer to an object: the words between the angle brackets are the member's type, so they
    are kept. ``0x05be7191`` is what ROOT 6.40.04 computes for a class holding only that
    member, so the checksum of the description is ROOT's own."""
    assert normalised("list<const TObject*>") == "list<const TObject*>"
    made = Catalogue({"ConstProbe": Declaration(
        "ConstProbe", 1, False, (), (Field("fV", "std::list<const TObject *>", "", ()),))})
    assert made.info("ConstProbe")[2] == (
        ("TStreamerSTL", "fV", "", 300, 24, 0, 0, (0, 0, 0, 0, 0),
         "list<const TObject*>", (2, 63)),)
    assert made.info("ConstProbe")[0] == 0x05BE7191


def test_a_class_a_template_argument_names_is_found_without_its_const():
    """Keeping the word must not lose the class: ``const TObject`` is ``TObject`` to the
    catalogue, so the container of pointers ROOT's own ``TObject`` is laid out by the ordinary
    rule for a container rather than refused for naming a class nothing describes."""
    made = Catalogue({"Holder": Declaration(
        "Holder", 1, False, (), (Field("fV", "list<const TObject*>", "", ()),))})
    assert made.known("const TObject") and made.known("TObject")
    assert made.declared("const TObject") is None, "it is ROOT's class, not the program's"
    assert made.info("Holder")[2][0][9] == (2, 63), "a list of object pointers, not a refusal"


def test_a_template_instance_member_is_spelled_as_root_spells_it():
    """``Event::fTemplateMember`` as the translation hands it over - ``template2<template1<int>>``
    - is the class ``template2<template1<int> >``: ROOT's spelling of the one class, in the
    element's type name and in the file's description of it. An instance whose templates declare
    no members has no elements, and ``0xf1819316`` is the checksum ROOT 6.40.04 gives it."""
    spelled = "template2<template1<int> >"
    made = Catalogue({
        "Event": Declaration("Event", 1, False, (), (
            Field("fTemplateMember", "template2<template1<int>>", "", ()),)),
        spelled: Declaration(spelled, 1, True, (), ()),
    })  # fmt: skip
    assert made.info("Event")[2] == (
        ("TStreamerObjectAny", "fTemplateMember", "", 62, 8, 0, 0, (0, 0, 0, 0, 0), spelled, ()),)
    assert made.info(spelled)[:2] == (0xF1819316, 1), "no members, and ROOT's checksum"
    assert made.info(spelled)[2] == ()
    assert set(made.entries()) == {"Event", spelled}
