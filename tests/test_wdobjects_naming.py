"""The names the writer puts in the bytes: a tag in front of a pointer, and whose spelling it is.

A tag holds a class name, and ROOT reads it back as one - ``ReadObjectAny`` hands it to
``TClass::GetClass``, which only knows the name ROOT's dictionary made. A template instance
has two spellings in a program: the one C++ requires of a nested ``>``, which the dictionary
uses, and the one the translation gives the same class, with the spaces out. Both are one
class to the catalogue; the bytes have to carry the dictionary's, which is what ROOT 6.40.04
wrote for ``roottest/root/io/bigevent``'s ``Event``.
"""

from __future__ import annotations

from xrdroot.wdeclared import Catalogue, Declaration, Field
from xrdroot.wdobjects import Instance, _Writer, write_instance
from xrdroot.writer import WBuffer

#: The name ROOT's dictionary gives ``template2< template1< int > >``, with the space C++
#: requires between the two closing brackets.
ROOTS_SPELLING = "template2<template1<int> >"
#: The name the translation of the same class gives it.
SPELLED_OUT = "template2<template1<int>>"
#: What ROOT 6.40.04 wrote for a tag naming that class: the tag's own byte count, the four
#: bytes no hash answers for, the name and its terminator - one byte longer than the
#: translation's spelling makes it.
ROOT_TAG_BYTES = bytes.fromhex("40000029ffffffff") + ROOTS_SPELLING.encode() + b"\x00"


def _a_holder_of_a_pointer(catalogue: Catalogue) -> Instance:
    """A ``PointerHolder`` holding one pointer to a template instance, as the bridge hands it
    over: the class named the way the translation spells it."""
    return Instance("PointerHolder", {"fP": Instance(SPELLED_OUT, {})})


def _a_catalogue() -> Catalogue:
    """``PointerHolder`` and the instance it points at, the instance under ROOT's spelling -
    which is the key a declaration reaches the catalogue by."""
    return Catalogue({
        "PointerHolder": Declaration("PointerHolder", 1, False, (),
                                     (Field("fP", ROOTS_SPELLING + "*", "", ()),)),
        ROOTS_SPELLING: Declaration(ROOTS_SPELLING, 1, True, (), ()),
    })  # fmt: skip


def test_a_pointer_to_a_template_instance_names_it_as_root_spells_it():
    """The tag ROOT writes is ``IsA()->GetName()``, which is the dictionary's spelling; the
    translation spells the same class without the space, so the pointer would go out naming a
    class ROOT's dictionary does not have."""
    buf = WBuffer()
    write_instance(buf, _a_holder_of_a_pointer(_a_catalogue()), _a_catalogue())
    assert ROOT_TAG_BYTES in bytes(buf.data)
    assert SPELLED_OUT.encode() + b"\x00" not in bytes(buf.data)


def test_a_pointer_to_a_class_the_catalogue_does_not_know_keeps_the_name_it_was_given():
    """Only a class the program declares has two spellings: ROOT's own are named once, by
    ROOT, and the writer passes the name it was handed straight through."""
    writer = _Writer(WBuffer(), Catalogue({}))
    assert writer._named("TH1F") == "TH1F"
