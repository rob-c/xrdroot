"""The classes C++ code declares, as ROOT's I/O sees them: described, written and read back.

A class a macro, a library or ACLiC declares is a Python class whose
translation says what its data members are (``_cxx_layout_``: its C++ name,
its bases, and each member's name, type as written, trailing comment and
array lengths) and, with a ``ClassDef``, its version (``Class_Version``). That
is all ROOT's dictionary knows of a class too, so an object of one is written
as ROOT would write it - :mod:`xrdroot.wdeclared` describes the class,
:mod:`xrdroot.wdobjects` lays out the bytes - and a record of it read back
becomes an object of it again: made by its default constructor, as ROOT
makes one, and given each member the file holds, converted to the type the
class declares it with now. That conversion is all the schema evolution a
class of plain members needs - an ``int`` written and a ``float`` read, a
member added since (left as the constructor made it) or dropped (ignored).
"""

from __future__ import annotations

import datetime
import functools
import math
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import Any, NamedTuple, cast
from weakref import WeakKeyDictionary

import numpy as np

from ...cppstd.containers import std
from ...pycxx.runtime.namespace import DECLARED
from ...pycxx.symbols import member_name
from ...wclasses import TYPES
from ...wdeclared import (
    ARRAY_TEMPLATES,
    MAPS,
    STL_KINDS,
    Catalogue,
    Declaration,
    Field,
    Spelled,
    bitset_width,
    class_name,
    parse_type,
)
from ...wdobjects import Clones, DeclaredObject, Instance, written_flat
from ...winfo import INFOS
from ...writer import WBuffer
from .objects import HEAP_BITS, TNamed, TObject

__all__ = [
    "declared_class",
    "declaration_of",
    "find_class",
    "catalogue_of",
    "instance_of",
    "payload_of",
    "made_from",
]

#: The bits a ``TObject``-derived integer type is wrapped to, by the C++ type it is.
WIDTHS = {
    "char": (8, True), "short": (16, True), "int": (32, True), "long": (64, True),
    "long long": (64, True), "unsigned char": (8, False), "unsigned short": (16, False),
    "unsigned int": (32, False), "unsigned long": (64, False),
    "unsigned long long": (64, False),
}  # fmt: skip
#: The floating types: what a member of one is read back as.
FLOATS = ("float", "double", "Double32_t", "Float16_t", "long double")
#: How many class descriptions are kept at once: a program declares a bounded number of
#: classes, and a description is derived from one class alone.
CATALOGUES_KEPT = 4096

#: ROOT's collections that hand themselves to a reader as the plain list of what they held -
#: they stream themselves, so no streamer information is what reads them - and that the
#: namespace has a class of its own for, which is what a member of one reads back as.
COLLECTIONS = ("TList", "THashList", "TSortedList", "TObjArray")
#: The names a string of ROOT's or of C++'s owns. A member of one is written as its text, by
#: the ``TStreamerString`` its element is, though ``TString`` has a layout in
#: :data:`~.xrdroot.winfo.INFOS` like any other class of ROOT's.
STRINGS = ("TString", "string", "basic_string")
#: The collections of those that keep a numbered slot per object, a slot never filled a hole.
SLOTTED = ("TObjArray",)
#: The elements whose member is an object of a class ROOT writes in place - by value, not behind
#: a pointer - so that what the writer is handed for it is the row of that class's members, as
#: :func:`root_row` builds one: a ``TLorentzVector``'s ``TVector3 fP``. An object behind a marker
#: is the writer's own business, a pointer to one arriving as a record already.
HELD_OBJECTS = ("TStreamerObject", "TStreamerObjectAny")


def declared_class(cls: Any) -> bool:
    """Was ``cls`` declared in C++ - does its translation describe its members?"""
    return isinstance(cls, type) and "_cxx_layout_" in cls.__dict__


def find_class(name: str) -> type | None:
    """The class C++ run so far declared as ``name`` - qualified or not - if there is one.

    The translation names a class in a namespace, or inside another class, by
    its own name alone, so ``ns::Deep`` is found as ``Deep``.

    A template instance is one class under two spellings: the declaration writes
    ``template2<template1<int> >`` as C++ requires of nested ``>``, and a streamer
    names it ``template2<template1<int>>``, so the spaces are out of both sides
    before they are compared - the bigevent roottest's ``Event::fTemplateMember``
    is reached only by the first and asked for only by the second.
    """
    held = DECLARED.get(name)
    known = _FOUND.get(name)
    if known is not None and known[0] is held:
        return cast(type, known[1])  # settled on, and nothing has replaced the entry
    found = _searched(name, DECLARED)
    if found is not None and found is held:
        # the name as C++ declared it, which is the answer a bulk read asks for again
        _FOUND[name] = (held, found)
    return found


#: The classes ``find_class`` has settled on by the name it was asked for, with the entry
#: ``DECLARED`` held at the time. A name declared again is another object, so a cached answer
#: is only given back while that object is still the one under the name - which is the case
#: a bulk write or read asks about, ten million times over.
_FOUND: dict[str, tuple[Any, Any]] = {}


def _searched(name: str, declared: dict[str, Any]) -> type | None:
    """``find_class`` by the rules its docstring gives, over everything declared so far."""
    wanted = _squashed_name(name)
    last = wanted.split("::")[-1]
    fallback = None
    for cls in list(declared.values()):
        if not declared_class(cls):
            continue
        own = _squashed_name(str(cls.__dict__["_cxx_layout_"][0]))
        if own == wanted:
            return cls  # type: ignore[no-any-return]
        if own.split("::")[-1] == last:
            fallback = cls
    return fallback


def _squashed_name(name: str) -> str:
    """A class name's spelling with the spaces out of it, which is what two spellings of one
    template instance agree on: ``A<B<int> >`` and ``A<B<int>>`` are the one class."""
    return name.strip().replace(" ", "")


def declaration_of(cls: type, name: str | None = None) -> Declaration:
    """What ``cls``'s translation says of it, as :mod:`xrdroot.wdeclared` takes a class."""
    return _declaration_of(cls, name)


@functools.lru_cache(maxsize=CATALOGUES_KEPT)
def _declaration_of(cls: Any, name: str | None) -> Declaration:
    """The declaration of ``cls``, kept: the fields of a class are read off its translation
    once, not once per record written - a class is what its translation says and cannot
    change underneath it. ``cls`` is ``Any`` because a class is a hashable key, not because a
    caller may pass anything."""
    own, bases, fields = cls.__dict__["_cxx_layout_"]
    versioned = "Class_Version" in cls.__dict__
    version = int(cls.Class_Version()) if versioned else 1
    made = tuple(Field(str(n), str(s), comment_of(c), tuple(d)) for n, s, c, d in fields)
    return Declaration(name or str(own), version, foreign(cls), _spelled_bases(cls, bases), made,
                       issubclass(cls, TObject), _class_title(cls))  # fmt: skip


def _spelled_bases(cls: Any, bases: Any) -> tuple[str, ...]:
    """The bases as C++ declares them, a template instance's arguments and all.

    The translator records a base by the name of the class it names, so that a class deriving
    from ``vector<UShort_t>`` - the bigevent roottest's ``UShortVector`` - arrives as deriving
    from ``std::vector``, with nothing left to say what the vector holds. What it holds is
    what a file's description of the base has to say, so the base is spelled here as the class
    of ``cppstd`` it is, which names itself with its arguments: the class ``cls`` derives from
    in Python, matched to the declared base by the class name before the angle bracket.
    """
    held = [_name_of(base) for base in cls.__bases__]
    spelled: list[str] = []
    for base in bases:
        stem = _stem(str(base))
        spelled.append(next((one for one in held if _stem(one) == stem), str(base)))
    return tuple(spelled)


def _name_of(base: Any) -> str:
    """What a class of the namespace calls itself in C++: ``vector<unsigned short>``."""
    return str(getattr(base, "__cpp_name__", "") or base.__name__)


def _stem(name: str) -> str:
    """A class name without its template arguments and namespace: ``vector<unsigned short>``'s
    ``vector``, which is what two spellings of one base agree on."""
    return _squashed_name(name).split("<")[0].rsplit("::", 1)[-1]


#: The doxygen markers a member's note opens with once its ``//`` or ``/*`` is off: a comment
#: written after the member it documents rather than above it. Doxygen's ``!`` and ``*`` are
#: doxygen's, not ROOT's - ROOT's transient mark is a bare ``!``.
DOXYGEN_TRAILERS = ("/<", "!<", "*<")


def comment_of(note: str) -> str:
    """A data member's trailing comment as ROOT reads it, which is what says whether the
    member is persistent.

    ROOT counts a member transient when its comment begins with ``!``, and doxygen's trailing
    comments begin with a marker instead: ``Int_t fPy; //!< text`` is a persistent member
    documented by ``text``, while ``Int_t fCache; ///<! text`` is a transient one. A note
    reaches here with its ``//`` or ``/*`` already off, so the marker is what it opens with -
    ``/<`` from ``///<``, ``*<`` from ``/**<``, and ``!<`` from ``//!<`` or ``/*!<``, the one
    form in which doxygen's ``!`` is not ROOT's. A ``!`` anywhere else, ``//! text``
    included, is ROOT's and stays.
    """
    text = str(note)
    if text[:2] in DOXYGEN_TRAILERS:
        text = text[2:]
    return text.lstrip()


def foreign(cls: type) -> bool:
    """Was ``cls`` declared without ``ClassDef`` - which gives it ``Class_Name`` as well as
    ``Class_Version`` - so that its objects carry its checksum where a version would go?"""
    return "Class_Name" not in cls.__dict__


def _class_title(cls: type) -> str:
    title = getattr(cls, "CLASS_TITLE", "")
    return "" if "CLASS_TITLE" not in cls.__dict__ else str(title)


def catalogue_of(cls: type) -> Catalogue:
    """Every declared class an object of ``cls`` can reach, described as a write needs.

    Kept, because a description is a pure function of the class and the same one is asked for
    again for every object written: a ``TClass::Streamer`` in a bulk loop of 10^7 records
    re-derived its class's elements - and their checksum - 10^7 times, which was most of the
    write. The class itself is the key, so a class declared again under the same name is
    described afresh.
    """
    return _described_by(cls)


@functools.lru_cache(maxsize=CATALOGUES_KEPT)
def _described_by(cls: type) -> Catalogue:
    found: dict[str, Declaration] = {}
    _gather(declaration_of(cls).name, cls, found)
    return _published(Catalogue(found), found)


def catalogue_for(classname: str) -> Catalogue:
    """Every declared class a value of the type ``classname`` can reach: a class's, or what a
    container holds.

    Kept for the reason :func:`catalogue_of` gives; the class the type names now is part of
    the key, so a type whose class has been declared again is described again.
    """
    return _described_for(classname, find_class(parse_type(classname).name))


@functools.lru_cache(maxsize=CATALOGUES_KEPT)
def _described_for(classname: str, cls: type | None) -> Catalogue:
    found: dict[str, Declaration] = {}
    _reach_type(parse_type(classname), found)
    return _published(Catalogue(found), found)


def _published(catalogue: Catalogue, found: dict[str, Declaration]) -> Catalogue:
    """Say how each of these classes is laid out in memory, for a reader of another file.

    ROOT's ``TClass`` holds a class's description whether an object of it is being written or
    read, so a file written by another vintage of the class is read with the description in
    memory rather than with the file's own account of it. Nothing below ``pyroot`` can reach a
    class the program declared, so the layout is published where the read can find it.
    """
    from ...interp import declare

    for name in found:
        declare(name, catalogue.info(name))
    return catalogue


def plain_value(value: Any, classname: str, catalogue: Catalogue) -> Any:
    """A value of the type ``classname`` as plain data, as a write takes it."""
    return _plain(value, parse_type(classname), catalogue)


def new_value(classname: str) -> Any:
    """A default-made value of the type ``classname``: a container of the kit, an object of a
    class C++ declared - or ``None`` for one nothing here makes."""
    found = parse_type(classname)
    if found.name in STL_KINDS:
        args = [arg.text() for arg in found.args]
        return getattr(std, found.name)[args[0] if len(args) == 1 else tuple(args)]()
    cls = find_class(found.name)
    if cls is None:
        from .reflection import _class_named

        cls = _class_named(found.name)
    return _new(cls) if cls is not None else None


def _gather(name: str, cls: type, found: dict[str, Declaration]) -> None:
    declared = declaration_of(cls, name)
    found[name] = declared
    for base in declared.bases:
        _reach(base, found)
    for field in declared.fields:
        _reach_type(parse_type(field.spelled), found)


def _reach_type(found: Spelled, into: dict[str, Declaration]) -> None:
    """Add every declared class a type reaches: the type itself - a template instance such as
    a macro's ``ArrayHolderT<int>`` is a class of its own - and whatever its arguments are."""
    for arg in found.args:
        _reach_type(arg, into)
    if found.name not in TYPES:
        # The name with its arguments and without the stars: a pointer to a class reads
        # the class, and a template instance is a class of its own.
        _reach(Spelled(found.name, found.args, 0).text(), into)


def _reach(name: str, into: dict[str, Declaration]) -> None:
    if name in into or name in INFOS:
        return
    cls = find_class(name)
    if cls is not None:
        _gather(name, cls, into)


# -- writing ------------------------------------------------------------------------------------


def payload_of(obj: Any, writer: Any = None) -> DeclaredObject | None:
    """What a file writes for ``obj``, if its class was declared in C++ - else ``None``.
    ``writer`` is the file being written into, which a ``TRef`` among the object's members
    names its process in - the current file's when not given."""
    cls = type(obj)
    if not declared_class(cls):
        return None
    catalogue = catalogue_of(cls)
    name = declaration_of(cls).name
    title = obj.GetTitle() if isinstance(obj, TObject) else ""
    token = WRITER.set(writer if writer is not None else _current_writer())
    try:
        return DeclaredObject(instance_of(obj, name, catalogue), catalogue, str(title))
    finally:
        WRITER.reset(token)


#: The file an object is being written into, for the references among its members.
WRITER: ContextVar[Any] = ContextVar("WRITER", default=None)


def _current_writer() -> Any:
    """The writable directory of the current file, if it is one being written."""
    from .directories import current_directory

    writable = getattr(current_directory(), "_writable", None)
    return writable() if callable(writable) else None


def flat_bytes(obj: Any, classname: str, catalogue: Catalogue) -> bytes | None:
    """The record of an object of a class whose members are all plain numbers, written
    straight from the object - or ``None`` for one that is more than that.

    The members are fetched from the object and packed under the one struct their class is
    written with, with no dictionary of them in between: roottest's ``WriteBuffer`` writes
    ten million records of nine numbers, and that dictionary was a third of what each cost.
    """
    if catalogue.declared(classname) is None:
        return None
    attributes = _plan(catalogue, classname).flat
    if attributes is None:
        return None
    out = WBuffer()
    values = [getattr(obj, attribute, None) for attribute in attributes]
    return bytes(out.data) if written_flat(out, classname, catalogue, values) else None


def instance_of(obj: Any, name: str, catalogue: Catalogue) -> Instance:
    """``obj`` as plain data, read by the declaration of its class ``name``.

    What the declaration says is worked out once per class and kept: an object written in a
    bulk loop - the 10^7 records of ``TClass::Streamer`` - used to parse each member's type,
    spell its class out again and look that class up, per member per record.
    """
    plan = _plan(catalogue, name)
    members: dict[str, Any] = {}
    for base in plan.bases:
        members[base] = _base_value(obj, base, catalogue)
    for field, attribute, found in plan.fields:
        value = getattr(obj, attribute, None)
        members[field.name] = value if found is None else _plain_member(value, field, catalogue,
                                                                      found)  # fmt: skip
    return Instance(name, members)


class _Plan(NamedTuple):
    """How one class's members come off an object, worked out from its declaration once."""

    bases: tuple[str, ...]
    #: Each persistent member: its field, the attribute the object keeps it in, and its type
    #: taken apart - or ``None`` for a number, which is written as the value it is.
    fields: tuple[tuple[Field, str, Spelled | None], ...]
    #: The attributes of a class whose members are all plain numbers and which has no base -
    #: a record that is one struct over the object's own values - or ``None`` for a class
    #: that is more than that.
    flat: tuple[str, ...] | None


#: Each catalogue's plans by class name. Held weakly: a catalogue is kept with the class it
#: describes, and a plan says nothing about anything outside it.
_PLANS: WeakKeyDictionary[Catalogue, dict[str, _Plan]] = WeakKeyDictionary()


def _plan(catalogue: Catalogue, name: str) -> _Plan:
    """How the class ``name``'s members come off an object, made the first time it is asked."""
    held = _PLANS.get(catalogue)
    if held is None:
        held = _PLANS[catalogue] = {}
    found = held.get(name)
    if found is None:
        found = held[name] = _planned(catalogue, name)
    return found


def _planned(catalogue: Catalogue, name: str) -> _Plan:
    """The plan for the class ``name``: whether each member is a number, which the writer
    takes as the value itself, or something to take apart first."""
    declared = catalogue.declared(name)
    assert declared is not None  # every class reached is in the catalogue
    fields = tuple(
        (field, _attribute(field.name), None if _a_number(field) else parse_type(field.spelled))
        for field in declared.fields
        if not field.comment.startswith("!")
    )
    flat = None
    if not declared.bases and all(found is None for _field, _at, found in fields):
        flat = tuple(attribute for _field, attribute, _found in fields)
    return _Plan(declared.bases, fields, flat)


def _a_number(field: Field) -> bool:
    """Is the member a plain number - as opposed to an object, a container or an array - which
    is plain data as it stands, so that :func:`_plain` would hand it back unchanged?"""
    found = parse_type(field.spelled)
    return not found.stars and not field.dims and found.name in TYPES


def _attribute(name: str) -> str:
    return member_name(name)


def _base_value(obj: Any, base: str, catalogue: Catalogue) -> Any:
    """The part of ``obj`` one base ``base`` is: an instance for a class the program declares,
    the container's contents for a base that is one of the standard library's - whose object
    *is* that container, so what it holds is written as a member of the type - and one of
    ROOT's classes by the members its layout names."""
    if catalogue.declared(base) is not None:
        return instance_of(obj, base, catalogue)
    found = parse_type(base)
    if found.name in STL_KINDS and not found.stars:
        return _plain(obj, found, catalogue)
    return root_row(obj, base)


def root_row(obj: Any, classname: str) -> dict[str, Any]:
    """The part of ``obj`` one of ROOT's classes ``classname`` is, as the members its layout
    names - each base's under the base's name, a ``TObject``'s bits and a ``TNamed``'s name and
    title alongside, as either way of writing them reads them."""
    row: dict[str, Any] = {}
    held = getattr(obj, "members", None)
    if isinstance(obj, TObject):
        # The heap's two bits are the reader's, never the file's: ROOT's writer masks them out
        # and only a reader puts them back, so a row ready to be written carries none.
        row["fBits"] = obj._bitword() & ~HEAP_BITS
        row["fUniqueID"] = obj.GetUniqueID()
    elif isinstance(held, Mapping) and classname == "TObject":
        # A drawing class keeps the members ROOT would write by name in ``members``, and is
        # not a ``TObject`` here though ROOT's ``TLine`` derives from one.
        row["fBits"] = int(held.get("fBits", 0))
        row["fUniqueID"] = int(held.get("fUniqueID", 0))
    if isinstance(obj, TNamed):
        row["fName"] = TNamed.GetName(obj)
        row["fTitle"] = TNamed.GetTitle(obj)
    for element in INFOS.get(classname, (0, 0, ()))[2]:
        member = element[1]
        if element[0] == "TStreamerBase":
            row[member] = root_row(obj, member)
        elif member not in row:
            row[member] = _root_member(obj, member, element[0], element[8])
    return row


def _root_member(obj: Any, member: str, kind: str = "", typename: str = "") -> Any:
    """A member of one of ROOT's bases, from its getter where it has one: ``fLineColor``.

    ROOT names most of them ``Get`` and the member's name without its ``f``, but not all of
    them - ``TObjArray::fLowerBound`` is read by ``LowerBound()`` - so the name alone is tried
    before the attribute, which a few classes keep for themselves instead. A member with
    neither is taken from the row the object keeps its members in, which is where a ``TAxis``
    holds ``fBits2`` - ROOT's second bit status word of an axis, which no ``Get`` of ROOT's
    asks for and which the axis here keeps for the histogram it belongs to.

    ``kind`` and ``typename`` are the element's, and are what say that the member is itself an
    object of one of ROOT's classes, held by value: what is written then is that class's own
    row of members, not the value a method of the member's name happens to answer.
    """
    value = _member_value(obj, member)
    if kind not in HELD_OBJECTS or typename in STRINGS or typename not in INFOS:
        return value
    if typename in COLLECTIONS:  # written as the entries it holds, not as a row of members
        return value
    if isinstance(value, Mapping):
        return value
    if _of_class(value, typename):
        return root_row(value, typename)
    kept = _kept_object(obj, member, typename)
    return value if kept is None else root_row(kept, typename)


#: How a member of one of ROOT's classes comes off the object - its getter, its own name, or
#: the row the object keeps - by the class of the object and the member. Kept because which
#: of them answers is a property of the class: a file of two million vectors asked it, and
#: built the getter's name from the member's, two million times over.
_ACCESSORS: dict[tuple[type, str], Callable[[Any], Any]] = {}


def _member_value(obj: Any, member: str) -> Any:
    """What ``obj`` answers for the member ``member``: its getter's, its own of that name, or
    the one the row it keeps its members in holds."""
    key = (type(obj), member)
    found = _ACCESSORS.get(key)
    if found is None:
        found = _ACCESSORS[key] = _probed(obj, member)
    return found(obj)


def _probed(obj: Any, member: str) -> Callable[[Any], Any]:
    """The way ``obj``'s class answers ``member``, for every object of that class after it."""
    getter = getattr(obj, "Get" + member[1:], None)
    if callable(getter):
        return lambda held: getattr(held, "Get" + member[1:])()
    terse = getattr(obj, member[1:], None)
    if callable(terse):
        return lambda held: getattr(held, member[1:])()
    if isinstance(getattr(obj, "_row", None), Mapping):
        def from_row(held: Any) -> Any:
            row = getattr(held, "_row", None)
            if isinstance(row, Mapping) and member in row:
                return row[member]
            return getattr(held, member, None)

        return from_row
    return lambda held: getattr(held, member, None)


def _of_class(value: Any, typename: str) -> bool:
    """Is ``value`` an object of the class ROOT's element calls ``typename``?"""
    return type(value).__name__ == typename


def _kept_object(obj: Any, member: str, typename: str) -> Any:
    """The object of ROOT's class ``typename`` that ``obj`` keeps for its member ``member``,
    for when no getter of ROOT's hands that member over.

    A ``TLorentzVector``'s ``TVector3 fP`` is such a member: ``P()`` - the member's name
    without its ``f``, which is what :func:`_member_value` tries before the attribute - answers
    the momentum's *length* rather than the vector the class keeps its components in, so the
    length is what is asked for and not what the file holds. The attribute a class keeps such
    an object in is the only one of that class, or is the one named after the member - ``_p``
    for ``fP``, which is all that tells two vectors apart.
    """
    held = getattr(obj, "__dict__", None)
    if not isinstance(held, dict):
        return None
    wanted = member.lstrip("f").lower()
    for name in ("_" + wanted, wanted):  # the one named after the member first
        found = held.get(name)
        if _of_class(found, typename):
            return found
    for value in held.values():
        if _of_class(value, typename):
            return value
    return None


def _plain_member(value: Any, field: Any, catalogue: Catalogue,
                  found: Spelled | None = None) -> Any:  # fmt: skip
    """A data member's value as plain data.

    An array of anything but a number - ``T a[N]`` and ``std::array<T,N>`` alike - holds one
    plain element per slot, which is what the writer of an array of objects, and of an array of
    containers, writes one after another; an array of numbers stays the values it is, and so
    does a string of characters, which is one text and not a run of them. The array level is
    the writer's: converting the member as the element's type would fold the two levels into
    one, and the bigevent roottest's ``vector<TNamed> *fVectorTnamed[6]`` arrives here as six
    vectors.
    """
    found = parse_type(field.spelled) if found is None else found
    template = found.name in ARRAY_TEMPLATES and bool(found.args)
    content = found.args[0] if template else found  # a std::array's, or the element of T[N]
    is_array = bool(field.dims) or template
    if is_array and found.name not in TYPES:  # an array of objects, containers or pointers
        return [_plain(element, content, catalogue) for element in (value if value is not None
                                                                    else ())]  # fmt: skip
    return _plain(value, found, catalogue)


def _plain(value: Any, found: Spelled, catalogue: Catalogue) -> Any:
    """A member's value as plain data: objects as instances, containers as lists or dicts - an
    object held by value that is not there yet, a slot ``resize`` made, as a default-made one.

    A pointer to a container is the container it points at, ROOT streams it as that and the
    translator hands over the contents, so the stars decide nothing here. They do for ``pair``:
    a bare ``pair<A,B>`` is a class the declaration describes, written through its members.
    """
    value = _present(value, found, catalogue)
    if value is None:
        return None
    if found.name in STL_KINDS or (found.name == "pair" and not found.stars):
        return _plain_container(value, found, catalogue)
    return _plain_object(value, found, catalogue)


def _present(value: Any, found: Spelled, catalogue: Catalogue) -> Any:
    """A member's value as there to write: an object held by value that is not made yet as
    a default-made one, and a pointer a macro set to 0 as the null it is."""
    if value is None and not found.stars and found.name in catalogue.declarations:
        return new_value(found.name)
    null = found.stars and isinstance(value, int) and value == 0
    return None if null else value


def _plain_object(value: Any, found: Spelled, catalogue: Catalogue) -> Any:
    """An object as plain data - one of a declared class as an instance, one of ROOT's as its
    members - and anything else as it is, a string class as its text.

    The class is looked up by the name a streamer spells, template arguments and all:
    a macro's ``ArrayHolderT<int>`` is declared under that name, where the spelled type
    of the member says only ``ArrayHolderT``.
    """
    name = class_name(found)
    if catalogue.declared(name) is not None and name not in STRINGS:
        return _object_of(value, name, catalogue)
    if isinstance(value, (list, tuple)) and name in INFOS and name not in STRINGS:
        return _plain_run(value, found, catalogue)
    if isinstance(value, TObject):
        return _root_object(value, found.name, catalogue)
    if _held_by_root(name, value):
        # Written by the layout alone: nothing else here knows the class, and it is no
        # ``TObject`` to be asked what class it is - a drawing class, a ``TAttLine``.
        return root_row(value, name)
    return value if isinstance(value, (int, float, np.ndarray, list)) else _scalar(value)


def _plain_run(value: Any, found: Spelled, catalogue: Catalogue) -> list[Any]:
    """A counted array of one of ROOT's own objects, ``T *f //[n]`` and ``T **f //[n]``: one
    plain object per slot, which is what ROOT streams one after another - in place for one
    star, behind a pointer that names the class for two. The bigevent roottest's ``fPaxis``
    and ``fQaxis`` arrive here as the ``TAxis`` objects the macro made."""
    return [_plain_object(element, found, catalogue) for element in value]


def _held_by_root(name: str, value: Any) -> bool:
    """Does ``value`` stand for one of ROOT's classes the writer carries the layout of, which
    is written by what that layout names rather than by what the value keeps?

    Not every one of them is a ``TObject`` here: a drawing class such as ``TLine`` keeps its
    members by name for the canvas, and ``TAttLine`` its three in the group it belongs to, and
    ROOT streams either through the layout's ``Get`` of each member.
    """
    plain = (str, bytes, int, float, np.ndarray, list, tuple, Mapping)
    return name in INFOS and name not in STRINGS and not isinstance(value, plain)


def _root_object(value: Any, classname: str, catalogue: Catalogue) -> Any:
    """One of ROOT's objects among a declared class's members: a ``TClonesArray`` as the
    objects it holds, one the writer takes - a histogram, a ``TRef``, a ``TBits`` - as it takes
    it, and any other by the members its layout names."""
    from .collections import TClonesArray
    from .records import recorded
    from .timing import TDatime

    if isinstance(value, TDatime):
        # A ``TDatime`` streams itself as one packed word and no record, so what is handed on
        # is the moment it stands for - the value reading one back gives.
        return _moment(value)
    if isinstance(value, TClonesArray):
        return _clones(value, catalogue)
    made = recorded(value, WRITER.get())
    if made is not value:
        return made
    return root_row(value, classname) if classname in INFOS else value


def _clones(value: Any, catalogue: Catalogue) -> Clones:
    """A ``TClonesArray`` of objects of a declared class, each as an instance of it."""
    held = str(value._class)
    if held not in catalogue.declarations:
        cls = find_class(held)
        if cls is not None:
            catalogue.declarations.update(catalogue_of(cls).declarations)
    items = [None if item is None else instance_of(item, held, catalogue)
             for item in list(value)[: value.GetEntriesFast()]]  # fmt: skip
    return Clones(held, str(getattr(value, "_name", "")) or held + "s", items)


def _moment(value: Any) -> datetime.datetime:
    """A ``TDatime`` as the moment it stands for, from the two numbers ROOT lets one be read
    by: its date as ``YYYYMMDD`` and its time as ``HHMMSS``."""
    date, clock = int(value.GetDate()), int(value.GetTime())
    return datetime.datetime(
        date // 10000, date // 100 % 100, date % 100,
        clock // 10000, clock // 100 % 100, clock % 100,
    )  # fmt: skip


def _plain_container(value: Any, found: Spelled, catalogue: Catalogue) -> Any:
    """A container as a list of what it holds - a map's as pairs, not a dict, since a multimap
    holds a key as often as it was inserted - and a pair as the record of its two members."""
    if found.name == "pair":
        first, second = value
        members = {"first": _plain(first, found.args[0], catalogue),
                   "second": _plain(second, found.args[1], catalogue)}  # fmt: skip
        return Instance(class_name(found), members)
    if found.name in MAPS:
        key, held = found.args
        return [(_plain(k, key, catalogue), _plain(v, held, catalogue)) for k, v in _pairs(value)]
    return [_plain(item, found.args[0], catalogue) for item in value]


def _object_of(value: Any, declared: str, catalogue: Catalogue) -> Instance:
    """An object as plain data - by its own class, when it is one derived from the declared."""
    actual = type(value)
    name = declaration_of(actual).name if declared_class(actual) else declared
    if name != declared and catalogue.declared(name) is None:
        catalogue.declarations.update(catalogue_of(actual).declarations)
    return instance_of(value, name, catalogue)


def _pairs(value: Any) -> list[tuple[Any, Any]]:
    items = value.items() if hasattr(value, "items") else value
    return [(pair[0], pair[1]) for pair in items]


def _scalar(value: Any) -> Any:
    """A string class as the text it holds, anything else as it is."""
    if isinstance(value, (str, np.generic)) or not hasattr(value, "__str__"):
        return value
    return str(value) if type(value).__name__ in ("TString", "string") else value


# -- reading ------------------------------------------------------------------------------------


def made_from(classname: str, row: Any) -> Any:
    """An object of the declared class ``classname`` made from a record read - or ``None``
    when no class of that name was declared."""
    cls = find_class(classname)
    if cls is None or not isinstance(row, Mapping):
        return None
    made = _new(cls)
    _fill(made, cls, row)
    return made


def _new(cls: type) -> Any:
    """A default-made object, as ROOT makes one to read into."""
    try:
        return cls()
    except TypeError:
        return object.__new__(cls)


def _fill(obj: Any, cls: type, row: Mapping[str, Any]) -> None:
    """``obj`` given the members of ``row``, by the declaration of ``cls``."""
    _own, bases, _fields = cls.__dict__["_cxx_layout_"]
    for base in bases:
        _fill_base(obj, cls, str(base), row.get(str(base)))
    for name, attribute, made in _fills_of(cls):
        if name in row:
            setattr(obj, attribute, made(row[name], getattr(obj, attribute, None)))


def _fills_of(cls: type) -> tuple[tuple[str, str, Callable[[Any, Any], Any]], ...]:
    """The members of ``cls`` a read fills: its name in the row, its attribute, and how the
    value it was read as becomes the member's own.

    Kept per class - a ``TClass::Streamer`` in roottest's bulk loop fills ten million
    records of one class of nine numbers, and working out which attribute a name is and
    what type its member has is not something a record changes.
    """
    found = _FILLS.get(cls)
    if found is None:
        found = _FILLS[cls] = tuple(
            (str(name), _attribute(str(name)), _member_maker(parse_type(str(spelled)), dims))
            for name, spelled, comment, dims in cls.__dict__["_cxx_layout_"][2]
            if not str(comment).startswith("!")
        )
    return found


#: Each class's fill plan, held weakly: a plan says nothing about anything outside the class.
_FILLS: WeakKeyDictionary[type, tuple[tuple[str, str, Callable[[Any, Any], Any]], ...]] = (
    WeakKeyDictionary()
)

#: The types a member's value is made from with nothing but the value itself, less the one
#: whose reading is not a number a stream gives: a ``char`` may be the string it holds.
_PLAIN = frozenset(name for name in TYPES if name not in ("char", "Char_t"))


def _member_maker(found: Spelled, dims: Any = ()) -> Callable[[Any, Any], Any]:
    """How a member of the type ``found`` is given the value read for it.

    A member that is one number - nine members in ten of a class of numbers - goes straight
    to the arithmetic its C++ type's reading is, where :func:`converted` would first ask
    whether it is a pointer, a container, a bitset or a string. A declaration of an array of
    them is not one number, and keeps the reading it had.
    """
    if found.stars or dims or found.name not in _PLAIN:
        return lambda value, current: converted(value, found, current)
    return _number_maker(found.name)


def _number_maker(typename: str) -> Callable[[Any, Any], Any]:
    """``_number``'s work for a member that is a number and not an array of them."""
    c_type = TYPES[typename][1]
    made_number = _converter(c_type)
    if c_type in FLOATS or c_type == "bool":
        return _converted_number(typename, made_number)
    # An integer member of a record read out of a file is a Python int - what ``struct``
    # gives - so the mask and the sign are worked out here, as ``_scalar_number`` works them
    # out, without a call and two checks that a value of that type never needs.
    bits, signed = WIDTHS[_c_name(c_type)]
    mask, sign, width = (1 << bits) - 1, 1 << (bits - 1), 1 << bits

    def made(value: Any, current: Any) -> Any:
        if type(value) is int:
            whole = value & mask
            return whole - width if signed and whole >= sign else whole
        if value is None:
            return None
        if isinstance(current, np.ndarray) or isinstance(value, (list, tuple, np.ndarray)):
            return _number(value, typename, current)
        return made_number(value)

    return made


def _converted_number(
    typename: str, made_number: Callable[[Any], Any]
) -> Callable[[Any, Any], Any]:
    """A member of a type whose value is a conversion rather than a mask: a float, a bool."""

    def made(value: Any, current: Any) -> Any:
        if value is None:
            return None
        if isinstance(current, np.ndarray) or isinstance(value, (list, tuple, np.ndarray)):
            return _number(value, typename, current)
        return made_number(value)

    return made


def _fill_base(obj: Any, cls: type, base: str, row: Any) -> None:
    if not isinstance(row, Mapping):
        return
    declared = find_class(base)
    if declared is not None and isinstance(obj, declared):
        _fill(obj, declared, row)
        return
    fill_root(obj, row)


#: How ``TObject``'s and ``TNamed``'s members are given back to an object of the namespace's.
_ROOT_MEMBERS: dict[str, Any] = {
    "fBits": lambda obj, value: obj._store_bits(int(value)),
    "fUniqueID": lambda obj, value: obj.SetUniqueID(int(value)),
    "fName": lambda obj, value: TNamed.SetName(obj, str(value)),
    "fTitle": lambda obj, value: TNamed.SetTitle(obj, str(value)),
}


def fill_root(obj: Any, row: Mapping[str, Any]) -> None:
    """The part of ``obj`` one of ROOT's classes is, given the members read: a ``TObject``'s
    bits, a ``TNamed``'s name and title, each base's members, and every other member by its
    setter - ``fLineColor`` by ``SetLineColor``."""
    for key, value in row.items():
        special = _ROOT_MEMBERS.get(key)
        if special is not None:
            special(obj, value)
        elif isinstance(value, Mapping):
            fill_root(obj, value)
        else:
            setter = getattr(obj, "Set" + key[1:], None)
            if callable(setter):
                setter(value)


def converted(value: Any, found: Spelled, current: Any) -> Any:
    """A value read, as a member of the type ``found`` holds it now - into ``current``, the
    member as the constructor left it, where that is a container or an object."""
    if value is None:
        return None
    if found.stars and (found.name in TYPES or found.name in WIDTHS):
        # A pointer to numbers is an array whose length the stream gives, not the shape the
        # class's constructor happened to leave there: ROOT hands the member a new array of
        # the count the file writes, so reading an entry with more or fewer than the one
        # before it does not keep the first one's length.
        return _array_of(value, found.name)
    if found.stars and found.name not in TYPES:
        return _pointed(value, found, current)
    if found.name in STL_KINDS:
        return _refilled(value, found, current)
    width = bitset_width(found)
    if width is not None:
        return _bitset_of(width, value)
    return _held(value, found, current)


def _bitset_of(width: int, value: Any) -> Any:
    """A member's ``std::bitset<N>`` from the bits read for it, lowest first.

    The member holds a bitset rather than the row of numbers the bits are on file, since a
    macro compares one with another and prints it as C++ does.
    """
    handed = getattr(value, "to_ulong", None)
    if handed is not None:
        return std.bitset[width](handed())
    return std.bitset[width](sum(1 << at for at, bit in enumerate(value) if bit))


def _text_of(value: Any) -> str | None:
    """A ``char[N]`` read as the string it holds - up to the first NUL, as ``strlen`` reads it,
    a byte that is not UTF-8 carried through as its own character - or ``None``."""
    if isinstance(value, bytes):
        end = value.find(b"\x00")
        return value[: len(value) if end < 0 else end].decode("utf-8", "surrogateescape")
    if isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == np.int8:
        return _text_of(value.tobytes())
    return None


def _array_of(value: Any, typename: str) -> Any:
    """A pointer's numbers as the array ROOT reads one into - of the length the stream gave."""
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value)
    return _number(value, typename, None)


def _held(value: Any, found: Spelled, current: Any) -> Any:
    """A value read for a member that is no container: text - a string class's, or a
    ``char``'s - a number, or an object."""
    if found.name == "char" and not isinstance(value, (int, np.number)):
        # ``char name[N]`` is the C string it holds, as ROOT's TLeafC reads one: a macro
        # compares it with strcmp and searches it with strstr, not a row of numbers.
        text = _text_of(value)
        if text is not None:
            return text
    if found.name in ("string", "TString") or isinstance(value, str):
        return _text(value, found, current)
    if found.name in TYPES or found.name in WIDTHS:
        return _number(value, found.name, current)
    return _object(value, found, current)


def _pointed(value: Any, found: Spelled, current: Any) -> Any:
    """A pointer's object: of the class the stream named, which may be a derived one - a
    ``BaseABC *fABC`` holding a ``Derived`` is a ``Derived``, as ROOT gives it back.

    Only a record - an object as its members - carries the class it was written as. What a
    collection comes back as carries the class of its *elements*, which is not what the
    member is, so there the member's own type names it.
    """
    streamed = str(getattr(value, "classname", "")) if isinstance(value, Mapping) else ""
    named = streamed or found.name
    made = made_from(named, value)
    if made is not None:
        return made
    known = _dressed(named, value)
    if known is not None:
        return known
    held = _collection(named, value, current)
    if held is not None:
        return held
    if named == "TClonesArray" and isinstance(value, list):
        return _clones_of(str(getattr(value, "classname", "")), value, current)
    return value


def _dressed(named: str, value: Any) -> Any:
    """A pointer to one of ROOT's own classes as that class - ``TH1F *fH`` as the ``TH1F``
    ``f.Get("fH")`` would give - or ``None`` for a class this namespace does not have.

    The file leaves such a member as the members it was written as, since the class name is
    the member's type rather than something the stream said; the class makes itself from
    them, which is what reading the same object as a key of the file does, and the ROOT
    object standing for what comes back is the one a macro holds.
    """
    from ...kinds import dress
    from .wrapping import from_members, wrap

    made = from_members(named, value)  # a class the kit makes again from a record's members
    if made is not None:
        return made
    dressed = dress(named, value) if isinstance(value, dict) else value
    wrapped = wrap(dressed)
    return None if wrapped is value else wrapped


def _clones_of(classname: str, value: list[Any], current: Any) -> Any:
    """A ``TClonesArray`` member: each record made again as an object of the class the array
    names once at its front, in the array the macro's pointer holds.

    The reader hands a ``TClonesArray`` back as what it held and the one class they are of,
    as a branch of one does; the objects are made from that class's declaration.
    """
    from .collections import TClonesArray

    classname = classname or str(getattr(current, "_class", ""))
    made = current if isinstance(current, TClonesArray) else TClonesArray(classname)
    made.Clear()
    for at, members in enumerate(value):
        obj = made_from(classname, members)
        made.AddAt(obj if obj is not None else members, at)
    return made


def _collection(named: str, value: Any, current: Any) -> Any:
    """A ROOT collection read back - as the plain list of what it held - made into the
    namespace's collection again, or ``None`` for one that is not such a collection.

    ROOT reads a collection into the object it is handed, name and all, so the name on
    file is the name either object ends up with: an array refilled where it stands loses
    the one its constructor gave it, as it does in ROOT, and one made here because the
    member was null is named from the same bytes.

    A slot a ``TObjArray`` never filled stays a hole, so its entries go back by index; a
    list is a run of what it held, which is what it is given.
    """
    if named not in COLLECTIONS or not isinstance(value, list):
        return None
    cls = _collection_class(named)
    if cls is None:
        return None
    made: Any = current if isinstance(current, cls) else cls()
    made.Clear()
    name = str(getattr(value, "name", ""))
    if name:
        made.SetName(name)
    for at, item in enumerate(value):
        if item is not None:
            _add_entry(made, _entry(item), at if named in SLOTTED else None)
    return made


def _collection_class(named: str) -> type | None:
    """The namespace's class standing for one of ROOT's collections, if it has one yet."""
    from .reflection import _class_named

    return _class_named(named)


def _add_entry(made: Any, item: Any, at: int | None) -> None:
    """One entry into a collection: into its own slot if it numbers them, else on the end."""
    if at is None:
        made.Add(item)
    else:
        made.AddAt(item, at)


def _entry(item: Any) -> Any:
    """One object a collection held: made into an object of the class the stream named it
    as, given the members read - or kept as the record it was read as, for a class this kit
    has none of."""
    named = str(getattr(item, "classname", ""))
    made = made_from(named, item) if named else None
    if made is not None or not named or not isinstance(item, Mapping):
        return item if made is None else made
    return _recorded_object(named, item)


def _recorded_object(named: str, row: Mapping[str, Any]) -> Any:
    """One of ROOT's own objects given the members a record of it holds."""
    cls = _collection_class(named)
    if cls is None:
        return row
    made = _new(cls)
    fill_root(made, row)
    return made


def _object(value: Any, found: Spelled, current: Any) -> Any:
    if found.name == "pair":
        first, second = found.args
        one, two = _pair_of(value)
        made = std.pair[first.text(), second.text()]
        return made(converted(one, first, None), converted(two, second, None))
    held = _collection(found.name, value, current)
    if held is not None:
        return held
    cls = find_class(found.name)
    if not isinstance(value, Mapping):
        return value
    from .wrapping import from_members

    if cls is None:  # one of ROOT's classes, held by value
        made = from_members(found.name, value)
        if made is not None:  # one the kit makes again from a record's members
            return made
        if current is None:
            return value
        fill_root(current, value)
        return current
    target = current if isinstance(current, cls) else _new(cls)
    _fill(target, cls, value)
    return target


def _text(value: Any, found: Spelled, current: Any) -> Any:
    text = str(value)
    if found.name == "TString" and current is not None:
        return type(current)(text)
    return text


def _number(value: Any, typename: str, current: Any) -> Any:
    """A number, or an array of them, as the C++ type ``typename`` holds it."""
    if isinstance(current, np.ndarray):
        flat = np.asarray(value).ravel()
        target = current.reshape(-1)
        count = min(len(flat), len(target))
        target[:count] = flat[:count]
        return current
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value)
    return _scalar_number(value, TYPES.get(typename, (0, typename, 0))[1])


def _scalar_number(value: Any, typename: str) -> Any:
    return _converter(typename)(value)


@functools.cache
def _converter(typename: str) -> Callable[[Any], Any]:
    """How a number of the C++ type ``typename`` is made from what a stream read for it.

    The width and the sign a reading masks and unwraps by are the same for every member of
    that type, so they are worked out once: a record read ten million times is nine numbers
    of one class apiece, and looking the width up for each of them was most of the member.
    """
    if typename == "bool":
        return _as_bool
    if typename in FLOATS:
        return _as_float
    bits, signed = WIDTHS[_c_name(typename)]  # every integer a streamer names has a width
    mask, sign, width = (1 << bits) - 1, 1 << (bits - 1), 1 << bits

    def whole(value: Any) -> Any:
        number = _unwrapped(value)
        number = math.trunc(number) if isinstance(number, float) else int(number)
        number &= mask
        return number - width if signed and number >= sign else number

    return whole


def _unwrapped(value: Any) -> Any:
    """A number a stream gave, as the Python number it stands for."""
    return value.item() if isinstance(value, np.generic) else value


def _as_bool(value: Any) -> bool:
    return bool(_unwrapped(value))


def _as_float(value: Any) -> float:
    return float(_unwrapped(value))


def _c_name(typename: str) -> str:
    """ROOT's streamer names for the integers as the C++ types they are."""
    return {"Long64_t": "long long", "ULong64_t": "unsigned long long"}.get(typename, typename)


def _refilled(value: Any, found: Spelled, current: Any) -> Any:
    """A container member given the items read, each converted to what it holds now."""
    if current is None or not hasattr(current, "clear"):
        return value
    current.clear()
    items = list(value.items()) if isinstance(value, Mapping) else list(value)
    if found.name in MAPS:
        key_type, value_type = found.args
        for key, item in map(_pair_of, items):
            _put(current, converted(key, key_type, None), converted(item, value_type, None))
        return current
    add = getattr(current, "push_back", None) or current.insert
    for item in items:
        add(converted(item, found.args[0], None))
    return current


def _pair_of(item: Any) -> tuple[Any, Any]:
    """A pair read - a tuple, or a record of ``first`` and ``second`` - as two values."""
    if isinstance(item, Mapping):
        return item.get("first"), item.get("second")
    first, second = item
    return first, second


def _put(target: Any, key: Any, value: Any) -> None:
    """A key and its value into a map - or a multimap, which only inserts."""
    if hasattr(target, "__setitem__"):
        target[key] = value
    else:
        target.insert((key, value))
