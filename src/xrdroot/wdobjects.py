"""Objects of the classes a program declares, written into a ROOT file as ROOT writes them.

An object is a record - its byte count, then its version, or for a class
without ``ClassDef`` a zero and its checksum - and then its elements in the
order :mod:`.wdeclared` describes them: each base as a record of its own
(``TObject`` and ``TNamed`` as they stream themselves), numbers big-endian and
packed floats squeezed by their comment, a counted array behind a marker
byte, strings by length, and containers in a record of their own. A container
of objects goes *member-wise*, as ROOT 6 writes it: the version of what it
holds, the count, then each member of every object in turn, a map's keys
before its values. A pointer that may be null is a zero, or the object
naming its class. An array of ROOT's own - a ``TArrayF`` or a ``TArrayI``
- is its count word and then its values, with no version in front of them:
ROOT streams one by hand.

What is written is an :class:`Instance`: a class name and its members as plain
values - numbers, strings, sequences, mappings, and further instances - which
the layer that knows the program's objects builds. :class:`DeclaredObject`
hands a file one, with the descriptions of every class it reached.
"""

from __future__ import annotations

import datetime
import struct
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple
from weakref import WeakKeyDictionary

import numpy as np

from .buffer import Listed, Written
from .errors import UnsupportedFeatureError
from .interp import MEMBER_WISE
from .wdeclared import (
    ANY,
    CHAR_STAR,
    MAPS,
    OBJECT,
    OFFSET_L,
    OFFSET_P,
    STL,
    STL_KINDS,
    STL_P,
    Catalogue,
    Spelled,
    _squash,
    bitset_width,
    class_name,
    complex_declaration,
    pair_name,
    parse_type,
)
from .winfo import INFOS, Element
from .wpacking import _PLAIN_STYPES, _packer, pack_member, pack_one
from .writer import Record, WBuffer, _payload, _record, info_entry

__all__ = ["Clones", "Instance", "DeclaredObject", "flat_record", "write_instance",
           "write_members", "write_value", "written_flat"]

#: ``TObject``'s ``kIsOnHeap`` and ``kNotDeleted``, which ROOT keeps out of what it writes.
MEMORY_BITS = 0x03000000
#: The version ROOT 6.40 gives the record round a container or string inside an object:
#: ``TStreamerInfo``'s own class version.
STL_VERSION = 10
#: How ``cppstd`` writes the name a ``std::complex`` gives itself: ``complex<float>``.
COMPLEX_PREFIX = "complex<"
#: The names a string of C++'s owns, which a ``kStreamLoop`` member holds as text - the rest
#: of what such a member can hold being objects.
STRING_NAMES = ("TString", "string", "basic_string")

#: ROOT's collections that stream themselves rather than by the layout a file describes them
#: with - so one read back out of a file is the entries and the name it held, not members - and
#: the class whose ``Streamer`` writes each of them, whose version the record carries.
#: ``THashList`` derives from ``TList`` and overrides nothing, so a file that numbers its own
#: ``THashList`` layout at 0 still writes version 5 of ``TList`` in front of its entries.
STREAMED_COLLECTIONS = {"TObjArray": "TObjArray", "TList": "TList", "THashList": "TList"}



class Clones(NamedTuple):
    """A ``TClonesArray``: its name, the class it holds and the objects in it, ``None`` for an
    empty slot - each an :class:`Instance` of that class."""

    classname: str
    name: str
    items: list[Any]


class Instance(NamedTuple):
    """One object of a class: its class's name, and its members (each base's under its name)."""

    classname: str
    members: Mapping[str, Any]


class DeclaredObject:
    """An object of a program's class, as a key of a file holds it - what
    :meth:`~.writer.WritableDirectory.write` takes through ``root_payload``."""

    def __init__(self, instance: Instance, catalogue: Catalogue, title: str = "") -> None:
        self.instance = instance
        self.catalogue = catalogue
        self.classname = instance.classname
        self.title = title

    def root_payload(
        self, keylen: int
    ) -> tuple[bytes, tuple[str, ...], dict[tuple[str, int], bytes]]:
        """The object's bytes, ROOT's classes it reached, and the descriptions of the
        program's classes it reached, for the file's streamer record."""
        buf = WBuffer()
        write_instance(buf, self.instance, self.catalogue)
        carried = {
            (name, version): info_entry(name, sum_, version, elements)
            for name, (sum_, version, elements) in self.catalogue.entries().items()
        }
        return bytes(buf.data), tuple(self.catalogue.used), carried


def write_instance(buf: WBuffer, instance: Instance, catalogue: Catalogue) -> None:
    """``instance`` written in place, as a member held by value is."""
    _Writer(buf, catalogue).record(instance.classname, instance)


def write_members(buf: WBuffer, instance: Instance, catalogue: Catalogue) -> None:
    """``instance``'s members alone, with no record round them: what one entry of an unsplit
    branch of its class holds, streamed by the class's description."""
    _Writer(buf, catalogue).members(instance.classname, instance.members)


def write_value(buf: WBuffer, classname: str, value: Any, catalogue: Catalogue) -> None:
    """A value of ``classname`` standing on its own: an object's record, or a container's -
    its record, count and items, as a member that is one writes it."""
    found = parse_type(classname)
    writer = _Writer(buf, catalogue)
    if bitset_width(found) is not None or (found.name in STL_KINDS and not found.stars):
        writer.collection(found, value)
    else:
        writer.record(_describe_complex(catalogue, found) or classname, value)


def _describe_complex(catalogue: Catalogue, found: Spelled) -> str | None:
    """The name a bare ``std::complex`` is written under, once the catalogue describes it -
    ``WriteObjectAny`` is handed the class by name and nothing else knows its layout."""
    made = complex_declaration(found)
    if made is None:
        return None
    catalogue.declarations.setdefault(made[0], made[1])
    return made[0]


#: How one member of a class, or a run of them, is written: the writer, and its members.
Step = Callable[["_Writer", Mapping[str, Any]], None]


def _steps_of(
    catalogue: Catalogue, classname: str, elements: tuple[Element, ...]
) -> tuple[Step, ...]:
    """A class's elements as the fewest writes that write them in the order they come.

    Consecutive plain numbers go under one ``struct``, which is what makes a class of nine
    ``int`` members - the record roottest's ``WriteBuffer`` streams ten million times - one
    ``pack`` rather than nine calls into the writer. Everything else keeps its own step, and
    the order of the writes, which is the order of the bytes, is unchanged.

    Kept per class: a class's elements are the same every time it is written, and a bulk
    write of one class asks for them ten million times.
    """
    held = _STEPS.get(catalogue)
    if held is None:
        held = _STEPS[catalogue] = {}
    found = held.get(classname)
    if found is None:
        found = held[classname] = _planned(elements)
    return found


#: Each catalogue's steps by class name. Held weakly: a step says nothing outside its class.
_STEPS: WeakKeyDictionary[Catalogue, dict[str, tuple[Step, ...]]] = WeakKeyDictionary()

#: The structs a run of numbers is written with, by the format each was built from.
_RUNS: dict[str, struct.Struct] = {}

#: The streamer type of a member that is one plain number, which a run can gather.
_BASIC_KIND = "TStreamerBasicType"


def _run_form(members: list[Element]) -> struct.Struct:
    """The struct that writes a run of plain numbers, built once however many runs use it."""
    code = "".join(_packer(element[3]).format.removeprefix(">") for element in members)
    found = _RUNS.get(code)
    if found is None:
        found = _RUNS[code] = struct.Struct(">" + code)
    return found


def _planned(elements: tuple[Element, ...]) -> tuple[Step, ...]:
    """A class's elements as steps, each run of plain numbers gathered into one write."""
    steps: list[Step] = []
    run: list[Element] = []
    for element in elements:
        if _a_plain_number(element):
            run.append(element)
            continue
        _flushed(steps, run)
        steps.append(_one_element(element))
    _flushed(steps, run)
    return tuple(steps)


def _a_plain_number(element: Element) -> bool:
    """Is this member one number of a type ``struct`` writes by itself?

    A packed float is squeezed by the recipe in its declaration and is not one of these; an
    array, a counted array and a ``char *`` are not one number either, and keep their own
    writers.
    """
    stype = element[3]
    return (
        element[0] == _BASIC_KIND
        and stype in _PLAIN_STYPES
        and not OFFSET_L < stype < OFFSET_P
    )


def _flushed(steps: list[Step], run: list[Element]) -> None:
    """Add a run of plain numbers to the plan, as one write or as members of their own."""
    if len(run) > 1:
        steps.append(_written_run(run))
    elif run:
        steps.append(_one_element(run[0]))
    run.clear()


def _one_element(element: Element) -> Step:
    """One member, written by the writer its streamer type has."""
    write = _ELEMENTS[element[0]]
    label = element[1]

    def step(writer: _Writer, members: Mapping[str, Any]) -> None:
        write(writer, element, members.get(label), members)

    return step


def _written_run(run: list[Element]) -> Step:
    """A run of plain numbers written by the one ``struct`` that covers them.

    A number the ``struct`` refuses - a float where an ``int`` goes, a number too wide for
    its member - takes the members one at a time, written as they always were, so that what
    lands on file is NumPy's truncation or wrap and not a refusal.
    """
    members, labels = tuple(run), tuple(element[1] for element in run)
    form = _run_form(run)

    def step(writer: _Writer, row: Mapping[str, Any]) -> None:
        try:
            writer.buf.raw(form.pack(*[_number(row.get(label)) for label in labels]))
        except (struct.error, TypeError, OverflowError):
            for element, label in zip(members, labels, strict=True):
                _Writer._basic(writer, element, row.get(label), row)

    return step


#: Each catalogue's flat records by class name: what :func:`flat_record` worked out for it.
_FLAT: WeakKeyDictionary[
    Catalogue, dict[str, tuple[int, int | None, struct.Struct, tuple[str, ...]] | None]
] = WeakKeyDictionary()


def flat_record(
    catalogue: Catalogue, classname: str
) -> tuple[int, int | None, struct.Struct, tuple[str, ...]] | None:
    """How a class of plain numbers is written in one go - its version, the checksum a
    foreign class carries after it, the struct over its members and their names - or
    ``None`` for a class that is more than that: a base, a string, a container, an object.

    ``TClass::Streamer`` in roottest's bulk loop writes ten million records of nine ints,
    and each of them went through a dictionary of its members to reach the one struct that
    writes them all.
    """
    held = _FLAT.get(catalogue)
    if held is None:
        held = _FLAT[catalogue] = {}
    if classname in held:
        return held[classname]
    plan = catalogue.written_as(classname)
    made = None
    if plan is not None:
        version, sum_, elements = plan
        if elements and all(_a_plain_number(element) for element in elements):
            made = (version, sum_, _run_form(list(elements)),
                    tuple(element[1] for element in elements))  # fmt: skip
    held[classname] = made
    return made


def written_flat(
    buf: WBuffer, classname: str, catalogue: Catalogue, values: Sequence[Any]
) -> bool:
    """``values`` written into ``buf`` as a record of ``classname``, when that class is a run
    of plain numbers - ``False`` when it is not, or a value the struct refuses, and the
    writer's own steps must write it."""
    made = flat_record(catalogue, classname)
    if made is None or len(values) != len(made[3]):
        return False
    version, sum_, form, _labels = made
    at = buf.start(version)
    if sum_ is not None:
        buf.u32(sum_)
    try:
        buf.raw(form.pack(*[_number(value) for value in values]))
    except (struct.error, TypeError, OverflowError):
        return False  # a value the struct will not take, written the way it always was
    buf.end(at)
    return True


class _Writer:
    """Writes objects into one buffer, by the descriptions of one catalogue."""

    def __init__(self, buf: WBuffer, catalogue: Catalogue) -> None:
        self.buf = buf
        self.catalogue = catalogue

    # -- objects ------------------------------------------------------------------------

    def record(self, classname: str, value: Any) -> None:
        """An object of ``classname`` in place: its record, then its members."""
        packed = PACKED_WORDS.get(classname)
        if packed is not None:
            self.buf.u32(packed(value))
            return
        if isinstance(value, Clones):
            self._clones(value)
            return
        if isinstance(value, (Written, Listed)):
            self.streamed(classname, value)
            return
        stype = ARRAY_STREAMS.get(classname)
        if stype is not None:
            self._root_array(classname, stype, value)
            return
        plan = self.catalogue.written_as(classname)
        if plan is None:
            self._root_record(classname, value)
            return
        version, sum_, elements = plan
        at = self.buf.start(version)
        if sum_ is not None:
            self.buf.u32(sum_)
        members = _members(value)
        for step in _steps_of(self.catalogue, classname, elements):
            step(self, members)
        self.buf.end(at)

    def _root_array(self, classname: str, stype: int, value: Any) -> None:
        """``TArrayF::Streamer`` and the arrays it is one of: the count the ``TArray`` base
        streams, and then the values, with nothing else round them.

        ROOT writes no version in front of these: a version word is what a class past its
        first carries, every array here is at 1, and the count word is already the number a
        reader needs - so a ``TArrayF`` of five is ``00 00 00 05`` and five big-endian
        floats, standing where ROOT's own ``TFile`` wrote one. Nor is there a marker byte in
        front of the values, which is how the ``TStreamerBasicPointer`` a file describes
        ``fArray`` with would read them: the class's hand-written ``Streamer`` is what writes
        and reads these bytes, and the description is only what a file says about them.
        """
        self.catalogue.info(classname)  # the record the file carries must describe it
        values = [] if value is None else _array_values(value)
        self.buf.i32(len(values))
        self.buf.raw(pack_member(stype, "", _numeric(values)))

    def _root_record(self, classname: str, value: Any) -> None:
        """An object of one of ROOT's classes, written by the layout the writer carries."""
        if value is not None and not isinstance(value, (Mapping, Instance)):
            # One of ROOT's own written by the writer: a histogram, a TBits, a TRef.
            _classname, payload, used = _payload(value)
            self.catalogue.used.update((name, None) for name in used if name in INFOS)
            self.buf.raw(payload)
            return
        row = dict(_members(value)) if value is not None else {}
        _record(self.buf, classname, row, self.catalogue.used)

    def _clones(self, clones: Clones) -> None:
        """``TClonesArray::Streamer``: its ``TObject``, name, class and version, how many and
        the lower bound, then each slot - a zero, or a one and the object as it streams."""
        version, _sum = self.catalogue.version_of(clones.classname)
        self.catalogue.used["TClonesArray"] = None
        at = self.buf.start(INFOS["TClonesArray"][1])
        self.buf.tobject(0)
        self.buf.string(clones.name)
        self.buf.string(f"{clones.classname};{version}")
        self.buf.i32(len(clones.items))
        self.buf.i32(0)
        for item in clones.items:
            self.buf.u8(0 if item is None else 1)
            if item is not None:
                self.record(clones.classname, item)
        self.buf.end(at)

    def streamed(self, classname: str, value: Any) -> None:
        """``TObjArray::Streamer`` and ``TList::Streamer``: a version, the ``TObject``, the name
        the collection was given, how many entries it holds - and the lower bound a
        ``TObjArray`` keeps - then each entry naming the class it was read as, with a
        ``TList``'s option after it.

        A collection read out of a file is the entries it held, not the members its layout
        describes - ROOT writes it by hand, which is why its description says nothing about
        the bytes of one - so writing the entries back is what makes it a collection again.
        ``buffer.Written`` keeps no lower bound and ``buffer.Listed`` no name, so those go out
        as the empty ones a freshly made array and list start with. Which record is written
        follows what was read, since that is where the shape is: an array is an array of
        entries and a list is entries with the option each was added under.
        """
        held = STREAMED_COLLECTIONS.get(classname)
        listed = isinstance(value, Listed)
        if held is None or (held == "TList") != listed:
            raise UnsupportedFeatureError(
                f"a {classname} read out of a file is written back only where it is one of "
                f"ROOT's collections that stream themselves - "
                f"{', '.join(sorted(STREAMED_COLLECTIONS))} - and its entries are objects of "
                f"classes that file describes; write what you want as the object of the "
                f"class holding it instead"
            )
        at = self.buf.start(INFOS[held][1])
        self.buf.tobject()
        self.buf.string(str(getattr(value, "name", "")))
        self.buf.i32(len(value))
        if listed:
            for item, option in zip(value, value.options, strict=True):
                self.streamed_item(item)
                self.buf.string(str(option))
        else:
            self.buf.i32(0)  # fLowerBound, which reading an array does not keep
            for item in value:
                self.streamed_item(item)
        self.buf.end(at)

    def streamed_item(self, item: Any) -> None:
        """One entry of such a collection: a zero where the slot was empty, or the object
        naming the class the file said it was, which is how those entries are read."""
        if item is None:
            self.buf.u32(0)
            return
        named = str(getattr(item, "classname", ""))
        if not named:
            raise UnsupportedFeatureError(
                f"an entry of a collection read out of a file is written back only where it "
                f"names the class the file said it was, and a {type(item).__name__} does not; "
                f"write the collection this reader made of the file instead"
            )
        self.pointer(named, item)

    def members(self, classname: str, members: Mapping[str, Any]) -> None:
        for element in self.catalogue.info(classname)[2]:
            self.element(element, members)

    def element(self, element: Element, members: Mapping[str, Any]) -> None:
        """One member of one object, laid out as its element describes."""
        # Every element a catalogue describes a program's class with has a writer here.
        _ELEMENTS[element[0]](self, element, members.get(element[1]), members)

    def pointer(self, classname: str, value: Any) -> None:
        """A pointer that may be null: a zero, or the object naming its class.

        A ``TClonesArray`` names itself, not the class its elements are of: the array says
        once, inside, that its objects are of that class, where the tag in front of it says
        what the array itself is - and ROOT reads an object of the wrong class otherwise.
        """
        if value is None:
            self.buf.u32(0)
            return
        named = str(getattr(value, "classname", classname))
        actual = classname if isinstance(value, Clones) else self._named(named)
        at = self.buf.tag(actual)
        self.record(actual, value)
        self.buf.end(at)

    def _named(self, classname: str) -> str:
        """``classname`` as the catalogue spells the class - the name ROOT's dictionary gave
        it, which is the one a tag in front of a pointer carries.

        ``TBufferFile::WriteObjectAny`` writes ``obj->IsA()->GetName()``, and ROOT's dictionary
        spells a template instance the way C++ must, ``template2<template1<int> >``, while the
        translation of the very class spells it ``template2<template1<int>>``. Both are one
        class to the catalogue, which answers to either, but the tag has to be the spelling
        ROOT reads back, since ROOT looks the name up in the dictionary it made - so an
        instance the bridge named the translation's way goes out under the catalogue's.
        """
        return self.catalogue.spelled.get(_squash(classname), classname)

    # -- elements -----------------------------------------------------------------------

    def _base(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        stype = element[3]
        row = _members(value) if value is not None else {}
        if stype == 66:  # TObject, which streams itself
            self.buf.tobject(int(row.get("fBits", 0)) & ~MEMORY_BITS, int(row.get("fUniqueID", 0)))
        elif stype == 67:  # TNamed, likewise
            bits = int(row.get("fBits", 0)) & ~MEMORY_BITS
            self.buf.named(str(row.get("fName", "")), str(row.get("fTitle", "")), bits)
        else:
            self.record(element[1], value)

    def _basic(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        stype, title = element[3], element[2]
        if stype == CHAR_STAR:
            self._char_star(value)
            return
        if OFFSET_L < stype < OFFSET_P:
            value = _characters(value, element[5]) if isinstance(value, (str, bytes)) else value
            values = np.asarray(_numbers(value), dtype=object).ravel()
            if len(values) != element[5]:
                raise ValueError(
                    f"{element[1]} holds {len(values)} values where {element[5]} were declared"
                )
            self.buf.raw(pack_member(stype - OFFSET_L, title, _numeric(values)))
            return
        self.buf.raw(pack_one(stype, title, _number(value)))

    def _char_star(self, value: Any) -> None:
        """``TBuffer::WriteCharStar``: the length, then the characters - a null one, nothing."""
        text = b"" if value is None else _text(value).encode("utf-8", "surrogateescape")
        self.buf.i32(len(text))
        self.buf.raw(text)

    def _counted(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        """``x[n]``: the marker byte, then the values it says are there.

        Nothing to write is the marker alone, whatever the counter says: the counter is a
        member of its own, so a null pointer whose class still says how long it would be is
        written as that marker and no values, which is how ROOT writes one and how reading a
        file gives it back.
        """
        counter = str(element[9][1])
        count = int(_number(members.get(counter, 0)))
        values = list(_numbers(value))[:count] if value is not None else []
        if not values:
            self.buf.u8(0)
            return
        if len(values) < count:
            held = len(values)
            raise ValueError(f"{element[1]} holds {held} values where {counter} says {count}")
        self.buf.u8(1)
        self.buf.raw(pack_member(element[3] - OFFSET_P, element[2], _numeric(values)))

    def _tstring(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        """A ``TString`` member - or the array ``TString f[N]`` ROOT lays out as that same
        element with the array offset on its type, whose N strings it writes one after
        another, each with the length byte the one is written with. A slot the object holds
        no string for is the empty one ROOT's own default-made ``TString`` is."""
        length = int(element[5])  # the N of the array, which a lone string has none of
        if not length:
            self.buf.string(_text(value))
            return
        held = [_text(one) for one in value or ()]
        if len(held) > length:
            raise ValueError(
                f"{element[1]} holds {len(held)} strings where its array has room for {length}"
            )
        for one in (*held, *("" for _ in range(length - len(held)))):
            self.buf.string(one)

    def _stl_string(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        at = self.buf.start(STL_VERSION)
        self.buf.string(_text(value))
        self.buf.end(at)

    def _stream_loop(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        """``kStreamLoop``: as many of the one thing as the counter member holds, one after
        another inside a single record - the record a container's is - and no marker for the
        pointer, which ROOT only streams as the record it fetched them from.

        ``TString *name //[n]`` holds the strings themselves. A counted pointer to a class
        holds objects: ROOT writes each in place for one star, as it writes the object a
        member held by value is, and for two stars each as a pointer naming its class - the
        two shapes :func:`~.interp._loop_step` reads back. A count of nothing is the record
        alone, the same bytes a null pointer writes, so nothing here asks whether the pointer
        is set.
        """
        typename = element[8]
        noun = "strings" if typename.rstrip("*") in STRING_NAMES else "objects"
        counter = str(element[9][1])
        count = int(_number(members.get(counter, 0)))
        items = list(() if value is None else value)[:count]
        if len(items) < count:
            raise ValueError(
                f"{element[1]} holds {len(items)} {noun} where {counter} says {count}"
            )
        at = self.buf.start(STL_VERSION)
        for item in items:
            self._loop_item(typename, item)
        self.buf.end(at)

    def _loop_item(self, typename: str, item: Any) -> None:
        """One item of a ``kStreamLoop``: the string itself, the object in place for a
        ``MyClass *x //[n]``, or for ``MyClass **x`` the pointer that names its class - which
        is a zero, where the pointer is null, rather than an object."""
        if typename.endswith("**"):
            self.pointer(class_name(parse_type(typename)), item)
        elif typename.rstrip("*") in STRING_NAMES:
            self.buf.string(_text(item))
        else:
            self.record(typename.rstrip("*"), item)

    def _stl(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        # A container base names the container in the element's name and gives no type name
        # of its own; a member's element carries both, so either says what the container is.
        found = parse_type(element[8] or element[1])
        if element[3] in (STL + OFFSET_L, STL_P + OFFSET_L):
            self.container_array(found, value)
            return
        if element[3] == STL_P and found.name in ("string", "basic_string", "TString"):
            # A pointer to a string: the string itself, an empty one where it is null, which
            # is how ROOT streams a container pointer.
            at = self.buf.start(STL_VERSION)
            self.buf.string(_text(value))
            self.buf.end(at)
            return
        self.collection(found, value)

    def _object(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        """An object a class holds whole, as ``TLorentzVector fLorentz`` and the ``TVector3``
        in its ``fP`` are: a record of its own, its length and version in front of its
        members, standing where the member goes and with no pointer to it - which is what
        makes the vector inside the four-vector one record inside another.

        What arrives is the members a layer made of that object, never the object itself: the
        layer above is what knows a program's objects, and asking a ``TLorentzVector`` here
        for the ``TVector3`` it keeps would be asking this component for a ROOT name. Those
        members go to :meth:`record` as they are, so a class the program declares is written
        by its declaration and one of ROOT's by the layout this writer carries.

        An array of them, ``T a[N]``, is that record once per element, in turn, and no count
        of its own anywhere: how long the array is, the declaration said.
        """
        if element[3] in (OBJECT + OFFSET_L, ANY + OFFSET_L):
            for one in _run_of(value, element):
                self.record(element[8], one)
            return
        self.record(element[8], value)

    def _pointed(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        """``T *member``, or the array of them ``T *member[N]``, which ROOT describes as that
        same pointer element with no array offset on its type and the N it holds as its
        length - the oracle gives ``TAxis *fVaxis[3]`` type 64, ``kObjectP``, a length of 3
        and three as its maximum index - writing one pointer per element, in turn.

        Read as the one pointer such an element otherwise looks like, the array would be the
        single tag of the first element's object and nothing of the other two.
        """
        classname = element[8].rstrip("*")
        length = int(element[5])  # the N of the array, which a lone pointer has none of
        if not length:
            self._one_pointer(element, classname, value)
            return
        held = list(() if value is None else value)
        if len(held) > length:
            raise ValueError(
                f"{element[1]} holds {len(held)} pointers where its array has room for {length}"
            )
        for one in (*held, *(None for _ in range(length - len(held)))):
            self._one_pointer(element, classname, one)

    def _one_pointer(self, element: Element, classname: str, value: Any) -> None:
        """One pointer of such an element: the object in place, where the comment promises it
        is never null, or the tag naming the class it points at - a zero where it is null."""
        if element[3] in (63, 68):  # promised never null: in place, as if held by value
            self.record(classname, value)
        else:
            self.pointer(classname, value)

    # -- containers ---------------------------------------------------------------------

    def collection(self, found: Spelled, value: Any) -> None:
        """A container member: its record, then its count and what it holds."""
        width = bitset_width(found)
        if width is not None:
            # A bitset is written as the container of its bits: how many there are, then a byte
            # each, the lowest bit first - the order C++ hands them out in.
            at = self.buf.start(STL_VERSION)
            self.items(Spelled("bool", (), 0), _bits(width, value))
            self.buf.end(at)
            return
        items, content = _contents(found, value)
        if self._member_wise(content):
            classname = class_name(content)
            at = self.buf.start(STL_VERSION | MEMBER_WISE)
            version, sum_ = self.catalogue.version_of(classname)
            self.buf.i16(version)
            if sum_ is not None:
                self.buf.u32(sum_)
            self.buf.i32(len(items))
            rows = [_members(item) for item in items]
            for element in self.catalogue.info(classname)[2]:
                self.column(element, rows)
            self.buf.end(at)
            return
        at = self.buf.start(STL_VERSION)
        self.items(content, items)
        self.buf.end(at)

    def _member_wise(self, content: Spelled) -> bool:
        """Is a container of ``content`` written member by member: is it of objects?"""
        if content.stars or content.name in STL_KINDS:
            return False
        return self.catalogue.basic(content) is None and content.name not in ("string", "TString")

    def container_array(self, found: Spelled, value: Any) -> None:
        """An array of containers, as ``vector<double> f[4]`` is: ROOT writes the whole array
        into one record - its version, then each element's count and what it holds, in turn -
        rather than giving every element a record of its own, as a container nested in a
        container is written.

        An element holding objects goes member by member, one element after another inside
        that one record: the class's version once for the array, then each element's count
        and the column each of its members is, as the oracle writes
        ``vector<TNamed> *f[2]`` - ``400a``, ``0001`` for ``TNamed``'s version, a count, the
        ``TObject`` base of each object and then a column per string.
        """
        width = bitset_width(found)
        if width is not None:  # a bitset's elements are its bits, and no object among them
            at = self.buf.start(STL_VERSION)
            for one in value or ():
                self.items(Spelled("bool", (), 0), _bits(width, one))
            self.buf.end(at)
            return
        content = _contents(found, None)[1]
        if self._member_wise(content):
            self.member_wise_array(found, content, value)
            return
        at = self.buf.start(STL_VERSION)
        for one in value or ():
            items, content = _contents(found, one)
            self.items(content, items)
        self.buf.end(at)

    def member_wise_array(self, found: Spelled, content: Spelled, value: Any) -> None:
        """``vector<Obj> *f[N]``: one record for the whole array - the container's version
        with its member-wise bit, the element class's version the one time the array needs
        it, then each element's count and its objects' members as columns, in the array's
        order."""
        classname = class_name(content)
        elements = self.catalogue.info(classname)[2]
        at = self.buf.start(STL_VERSION | MEMBER_WISE)
        version, sum_ = self.catalogue.version_of(classname)
        self.buf.i16(version)
        if sum_ is not None:
            self.buf.u32(sum_)
        for one in value or ():
            items, _held = _contents(found, one)
            rows = [_members(item) for item in items]
            self.buf.i32(len(items))
            for element in elements:
                self.column(element, rows)
        self.buf.end(at)

    def items(self, content: Spelled, items: list[Any]) -> None:
        """A count and the items one after another, as a container of other things holds them."""
        self.buf.i32(len(items))
        basic = self.catalogue.basic(content) if not content.stars else None
        if basic is not None:
            self.buf.raw(pack_member(basic[0], "", _numeric([_number(i) for i in items])))
            return
        for item in items:
            self.item(content, item)

    def item(self, content: Spelled, item: Any) -> None:
        """One item of a container written object-wise: a pointer, a string, or a container -
        its count and items, no record - which is all such a container holds but numbers."""
        if content.stars:
            self.pointer(class_name(content), item)
        elif content.name in ("string", "TString"):
            self.buf.string(_text(item))
        else:
            inner, held = _contents(content, item)
            if self._member_wise(held):
                raise UnsupportedFeatureError(
                    f"a {content.text()} inside a container holds objects, which ROOT writes "
                    f"member-wise inside its own record; this writer does not"
                )
            self.items(held, inner)

    def column(self, element: Element, rows: list[Mapping[str, Any]]) -> None:
        """One member of every object of a member-wise container, in turn."""
        kind, name = element[0], element[1]
        if kind == "TStreamerBase" and self.catalogue.declared(name) is not None:
            # A base of the program's: its own members, each a column, with no record of its own.
            bases = [_members(row.get(name)) for row in rows]
            for inner in self.catalogue.info(name)[2]:
                self.column(inner, bases)
        elif kind == "TStreamerBasicType" and element[3] < OFFSET_L:
            values = [_number(row.get(name)) for row in rows]
            self.buf.raw(pack_member(element[3], element[2], _numeric(values)))
        elif kind in ("TStreamerSTL", "TStreamerSTLstring"):
            self._stl_column(element, rows)
        else:
            for row in rows:
                self.element(element, row)

    def _stl_column(self, element: Element, rows: list[Mapping[str, Any]]) -> None:
        """Every object's container, or string, of one member, inside the one record - which
        an empty container, holding no objects, does not even write."""
        if not rows:
            return
        at = self.buf.start(STL_VERSION)
        found = parse_type(element[8])
        for row in rows:
            self.item(found, row.get(element[1]))
        self.buf.end(at)


#: ROOT's arrays, whose ``Streamer`` is written by hand, and the streamer type of the numbers
#: each holds: ``TArrayF::Streamer`` streams the ``fN`` of its ``TArray`` base and then its
#: ``fArray`` - the shape :meth:`_Writer._root_array` writes. These are the ones the catalogue
#: carries a layout for, so that a file holding one of them says what it is.
ARRAY_STREAMS: dict[str, int] = {
    "TArrayF": 5,  # kFloat_t
    "TArrayI": 3,  # kInt_t
    "TArrayD": 8,  # kDouble_t
}


def _array_values(value: Any) -> list[Any]:
    """The values a ``TArray`` holds: the ``fArray`` ROOT's array keeps them in, or the
    numbers of a plain sequence - a translated macro hands over one of ROOT's own, whose
    ``fArray`` is the values and nothing else of it is a member anything writes."""
    held = value if isinstance(value, (list, tuple, np.ndarray)) else getattr(value, "fArray",
                                                                              None)  # fmt: skip
    if held is None:
        raise UnsupportedFeatureError(
            f"a {type(value).__name__} stands where a ROOT array should be written, and an "
            f"array is written as the values it holds; hand over ROOT's own array, whose "
            f"fArray is those values, or the run of numbers itself"
        )
    return list(np.asarray(held).ravel())


#: How each kind of element is written.
_ELEMENTS: dict[str, Callable[[_Writer, Element, Any, Mapping[str, Any]], None]] = {
    "TStreamerBase": _Writer._base,
    "TStreamerBasicType": _Writer._basic,
    "TStreamerBasicPointer": _Writer._counted,
    "TStreamerString": _Writer._tstring,
    "TStreamerSTLstring": _Writer._stl_string,
    "TStreamerLoop": _Writer._stream_loop,
    "TStreamerSTL": _Writer._stl,
    "TStreamerObject": _Writer._object,
    "TStreamerObjectAny": _Writer._object,
    "TStreamerObjectPointer": _Writer._pointed,
    "TStreamerObjectAnyPointer": _Writer._pointed,
}


def _run_of(value: Any, element: Element) -> list[Any]:
    """The objects of a member declared ``T a[N]``, in the order the declaration gives them.

    A member that is an array of objects is a run and never one object: a mapping or an
    instance where the run goes would be walked as its keys or its fields and written as a
    record each, which is bytes no reader of the class can make sense of - so it is refused,
    naming the member, rather than written as though it were the array.
    """
    if value is None:
        return []
    if isinstance(value, (Mapping, Instance, Record)):
        raise UnsupportedFeatureError(
            f"{element[1]} is an array of {element[8]} and holds the members of one object "
            f"where the run of them belongs; hand over the objects the array holds, in the "
            f"order the declaration gives them"
        )
    return list(value)


def _contents(found: Spelled, value: Any) -> tuple[list[Any], Spelled]:
    """What a container holds, and the type of each: a map's as the pairs it is made of."""
    if found.name in MAPS:
        pairs = value.items() if isinstance(value, Mapping) else (value or ())
        items: list[Any] = [{"first": key, "second": item} for key, item in pairs]
        return items, parse_type(pair_name(found))
    return list(value if value is not None else ()), found.args[0]


def _bits(width: int, value: Any) -> list[int]:
    """``width`` bits of a bitset, lowest first, as ROOT writes one: a byte each."""
    if value is None:
        return [0] * width
    number = int(value.to_ulong()) if hasattr(value, "to_ulong") else int(value)
    return [(number >> bit) & 1 for bit in range(width)]


def _members(value: Any) -> Mapping[str, Any]:
    """An object's members as a mapping: an instance's, a mapping's, a record's - one of ROOT's
    core classes handed over by the layer that knows it, which members are what the class's own
    streamer information describes - or a ``std::complex``'s two numbers, which is all a class
    of C++'s own declaration is here - no layer builds an instance of one."""
    if isinstance(value, (Instance, Record)):
        return value.members
    if isinstance(value, Mapping):
        return value
    if str(getattr(value, "__cpp_name__", "")).startswith(COMPLEX_PREFIX):
        return {"_real": value.real(), "_imag": value.imag()}
    raise UnsupportedFeatureError(
        f"a {type(value).__name__} stands where an object of a class should be written"
    )


def _number(value: Any) -> Any:
    """A number as written: a one-character string is the character's code."""
    if type(value) is int or type(value) is float:
        return value  # what a member of a declared class nearly always is
    if isinstance(value, (str, bytes)):
        return ord(value[:1] or "\0") if value else 0
    return 0 if value is None else value


def _numbers(value: Any) -> list[Any]:
    return [_number(item) for item in np.asarray(value, dtype=object).ravel()]


def _numeric(values: Any) -> np.ndarray[Any, Any]:
    """Values as one NumPy array of their kind, booleans and integers as they are."""
    found = np.asarray([item.item() if isinstance(item, np.generic) else item for item in values])
    return found.astype(np.float64) if found.dtype == object else found


def _characters(text: Any, length: int) -> list[int]:
    """A ``char x[n]`` holding a string: its bytes, NUL-padded to the array's length."""
    raw = text if isinstance(text, bytes) else text.encode("utf-8", "surrogateescape")
    return list(raw[:length]) + [0] * max(0, length - len(raw))


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _packed_datime(value: Any) -> int:
    """ROOT's packed date word for a moment - the inverse of :func:`~.buffer.as_datetime`,
    whose years count from 1995."""
    if not isinstance(value, datetime.datetime):
        raise UnsupportedFeatureError(
            f"a TDatime is written as the moment it stands for, and a {type(value).__name__} "
            f"is not one; hand over the datetime a TDatime stands for"
        )
    when = value
    return (
        ((when.year - 1995) << 26)
        | (when.month << 22)
        | (when.day << 17)
        | (when.hour << 12)
        | (when.minute << 6)
        | when.second
    )

#: The classes whose whole streamer is one packed word - no record and no version round them,
#: which ROOT unpacks the way :func:`~.buffer.as_datetime` does - and how that word is made of
#: the value handed over, so why a ``TDatime`` member reaches here as the moment it stands for.
PACKED_WORDS: dict[str, Callable[[Any], int]] = {"TDatime": _packed_datime}
