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
naming its class.

What is written is an :class:`Instance`: a class name and its members as plain
values - numbers, strings, sequences, mappings, and further instances - which
the layer that knows the program's objects builds. :class:`DeclaredObject`
hands a file one, with the descriptions of every class it reached.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np

from .errors import UnsupportedFeatureError
from .interp import MEMBER_WISE
from .wdeclared import (
    MAPS,
    OFFSET_L,
    OFFSET_P,
    STL_KINDS,
    Catalogue,
    Spelled,
    class_name,
    pair_name,
    parse_type,
)
from .winfo import Element
from .wpacking import pack_member
from .writer import WBuffer, _record, info_entry

__all__ = ["Instance", "DeclaredObject", "write_instance", "write_value"]

#: ``TObject``'s ``kIsOnHeap`` and ``kNotDeleted``, which ROOT keeps out of what it writes.
MEMORY_BITS = 0x03000000
#: The version ROOT 6.40 gives the record round a container or string inside an object:
#: ``TStreamerInfo``'s own class version.
STL_VERSION = 10


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


def write_value(buf: WBuffer, classname: str, value: Any, catalogue: Catalogue) -> None:
    """A value of ``classname`` standing on its own: an object's record, or a container's -
    its record, count and items, as a member that is one writes it."""
    found = parse_type(classname)
    writer = _Writer(buf, catalogue)
    if found.name in STL_KINDS and not found.stars:
        writer.collection(found, value)
    else:
        writer.record(classname, value)


class _Writer:
    """Writes objects into one buffer, by the descriptions of one catalogue."""

    def __init__(self, buf: WBuffer, catalogue: Catalogue) -> None:
        self.buf = buf
        self.catalogue = catalogue

    # -- objects ------------------------------------------------------------------------

    def record(self, classname: str, value: Any) -> None:
        """An object of ``classname`` in place: its record, then its members."""
        if self.catalogue.declared(classname) is None:
            self._root_record(classname, value)
            return
        version, sum_ = self.catalogue.version_of(classname)
        at = self.buf.start(version)
        if sum_ is not None:
            self.buf.u32(sum_)
        self.members(classname, _members(value))
        self.buf.end(at)

    def _root_record(self, classname: str, value: Any) -> None:
        """An object of one of ROOT's classes, written by the layout the writer carries."""
        row = dict(_members(value)) if value is not None else {}
        _record(self.buf, classname, row, self.catalogue.used)

    def members(self, classname: str, members: Mapping[str, Any]) -> None:
        for element in self.catalogue.info(classname)[2]:
            self.element(element, members)

    def element(self, element: Element, members: Mapping[str, Any]) -> None:
        """One member of one object, laid out as its element describes."""
        # Every element a catalogue describes a program's class with has a writer here.
        _ELEMENTS[element[0]](self, element, members.get(element[1]), members)

    def pointer(self, classname: str, value: Any) -> None:
        """A pointer that may be null: a zero, or the object naming its class."""
        if value is None:
            self.buf.u32(0)
            return
        actual = value.classname if isinstance(value, Instance) else classname
        at = self.buf.tag(actual)
        self.record(actual, value)
        self.buf.end(at)

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
        if OFFSET_L < stype < OFFSET_P:
            values = np.asarray(_numbers(value), dtype=object).ravel()
            if len(values) != element[5]:
                raise ValueError(
                    f"{element[1]} holds {len(values)} values where {element[5]} were declared"
                )
            self.buf.raw(pack_member(stype - OFFSET_L, title, _numeric(values)))
            return
        self.buf.raw(pack_member(stype, title, _numeric([_number(value)])))

    def _counted(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        counter = str(element[9][1])
        count = int(_number(members.get(counter, 0)))
        values = list(_numbers(value))[:count] if value is not None else []
        if len(values) < count:
            held = len(values)
            raise ValueError(f"{element[1]} holds {held} values where {counter} says {count}")
        self.buf.u8(1 if count else 0)
        if count:
            self.buf.raw(pack_member(element[3] - OFFSET_P, element[2], _numeric(values)))

    def _tstring(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        self.buf.string(_text(value))

    def _stl_string(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        at = self.buf.start(STL_VERSION)
        self.buf.string(_text(value))
        self.buf.end(at)

    def _stl(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        self.collection(parse_type(element[8]), value)

    def _object(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        self.record(element[8], value)

    def _pointed(self, element: Element, value: Any, members: Mapping[str, Any]) -> None:
        classname = element[8].rstrip("*")
        if element[3] in (63, 68):  # promised never null: in place, as if held by value
            self.record(classname, value)
        else:
            self.pointer(classname, value)

    # -- containers ---------------------------------------------------------------------

    def collection(self, found: Spelled, value: Any) -> None:
        """A container member: its record, then its count and what it holds."""
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


#: How each kind of element is written.
_ELEMENTS: dict[str, Callable[[_Writer, Element, Any, Mapping[str, Any]], None]] = {
    "TStreamerBase": _Writer._base,
    "TStreamerBasicType": _Writer._basic,
    "TStreamerBasicPointer": _Writer._counted,
    "TStreamerString": _Writer._tstring,
    "TStreamerSTLstring": _Writer._stl_string,
    "TStreamerSTL": _Writer._stl,
    "TStreamerObject": _Writer._object,
    "TStreamerObjectAny": _Writer._object,
    "TStreamerObjectPointer": _Writer._pointed,
    "TStreamerObjectAnyPointer": _Writer._pointed,
}


def _contents(found: Spelled, value: Any) -> tuple[list[Any], Spelled]:
    """What a container holds, and the type of each: a map's as the pairs it is made of."""
    if found.name in MAPS:
        pairs = value.items() if isinstance(value, Mapping) else (value or ())
        items: list[Any] = [{"first": key, "second": item} for key, item in pairs]
        return items, parse_type(pair_name(found))
    return list(value if value is not None else ()), found.args[0]


def _members(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Instance):
        return value.members
    if isinstance(value, Mapping):
        return value
    raise UnsupportedFeatureError(
        f"a {type(value).__name__} stands where an object of a class should be written"
    )


def _number(value: Any) -> Any:
    """A number as written: a one-character string is the character's code."""
    if isinstance(value, (str, bytes)):
        return ord(value[:1] or "\0") if value else 0
    return 0 if value is None else value


def _numbers(value: Any) -> list[Any]:
    return [_number(item) for item in np.asarray(value, dtype=object).ravel()]


def _numeric(values: Any) -> np.ndarray[Any, Any]:
    """Values as one NumPy array of their kind, booleans and integers as they are."""
    found = np.asarray([item.item() if isinstance(item, np.generic) else item for item in values])
    return found.astype(np.float64) if found.dtype == object else found


def _text(value: Any) -> str:
    return "" if value is None else str(value)
