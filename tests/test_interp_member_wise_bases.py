"""A member-wise container of a class that has a base: every member of every object is read.

``roottest/root/io/bigevent``'s ``Event::fVectorTLine`` is a ``vector<TLine>``, and a ``TLine``
is three bases and four doubles: ``TObject`` (which streams itself), ``TAttLine`` and
``TAttBBox2D``. ROOT writes a base as a record of its own per object when its class has a
compiled ``Streamer`` of ROOT's - the oracle's ``vector<TLine>`` holds one for ``TAttLine``,
``0008 0002 0001 0001 0001`` - and lays its members among the rest only when nothing but a
streamer info describes it, as the HistFactory classes are described. A reader that flattened
every base read the three shorts of a ``TAttLine`` and then looked for the next object inside
a record header, so the collection ended six bytes an object short: the ``read`` half of that
roottest said its ``fVectorTLine`` record "ended 480 bytes from where it said it would".
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any, cast

from xrdroot import interp
from xrdroot.buffer import BYTE_COUNT_MASK, Buffer
from xrdroot.rdobjects import _member
from xrdroot.streamers import Member

if TYPE_CHECKING:
    from xrdroot.file import Source


def _element(kind: str, name: str, stype: int, typename: str, length: int = 0) -> Member:
    """One streamer element as a reader takes it: only what these shapes read is spelled out."""
    return _member((kind, name, "", stype, 0, length, 0, (0, 0, 0, 0, 0), typename, ()))


def _base(name: str, stype: int = 0) -> Member:
    """A base class's element: a plain base, or one ROOT streams itself (``TObject``'s 66)."""
    return _element("TStreamerBase", name, stype, "BASE")


def _number(name: str, stype: int, typename: str) -> Member:
    return _element("TStreamerBasicType", name, stype, typename)


class _File:
    """A source that describes the classes these tests need and nothing else."""

    name = "fixture.root"

    def __init__(self, described: dict[str, dict[str, Member]]) -> None:
        self._described = described

    def streamers(self) -> dict[str, dict[str, Member]]:
        return self._described


def _record(version: int, body: bytes = b"") -> bytes:
    return struct.pack(">IH", BYTE_COUNT_MASK | (2 + len(body)), version) + body


def _member_wise(count: int, columns: bytes, version: int = 1) -> bytes:
    """One ``vector<Held>`` record written field by field: the container's version with its
    member-wise bit, the class's own version the once, then a count and every object's
    members as columns."""
    return _record(interp.MEMBER_WISE | 1, struct.pack(">hi", version, count) + columns)


def _container(held: str) -> Member:
    return _element("TStreamerSTL", f"vector<{held}>", 500, f"vector<{held}>")


def _read(described: dict[str, dict[str, Member]], held: str, written: bytes) -> list[Any]:
    step = interp._container_step(_container(held), cast("Source", _File(described)), ())
    assert step is not None
    buf = Buffer(written)
    found = step(buf, {})
    assert buf.pos == len(written)  # the record was read to the end it named, not past it
    return list(found)


#: ``Held`` as a ``TLine`` is: a base ROOT streams itself, a base of one of ROOT's own classes
#: (a ``TAttLine``, which this library carries a layout for), and the class's own number.
LINES = {
    "Held": {
        "TObject": _base("TObject", 66),
        "TAttLine": _base("TAttLine"),
        "fX": _number("fX", 8, "double"),
    },
    "TAttLine": {"fLineColor": _number("fLineColor", 2, "short")},
}


def test_a_base_of_one_of_roots_own_classes_is_read_as_the_record_it_was_written_as() -> None:
    """The ``TAttLine`` of a ``TLine``: ROOT has a compiled ``Streamer`` for it, so a file has
    a record of it per object and the reader must not flatten it into a bare column."""
    written = _member_wise(
        1,
        struct.pack(">HII", 1, 0, 0)  # the TObject base, which streams itself
        + _record(2, struct.pack(">h", 3))  # TAttLine's own record: its three shorts
        + struct.pack(">d", 1.5),  # the class's own member
    )
    assert _read(LINES, "Held", written) == [
        {"TObject": {"fUniqueID": 0, "fBits": 0}, "TAttLine": {"fLineColor": 3}, "fX": 1.5}
    ]


#: The same shape with a base that only a streamer info describes - the HistFactory classes a
#: file carries at version 0 with a checksum, and every class a program declares, are these.
FILE_ONLY = {
    "Held": {"Bare": _base("Bare"), "fX": _number("fX", 8, "double")},
    "Bare": {"fLineColor": _number("fLineColor", 2, "short")},
}


def test_a_base_only_the_file_describes_is_read_as_the_columns_it_was_written_as() -> None:
    """The other half of the same rule: a base with no ``Streamer`` of ROOT's behind it has
    its members laid among the rest, with no record standing in front of them."""
    written = _member_wise(1, struct.pack(">h", 3) + struct.pack(">d", 1.5))
    assert _read(FILE_ONLY, "Held", written) == [{"Bare": {"fLineColor": 3}, "fX": 1.5}]


#: One level deeper: ``Held``'s base ``Bare`` has a base of its own, which is flattened into
#: ``Bare``'s columns as ``Bare`` is flattened into ``Held``'s.
NESTED = {
    "Held": {"Bare": _base("Bare"), "fX": _number("fX", 8, "double")},
    "Bare": {"Style": _base("Style"), "fLineColor": _number("fLineColor", 2, "short")},
    "Style": {"fWidth": _number("fWidth", 2, "short")},
}


def test_a_base_of_a_base_is_read_as_columns_too() -> None:
    """A base with a base of its own is the reader's recursion as much as the writer's: both
    of its levels are columns, and every one of their members is read."""
    written = _member_wise(
        1,
        struct.pack(">h", 4)  # Style's member, a column first because it is the first base
        + struct.pack(">h", 3)  # Bare's own member
        + struct.pack(">d", 1.5),
    )
    assert _read(NESTED, "Held", written) == [
        {"Bare": {"Style": {"fWidth": 4}, "fLineColor": 3}, "fX": 1.5}
    ]


#: A base of a base that is one of ROOT's own: only the level with a ``Streamer`` behind it
#: has a record, and the level above it is columns as before.
MIXED = {
    "Held": {"Bare": _base("Bare"), "fX": _number("fX", 8, "double")},
    "Bare": {"TAttLine": _base("TAttLine"), "fLineColor": _number("fLineColor", 2, "short")},
    "TAttLine": {"fLineWidth": _number("fLineWidth", 2, "short")},
}


def test_a_base_of_a_base_that_is_one_of_roots_own_keeps_its_own_record() -> None:
    """Two levels down, the same distinction: a ``TAttLine`` inside ``Bare`` is a record
    where ``Bare``'s own members are columns."""
    written = _member_wise(
        1,
        _record(2, struct.pack(">h", 5))  # TAttLine's record, where its member would go
        + struct.pack(">h", 3)
        + struct.pack(">d", 1.5),
    )
    assert _read(MIXED, "Held", written) == [
        {"Bare": {"TAttLine": {"fLineWidth": 5}, "fLineColor": 3}, "fX": 1.5}
    ]
