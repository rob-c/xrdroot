"""What the file says its own classes look like.

Every ROOT file carries a description of the C++ classes it was written from:
for each class, the members in declaration order, each with its type spelled
out. A split branch names only the member, so this is the only place the type
of ``StlVecI16`` can be learnt - and learning it is the difference between a
column that reads and one that is refused.

It is read once, on the first branch that needs it, and not at all by a file
whose branches all say what they hold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .buffer import Buffer

if TYPE_CHECKING:
    from .file import Source

__all__ = ["Member", "read_streamers"]

#: Enough of the record at ``fSeekInfo`` to read its key.
KEY_WINDOW = 512


class Member:
    """One data member of one class, as the file describes it."""

    __slots__ = ("name", "title", "stype", "typename", "length", "count", "dims")

    def __init__(
        self,
        name: str,
        title: str,
        stype: int,
        typename: str,
        length: int,
        count: str = "",
        dims: tuple[int, ...] = (),
    ) -> None:
        self.name = name
        #: The declaration's trailing comment, where a packed float keeps its range.
        self.title = title
        self.stype = stype
        self.typename = typename
        self.length = length
        #: The member holding how many of these there are, for a counted pointer.
        self.count = count
        #: The lengths of a fixed array's dimensions, outermost first - ``fMatrix[4][4]``'s
        #: ``(4, 4)`` - which is the shape its ``length`` values were declared in.
        self.dims = dims

    def __repr__(self) -> str:
        return f"<Member {self.typename} {self.name}>"


def read_element(buf: Buffer, depth: int = 1, counted: bool = False) -> Member:
    """A ``TStreamerElement``, under whichever subclass wrote it.

    Every one of them is the same record with more fields bolted underneath,
    and the fields this reader wants are in the part they share. ``depth`` is
    how many records deep that shared part is: one for nearly every subclass,
    two for the single one that derives from another subclass rather than
    from the element itself. ``counted`` is for the two subclasses that add
    the name of the member saying how many there are.
    """
    _version, end = buf.header()
    for _ in range(depth - 1):
        buf.header()
    base, inner = buf.header()
    name, title = buf.named()
    stype = buf.i32()
    buf.i32()  # how many bytes the member takes in memory
    length = buf.i32()
    dimensions = buf.i32()
    sizes = buf.i32s(buf.i32() if base == 1 else 5)
    typename = buf.string()
    buf.resume(inner)
    count = ""
    if counted:
        buf.i32()  # the version of the class the counter belongs to
        count = buf.string()
    buf.resume(end)
    return Member(name, title, stype, typename, length, count, _dims(dimensions, sizes, length))


def _dims(dimensions: int, sizes: Sequence[int], length: int) -> tuple[int, ...]:
    """The lengths of a fixed array's dimensions, or none for one that is not shaped.

    ROOT writes how many dimensions the member has and, after it, the maximum
    index of each, so ``m[4][4]`` is two dimensions of four. A description that
    does not add up - no dimensions, one, or lengths that do not come to the
    values written - says nothing about a shape, and the values stay flat.
    """
    if dimensions < 2 or len(sizes) < dimensions:
        return ()
    found = tuple(int(size) for size in sizes[:dimensions])
    return found if all(size > 0 for size in found) and math.prod(found) == length else ()


def read_counted(buf: Buffer) -> Member:
    """A ``TStreamerBasicPointer`` or ``TStreamerLoop``, which name their counter."""
    return read_element(buf, counted=True)


def read_info(buf: Buffer) -> tuple[str, dict[str, Member]]:
    """A ``TStreamerInfo``: one class, and the members it was declared with."""
    _version, end = buf.header()
    name, _title = buf.named()
    buf.u32()  # the checksum ROOT uses to tell two versions of a class apart
    buf.i32()  # the class version
    elements = buf.any(CLASSES)
    members = {m.name: m for m in elements if isinstance(m, Member)} if elements else {}
    buf.resume(end)
    return name, members


#: The ``TStreamerElement`` subclasses, all read as the element under them.
ELEMENTS = (
    "TStreamerElement",
    "TStreamerBase",
    "TStreamerBasicType",
    "TStreamerBasicPointer",
    "TStreamerLoop",
    "TStreamerObject",
    "TStreamerObjectAny",
    "TStreamerObjectPointer",
    "TStreamerObjectAnyPointer",
    "TStreamerString",
    "TStreamerSTL",
    "TStreamerArtificial",
)


def read_stl_string(buf: Buffer) -> Member:
    """A ``TStreamerSTLstring``, which is a ``TStreamerSTL`` with a lid on it.

    It is the one subclass that derives from another subclass, so its shared
    part sits one record deeper than everything else's.
    """
    return read_element(buf, 2)


def read_objarray(buf: Buffer) -> list[Any]:
    """The ``TObjArray`` a ``TStreamerInfo`` keeps its members in."""
    return buf.objarray(CLASSES)


CLASSES: dict[str, Any] = dict.fromkeys(ELEMENTS, read_element)
CLASSES["TStreamerBasicPointer"] = read_counted
CLASSES["TStreamerLoop"] = read_counted
CLASSES["TStreamerSTLstring"] = read_stl_string
CLASSES["TStreamerInfo"] = read_info
CLASSES["TObjArray"] = read_objarray


#: One class as a file describes it in full: its checksum, version and elements, each element
#: as :data:`~.winfo.INFOS` holds one.
Described = tuple[int, int, tuple[tuple[Any, ...], ...]]


def _element_in_full(kind: str) -> Any:
    """A reader of one ``TStreamerElement`` subclass, keeping everything it says."""

    def read(buf: Buffer) -> tuple[Any, ...]:
        _version, end = buf.header()
        inner = buf.header()[1] if kind == "TStreamerSTLstring" else None
        base, shared = buf.header()
        name, title = buf.named()
        stype, size, length, dims = buf.i32(), buf.i32(), buf.i32(), buf.i32()
        maxidx = tuple(buf.i32s(buf.i32() if base == 1 else 5))
        typename = buf.string()
        buf.resume(shared)
        extras = _EXTRAS.get(kind, lambda buf: ())(buf)
        buf.resume(inner)
        buf.resume(end)
        return (kind, name, title, stype, size, length, dims, maxidx, typename, extras)

    return read


#: What each subclass writes after the element it builds on.
_EXTRAS: dict[str, Any] = {
    "TStreamerBase": lambda buf: (buf.i32(),),
    "TStreamerBasicPointer": lambda buf: (buf.i32(), buf.string(), buf.string()),
    "TStreamerLoop": lambda buf: (buf.i32(), buf.string(), buf.string()),
    "TStreamerSTL": lambda buf: (buf.i32(), buf.i32()),
    "TStreamerSTLstring": lambda buf: (buf.i32(), buf.i32()),
}


def _info_in_full(buf: Buffer) -> tuple[str, Described]:
    _version, end = buf.header()
    name, _title = buf.named()
    checksum, version = buf.u32(), buf.i32()
    elements = buf.any(FULL) or []
    buf.resume(end)
    return name, (checksum, version, tuple(e for e in elements if isinstance(e, tuple)))


#: The readers that keep every element whole, for :func:`read_infos`.
FULL: dict[str, Any] = {kind: _element_in_full(kind) for kind in (*ELEMENTS, "TStreamerSTLstring")}
FULL["TStreamerInfo"] = _info_in_full
FULL["TObjArray"] = lambda buf: buf.objarray(FULL)


def read_infos(source: Source) -> list[tuple[str, Described]]:
    """Every class this file describes, in full and in the order it does - a class twice
    when the file holds objects of two of its versions."""
    from .file import Key

    seek, nbytes = source.info
    if seek <= 0 or nbytes <= 0:
        return []
    key = Key(Buffer(source.read(seek, min(nbytes, KEY_WINDOW))))
    return [info for info in key.buffer(source).tlist(FULL) if isinstance(info, tuple)]


def read_streamers(source: Source) -> dict[str, dict[str, Member]]:
    """Every class this file describes, by name.

    A file written without streamer information - which a tree of plain
    numbers does not need - gives back nothing, and the branches that would
    have wanted it say so for themselves.
    """
    from .file import Key

    seek, nbytes = source.info
    if seek <= 0 or nbytes <= 0:
        return {}
    key = Key(Buffer(source.read(seek, min(nbytes, KEY_WINDOW))))
    found: dict[str, dict[str, Member]] = {}
    for info in key.buffer(source).tlist(CLASSES):
        if isinstance(info, tuple):
            found.setdefault(info[0], info[1])
    return found
