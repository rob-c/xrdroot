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

__all__ = ["Member", "read_layouts", "read_streamers", "reconcile"]

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


#: The file :func:`read_infos` last described the layouts of, which ROOT's
#: ``TStreamerInfo::BuildCheck`` warning names. That warning fires while the file is being
#: opened, and the caller which notes those layouts (:func:`.reflection.note_file_infos`) is
#: handed the layouts alone, so the name of the file they came from is kept here.
LAST_SOURCE = ""


def read_infos(source: Source) -> list[tuple[str, Described]]:
    """Every class this file describes, in full and in the order it does - a class twice
    when the file holds objects of two of its versions."""
    from .file import Key

    global LAST_SOURCE

    seek, nbytes = source.info
    LAST_SOURCE = str(getattr(source, "name", "") or "")
    if seek <= 0 or nbytes <= 0:
        return []
    key = Key(Buffer(source.read(seek, min(nbytes, KEY_WINDOW))))
    return [info for info in key.buffer(source).tlist(FULL) if isinstance(info, tuple)]


def read_layouts(source: Source) -> dict[str, list[Described]]:
    """Every class this file describes, by name, keeping every version it recorded.

    One name can hold more than one layout: a file merged from two vintages of a
    program, or appended to after a class changed, records a layout per version,
    and each object must be read by the layout of the version its own record
    names. Collapsing them - which is what :func:`read_streamers` does, because a
    reader that only wants to know what a member's name means wants one - misreads
    every object written by the vintage that came second.
    """
    found: dict[str, list[Described]] = {}
    for name, layout in read_infos(source):
        found.setdefault(name, []).append(layout)
    return found


#: ``TStreamerBase``: the element kind that names a base class. Its element name is the
#: class and the one number it keeps past the shared part is the version it was declared
#: with - which is the whole of what the warning below needs.
BASE = "TStreamerBase"


def _bases(layout: Described) -> dict[str, int]:
    """The base classes one layout records, and the version each of them was declared with."""
    return {element[1]: element[9][0] for element in layout[2] if element[0] == BASE and element[9]}


def reconcile(
    name: str, mine: Described, on_file: list[Described], filename: str
) -> tuple[Described | None, str]:
    """Which layout a file's objects of one class are read with, and what ROOT says about it.

    ROOT reads a record by the layout the file recorded for the version the record
    names - but not when that version is also the version the class has in memory:
    then a differing checksum means the file was written by another build of the
    same class version, and ROOT reads by the *in-memory* layout, because that is
    the one the objects in memory have and the only one that makes the two agree.
    Reading by the file's layout instead is silently wrong rather than loudly: it
    consumes a plausible number of bytes, so the values come out as the file meant
    them and nothing is noticed until an object is compared with the same object
    written today.

    ``mine`` is the layout the class has in memory, ``on_file`` every layout the
    file records for it, and the answer is the layout to read with - ``None`` when
    the file records none at that version, where the version a record names picks
    one from ``on_file`` and there is nothing to settle - plus the warning text,
    which is empty when the two agree. The caller puts
    ``Warning in <CreateReadMemberWiseActions>: `` in front of it.
    """
    same = next((layout for layout in on_file if layout[1] == mine[1]), None)
    if same is None or same[0] == mine[0]:
        return same, ""
    return mine, _not_in_sync(name, mine, same, filename)


def _not_in_sync(name: str, mine: Described, on_file: Described, filename: str) -> str:
    """ROOT's warning for a class whose version survived but whose layout did not.

    Every line but ROOT's own prefix is indented by three spaces, which is how the
    message reads in ``execMixedBaseClass_v2.ref``: the last sentence names the base
    class that changed, since that is what moved the members the record's objects
    were written with, and it is left out when no base class's version differs.
    """
    version = mine[1]
    lines = [
        f"   The StreamerInfo of class {name} read from file {filename}",
        f"   has the same version (={version}) as the active class but a different checksum.",
        f"   You should update the version to ClassDef({name},{version + 1}).",
    ]
    theirs = _bases(on_file)
    for base, declared in _bases(mine).items():
        if theirs.get(base, declared) != declared:
            lines += [
                "   The objects on this file might not be readable because:",
                f"   The in-memory layout version {version} for class '{name}' has a base "
                f"class ({base}) with version {declared} but the on-file layout version "
                f"{version} recorded the version number {theirs[base]} for this base "
                f"class ({base}).",
            ]
            break
    return "\n".join(lines)


def read_streamers(source: Source) -> dict[str, dict[str, Member]]:
    """Every class this file describes, by name, as the first layout it recorded.

    A file written without streamer information - which a tree of plain
    numbers does not need - gives back nothing, and the branches that would
    have wanted it say so for themselves. Use :func:`read_layouts` instead
    where two versions of one class must be told apart.
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
