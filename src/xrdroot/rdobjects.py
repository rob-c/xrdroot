"""Objects of the classes a program declares, read back from bytes laid out as ROOT lays them.

A key of a file is read by the descriptions the file carries. Bytes written
into a buffer in memory - ``TBufferFile`` - carry none, so they are read by
the descriptions :mod:`.wdeclared` makes of the program's classes, handed to
the same reader a file's are (:mod:`.interp`) as though a file described
them. A standard container standing on its own - a ``vector<int>`` written
into a buffer - reads as the one member of a class made up to hold it.

How a class reads is a pure function of its name and the catalogue that
describes it, so both the layouts a catalogue answers with and the reader a
class is read by are kept: ``TClass::Streamer`` in a bulk loop - roottest's
``ReadBuffer`` reads 10^7 records of one declared class - used to lay out
ROOT's own eighty classes and this program's for every record, which was
nearly all of the read.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from . import interp
from .buffer import Buffer
from .interp import _members, _streamed
from .rootarrays import ARRAYS as ARRAY_CLASSES
from .streamers import Member
from .wdeclared import MAPS, STL_KINDS, Catalogue, Spelled, pair_name, parse_type
from .winfo import INFOS, Element

__all__ = ["DescribedSource", "Members", "flat_reader_for", "reader_for"]

#: The streamer type a file gives a container member, which its reader goes by.
STREAMER = 500

#: The streamer type ROOT gives a container that is pointed at rather than held, a
#: ``vector<TObject> *fVectorTobject``: the container written where the pointer stands, with
#: no tag in front of it to say a pointer was there at all.
STL_POINTER = 71

#: The two streamer types of a member that is one object written where it stands - ROOT's
#: ``kObject`` and ``kObjectAny``. The record goes where the member is with no class name in
#: front of it, so the description the class is laid out by is the only place that says what
#: class the object is. A pointer to one is ``kObjectPtr``, which names its class itself.
HELD_OBJECTS = (61, 62)

#: How many catalogues' layouts are kept at once. A program declares a bounded number of
#: classes, so it reads a bounded number of catalogues - one per class declared again - and a
#: layout is a dictionary of every class ROOT carries, so this is the width of the memory kept.
SOURCES_KEPT = 32
#: How many readers are kept at once: one per class read from each catalogue, which is the
#: number of classes a program declares or holds in a file.
READERS_KEPT = 256


def _member(element: Element) -> Member:
    """One element as a file's description of a class gives it to the reader.

    A container is read by the type it names rather than by a class of its own, and that is
    the same bytes whether the member is a container or a pointer to one: the bigevent
    roottest's ``Event::fVectorTobject`` is a ``vector<TObject> *`` whose three objects come
    after a record and a count, with nothing saying the pointer was there, so the pointer is
    taken off the type and the container read where the pointer stands.
    """
    kind, name, title, stype, _size, length, _dims, _maxidx, typename, extras = element
    if kind in ("TStreamerSTL", "TStreamerSTLstring") and stype in (300, STL_POINTER):
        stype = STREAMER
        typename = typename.rstrip("*")  # a pointer to a container holds the container
    count = str(extras[1]) if kind in ("TStreamerBasicPointer", "TStreamerLoop") else ""
    return Member(name, title, stype, typename, length, count, _shape(_dims, _maxidx))


def _shape(dims: int, max_index: tuple[int, ...]) -> tuple[int, ...]:
    """A member's array lengths as the catalogue records them, for the reader's own shape."""
    return tuple(int(size) for size in max_index[:dims]) if dims > 1 else ()


class DescribedSource:
    """What the reader asks a file for - how each class is laid out - answered from a catalogue
    of the program's classes and the layouts of ROOT's own.

    The answer is worked out once and kept, because it is asked for again for every record read
    and it changes only when the catalogue describes another class: the count of what the
    catalogue knows is the sign of that, so a class described later is in the layouts a reader
    made afterwards works from, while a record read by a reader already made costs nothing.
    """

    def __init__(
        self,
        catalogue: Catalogue,
        extra: dict[str, dict[str, Member]] | None = None,
        base: DescribedSource | None = None,
    ) -> None:
        self.catalogue = catalogue
        self._extra = extra or {}
        #: The source whose layouts these extend, for the class made up to hold a container.
        self._base = base
        self._found: dict[str, dict[str, Member]] | None = None
        #: What the catalogue knew when ``_found`` was worked out.
        self._described: tuple[int, int] | None = None

    def streamers(self) -> dict[str, dict[str, Member]]:
        if self._base is not None:  # a container's holder, on the layouts of the catalogue
            found = dict(self._base.streamers())
            found.update(self._extra)
            return found
        key = (len(self.catalogue.declarations), len(self.catalogue.infos))
        if self._found is None or self._described != key:
            self._found = _described(self.catalogue)
            self._described = key
        return self._found


def _described(catalogue: Catalogue) -> dict[str, dict[str, Member]]:
    """Every class laid out as the catalogue describes it: ROOT's own, and the program's."""
    found = {name: _layout(info[2]) for name, info in INFOS.items()}
    found.update((name, _layout(info[2])) for name, info in catalogue.entries().items())
    return found


def _layout(elements: tuple[Element, ...]) -> dict[str, Member]:
    return {element[1]: _member(element) for element in elements}


@functools.lru_cache(maxsize=SOURCES_KEPT)
def _source(catalogue: Catalogue) -> DescribedSource:
    """The layouts one catalogue answers with, kept for the readers made from it."""
    return DescribedSource(catalogue)


#: The made-up class a container standing on its own is read as the one member of.
HOLDER = "<standing container>"


def reader_for(classname: str, catalogue: Catalogue) -> Callable[[Buffer], Any]:
    """How an object of ``classname`` reads from bytes: its record and members - or, for a
    container, its record and items.

    Kept per class and catalogue, because what it is is nothing the bytes can change: a
    ``TClass::Streamer`` in a bulk loop of 10^7 records used to lay every class out again and
    step each of its members again for every one of them, which was most of the read.
    """
    return _reader_for(classname, catalogue)


@functools.lru_cache(maxsize=READERS_KEPT)
def _reader_for(classname: str, catalogue: Catalogue) -> Callable[[Buffer], Any]:
    found = parse_type(classname)
    if found.name in STL_KINDS and not found.stars:
        _describe_contents(found, catalogue)
        held = {HOLDER: {"value": Member("value", "", STREAMER, classname, 0)}}
        holder: Any = DescribedSource(catalogue, held, _source(catalogue))
        read = _members(HOLDER, holder)  # what the reader asks of the made-up holder
        return lambda buf: read(buf)["value"]
    catalogue.info(classname)
    source: Any = _source(catalogue)
    return _array_read(source, classname, _streamed(_members(classname, source)))


def _array_members(source: Any, classname: str) -> tuple[str, ...]:
    """Which of a class's members are one of ROOT's array classes, by name.

    ROOT describes a ``TArrayF`` as a class of two members - the counter ``fN`` of the base
    ``TArray`` it derives from, and the values ``fArray`` counted by it - but gives the class
    no streamer of its own, so a member of one, held by value or pointed at, is written as
    those two members and nothing else: no record round them, and no class name in front of
    the one held by value. The reader reads that as the row of numbers it is, and these are
    the members to give back as the class the namespace has, which is what a macro reaches
    ``fArray`` and ``GetArray`` on.
    """
    layout = source.streamers().get(classname) or {}
    return tuple(
        name for name, member in layout.items()
        if parse_type(member.typename).name in ARRAY_CLASSES  # not a base, which types ``BASE``
    )  # fmt: skip


def _class_of(source: Any, classname: str) -> tuple[tuple[str, str], ...]:
    """Which of a class's members hold an object of one of ROOT's own classes by value, and the
    class each of them is: the ``fP`` of a ``TLorentzVector`` is a ``TVector3``.

    ROOT writes that object's record where the member stands, with nothing in the bytes saying
    which class it is, so what the reader hands back for it is a dictionary of members and no
    name of a class - which is not something a macro's ``fLorentz.Px()`` can be read off. Only
    a class the description lays out here is named: one the program declared is not, since
    what it is was already known from the declaration that made it.
    """
    layout = source.streamers().get(classname) or {}
    return tuple(
        (name, parse_type(member.typename).name)
        for name, member in layout.items()
        if member.stype in HELD_OBJECTS and parse_type(member.typename).name in INFOS
    )  # fmt: skip


def _cell_of(member: Member) -> str | None:
    """The class a container member holds *by value*, or ``None`` for a member that is no
    such container.

    ROOT describes a container of objects by the type it names - ``vector<TObject>`` - so what
    the items are is in the spelling and nowhere else, and the kind of content it is says how
    they were written: ``kObject``, the object itself in the record, against ``kObjectPtr``,
    a pointer, whose item names its own class in the bytes and may well be of a subclass. Only
    the first is a class the declaration can name, so only it is named here; a map is none of
    them, its items being the pairs its two types make.
    """
    if member.stype != STREAMER:
        return None
    found = parse_type(member.typename)
    if found.name in MAPS or found.name not in STL_KINDS or len(found.args) != 1:
        return None
    content = found.args[0]
    return content.name if not content.stars and content.name in INFOS else None


def _cells_of(source: Any, classname: str) -> tuple[tuple[str, str], ...]:
    """Which of a class's members are a container of one of ROOT's own classes by value, and
    the class each of them holds - the ``vector<TObject> *fVectorTobject`` of the bigevent
    roottest's ``Event``, whose three objects the bytes give as three rows of numbers.

    Nothing in the bytes says which class an item is, since the container was declared as one:
    what the reader hands back for the lot is dictionaries of members, and the description is
    the only place where the class of them is written down - so a macro asking an item for what
    its class has has nothing to ask.
    """
    layout = source.streamers().get(classname) or {}
    return tuple(
        (name, cell) for name, member in layout.items() if (cell := _cell_of(member)) is not None
    )


class Members(dict[str, Any]):
    """The members of one of ROOT's own classes, read from a record that held it by value - or
    from one item of a container that held them all so.

    A dictionary of the members the class is described with, and the name of the class, which
    the bytes did not carry: ``TLorentzVector::fP`` is a ``TVector3``, and the three numbers it
    holds are that class's ``fX``, ``fY`` and ``fZ`` - what the namespace's ``TVector3`` is
    made from, and what a macro reaches ``Px()`` on.
    """

    def __init__(self, members: dict[str, Any], classname: str) -> None:
        super().__init__(members)
        #: The class of ROOT's the members are of, as the description named the member's type.
        self.classname = classname


def _array_read(
    source: Any, classname: str, read: Callable[[Buffer], Any]
) -> Callable[[Buffer], Any]:
    """A class's record read, with each member that is one of ROOT's arrays handed back as the
    members that class is described with, each that is one of ROOT's own objects handed back as
    those members and the name of their class, and each that is a container of them - every
    item of it so - or ``read`` itself, for a class holding none of the three, which is every
    class but the few that keep a ROOT array, object or container of them in them."""
    names = _array_members(source, classname)
    held = _class_of(source, classname)
    cells = _cells_of(source, classname)
    if not names and not held and not cells:
        return read

    def value(buf: Buffer) -> dict[str, Any]:
        row: dict[str, Any] = read(buf)
        for name in names:
            values = row.get(name)
            if values is not None:  # a pointer to one that was never made is still null
                row[name] = {"TArray": {"fN": len(values)}, "fArray": values}
        for name, of_class in held:
            members = row.get(name)
            if isinstance(members, dict):  # a member that streams itself is no dict - a TDatime
                row[name] = Members(members, of_class)
        for name, of_class in cells:
            items = row.get(name)
            if items:  # a container that was never made holds nothing to name
                row[name] = [Members(item, of_class) for item in items]
        return row

    return value


def flat_reader_for(
    classname: str, catalogue: Catalogue
) -> tuple[Callable[[Buffer], tuple[Any, ...]], tuple[str, ...]] | None:
    """How an object of ``classname`` reads as one struct, with its members' names - or
    ``None`` when it is more than a run of plain numbers.

    Kept like the reader itself, because what it is no bytes can change.
    """
    return _flat_reader_for(classname, catalogue)


@functools.lru_cache(maxsize=READERS_KEPT)
def _flat_reader_for(
    classname: str, catalogue: Catalogue
) -> tuple[Callable[[Buffer], tuple[Any, ...]], tuple[str, ...]] | None:
    found = parse_type(classname)
    if found.name in STL_KINDS and not found.stars:
        return None
    catalogue.info(classname)
    source: Any = _source(catalogue)
    return interp.flat_reader(classname, source)


def _describe_contents(found: Spelled, catalogue: Catalogue) -> None:
    """See the classes a container holds described: a map's pairs, or what it holds."""
    if found.name in MAPS:
        catalogue.describe(pair_name(found))
    else:
        catalogue.content_type(found.args[0])
