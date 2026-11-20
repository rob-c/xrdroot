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
from .streamers import Member
from .wdeclared import MAPS, STL_KINDS, Catalogue, Spelled, pair_name, parse_type
from .winfo import INFOS, Element

__all__ = ["DescribedSource", "flat_reader_for", "reader_for"]

#: The streamer type a file gives a container member, which its reader goes by.
STREAMER = 500

#: The streamer type ROOT gives a container that is pointed at rather than held, a
#: ``vector<TObject> *fVectorTobject``: the container written where the pointer stands, with
#: no tag in front of it to say a pointer was there at all.
STL_POINTER = 71

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
    return _streamed(_members(classname, source))


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
