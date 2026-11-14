"""Objects of the classes a program declares, read back from bytes laid out as ROOT lays them.

A key of a file is read by the descriptions the file carries. Bytes written
into a buffer in memory - ``TBufferFile`` - carry none, so they are read by
the descriptions :mod:`.wdeclared` makes of the program's classes, handed to
the same reader a file's are (:mod:`.interp`) as though a file described
them. A standard container standing on its own - a ``vector<int>`` written
into a buffer - reads as the one member of a class made up to hold it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .buffer import Buffer
from .interp import _members, _streamed
from .streamers import Member
from .wdeclared import MAPS, STL_KINDS, Catalogue, Spelled, pair_name, parse_type
from .winfo import INFOS, Element

__all__ = ["DescribedSource", "reader_for"]

#: The streamer type a file gives a container member, which its reader goes by.
STREAMER = 500


def _member(element: Element) -> Member:
    """One element as a file's description of a class gives it to the reader."""
    kind, name, title, stype, _size, length, _dims, _maxidx, typename, extras = element
    if kind in ("TStreamerSTL", "TStreamerSTLstring") and stype == 300:
        stype = STREAMER
    count = str(extras[1]) if kind in ("TStreamerBasicPointer", "TStreamerLoop") else ""
    return Member(name, title, stype, typename, length, count)


class DescribedSource:
    """What the reader asks a file for - how each class is laid out - answered from a catalogue
    of the program's classes and the layouts of ROOT's own."""

    def __init__(self, catalogue: Catalogue, extra: dict[str, dict[str, Member]] | None = None):
        self.catalogue = catalogue
        self._extra = extra or {}

    def streamers(self) -> dict[str, dict[str, Member]]:
        found = {name: _layout(info[2]) for name, info in INFOS.items()}
        found.update((name, _layout(info[2])) for name, info in self.catalogue.entries().items())
        found.update(self._extra)
        return found


def _layout(elements: tuple[Element, ...]) -> dict[str, Member]:
    return {element[1]: _member(element) for element in elements}


#: The made-up class a container standing on its own is read as the one member of.
HOLDER = "<standing container>"


def reader_for(classname: str, catalogue: Catalogue) -> Callable[[Buffer], Any]:
    """How an object of ``classname`` reads from bytes: its record and members - or, for a
    container, its record and items."""
    found = parse_type(classname)
    if found.name in STL_KINDS and not found.stars:
        _describe_contents(found, catalogue)
        held = {HOLDER: {"value": Member("value", "", STREAMER, classname, 0)}}
        holder: Any = DescribedSource(catalogue, held)  # what the reader asks a file, answered
        read = _members(HOLDER, holder)
        return lambda buf: read(buf)["value"]
    catalogue.info(classname)
    source: Any = DescribedSource(catalogue)
    return _streamed(_members(classname, source))


def _describe_contents(found: Spelled, catalogue: Catalogue) -> None:
    """See the classes a container holds described: a map's pairs, or what it holds."""
    if found.name in MAPS:
        catalogue.describe(pair_name(found))
    else:
        catalogue.content_type(found.args[0])
