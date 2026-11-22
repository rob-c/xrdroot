"""Writing a ROOT file: keys, objects, and the description that reads them back.

A file written here is the real thing - a header, a directory, one key per
object, a key list, a free list, and the streamer information saying what the
classes look like - laid out the way ROOT 6.08 laid out the donor files the
layouts in :mod:`.winfo` were harvested from. Anything that reads ROOT files
by their own self-description, this library included, walks the result by the
same map it was written from.

What can be written is what can be written *correctly*: histograms, graphs,
strings and arrays of numbers. Anything else is refused by name rather than
guessed at, because a plausible-looking file that ROOT misreads is worse than
an error message. The same goes for the parts of an object this writer cannot
carry: a histogram's list of functions is written when what it holds is
functions - the fits ROOT hangs there - and refused, rather than silently
dropped, when it holds anything else.

Records go out as they are made: only the hundred-odd bytes of header at the
front are held back, because they point at the bookkeeping written last, and
they are filled in when the file closes. A file this writes can therefore be
far larger than memory. A ``with`` block that raises takes back whatever it
had written - the target is cut back to where the file began - so what is
left behind is nothing rather than half a file. A target that cannot seek,
which no header could be filled in on afterwards, is the one exception: that
file is kept in memory and written in one piece at a clean close.

Directories are ROOT's own: a ``TDirectory`` record per directory, each with
its key list, and a file that grows past 2 GB changes to ROOT's wide layout
for the records past that point, as ROOT does, rather than being refused.
"""

from __future__ import annotations

import array
import datetime
import struct
import uuid
from collections.abc import Callable, Iterator, Mapping
from typing import IO, TYPE_CHECKING, Any

import numpy as np

from .buffer import BYTE_COUNT_MASK, CLASS_MASK, IS_REFERENCED, MAP_OFFSET, NEW_CLASS_TAG, Buffer
from .compression import CODES, LEVELS, compress, spelled
from .efficiency import EFFICIENCIES, Efficiency
from .errors import UnsupportedFeatureError
from .file import Key
from .function import FUNCTIONS, Function
from .graph import GRAPHS, Graph
from .hist import HISTOGRAMS, Histogram
from .interp import MEMBER_WISE, OFFSET_L, OFFSET_P
from .profile import PROFILES
from .rootarrays import ARRAYS
from .stacks import MultiGraph
from .transport import parse
from .winfo import INFOS, SUBVERSIONS, WRITER_VERSION, Element

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .rntuple.writer import WritableRNTuple
    from .wtree import WritableTree

__all__ = ["create", "ObjString", "Record", "WritableFile", "WritableDirectory"]

#: The bits a freshly made object carries: on the heap, and not deleted.
BITS = 0x03000000
#: The most bytes ``TKey`` writes an object in raw whatever the compression setting.
SMALL_OBJECT = 256
#: The bits of an object's identifier that number it within its process.
NUMBER_MASK = 0xFFFFFF
#: ``TCollection::kUseRWLock``, which ROOT sets on the list of functions every
#: histogram it makes carries.
USE_RWLOCK = 1 << 16
#: What every ROOT file starts with.
MAGIC = b"root"
#: Where the first record starts; the hundred bytes before it are the header.
BEGIN = 100
#: Where ROOT says big-file territory starts: a record that points past here
#: needs the wide layout, its places in the file in eight bytes rather than
#: four. It is read when each record is written, never copied, so a test can
#: lower it and see the wide layout without writing 2 GB to get there.
BIG = 2_000_000_000
#: How far ROOT moves the end of its free list each time a file passes it.
GROW = 1_000_000_000
#: The key version this writer emits for a record with small offsets.
KEY_VERSION = 4
#: The version of a directory's record, and of each free-list entry.
DIRECTORY_VERSION = 5
FREE_VERSION = 1
#: What a key, a directory record or a free-list entry adds to its version
#: to say it is wide, and what the header adds to the file's.
WIDE = 1000
WIDE_FILE = 1_000_000
#: How much longer a wide key is: two places in eight bytes, not four.
WIDER = 8
#: How long a directory's record is, small or wide: ROOT leaves room for the
#: wide places in every record, so a directory never has to move.
DIRECTORY_BYTES = 60
#: How long a free-list entry is, small and wide.
FREE_SMALL = 10
FREE_WIDE = 18
#: The most a gap's marker can say it spans, being four signed bytes.
MAX_GAP = 0x7FFFFFFF
#: How many bytes of one tree column gather before that column writes a
#: basket. ROOT's own default, and a reasonable trade: bigger baskets read
#: faster in bulk, smaller ones let a reader take a few entries cheaply.
BASKET_BYTES = 32_000
#: How many bytes of finished records gather before they go to the target in
#: one write. A remote file pays a round trip per write, and a basket is a few
#: tens of kilobytes, so writing each the moment it is made would spend the
#: time on latency rather than on bytes.
WRITE_BEHIND = 8 << 20

#: The struct code for each basic streamer type this writer packs.
FORMS = {
    1: "b",
    2: "h",
    3: "i",
    4: "q",
    5: "f",
    6: "i",
    8: "d",
    11: "B",
    12: "H",
    13: "I",
    14: "Q",
    15: "I",
    16: "q",
    17: "Q",
    18: "B",
}

#: The ``TArray`` class for each :mod:`array` typecode ROOT has a class for.
ARRAY_CLASSES = {
    "b": "TArrayC",
    "h": "TArrayS",
    "i": "TArrayI",
    "q": "TArrayL64",
    "f": "TArrayF",
    "d": "TArrayD",
}

#: The lists this writer will write - empty, because what a list holds could
#: be anything at all, and only nothing is nothing in every layout.
LISTS = ("TList", "THashList")

#: Members the layouts this writer carries have and an object read from an
#: older file lacks, with what ROOT gives a freshly made object: a histogram
#: from ROOT 6.08 keeps no ``fStatOverflows``, and ``kNeutral`` - follow the
#: global setting, which leaves the flow out of the statistics - is ROOT's.
DEFAULTS = {"fStatOverflows": 2}

#: The one STL member of any class this writer carries: the prior a
#: ``TEfficiency`` keeps bin by bin, a vector of pairs of doubles.
PAIRS = "vector<pair<double,double> >"
#: The record ROOT 6.24 writes that vector in: its version with the
#: member-wise bit set, because the firsts and the seconds go in two blocks,
#: and the checksum it gives ``pair<double,double>``, which has no version.
PAIRS_VERSION = MEMBER_WISE | 9
PAIR_CHECKSUM = 0xD7BED200

#: The version ROOT's collection proxy gives the record round an STL
#: container written inside an object.
STL_VERSION = 9

#: The objects this writer writes by their members, and the classes they are.
WRITABLE = (Histogram, Graph, Efficiency, Function, MultiGraph)
OBJECTS = (*HISTOGRAMS, *PROFILES, *EFFICIENCIES, *GRAPHS, *FUNCTIONS, "TMultiGraph")


def packed_now() -> int:
    """The moment, in ROOT's packed date word - the inverse of ``as_datetime``."""
    now = datetime.datetime.now()
    return (
        ((now.year - 1995) << 26)
        | (now.month << 22)
        | (now.day << 17)
        | (now.hour << 12)
        | (now.minute << 6)
        | now.second
    )


class WBuffer:
    """Bytes being written, with the record bookkeeping ROOT wraps them in.

    A record is a byte count that can only be known once the record is over,
    so :meth:`start` leaves a hole and :meth:`end` comes back to fill it -
    and :meth:`tag` does the same for an object that names its class first.
    """

    __slots__ = ("backrefs", "data", "origin", "_first")

    def __init__(self, origin: int = 0) -> None:
        self.data = bytearray()
        #: What a reference back to a place in here has added to it: the length of the key
        #: in front of the payload when this buffer is one - ROOT counts a reference from
        #: the start of the record, key and all, because it streams both into one buffer.
        self.origin = origin
        #: Whether a class named again is written as a reference back to the first spelling.
        #: ROOT writes it that way; this writer spells every name out unless a record says
        #: so, which means its class references can only be right when one buffer holds the
        #: whole record - the ``StreamerInfo`` list is the one.
        self.backrefs = False
        self._first: dict[str, int] = {}

    def raw(self, chunk: bytes) -> None:
        self.data += chunk

    def u8(self, value: int) -> None:
        self.data += struct.pack(">B", value)

    def u16(self, value: int) -> None:
        self.data += struct.pack(">H", value)

    def i16(self, value: int) -> None:
        self.data += struct.pack(">h", value)

    def u32(self, value: int) -> None:
        self.data += struct.pack(">I", value)

    def i32(self, value: int) -> None:
        self.data += struct.pack(">i", value)

    def i64(self, value: int) -> None:
        self.data += struct.pack(">q", value)

    def seek(self, value: int, wide: bool) -> None:
        """A place in the file: eight bytes in the wide layout, four otherwise."""
        if wide:
            self.i64(value)
        else:
            self.i32(value)

    def string(self, text: str) -> None:
        """A ``TString``: one length byte, or ``255`` and then four."""
        raw = text.encode("utf-8", "surrogateescape")
        if len(raw) >= 255:
            self.u8(255)
            self.i32(len(raw))
        else:
            self.u8(len(raw))
        self.data += raw

    def cstring(self, text: str) -> None:
        """A NUL-terminated name, which is how class names are written."""
        self.data += text.encode("utf-8", "surrogateescape") + b"\x00"

    def start(self, version: int) -> int:
        """Open a record of ``version``; :meth:`end` closes what this returns."""
        index = len(self.data)
        self.data += b"\x00\x00\x00\x00"
        self.u16(version)
        return index

    def tag(self, classname: str) -> int:
        """Open a counted object that names its class, the way ``any`` reads one.

        The class is spelled out every time rather than referred back to,
        which is longer and legal, and means no bookkeeping can be wrong -
        except where :attr:`backrefs` asks for ROOT's own shorter spelling.
        """
        index = len(self.data)
        self.data += b"\x00\x00\x00\x00"
        first = self._first.get(classname) if self.backrefs else None
        if first is None:
            self.u32(NEW_CLASS_TAG)
            self.cstring(classname)
            if self.backrefs:
                self._first[classname] = index
        else:
            self.u32(CLASS_MASK | (self.origin + first + 4 + MAP_OFFSET))
        return index

    def end(self, index: int) -> None:
        """Close a record or a tag by filling in how long it turned out to be."""
        count = (len(self.data) - index - 4) | BYTE_COUNT_MASK
        struct.pack_into(">I", self.data, index, count)

    def tobject(self, bits: int = BITS, unique: int = 0) -> None:
        """A ``TObject`` base: version, identifier and bits, no record round it.

        The referenced bit is cleared whatever came in: it promises two more
        bytes this writer does not write, and an object in a new file is not
        on anyone's reference list anyway.
        """
        self.u16(1)
        self.u32(unique)
        self.u32(bits & ~IS_REFERENCED)

    def named(self, name: str, title: str, bits: int = BITS) -> None:
        """A ``TNamed`` base: a record holding the object bits and two strings."""
        index = self.start(1)
        self.tobject(bits)
        self.string(name)
        self.string(title)
        self.end(index)


def _seekable(handle: IO[bytes]) -> bool:
    """Whether the header at the front can be written after everything else."""
    probe = getattr(handle, "seekable", None)
    return bool(probe()) if callable(probe) else False


class _Output:
    """Where a file's bytes go: straight out, except the header at the front.

    Everything a file holds is written once and never touched again, save the
    header and the file's own record in the first few hundred bytes, which say
    where the bookkeeping written last ended up. Those are kept here as
    :attr:`head` and written over the placeholder left for them at the close.
    Everything after them goes to the target as it is made, a few megabytes
    at a time.

    A target that cannot seek could not have its header filled in, so for one
    of those everything is kept and written in order at the end instead.
    Places in the file are counted from where the target stood when writing
    began, which is the start of the file for any target but a caller's own
    handle that already held something.

    The few other bytes written over after the fact - a directory's record,
    told at the close where its key list went - are kept by :meth:`patch`
    until :meth:`finish`, and written before the header, so that the header
    is the very last thing to change.
    """

    __slots__ = ("head", "size", "_handle", "_base", "_keep", "_streaming", "_pending", "_patches")

    def __init__(self, handle: IO[bytes], reserved: int) -> None:
        self._handle = handle
        self._streaming = _seekable(handle)
        self._base = handle.tell() if self._streaming else 0
        #: Where the target is cut back to if the file is abandoned.
        self._keep = self._base
        #: The bytes at the front, filled in last.
        self.head = bytearray(reserved)
        #: How long the file is so far, which is where the next record goes.
        self.size = reserved
        self._pending = bytearray(reserved) if self._streaming else bytearray()
        self._patches: list[tuple[int, bytes]] = []

    @classmethod
    def resume(cls, handle: IO[bytes], head: bytes, end: int) -> _Output:
        """Carry on at the end of a file that is already there, its ``head`` in hand.

        Nothing of what was there is written over until :meth:`finish`, and
        abandoning cuts the file back to ``end``: an update that fails leaves
        the file byte for byte as it found it.
        """
        out = cls.__new__(cls)
        out._handle = handle
        out._streaming = True
        out._base = 0
        out._keep = end
        out.head = bytearray(head)
        out.size = end
        out._pending = bytearray()
        out._patches = []
        handle.seek(end)
        return out

    def patch(self, at: int, chunk: bytes) -> None:
        """Write ``chunk`` over what is at ``at``, when the file is finished."""
        if at + len(chunk) <= len(self.head):
            self.head[at : at + len(chunk)] = chunk
        else:
            self._patches.append((at, bytes(chunk)))

    def append(self, chunk: bytes) -> None:
        """Put ``chunk`` at the end of the file, sending it once enough gathers."""
        self._pending += chunk
        self.size += len(chunk)
        if self._streaming and len(self._pending) >= WRITE_BEHIND:
            self._send()

    def _send(self) -> None:
        # Where the gathered bytes go is said every time, because a file
        # being updated is read from between writes, and reading moves it.
        self._handle.seek(self._base + self.size - len(self._pending))
        self._handle.write(self._pending)
        self._pending = bytearray()

    def finish(self) -> None:
        """Write what is still gathered, then the patches, then the header."""
        if not self._streaming:
            whole = self.head + self._pending
            for at, chunk in self._patches:
                whole[at : at + len(chunk)] = chunk
            self._handle.write(whole)
        else:
            self._send()
            for at, chunk in self._patches:
                self._handle.seek(self._base + at)
                self._handle.write(chunk)
            self._handle.seek(self._base)
            self._handle.write(self.head)
            self._handle.seek(self._base + self.size)
        self._pending = bytearray()
        if hasattr(self._handle, "flush"):
            self._handle.flush()

    def abandon(self) -> None:
        """Take back everything written, leaving the target as writing found it."""
        self._pending = bytearray()
        if self._streaming:
            self._handle.seek(self._keep)
            self._handle.truncate()


def _find(row: dict[str, Any], name: str) -> Any:
    """A counter's value, wherever the class or one of its bases keeps it."""
    if name in row:
        return row[name]
    for value in row.values():
        if isinstance(value, dict):
            try:
                return _find(value, name)
            except KeyError:
                continue
    raise KeyError(name)


def _numbers(buf: WBuffer, typecode: str, values: Any) -> None:
    """A run of values of one type, big-endian, as every ROOT number is."""
    buf.raw(np.asarray(values, dtype=np.dtype(typecode).newbyteorder(">")).tobytes())


def _array(buf: WBuffer, classname: str, values: Any) -> None:
    """A ``TArray``: its length and its values, with no record round them."""
    values = np.asarray(values if values is not None else ())
    buf.i32(len(values))
    _numbers(buf, ARRAYS[classname].typecode, values)


def _row(value: Any) -> Any:
    """The members an object is written from: a function of code sampled, as ROOT samples it."""
    if isinstance(value, Function):
        return value.written_members()
    return value.members


def _list(
    buf: WBuffer,
    classname: str,
    entries: Any,
    used: dict[str, None],
    bits: int = BITS,
    name: str = "",
) -> None:
    """A ``TList`` of what this writer writes - fits, a multigraph's graphs, a macro's
    ``TObjString``s - or of nothing.

    Each entry is the object naming its class, then the option it was added
    with, which is written as none. Anything else is refused: what a list
    carries could be anything at all, and writing it wrongly would be worse
    than refusing.
    """
    held = list(entries or ())
    strangers = [type(entry).__name__ for entry in held if not _streamable(entry)]
    if strangers:
        raise UnsupportedFeatureError(
            f"a {classname} holding {', '.join(strangers)} is not written: a list of "
            f"histograms, functions, graphs or ROOT's own named objects and strings is, and "
            f"what else a list carries could be anything at all, which writing wrongly would "
            f"be worse than refusing; take them out first"
        )
    index = buf.start(5)
    buf.tobject(bits)
    buf.string(name)
    buf.i32(len(held))
    for entry in held:
        _listed(buf, entry, used)
        buf.string("")
    buf.end(index)


def _streamable(entry: Any) -> bool:
    """Whether a list may hold ``entry``: an object this writer writes, or one whose
    class it carries a layout for - a drawn histogram's ``TPaveStats`` and its lines."""
    if isinstance(entry, (*WRITABLE, Record)):
        return True
    return getattr(entry, "classname", None) in INFOS and isinstance(
        getattr(entry, "members", None), dict
    )


def _listed(buf: WBuffer, entry: Any, used: dict[str, None]) -> None:
    """One object a list holds, naming its class: a bare ``TObject`` has no byte count."""
    tag = buf.tag(entry.classname)
    hand = BY_HAND.get(entry.classname)
    if hand is not None:
        used.update(dict.fromkeys(hand(buf, _row(entry))))
    elif entry.classname == "TObject":
        used["TObject"] = None
        _write_tobject(buf, _row(entry))
    else:
        _record(buf, entry.classname, _row(entry), used)
    buf.end(tag)


def _record(buf: WBuffer, classname: str, row: Any, used: dict[str, None]) -> None:
    """One object written in place: its record, then its members in order.

    Every class actually written lands in ``used``, so an object hung behind
    a pointer - a graph's fitted histogram, say - gets described by the
    file's streamer information exactly as a key of the file would be.
    """
    if classname in LISTS:
        _list(buf, classname, row, used)
        return
    info = INFOS.get(classname)
    if info is None:
        raise UnsupportedFeatureError(
            f"a {classname} inside what is being written is not a class this "
            f"writer carries a layout for"
        )
    if not isinstance(row, dict):
        raise UnsupportedFeatureError(
            f"a {classname} here is a {type(row).__name__} rather than the dict "
            f"of members writing one starts from"
        )
    used[classname] = None
    index = buf.start(info[1])
    for kind, element in _steps_of(classname, info):
        _written_element(buf, kind, element, row, used)
    buf.end(index)


def _pointed(buf: WBuffer, typename: str, value: Any, used: dict[str, None]) -> None:
    """A pointer that may be null: nothing, or an object naming its class."""
    if value is None or (isinstance(value, (list, tuple)) and not value):
        buf.u32(0)
        return
    classname = typename.rstrip("*")
    if isinstance(value, WRITABLE):
        classname, value = value.classname, _row(value)
    index = buf.tag(classname)
    _record(buf, classname, value, used)
    buf.end(index)


#: The streamer types one element kind covers: the bases and the string, the objects and
#: pointers, the numbers, and STL's own.
BASE_STYPES = frozenset({0, 65, 66, 67})
OBJECT_STYPES = frozenset({61, 62, 63, 64, 68, 69})
NUMBER_STYPES = frozenset(
    set(FORMS) | {OFFSET_L + kind for kind in FORMS} | {OFFSET_P + kind for kind in FORMS}
)
STL_STYPE = 500

#: Which write each of a class's elements takes, by class name: worked out once, because a
#: class's elements are the same every time it is written and a file of ten thousand
#: histograms writes each class's elements ten thousand times over. ``""`` is a member this
#: writer does not lay out, which writing one refuses by name.
_STEPS: dict[str, tuple[tuple[str, Element], ...]] = {}


def _steps_of(classname: str, info: Any) -> tuple[tuple[str, Element], ...]:
    """Each of a class's elements with the kind of write it takes, in the order they come."""
    found = _STEPS.get(classname)
    if found is None:
        found = _STEPS[classname] = tuple(
            (_kind_of(element), element) for element in info[2]
        )
    return found


def _kind_of(element: Element) -> str:
    """The write one element takes: a base, a histogram's locked list, an object, a number,
    an STL member - or nothing this writer lays out."""
    stype, name, typename = element[3], element[1], element[8]
    if stype in BASE_STYPES:
        return "base"
    if name == "fFunctions":
        return "list"  # a histogram's, which ROOT locks - or an object when it is not one
    if stype in OBJECT_STYPES:
        return "object"
    if stype in NUMBER_STYPES:
        return "number"
    if _an_stl_member(stype, typename):
        return "stl"
    return ""


def _an_stl_member(stype: int, typename: str) -> bool:
    """Is this element an STL member of a kind this writer lays out?"""
    if stype != STL_STYPE:
        return False
    return typename == PAIRS or typename in STL_WRITERS


def _element(buf: WBuffer, element: Element, row: dict[str, Any], used: dict[str, None]) -> None:
    """One member of one class, laid out exactly as its element describes - decided afresh,
    which is what writing one member of a class by hand does."""
    _written_element(buf, _kind_of(element), element, row, used)


def _written_element(
    buf: WBuffer, kind: str, element: Element, row: dict[str, Any], used: dict[str, None]
) -> None:
    """One member of one class, written by the write its kind was decided to take."""
    name, stype, alen, typename, extras = element[1], element[3], element[5], element[8], element[9]
    value = row.get(name, DEFAULTS.get(name))
    if kind == "base":
        _base_element(buf, name, stype, value, used, row)
    elif kind == "list" and "fNcells" in row:  # a histogram's fFunctions, which ROOT locks
        _list(buf, "TList", value, used, BITS | USE_RWLOCK)
    elif kind == "object" or kind == "list":
        _object_element(buf, stype, typename, value, used)
    elif kind == "number":
        _number_element(buf, name, stype, alen, extras, value, row)
    elif kind == "stl":
        _stl_element(buf, name, typename, value)
    else:
        raise UnsupportedFeatureError(
            f"{name} is of streamer type {stype}, which this writer does not lay out"
        )


def _stl_element(buf: WBuffer, name: str, typename: str, value: Any) -> None:
    """An STL member: the prior a ``TEfficiency`` keeps, or a function's vectors and map."""
    if typename == PAIRS:
        _pairs(buf, value)
        return
    index = buf.start(STL_VERSION)
    STL_WRITERS[typename](buf, name, value)  # which kind it is was decided when it was planned
    buf.end(index)


def _base_element(
    buf: WBuffer,
    name: str,
    stype: int,
    value: Any,
    used: dict[str, None],
    row: dict[str, Any],
) -> None:
    if stype in (0, 66, 67):
        value = _flat_base(row, name, value)
    if stype == 66:  # the TObject base, which streams itself
        _write_tobject(buf, value)
    elif stype == 67:  # the TNamed base, likewise
        _write_tnamed(buf, value)
    elif stype == 0:  # any other base, written under its own name
        _write_base(buf, name, value, used)
    else:  # stype 65: a TString member, its bytes and no record - the only kind left
        buf.string(str(value if value is not None else ""))


def _flat_base(row: dict[str, Any], name: str, value: Any) -> Any:
    """A base class's members: its own row, or the object's when it keeps them flat.

    The graphics layer's paves are one dict of every member they inherit -
    ``fX1NDC`` beside ``fBorderSize`` beside ``fTextFont`` - rather than the
    row per base class a file holds, so a base with none of its own is the
    object itself; a base whose members are nowhere is written empty, which
    every layout this writer carries allows for an interface class.
    """
    if value is not None:
        return value
    return row if any(element[1] in row for element in INFOS[name][2]) else {}


def _pairs(buf: WBuffer, value: Any) -> None:
    """A vector of pairs of doubles, the way ROOT writes one inside an object.

    Member-wise, as every container of pairs is: the record, the checksum
    standing in for the pair's missing version, the count, and then every
    first followed by every second.
    """
    pairs = np.asarray(value if value is not None else (), dtype=np.float64).reshape(-1, 2)
    index = buf.start(PAIRS_VERSION)
    buf.i16(0)
    buf.u32(PAIR_CHECKSUM)
    buf.u32(len(pairs))
    _numbers(buf, "d", pairs[:, 0])
    _numbers(buf, "d", pairs[:, 1])
    buf.end(index)


def _doubles(buf: WBuffer, name: str, value: Any) -> None:
    """A ``vector<double>``: how many, then each."""
    values = np.asarray(value if value is not None else (), dtype=np.float64).ravel()
    buf.u32(len(values))
    _numbers(buf, "d", values)


def _strings(buf: WBuffer, name: str, value: Any) -> None:
    """A ``vector<string>``: how many, then each as a length and its bytes."""
    values = list(value or ())
    buf.u32(len(values))
    for text in values:
        buf.string(str(text))


def _parameter_names(buf: WBuffer, name: str, value: Any) -> None:
    """A ``TFormula``'s ``map<TString,int,TFormulaParamOrder>``, pair by pair.

    The pairs go in the order the members hold them, which a formula made
    here keeps in ``TFormulaParamOrder`` and one read from a file keeps as
    ROOT wrote it.
    """
    pairs = dict(value or {})
    buf.u32(len(pairs))
    for key, index in pairs.items():
        buf.string(str(key))
        buf.i32(int(index))


def _no_objects(buf: WBuffer, name: str, value: Any) -> None:
    """A ``vector<TObject*>`` - a formula's linear parts - which is written only empty."""
    if value:
        raise UnsupportedFeatureError(
            f"{name} holds {len(value)} objects, the parts of a linear fit, which this "
            f"writer does not lay out; a formula made or read here has none"
        )
    buf.u32(0)


#: The STL members of the function classes, by the type the layout names.
STL_WRITERS: dict[str, Callable[[WBuffer, str, Any], None]] = {
    "vector<double>": _doubles,
    "vector<string>": _strings,
    "map<TString,int,TFormulaParamOrder>": _parameter_names,
    "vector<TObject*>": _no_objects,
}


def _write_tobject(buf: WBuffer, value: Any) -> None:
    """``TObject::Streamer``: and for an object a ``TRef`` may point at - ``kIsReferenced``,
    with ``pidf`` saying which of the file's ``ProcessID`` records numbered it - its number
    within that process, then the record's index."""
    bits = value if isinstance(value, dict) else {}
    word, unique = int(bits.get("fBits", BITS)), int(bits.get("fUniqueID", 0))
    process = bits.get("pidf")
    if process is None or not word & IS_REFERENCED:
        buf.tobject(word, unique)
        return
    buf.u16(1)
    buf.u32(unique & NUMBER_MASK)
    buf.u32(word)
    buf.u16(int(process))


def _write_tnamed(buf: WBuffer, value: Any) -> None:
    named = value if isinstance(value, dict) else {}
    # The bits go out as they came in: a TEfficiency keeps its settings there.
    bits = int(named.get("fBits", BITS))
    if named.get("pidf") is None:
        buf.named(str(named.get("fName", "")), str(named.get("fTitle", "")), bits)
        return
    index = buf.start(1)
    _write_tobject(buf, named)
    buf.string(str(named.get("fName", "")))
    buf.string(str(named.get("fTitle", "")))
    buf.end(index)


def _write_base(buf: WBuffer, name: str, value: Any, used: dict[str, None]) -> None:
    if name in ARRAYS:
        _array(buf, name, value)
    else:
        _record(buf, name, value if value is not None else {}, used)


def _object_element(
    buf: WBuffer, stype: int, typename: str, value: Any, used: dict[str, None]
) -> None:
    if stype in (61, 62):  # an object held by value
        if typename in ARRAYS:
            _array(buf, typename, value)
        else:
            _record(buf, typename, value if value is not None else {}, used)
    elif stype in (63, 68):  # a pointer promised never null, written in place
        _record(buf, typename.rstrip("*"), value if value is not None else [], used)
    else:  # 64, 69: a pointer that may be null - the only kind left
        _pointed(buf, typename, value, used)


def _number_element(
    buf: WBuffer,
    name: str,
    stype: int,
    alen: int,
    extras: tuple[Any, ...],
    value: Any,
    row: dict[str, Any],
) -> None:
    if stype in FORMS:  # one number
        buf.raw(struct.pack(">" + FORMS[stype], value))
    elif stype - OFFSET_L in FORMS:  # a fixed-size array, x[10]
        values = list(value if value is not None else ())
        if len(values) != alen:
            raise ValueError(f"{name} holds {len(values)} values where {alen} were declared")
        _numbers(buf, FORMS[stype - OFFSET_L], values)
    else:  # a counted array, x[n], behind its marker - the only kind left
        _counted(buf, name, FORMS[stype - OFFSET_P], str(extras[1]), value, row)


def _counted(
    buf: WBuffer, name: str, form: str, counter: str, value: Any, row: dict[str, Any]
) -> None:
    """A counted array, ``x[n]``: the marker a pointer carries, then ``n`` values.

    ROOT marks a null pointer with a zero, and the pointer of an array of
    nothing is null, so an empty one is marked that way too.
    """
    try:
        count = int(_find(row, counter))
    except KeyError:
        raise ValueError(f"{name} is counted by {counter}, which is not here") from None
    values = list(value if value is not None else ())
    if len(values) < count:
        raise ValueError(f"{name} holds {len(values)} values where {counter} says {count}")
    buf.u8(1 if count else 0)
    _numbers(buf, form, values[:count])


def _held(kind: str, name: str, stype: int, typename: str) -> str | None:
    """The class one element brings in with it: a base, an object, a string, or none."""
    if kind == "TStreamerBase":
        return name
    if stype in (61, 62, 63, 64, 68, 69):
        return typename.rstrip("*")
    if stype == 65:
        return "TString"
    return None


def _closure(seeds: dict[str, None]) -> list[str]:
    """Every class the seeds reach through bases and members, in met order."""
    ordered = dict.fromkeys(seeds)
    queue = list(ordered)
    while queue:
        info = INFOS.get(queue.pop(0))
        if info is None:
            continue
        for kind, name, _t, stype, _s, _a, _d, _m, typename, _x in info[2]:
            held = _held(kind, name, stype, typename)
            if held is not None and held in INFOS and held not in ordered:
                ordered[held] = None
                queue.append(held)
    return list(ordered)


def _streamers(
    used: dict[str, None],
    carried: Mapping[tuple[str, int], bytes] | None = None,
    keylen: int = 0,
) -> bytes:
    """The ``StreamerInfo`` record: what this file says its classes look like.

    Emitted from :data:`~.winfo.INFOS` verbatim, checksums and all, so the
    file describes its classes exactly as the ROOT that the descriptions were
    harvested from would have - and then whatever descriptions ``carried``
    brings from other files, for records copied from them as they were.

    ``keylen`` is how long the key in front of this record turns out, which is
    what the references back to a class named a second time are counted from.
    """
    names = _closure(used)
    extra = carried_entries(names, carried)
    buf = WBuffer(keylen)
    buf.backrefs = True
    index = buf.start(5)
    buf.tobject()
    buf.string("")
    buf.i32(len(names) + len(extra))
    for name in names:
        _entry_into(buf, name, *INFOS[name])
    buf.raw(b"".join(extra))
    buf.end(index)
    return bytes(buf.data)


def carried_entries(
    names: list[str], carried: Mapping[tuple[str, int], bytes] | None
) -> list[bytes]:
    """The carried descriptions of every class and version ``names`` leave out."""
    known = {(name, INFOS[name][1]) for name in names}
    return [entry for key, entry in (carried or {}).items() if key not in known]


def _info_entries(names: list[str]) -> bytes:
    """One ``TStreamerInfo`` list entry per class, each naming its classes in
    full rather than referring back, so the bytes mean the same wherever in
    a list they land - which is what lets an update add them to a list that
    is already there."""
    buf = WBuffer()
    for name in names:
        _entry_into(buf, name, *INFOS[name])
    return bytes(buf.data)


def info_entry(name: str, checksum: int, version: int, elements: tuple[Element, ...]) -> bytes:
    """One class's ``TStreamerInfo`` list entry, its elements in full, its option after it.

    Every class it names is spelled out: an entry on its own cannot know where
    in the list it will land, so a reference back would point nowhere.
    """
    buf = WBuffer()
    _entry_into(buf, name, checksum, version, elements)
    return bytes(buf.data)


def _entry_into(
    buf: WBuffer, name: str, checksum: int, version: int, elements: tuple[Element, ...]
) -> None:
    """One list entry written into ``buf``, which may be holding the entries before it."""
    tag = buf.tag("TStreamerInfo")
    info = buf.start(9)
    buf.named(name, "")
    buf.u32(checksum)
    buf.i32(version)
    held = buf.tag("TObjArray")
    arr = buf.start(3)
    buf.tobject()
    buf.string("")
    buf.i32(len(elements))
    buf.i32(0)
    for element in elements:
        _info_element(buf, element)
    buf.end(arr)
    buf.end(held)
    buf.end(info)
    buf.end(tag)
    buf.u8(0)  # the option string every list entry carries, empty


#: ``kSTL``, a container member's streamer type in memory, and ``kStreamer``, what
#: ``TStreamerSTL`` writes in its place in a file, as ROOT has since before version 6.
STL, STREAMER = 300, 500
#: ``TStreamerElement::kDoNotDelete``, and the containers of pairs it is set on.
DO_NOT_DELETE = 1 << 13
MAP_KINDS = (4, 5, 12, 13)

#: ``TStreamerElement::kHasRange``: the element's title holds a packing range, which ROOT
#: reads the title for only when this bit says to - without it the range is not there.
HAS_RANGE = 1 << 6
#: The streamer types a range packs: ``Double32_t`` and ``Float16_t``.
PACKED_TYPES = (9, 19)


def _ranged(stype: int, title: str) -> bool:
    """Is this a packed float whose title spells out a range or a bit count?"""
    from .interp import _range_parts

    return stype in PACKED_TYPES and len(_range_parts(title)) > 1


def _element_bits(kind: str, stype: int, title: str, extras: tuple[object, ...]) -> int:
    """The bits an element carries: a packed float's range, and ``kDoNotDelete`` on a map and
    a container of pointers, whose items reading does not hand to the container to delete."""
    bits = HAS_RANGE if _ranged(stype, title) else 0
    if kind == "TStreamerSTL" and stype == STL and (extras[0] in MAP_KINDS or extras[1] == 63):
        bits |= DO_NOT_DELETE
    return bits


def _info_element(buf: WBuffer, element: Element) -> None:
    """One ``TStreamerElement``, written the way the donor files write them."""
    kind = element[0]
    tag = buf.tag(kind)
    if kind == "TStreamerSTLstring":  # a TStreamerSTL with a lid on it, one record deeper
        lid = buf.start(SUBVERSIONS[kind])
        _element_record(buf, element, "TStreamerSTL")
        buf.end(lid)
    else:
        _element_record(buf, element, kind)
    buf.end(tag)


def _element_record(buf: WBuffer, element: Element, kind: str) -> None:
    """The record of one element, as the subclass ``kind`` writes it."""
    _kind, name, title, stype, size, alen, adim, maxidx, typename, extras = element
    sub = buf.start(SUBVERSIONS[kind])
    base = buf.start(4)  # the TStreamerElement the subclass builds on
    buf.named(name, title, BITS | _element_bits(kind, stype, title, extras))
    buf.i32(STREAMER if kind == "TStreamerSTL" and stype == STL else stype)
    buf.i32(size)
    buf.i32(alen)
    buf.i32(adim)
    for index in maxidx:
        buf.i32(index)
    buf.string(typename)
    buf.end(base)
    if kind == "TStreamerBase":
        buf.i32(int(str(extras[0])))
    elif kind in ("TStreamerBasicPointer", "TStreamerLoop"):
        buf.i32(int(str(extras[0])))
        buf.string(str(extras[1]))
        buf.string(str(extras[2]))
    elif kind == "TStreamerSTL":
        buf.i32(int(str(extras[0])))
        buf.i32(int(str(extras[1])))
    buf.end(sub)


class ObjString(str):
    """A string written as ROOT's ``TObjString`` rather than as a ``std::string``.

    A plain ``str`` is written as the ``std::string`` a Python user most
    likely means; a key ROOT itself writes as a ``TObjString`` - TMVA's
    ``TrainingPath``, say - is written as one by wrapping it in this.
    """


class Record:
    """An object of a class this writer carries a layout for, written from its members.

    The classes ROOT's core streams for itself - a ``TObject``, a ``TNamed``,
    a ``TObjString``, and a ``TList`` or ``THashList`` holding any of these -
    are no xrdroot object of their own: whatever makes one hands over its
    class and its members, as a reader would have read them back, and they
    are laid out by the class's streamer information. A list's members are
    the objects it holds, in order, and ``name`` is a list's own name.
    """

    __slots__ = ("classname", "members", "title", "name")

    def __init__(self, classname: str, members: Any, title: str = "", name: str = "") -> None:
        known = classname in INFOS or classname in BY_HAND
        if not known or classname in UNRECORDED:
            raise UnsupportedFeatureError(
                f"a {classname} is not a class this writer carries a layout for; it writes "
                f"a TObject, a TNamed, a TObjString, a TList of them, a TRef, a TRefArray, "
                f"a TBits and a TProcessID from their members"
            )
        self.classname = classname
        self.members = members
        self.title = title
        self.name = name

    def __repr__(self) -> str:
        return f"<Record {self.classname} {self.title!r}>"


#: Classes described here that are parts of others rather than objects written whole.
UNRECORDED = frozenset({"TCollection", "TSeqCollection", "TString"})


def _tref(buf: WBuffer, members: dict[str, Any]) -> tuple[str, ...]:
    """``TRef::Streamer``: its ``TObject``, no byte count, then the index of its process."""
    _write_tobject(buf, members.get("TObject", {}))
    buf.u16(int(members.get("pidf", 0)))
    return ()


def _trefarray(buf: WBuffer, members: dict[str, Any]) -> tuple[str, ...]:
    """``TRefArray::Streamer``: its ``TObject`` and name, how many, the lower bound, the
    index of its process and each object's identifier."""
    uids = [int(uid) for uid in members.get("fUIDs", ())]
    index = buf.start(1)
    _write_tobject(buf, members.get("TObject", {}))
    buf.string(str(members.get("fName", "")))
    buf.i32(len(uids))
    buf.i32(int(members.get("fLowerBound", 0)))
    buf.u16(int(members.get("pidf", 0)))
    for uid in uids:
        buf.u32(uid)
    buf.end(index)
    return ()


def _tprocessid(buf: WBuffer, members: dict[str, Any]) -> tuple[str, ...]:
    """A ``TProcessID``: a ``TNamed`` - ``ProcessID0`` and on, titled with the UUID."""
    index = buf.start(1)
    _write_tnamed(buf, {"fBits": 0, **members})
    buf.end(index)
    return ("TNamed",)


def _tbits(buf: WBuffer, members: dict[str, Any]) -> tuple[str, ...]:
    """A ``TBits``: its ``TObject``, ``fNbits``, ``fNbytes`` and that many bytes."""
    raw = bytes(members.get("fAllBits", b""))
    index = buf.start(1)
    _write_tobject(buf, members.get("TObject", {}))
    buf.u32(int(members.get("fNbits", 8 * len(raw))))
    buf.u32(len(raw))
    buf.u8(1)  # the bytes are there: a pointer to an array, not a null one
    buf.raw(raw)
    buf.end(index)
    return ("TBits",)


#: Classes ROOT streams by hand rather than by their description, and how each is laid out;
#: what comes back is the described classes the record used.
BY_HAND: dict[str, Callable[[WBuffer, dict[str, Any]], tuple[str, ...]]] = {
    "TRef": _tref,
    "TRefArray": _trefarray,
    "TProcessID": _tprocessid,
    "TBits": _tbits,
}


def _record_payload(value: Record) -> tuple[str, bytes, tuple[str, ...]]:
    buf = WBuffer()
    used: dict[str, None] = {value.classname: None}
    hand = BY_HAND.get(value.classname)
    if hand is not None:
        described = hand(buf, value.members)
        return value.classname, bytes(buf.data), described
    if value.classname == "TObject":  # TObject::Streamer: no byte count round it
        _write_tobject(buf, value.members)
    elif value.classname in LISTS:
        _list(buf, value.classname, value.members, used, name=value.name)
    else:
        _record(buf, value.classname, value.members, used)
    return value.classname, bytes(buf.data), tuple(used)


def _objstring_payload(value: ObjString) -> tuple[str, bytes, tuple[str, ...]]:
    buf = WBuffer()
    used: dict[str, None] = {}
    _record(buf, "TObjString", {"fString": str(value)}, used)
    return "TObjString", bytes(buf.data), tuple(used)


def _payload(obj: Any) -> tuple[str, bytes, tuple[str, ...]]:
    """What one object writes as: its class, its bytes, the layouts it needs."""
    if isinstance(obj, ObjString):
        return _objstring_payload(obj)
    if isinstance(obj, Record):
        return _record_payload(obj)
    if isinstance(obj, str):
        return _string_payload(obj)
    if isinstance(obj, (array.array, np.ndarray)):
        return _array_payload(obj)
    if isinstance(obj, WRITABLE):
        return _object_payload(obj)
    if Histogram.recognises(obj):
        return _object_payload(Histogram.of(obj))
    raise UnsupportedFeatureError(
        f"a {type(obj).__name__} is not something this writer puts in a ROOT "
        f"file: it takes a Histogram, a Profile, an Efficiency, a Graph, a MultiGraph, "
        f"a Function, a Record of a TNamed, TObjString or TList, any "
        f"histogram that speaks the "
        f"plotting protocol (hist, boost-histogram), a (values, edges) pair "
        f"from numpy.histogram, a str, or a one-dimensional array of numbers"
    )


def _table(obj: Any) -> dict[str, Any] | None:
    """The columns of ``obj``, if it is a table: a dict of arrays, or a frame.

    A mapping of column name to values is one; so is a pandas or Polars
    DataFrame, which says its column names in ``columns``, and an Arrow table,
    which says them in ``column_names``.
    """
    if isinstance(obj, Mapping):
        return dict(obj)
    names = getattr(obj, "column_names", None)
    if names is None and not isinstance(obj, np.ndarray):
        names = getattr(obj, "columns", None)
    if names is None:
        return None
    return {str(name): np.asarray(obj[name]) for name in names}


def _string_payload(value: str) -> tuple[str, bytes, tuple[str, ...]]:
    buf = WBuffer()
    buf.string(value)
    return "string", bytes(buf.data), ()


def _array_payload(value: Any) -> tuple[str, bytes, tuple[str, ...]]:
    code = value.typecode if isinstance(value, array.array) else value.dtype.char
    if code == "l":  # its width is the platform's, so pick the class by it
        code = "q" if value.itemsize == 8 else "i"
    classname = ARRAY_CLASSES.get(code)
    if classname is None or np.ndim(value) != 1:
        raise UnsupportedFeatureError(
            f"an array of typecode {code!r} and {np.ndim(value)} dimensions has no ROOT "
            f"class: the TArrays are one-dimensional runs of signed integers b, h, i, "
            f"l and q, and floats f and d"
        )
    buf = WBuffer()
    buf.i32(len(value))
    _numbers(buf, ARRAYS[classname].typecode, value)
    return classname, bytes(buf.data), ()


def _object_payload(
    value: Histogram | Graph | Efficiency | Function | MultiGraph,
) -> tuple[str, bytes, tuple[str, ...]]:
    classname = value.classname
    if classname not in INFOS:
        writable = ", ".join(name for name in INFOS if name in OBJECTS)
        raise UnsupportedFeatureError(
            f"a {classname} is not a class this writer carries a layout for; "
            f"the ones it does are {writable}"
        )
    buf = WBuffer()
    used: dict[str, None] = {}
    _record(buf, classname, _row(value), used)
    return classname, bytes(buf.data), tuple(used)


#: An object's bytes for whatever streams one inside its own record, as a measurement does.
object_payload = _object_payload


def _keylen(classname: str, name: str, title: str, extra: int = 0) -> int:
    """How long the small form of the key in front of a record is: 26 fixed
    bytes, then three strings - and whatever else the class writes into its
    own key, which only a ``TBasket`` does. The wide form is :data:`WIDER`
    bytes longer."""
    return (
        26
        + extra
        + sum(1 + len(text.encode("utf-8", "surrogateescape")) for text in (classname, name, title))
    )


def _wide(*seeks: int) -> bool:
    """Whether a record pointing at these places needs ROOT's wide layout.

    ROOT decides record by record, and so does this: a key stays small until
    it is written past :data:`BIG`, or belongs to a directory that was -
    which is what ``TKey`` does once the file's end passes ``kStartBigFile``.
    A file that grows past 2 GB is then small keys at the front and wide ones
    after, exactly as ROOT's own big files are. Making every key wide from
    the start would be just as legal, and would cost eight bytes a key in
    the files that never get there, which is nearly all of them.
    """
    return max(seeks) > BIG


def _key_fields(
    buf: WBuffer,
    seek: int,
    pdir: int,
    sizes: tuple[int, int, int],
    cycle: int,
    datime: int,
    wide: bool = False,
) -> None:
    """The fixed part of a key, small or wide as the places it holds decide.

    ``sizes`` are the record's length on file, the object's length, and the
    key's own length, in the order the key holds them. ``wide`` asks for the
    wide layout wherever the key lands, which is how ROOT writes every basket
    and so what a basket copied as it was needs to keep its length.
    """
    wide = wide or _wide(seek, pdir)
    nbytes, objlen, keylen = sizes
    buf.i32(nbytes)
    buf.u16(KEY_VERSION + (WIDE if wide else 0))
    buf.i32(objlen)
    buf.u32(datime)
    buf.i16(keylen)
    buf.i16(cycle)
    buf.seek(seek, wide)
    buf.seek(pdir, wide)


def _tail(end: int) -> int:
    """Where the free space at the end of a file is said to stop.

    ROOT starts it at :data:`BIG` and moves it on :data:`GROW` bytes at a
    time as the file passes it, so this is the number ROOT would have written.
    """
    if end <= BIG:
        return BIG
    return BIG + GROW * -(-(end - BIG) // GROW)


def _free_entries(segments: list[tuple[int, int]]) -> bytes:
    """Free-list entries: each gap's first and last byte, wide past :data:`BIG`."""
    buf = WBuffer()
    for first, last in segments:
        wide = last > BIG
        buf.u16(FREE_VERSION + (WIDE if wide else 0))
        buf.seek(first, wide)
        buf.seek(last, wide)
    return bytes(buf.data)


def _merged(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Gaps in order, neighbours and overlaps joined, as ROOT keeps its free list."""
    joined: list[tuple[int, int]] = []
    for first, last in sorted(segments):
        if joined and first <= joined[-1][1] + 1:
            joined[-1] = (joined[-1][0], max(joined[-1][1], last))
        else:
            joined.append((first, last))
    return joined


def _checked(text: str, what: str) -> str:
    """A name or title a key can carry: short enough for its one length byte."""
    if not isinstance(text, str):
        raise ValueError(f"the {what} must be a str, not {type(text).__name__}")
    if len(text.encode("utf-8", "surrogateescape")) > 254:
        raise ValueError(f"the {what} {text[:40]!r}... is too long for a key to carry")
    return text


class WritableDirectory:
    """A directory in a ROOT file being written: a mapping from name to object.

        >>> with xrdroot.create("out.root") as f:            # doctest: +SKIP
        ...     run = f.mkdir("calibration/run4711")
        ...     run["gains"] = gains
        ...     f["calibration/run4711/offsets"] = offsets   # the same place

    A directory is what ROOT makes one: a ``TDirectory`` record with a key of
    its own in the directory above, and a key list of its own written when
    the file closes. The record goes out the moment the directory is made,
    because every key inside it says where it is, and is filled in at the
    close with where the key list landed - the order ROOT does it in.

    A path with a ``/`` in it names a place in a directory below this one,
    and whatever directories it passes through are made if they are not
    there, so ``f["a/b/h"] = h`` needs no ``mkdir`` first.
    """

    def __init__(
        self,
        file: WritableFile,
        path: str,
        title: str,
        seek: int,
        parent_seek: int,
        nname: int,
    ) -> None:
        self._file = file
        #: Where this is, counted from the top of the file: ``"calibration/run4711"``.
        self.path = path
        #: What this directory is called, and what it says it is.
        self.name = path.rpartition("/")[2]
        self.title = title
        #: Where its record's key is, where its parent's is, and how many
        #: bytes come before the record itself: what the record says of both.
        self._seek = seek
        self._parent_seek = parent_seek
        self._nname = nname
        self._ctime = file._datime
        self._uuid = uuid.uuid4().bytes
        #: The class, name and title the key in front of its key list carries.
        self._listed_as = ("TDirectory", self.name, title)
        self._keys: list[bytes] = []
        self._cycles: dict[str, int] = {}
        self._subdirs: dict[str, WritableDirectory] = {}
        #: Directories already on file, opened only when something goes in.
        self._found: dict[str, Callable[[], WritableDirectory]] = {}
        #: Whether its key list has to be written: anything new has one.
        self._dirty = True
        #: The key list this one's replaces, freed once the new one is out.
        self._replaced: tuple[int, int] | None = None

    def __repr__(self) -> str:
        state = "closed" if self.closed else f"{len(self._keys)} keys so far"
        return f"<WritableDirectory {self.path!r} in {self._file.name!r}, {state}>"

    def write(
        self, name: str, obj: Any, *, title: str | None = None, rntuple: bool = False
    ) -> None:
        """Write one object under ``name``, as :meth:`__setitem__` does.

            >>> f["events"] = {"energy": energies, "hits": hits}   # doctest: +SKIP
            >>> f["spectrum"] = hist.Hist(...)                      # doctest: +SKIP
            >>> f["runs/4711/spectrum"] = hist.Hist(...)            # doctest: +SKIP

        A table - a dict of arrays, a pandas or Polars DataFrame, an Arrow
        table - becomes a tree, a column per column, typed by what the arrays
        hold: numbers, rows of numbers of different lengths (a
        :class:`~.tree.Jagged`, an Awkward Array, a list of arrays), or
        strings. A histogram from ``hist``, ``boost-histogram`` or
        :func:`numpy.histogram` becomes the ROOT histogram it is. The title is
        taken from the object when it has one; a name written twice becomes a
        second cycle of itself, exactly as in ROOT, and reading the file back
        gives the newest. A name with a ``/`` in it goes into the directory
        it names, made if it is not there yet. ``rntuple=True`` writes a table
        as an RNTuple instead of a tree, a field per column: see :meth:`rntuple`.
        """
        here, leaf = self._place(name)
        here._write(leaf, obj, title, rntuple)

    def process_index(self, uuid: str) -> int:
        """Which of the file's ``ProcessID`` records the process of this UUID is - written now,
        at the top of the file, if the file has none for it yet: ``TFile::WriteProcessID``.

        An object a ``TRef`` may point at is written with this index, and so is
        every ``TRef`` and ``TRefArray``, which is how a reader finds the object
        a reference names among those its file holds.
        """
        top = self._file
        known = top._processes()
        if uuid not in known:
            index = len(known)
            known[uuid] = index
            name = f"ProcessID{index}"
            top._write(name, Record("TProcessID", {"fName": name, "fTitle": uuid}, uuid), uuid)
        return known[uuid]

    def _write(self, name: str, obj: Any, title: str | None, rntuple: bool = False) -> None:
        self._check_leaf(name)
        table = _table(obj)
        if rntuple:  # ROOT 7's columnar format rather than a TTree: see .rntuple
            from .rntuple.writer import write_table

            write_table(self, name, table, obj)
            return
        if table is not None:
            from .wtree import spec_of

            columns = {column: spec_of(column, values) for column, values in table.items()}
            self.tree(name, columns, title=title).extend(table)
            return
        if title is None:
            titled = isinstance(obj, (*WRITABLE, Record)) or hasattr(obj, "root_payload")
            title = obj.title if titled else ""
        _checked(title, "title")
        if hasattr(obj, "root_payload"):  # streams itself, knowing where its key's bytes end
            classname = str(obj.classname)
            payload, used, carried = obj.root_payload(_keylen(classname, name, title))
            for key, entry in carried.items():
                self._file._carried.setdefault(key, entry)
        else:
            classname, payload, used = _payload(obj)
        self._file._used.update(dict.fromkeys(used))
        self._put(classname, name, title, payload, self._next_cycle(name), listed=True)

    def __setitem__(self, name: str, obj: Any) -> None:
        self.write(name, obj)

    def tree(
        self,
        name: str,
        columns: Mapping[str, Any],
        *,
        title: str | None = None,
        basket_size: int = BASKET_BYTES,
        counters: Mapping[str, Any] | None = None,
        classname: str = "TTree",
        empty: bool = False,
    ) -> WritableTree:
        """A tree in this directory, to be filled entry by entry.

            >>> with xrdroot.create("out.root") as f:        # doctest: +SKIP
            ...     tree = f.tree("events", {"energy": float, "hits": ("i", 4)})
            ...     tree.fill(energy=12.5, hits=[3, 1, 4, 1])

        ``columns`` maps each column's name to what it holds: a Python
        ``bool``, ``int`` or ``float``, an :mod:`array` type code such as
        ``'f'`` for a narrower number, or a pair of either and how many
        values every entry holds - ``None`` there for a number that changes
        from entry to entry, or a counter's name to share one - or ``str``
        for a column of text. Entries go out a basket at a time as they
        gather - ``basket_size`` bytes of a column at a time - so the tree
        can be far larger than memory, and the tree's own record is written
        when the file closes. A name with a ``/`` in it puts the tree in the
        directory it names. A counter is a 32-bit int unless ``counters``
        gives its name another integer type, as ROOT's ``n/b`` would be.
        ``classname`` writes the tree as a ``TNtuple`` or ``TNtupleD`` instead,
        as ROOT writes one: the same tree, and how many variables it has.
        No columns at all is refused as a mistake, unless ``empty`` says it is
        meant: a tree of no branches, as ROOT writes one asked to.
        """
        from .wtree import WritableTree

        here, leaf = self._place(name)
        here._check_leaf(leaf)
        _checked(title or "", "title")
        cycle = here._next_cycle(leaf)
        tree = WritableTree(
            here, leaf, title or "", columns, basket_size, cycle, counters, classname=classname,
            empty=empty,
        )  # fmt: skip
        self._file._trees.append(tree)
        self._file._used.update(dict.fromkeys(tree.classes))
        for layout in tree.declared:  # a macro's classes, described as they were declared
            entry = info_entry(layout.name, layout.checksum, layout.version, layout.elements())
            self._file._carried.setdefault((layout.name, layout.version), entry)
        return tree

    def mkdir(self, path: str) -> WritableDirectory:
        """The directory at ``path`` below this one, made if it is not there.

            >>> run = f.mkdir("calibration/run4711")        # doctest: +SKIP
            >>> run["gains"] = gains                         # doctest: +SKIP

        Every directory along the way is made too if need be, and one that is
        already there - made earlier, or already in a file being updated - is
        given back as it is, so asking twice is harmless. A name that already
        holds an object is refused, because a directory of the same name would
        hide it from anything reading the file back. A path with any part a
        key could not be named is refused before anything along it is made.
        """
        parts = path.split("/") if isinstance(path, str) else [path]  # which then refuses
        for part in parts:
            self._check_name(part)
        here = self
        for part in parts:
            here = here._subdirectory(part)
        return here

    def _subdirectory(self, name: str) -> WritableDirectory:
        """The directory ``name`` right here: already made, on file, or new."""
        found = self._subdirs.get(name)
        if found is None and name in self._found:
            found = self._subdirs[name] = self._found.pop(name)()
        if found is not None:
            return found
        if name in self._cycles:
            raise ValueError(
                f"{name!r} in {self._where} already holds an object, and a directory "
                f"of that name would hide it; give the directory another name"
            )
        return self._make_subdirectory(name)

    def _make_subdirectory(self, name: str) -> WritableDirectory:
        """A new directory: its key here, and its record, to be filled in at the close.

        The record's parent is the file's top directory however deep this one
        is, because that is what ROOT writes there; the key's own directory,
        which is what a reader goes by, is the one it is really in.
        """
        seek, nbytes = self._put(
            "TDirectory", name, name, bytes(DIRECTORY_BYTES), 1, listed=True, packed=False
        )
        path = f"{self.path}/{name}" if self.path else name
        file = self._file
        made = WritableDirectory(file, path, name, seek, file._seek, nbytes - DIRECTORY_BYTES)
        self._subdirs[name] = made
        return made

    def _place(self, name: str) -> tuple[WritableDirectory, str]:
        """Which directory a path puts its last part in, made if need be, and that part."""
        if not isinstance(name, str):
            return self, name  # which the name check then refuses, by what it is
        head, slash, leaf = name.rpartition("/")
        if not slash:
            return self, leaf
        self._check_name(leaf)  # before any directory is made for it
        return self.mkdir(head), leaf

    @property
    def _where(self) -> str:
        return repr(self.path) if self.path else "the top of the file"

    def rntuple(self, name: str, fields: Mapping[str, Any], **options: Any) -> WritableRNTuple:
        """An RNTuple in this directory, to be filled a batch or an entry at a time.

            >>> with xrdroot.create("out.root") as f:        # doctest: +SKIP
            ...     ntuple = f.rntuple("events", {"n": np.int32, "pt": [np.float32]})
            ...     ntuple.extend({"n": counts, "pt": pts})

        ``fields`` maps each field's name to what it holds, and ``options``
        are ``cluster_size`` and ``page_size`` in bytes and a ``description``;
        :class:`~xrdroot.rntuple.writer.WritableRNTuple` says what each can be.
        A name with a ``/`` in it puts the RNTuple in the directory it names.
        """
        from .rntuple.writer import WritableRNTuple

        here, leaf = self._place(name)
        here._check_leaf(leaf)
        ntuple = WritableRNTuple(here, leaf, fields, here._next_cycle(leaf), **options)
        self._file._rntuples.append(ntuple)
        self._file._used["ROOT::RNTuple"] = None
        return ntuple

    def _check_name(self, name: str) -> None:
        """Whether a key could be written under this name, and read back by it."""
        if self._file._closed:
            raise ValueError("this file is closed; whatever it was going to hold is written")
        _checked(name, "name")
        if not name:
            raise ValueError("a key with no name could never be asked for; give it one")
        if ";" in name:
            raise ValueError(
                f"{name!r} has a ; in it, which is how a reader asks for an old cycle; "
                f"a name containing one could never be read back"
            )

    def _check_leaf(self, name: str) -> None:
        """Whether an object could go under this name, which a directory must not have."""
        self._check_name(name)
        if name in self._subdirs or name in self._found:
            raise ValueError(
                f"{name!r} is a directory in {self._where}, and an object of that name "
                f"would hide everything in it; write into it as {name}/..., or pick "
                f"another name"
            )

    def delete(self, name: str, cycle: int | None = None) -> int:
        """Take ``name`` out of this directory - every cycle, or only ``cycle`` - as
        ``TDirectory::Delete("name;*")`` does; how many keys went.

        The records go to the free list; a directory's key goes, and what it
        held with it, unread. The next cycle written under the name is one
        past the newest left, or the first.
        """
        kept, gone = [], 0
        for header in self._keys:
            key = Key(Buffer(header))
            if key.name == name and cycle in (None, key.cycle):
                self._file._release((key.seek_key, key.nbytes))
                gone += 1
            else:
                kept.append(header)
        if gone:
            self._keys = kept
            self._dirty = True
            self._forget(name)
        return gone

    def classname_of(self, name: str) -> str | None:
        """The class of what ``name`` holds here - its newest cycle's - or ``None``."""
        found = [Key(Buffer(header)) for header in self._keys]
        named = [key for key in found if key.name == name]
        return max(named, key=lambda key: key.cycle).classname if named else None

    def _forget(self, name: str) -> None:
        """What is known of ``name`` here, brought in line with the keys left."""
        self._subdirs.pop(name, None)
        self._found.pop(name, None)
        cycles = [Key(Buffer(header)).cycle for header in self._keys
                  if Key(Buffer(header)).name == name]  # fmt: skip
        if cycles:
            self._cycles[name] = max(cycles)
        else:
            self._cycles.pop(name, None)

    def _next_cycle(self, name: str) -> int:
        cycle = self._cycles.get(name, 0) + 1
        self._cycles[name] = cycle
        return cycle

    @property
    def closed(self) -> bool:
        return self._file._closed

    @property
    def _codes(self) -> int:
        """How ROOT spells this file's compression: the algorithm and the level."""
        file = self._file
        if file._algorithm is None:
            return 0
        return CODES[file._algorithm] * 100 + (file._level or 0)

    def _key_length(
        self, classname: str, name: str, title: str, extra: int = 0, wide: bool = False
    ) -> int:
        """How long the key of the next record put here will be.

        A tree has to know before it writes a record, because the record
        counts places from the start of its key; and whether that key is
        small or wide depends on where it lands, which is here and now -
        unless ``wide`` asks for the wide layout wherever it lands.
        """
        wide = wide or _wide(self._file._out.size, self._seek)
        return _keylen(classname, name, title, extra) + (WIDER if wide else 0)

    def _put(
        self,
        classname: str,
        name: str,
        title: str,
        payload: bytes,
        cycle: int,
        listed: bool,
        packed: bool = True,
        extra: bytes = b"",
        *,
        objlen: int | None = None,
        wide: bool = False,
        codes: int | None = None,
    ) -> tuple[int, int]:
        """One record in this directory: its key, then its payload, compressed
        when that is smaller.

        Where it went and how long it turned out come back, which is what a
        tree writes down about each of its baskets. ``extra`` is whatever the
        class keeps in its own key rather than its payload - only a
        ``TBasket`` does, and what it keeps is how to slice itself.

        The key lists, the free list and a directory's record go in raw
        whatever the file's setting, because ROOT - and the reader here -
        parses them without looking at the lengths that would say they were
        compressed.

        ``objlen`` says the payload is already as it goes on file - a record
        copied from another file, compressed or not - and is how long the
        object it holds is; nothing is squeezed then. ``wide`` asks for the
        wide key wherever the record lands, as :func:`_key_fields` has it.
        ``codes`` compresses the payload otherwise than the file does.
        """
        file = self._file
        seek = file._out.size
        body = payload
        if objlen is None:
            # TKey leaves an object of 256 bytes or fewer raw; a basket is packed whatever its size.
            packed = packed and (classname == "TBasket" or len(payload) > SMALL_OBJECT)
            body = file._squeeze(payload, codes) if packed else payload
            objlen = len(payload)
        keylen = self._key_length(classname, name, title, len(extra), wide)
        key = WBuffer()
        sizes = (keylen + len(body), objlen, keylen)
        _key_fields(key, seek, self._seek, sizes, cycle, file._datime, wide)
        key.string(classname)
        key.string(name)
        key.string(title)
        header = bytes(key.data) + extra
        if listed:
            self._keys.append(header)
            self._dirty = True
        file._out.append(header + body)
        return seek, keylen + len(body)

    def _walk(self) -> Iterator[WritableDirectory]:
        """This directory and every one below it, deepest first, as ROOT closes them."""
        for sub in self._subdirs.values():
            yield from sub._walk()
        yield self

    def _write_keys(self) -> None:
        """This directory's key list, then its record told where the list went."""
        keylist = WBuffer()
        keylist.i32(len(self._keys))
        for header in self._keys:
            keylist.raw(header)
        classname, name, title = self._listed_as
        seek, nbytes = self._put(
            classname, name, title, bytes(keylist.data), 1, listed=False, packed=False
        )
        self._file._release(self._replaced)
        self._file._out.patch(self._seek + self._nname, self._record(seek, nbytes))

    def _record(self, seek_keys: int, nbytes_keys: int) -> bytes:
        """The ``TDirectory`` record: when, how long its key list is, and where
        it, its parent and its keys are.

        It is wide when any of those places is past :data:`BIG`, and always
        :data:`DIRECTORY_BYTES` long either way - ROOT leaves room for the
        wide form in every file it writes, so a directory never has to move.
        """
        seeks = (self._seek, self._parent_seek, seek_keys)
        wide = _wide(*seeks)
        buf = WBuffer()
        buf.u16(DIRECTORY_VERSION + (WIDE if wide else 0))
        buf.u32(self._ctime)
        buf.u32(self._file._datime)
        buf.i32(nbytes_keys)
        buf.i32(self._nname)
        for place in seeks:
            buf.seek(place, wide)
        buf.u16(1)  # the version of the UUID after it
        buf.raw(self._uuid)
        buf.raw(bytes(DIRECTORY_BYTES - len(buf.data)))
        return bytes(buf.data)


class WritableFile(WritableDirectory):
    """A ROOT file being written: a mapping from name to object, then a close.

        >>> with xrdroot.create("out.root") as f:            # doctest: +SKIP
        ...     f["hist"] = xrdroot.Histogram.new("hist", edges, counts)

    The file is its own top directory, so everything a
    :class:`WritableDirectory` does - ``mkdir``, trees, paths with ``/`` in
    them - it does too. Records go out as they are made, so a file far larger
    than memory can be written; the header, which points at what is written
    last, is filled in at a clean close. A ``with`` block that raises takes
    back everything it wrote, on the principle that no file is better than
    half a file.
    """

    def __init__(
        self,
        handle: IO[bytes],
        name: str,
        owned: bool,
        algorithm: str | None,
        level: int | None,
        title: str = "",
    ) -> None:
        self._start(handle, owned, algorithm, level)
        label = _checked(name.rstrip("/").rpartition("/")[2] or name, "file name")
        # ROOT calls the file title the description of the file - the third
        # argument of ``new TFile(name, option, ftitle)`` - and it is the title
        # of the key in front of the file's own record. That key is written
        # more than once (the fBEGIN block, the top directory's key list, the
        # free list), and each copy carries the title, so a file with one is
        # four bytes longer per character than the same file without.
        titled = _checked(title, "file title")
        # The key in front of the file's own record: the fixed part, then
        # "TFile", the name, the title; the name and title again.
        keylen = _keylen("TFile", label, titled)
        nname = keylen + 2 + len(label.encode("utf-8", "surrogateescape"))
        nname += len(titled.encode("utf-8", "surrogateescape"))
        #: The header, then the room the file's own record takes; both are
        #: filled in at close, when everything they point at has a place.
        self._out = _Output(handle, BEGIN + nname + DIRECTORY_BYTES)
        super().__init__(self, "", titled, BEGIN, 0, nname)
        #: What the file calls itself: the last segment of where it is going.
        self.name = name
        self._listed_as = ("TFile", label, titled)
        #: The file's identity, which its top directory shares, as in ROOT.
        self._file_uuid = self._uuid
        head = WBuffer()  # the fBEGIN block: the file is a key like any other
        sizes = (nname + DIRECTORY_BYTES, nname - keylen + DIRECTORY_BYTES, keylen)
        _key_fields(head, BEGIN, 0, sizes, 1, self._datime)
        for text in ("TFile", label, titled, label, titled):
            head.string(text)
        self._out.head[BEGIN : BEGIN + len(head.data)] = head.data

    def _start(
        self, handle: IO[bytes], owned: bool, algorithm: str | None, level: int | None
    ) -> None:
        """What every file being written keeps, whether new or being updated."""
        self._handle = handle
        self._owned = owned
        self._algorithm = algorithm
        self._level = LEVELS[algorithm] if algorithm is not None and level is None else level
        self._datime = packed_now()
        self._closed = False
        #: The version of ROOT the header says wrote the file.
        self._version = WRITER_VERSION
        self._used: dict[str, None] = {}
        self._trees: list[WritableTree] = []
        self._rntuples: list[WritableRNTuple] = []
        #: The gaps the file already had, and the records this session let go.
        self._gaps: list[tuple[int, int]] = []
        self._freed: list[tuple[int, int]] = []
        #: Descriptions of classes carried over from other files, by class
        #: and version, each a ``TStreamerInfo`` list entry that stands on its
        #: own: what a record copied as it was needs the file to say about it.
        self._carried: dict[tuple[str, int], bytes] = {}
        #: The file's ``ProcessID`` records, by the UUID each is titled with: what an object
        #: a ``TRef`` may point at is written with the index of.
        self._pids: dict[str, int] | None = None
        #: The trees and ntuples a checkpoint has already given their own record, by object id,
        #: and the streamer-info record written last with how many classes it covered - so a
        #: second checkpoint writes only what is new and a close after one writes nothing twice.
        self._finished: set[int] = set()
        self._streamer_record: tuple[int, int] | None = None
        self._streamers_for = 0

    def _processes(self) -> dict[str, int]:
        """The file's ``ProcessID`` records - those it had already, when it is being updated."""
        if self._pids is None:
            self._pids = {}
            for raw in self._keys:
                key = Key(Buffer(raw))
                if key.classname == "TProcessID" and key.name.startswith("ProcessID"):
                    self._pids[key.title] = int(key.name[len("ProcessID") :] or 0)
        return self._pids

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"{len(self._keys)} keys so far"
        return f"<WritableFile {self.name!r}, {state}>"

    def _squeeze(self, payload: bytes, codes: int | None = None) -> bytes:
        """The payload compressed with the file's setting, if that made it smaller.

        ``codes`` is a setting of its own, as ROOT spells one, for what is
        compressed otherwise than the file: a branch given one with
        ``TBranch::SetCompressionSettings``.
        """
        algorithm, level = self._algorithm, self._level
        if codes is not None:
            algorithm, level = spelled(codes)
        if algorithm is None:
            return payload
        squeezed = compress(payload, algorithm, level)
        return squeezed if len(squeezed) < len(payload) else payload

    def _release(self, region: tuple[int, int] | None) -> None:
        """Give a replaced record's bytes, where and how many, to the free list."""
        if region is not None:
            seek, nbytes = region
            self._freed.append((seek, seek + nbytes - 1))

    def checkpoint(self) -> None:
        """Write the file as far as it goes and leave it open - what ROOT's ``TFile::Write`` does.

        Everything written so far becomes a file at this file's path that can be opened and
        read: the header, the directory's keys, each tree's own record and the streamer info.
        What is written next is appended, as ROOT appends it, and a second ``Write`` appends a
        second free list and header of its own. A program that writes a file and reads it back
        in the same run - as ROOT's own tests do - reads it here too.
        """
        self._assemble()
        self._out.finish()

    def close(self) -> None:
        """Lay out the bookkeeping, fill in the header, and let go."""
        if self._closed:
            return
        self._closed = True
        try:
            self._assemble()
            self._out.finish()
        finally:
            if self._owned:
                self._handle.close()

    def _finish_containers(self) -> None:
        """Give each tree and ntuple its own record, each one once.

        A record written twice is a second key, so a checkpoint and the close that follows it
        write it once between them: what is in ``_finished`` has already been written.
        """
        for tree in self._trees:
            if id(tree) not in self._finished:
                self._finished.add(id(tree))
                tree._finish()  # its baskets are in; now it can say where they went
        for ntuple in self._rntuples:
            if id(ntuple) not in self._finished:
                self._finished.add(id(ntuple))
                ntuple._finish()  # likewise its pages; now its header, footer and anchor

    def _assemble(self) -> None:
        """Everything that could only be placed once the objects were in.

        Each tree's record is written once and the streamer info only when a class new since
        the last one has been used, so assembling twice - a checkpoint and then the close, or
        two ``Write`` calls - writes nothing twice. The free list and the header are written
        every time, each replacing the one before it, as ROOT's own ``WriteFree`` does.
        """
        self._finish_containers()
        if self._streamer_record is None or self._streamers_for != len(self._used):
            self._streamer_record = self._write_streamers()
            self._streamers_for = len(self._used)
        for directory in self._walk():
            if directory._dirty:
                directory._write_keys()
        self._write_header(*self._write_free(), self._streamer_record)

    def _write_streamers(self) -> tuple[int, int]:
        """The file's ``StreamerInfo`` record, and where it went."""
        keylen = self._key_length("TList", "StreamerInfo", "Doubly linked list")
        info = _streamers(self._used, self._carried, keylen)
        return self._put("TList", "StreamerInfo", "Doubly linked list", info, 1, listed=False)

    def _write_free(self) -> tuple[int, tuple[int, int, int]]:
        """The free list, last of all: every gap, then the open end of the file.

        Its own length decides where the file ends, and where the file ends
        decides whether its last entry is wide, so the small entry is tried
        first and the wide one taken only when the small would end past
        :data:`BIG`. Each gap also gets ROOT's marker at its front - its
        length, negated - so that anything walking the file record by record
        steps over it.
        """
        gaps = _merged(self._gaps + self._freed)
        for first, last in gaps:
            if last - first >= 3:
                self._out.patch(first, struct.pack(">i", -min(last - first + 1, MAX_GAP)))
        classname, name, title = self._listed_as
        seek = self._out.size
        before = self._key_length(classname, name, title) + len(_free_entries(gaps))
        end = seek + before + FREE_SMALL
        if end > BIG:
            end = seek + before + FREE_WIDE
        body = _free_entries([*gaps, (end, _tail(end))])
        _, nbytes = self._put(classname, name, title, body, 1, listed=False, packed=False)
        return end, (seek, nbytes, len(gaps) + 1)

    def _write_header(self, end: int, free: tuple[int, int, int], info: tuple[int, int]) -> None:
        """The hundred bytes at the front, in the wide form once the file ends past BIG."""
        wide = end > BIG
        seek_free, nbytes_free, nfree = free
        seek_info, nbytes_info = info
        form = ">qqiiiBiqi" if wide else ">iiiiiBiii"
        head = self._out.head
        version = self._version + (WIDE_FILE if wide else 0)
        struct.pack_into(">4sii", head, 0, MAGIC, version, self._seek)
        fields = (nbytes_free, nfree, self._nname, 8 if wide else 4, self._codes)
        struct.pack_into(form, head, 12, end, seek_free, *fields, seek_info, nbytes_info)
        struct.pack_into(">H16s", head, 12 + struct.calcsize(form), 1, self._file_uuid)

    def __enter__(self) -> WritableFile:
        return self

    def __exit__(self, kind: object, *rest: object) -> None:
        if kind is not None:
            # Whatever went wrong, do not leave a plausible half-truth behind.
            self._closed = True
            try:
                self._out.abandon()
            finally:
                if self._owned:
                    self._handle.close()
            return
        self.close()


def create(
    target: Any,
    *,
    title: str = "",
    compression: str | None = "zlib",
    level: int | None = None,
    config: Any = None,
) -> WritableFile:
    """A new ROOT file at ``target``, ready to be given objects.

        >>> with xrdroot.create("counts.root") as f:         # doctest: +SKIP
        ...     f["signal"] = xrdroot.Histogram.new("signal", edges, counts)
        ...     f["scan"] = xrdroot.Graph.new("scan", xs, ys, yerr=bars)
        ...     f["runs/4711/signal"] = signal    # in a directory, made for it

    ``target`` is a local path, a URL of any scheme this library writes, or
    an already-open binary file, which is used as it is and left open.
    ``title`` is what the file says it is, as ROOT's
    ``new TFile(name, option, ftitle)`` has it, and is empty - like a file the
    command line alone makes - unless given; a title is the only thing ROOT
    keeps in the key of the file itself, and it is written four times over.
    ``compression`` is ``zlib`` unless said otherwise - ``lzma``, ``lz4``,
    ``zstd``, or ``None`` to store everything as it is - and each object is
    stored raw anyway when compressing it did not make it smaller.
    """
    _check_compression(compression)
    if hasattr(target, "write"):
        name = getattr(target, "name", "<file>")
        return WritableFile(target, name, False, compression, level, title)
    url = parse(target)
    if url.is_local:
        handle: IO[bytes] = open(url.path, "wb")
    else:
        from xrdclient.io import open_url

        # Persist on successful close: a process that dies part way through
        # leaves the server nothing, where cutting back could not be done.
        handle = open_url(url, "wb", config=config, posc=True)
    return WritableFile(handle, str(target), True, compression, level, title)


def _check_compression(compression: str | None) -> None:
    if compression is not None and compression not in CODES:
        raise ValueError(
            f"compression must be one of {', '.join(CODES)} or None, not {compression!r}"
        )
