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
carry - a histogram with fits attached refuses rather than silently dropping
them.

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
from xrdclient.url import parse

from .buffer import BYTE_COUNT_MASK, IS_REFERENCED, NEW_CLASS_TAG
from .compression import CODES, LEVELS, compress
from .errors import UnsupportedFeatureError
from .graph import GRAPHS, Graph
from .hist import HISTOGRAMS, Histogram
from .interp import ARRAYS, OFFSET_L, OFFSET_P
from .winfo import INFOS, SUBVERSIONS, WRITER_VERSION, Element

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .rntuple.writer import WritableRNTuple
    from .wtree import WritableTree

__all__ = ["create", "WritableFile", "WritableDirectory"]

#: The bits a freshly made object carries: on the heap, and not deleted.
BITS = 0x03000000
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

    __slots__ = ("data",)

    def __init__(self) -> None:
        self.data = bytearray()

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
        which is longer and legal, and means no bookkeeping can be wrong.
        """
        index = len(self.data)
        self.data += b"\x00\x00\x00\x00"
        self.u32(NEW_CLASS_TAG)
        self.cstring(classname)
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

    def named(self, name: str, title: str) -> None:
        """A ``TNamed`` base: a record holding the object bits and two strings."""
        index = self.start(1)
        self.tobject()
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


def _list(buf: WBuffer, classname: str, entries: Any) -> None:
    """An empty ``TList``, which is the only list this writer will write."""
    if entries:
        raise UnsupportedFeatureError(
            f"a {classname} holding {len(entries)} entries is not written: what a "
            f"list carries could be anything at all, and writing it wrongly would "
            f"be worse than refusing; empty it first"
        )
    index = buf.start(5)
    buf.tobject()
    buf.string("")
    buf.i32(0)
    buf.end(index)


def _record(buf: WBuffer, classname: str, row: Any, used: dict[str, None]) -> None:
    """One object written in place: its record, then its members in order.

    Every class actually written lands in ``used``, so an object hung behind
    a pointer - a graph's fitted histogram, say - gets described by the
    file's streamer information exactly as a key of the file would be.
    """
    if classname in LISTS:
        _list(buf, classname, row)
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
    for element in info[2]:
        _element(buf, element, row, used)
    buf.end(index)


def _pointed(buf: WBuffer, typename: str, value: Any, used: dict[str, None]) -> None:
    """A pointer that may be null: nothing, or an object naming its class."""
    if value is None or (isinstance(value, (list, tuple)) and not value):
        buf.u32(0)
        return
    classname = typename.rstrip("*")
    if isinstance(value, (Histogram, Graph)):
        classname, value = value.classname, value.members
    elif isinstance(value, (list, tuple)):
        _list(buf, classname, value)  # non-empty here, so this refuses by name
    index = buf.tag(classname)
    _record(buf, classname, value, used)
    buf.end(index)


def _element(buf: WBuffer, element: Element, row: dict[str, Any], used: dict[str, None]) -> None:
    """One member of one class, laid out exactly as its element describes."""
    _kind, name, _title, stype, _size, alen, _adim, _maxidx, typename, extras = element
    value = row.get(name)
    if _base_element(buf, name, stype, value, used):
        return
    if _object_element(buf, stype, typename, value, used):
        return
    if _number_element(buf, name, stype, alen, extras, value, row):
        return
    raise UnsupportedFeatureError(
        f"{name} is of streamer type {stype}, which this writer does not lay out"
    )


def _base_element(buf: WBuffer, name: str, stype: int, value: Any, used: dict[str, None]) -> bool:
    if stype == 66:  # the TObject base, which streams itself
        _write_tobject(buf, value)
    elif stype == 67:  # the TNamed base, likewise
        _write_tnamed(buf, value)
    elif stype == 0:  # any other base, written under its own name
        _write_base(buf, name, value, used)
    elif stype == 65:  # a TString member: its bytes, no record
        buf.string(str(value if value is not None else ""))
    else:
        return False
    return True


def _write_tobject(buf: WBuffer, value: Any) -> None:
    bits = value if isinstance(value, dict) else {}
    buf.tobject(int(bits.get("fBits", BITS)), int(bits.get("fUniqueID", 0)))


def _write_tnamed(buf: WBuffer, value: Any) -> None:
    named = value if isinstance(value, dict) else {}
    buf.named(str(named.get("fName", "")), str(named.get("fTitle", "")))


def _write_base(buf: WBuffer, name: str, value: Any, used: dict[str, None]) -> None:
    if name in ARRAYS:
        _array(buf, name, value)
    else:
        _record(buf, name, value if value is not None else {}, used)


def _object_element(
    buf: WBuffer, stype: int, typename: str, value: Any, used: dict[str, None]
) -> bool:
    if stype in (61, 62):  # an object held by value
        if typename in ARRAYS:
            _array(buf, typename, value)
        else:
            _record(buf, typename, value if value is not None else {}, used)
    elif stype in (63, 68):  # a pointer promised never null, written in place
        _record(buf, typename.rstrip("*"), value if value is not None else [], used)
    elif stype in (64, 69):  # a pointer that may be null
        _pointed(buf, typename, value, used)
    else:
        return False
    return True


def _number_element(
    buf: WBuffer,
    name: str,
    stype: int,
    alen: int,
    extras: tuple[Any, ...],
    value: Any,
    row: dict[str, Any],
) -> bool:
    if stype in FORMS:  # one number
        buf.raw(struct.pack(">" + FORMS[stype], value))
    elif stype - OFFSET_L in FORMS:  # a fixed-size array, x[10]
        values = list(value if value is not None else ())
        if len(values) != alen:
            raise ValueError(f"{name} holds {len(values)} values where {alen} were declared")
        _numbers(buf, FORMS[stype - OFFSET_L], values)
    elif stype - OFFSET_P in FORMS:  # a counted array, x[n], behind its marker
        counter = str(extras[1])
        try:
            count = int(_find(row, counter))
        except KeyError:
            raise ValueError(f"{name} is counted by {counter}, which is not here") from None
        values = list(value if value is not None else ())
        if len(values) < count:
            raise ValueError(f"{name} holds {len(values)} values where {counter} says {count}")
        buf.u8(1)
        _numbers(buf, FORMS[stype - OFFSET_P], values[:count])
    else:
        return False
    return True


def _closure(seeds: dict[str, None]) -> list[str]:
    """Every class the seeds reach through bases and members, in met order."""
    ordered = dict.fromkeys(seeds)
    queue = list(ordered)
    while queue:
        info = INFOS.get(queue.pop(0))
        if info is None:
            continue
        for kind, name, _t, stype, _s, _a, _d, _m, typename, _x in info[2]:
            if kind == "TStreamerBase":
                held = name
            elif stype in (61, 62, 63, 64, 68, 69):
                held = typename.rstrip("*")
            elif stype == 65:
                held = "TString"
            else:
                continue
            if held in INFOS and held not in ordered:
                ordered[held] = None
                queue.append(held)
    return list(ordered)


def _streamers(used: dict[str, None]) -> bytes:
    """The ``StreamerInfo`` record: what this file says its classes look like.

    Emitted from :data:`~.winfo.INFOS` verbatim, checksums and all, so the
    file describes its classes exactly as the ROOT that the descriptions were
    harvested from would have.
    """
    names = _closure(used)
    buf = WBuffer()
    index = buf.start(5)
    buf.tobject()
    buf.string("")
    buf.i32(len(names))
    buf.raw(_info_entries(names))
    buf.end(index)
    return bytes(buf.data)


def _info_entries(names: list[str]) -> bytes:
    """One ``TStreamerInfo`` list entry per class, each naming its classes in
    full rather than referring back, so the bytes mean the same wherever in
    a list they land - which is what lets an update add them to a list that
    is already there."""
    buf = WBuffer()
    for name in names:
        checksum, version, elements = INFOS[name]
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
    return bytes(buf.data)


def _info_element(buf: WBuffer, element: Element) -> None:
    """One ``TStreamerElement``, written the way the donor files write them."""
    kind, name, title, stype, size, alen, adim, maxidx, typename, extras = element
    tag = buf.tag(kind)
    sub = buf.start(SUBVERSIONS[kind])
    base = buf.start(4)  # the TStreamerElement the subclass builds on
    buf.named(name, title)
    buf.i32(stype)
    buf.i32(size)
    buf.i32(alen)
    buf.i32(adim)
    for index in maxidx:
        buf.i32(index)
    buf.string(typename)
    buf.end(base)
    if kind == "TStreamerBase":
        buf.i32(int(str(extras[0])))
    elif kind == "TStreamerBasicPointer":
        buf.i32(int(str(extras[0])))
        buf.string(str(extras[1]))
        buf.string(str(extras[2]))
    buf.end(sub)
    buf.end(tag)


def _payload(obj: Any) -> tuple[str, bytes, tuple[str, ...]]:
    """What one object writes as: its class, its bytes, the layouts it needs."""
    if isinstance(obj, str):
        return _string_payload(obj)
    if isinstance(obj, (array.array, np.ndarray)):
        return _array_payload(obj)
    if isinstance(obj, (Histogram, Graph)):
        return _object_payload(obj)
    if Histogram.recognises(obj):
        return _object_payload(Histogram.of(obj))
    raise UnsupportedFeatureError(
        f"a {type(obj).__name__} is not something this writer puts in a ROOT "
        f"file: it takes a Histogram, a Graph, any histogram that speaks the "
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


def _object_payload(value: Histogram | Graph) -> tuple[str, bytes, tuple[str, ...]]:
    classname = value.classname
    if classname not in INFOS:
        writable = ", ".join(name for name in INFOS if name in HISTOGRAMS or name in GRAPHS)
        raise UnsupportedFeatureError(
            f"a {classname} is not a class this writer carries a layout for; "
            f"the ones it does are {writable}"
        )
    buf = WBuffer()
    used: dict[str, None] = {}
    _record(buf, classname, value.members, used)
    return classname, bytes(buf.data), tuple(used)


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
    buf: WBuffer, seek: int, pdir: int, sizes: tuple[int, int, int], cycle: int, datime: int
) -> None:
    """The fixed part of a key, small or wide as the places it holds decide.

    ``sizes`` are the record's length on file, the object's length, and the
    key's own length, in the order the key holds them.
    """
    wide = _wide(seek, pdir)
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
        classname, payload, used = _payload(obj)
        if title is None:
            title = obj.title if isinstance(obj, (Histogram, Graph)) else ""
        _checked(title, "title")
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
        directory it names.
        """
        from .wtree import WritableTree

        here, leaf = self._place(name)
        here._check_leaf(leaf)
        _checked(title or "", "title")
        tree = WritableTree(here, leaf, title or "", columns, basket_size, here._next_cycle(leaf))
        self._file._trees.append(tree)
        self._file._used.update(dict.fromkeys(tree.classes))
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

    def _key_length(self, classname: str, name: str, title: str, extra: int = 0) -> int:
        """How long the key of the next record put here will be.

        A tree has to know before it writes a record, because the record
        counts places from the start of its key; and whether that key is
        small or wide depends on where it lands, which is here and now.
        """
        wide = _wide(self._file._out.size, self._seek)
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
        """
        file = self._file
        seek = file._out.size
        body = file._squeeze(payload) if packed else payload
        keylen = self._key_length(classname, name, title, len(extra))
        key = WBuffer()
        sizes = (keylen + len(body), len(payload), keylen)
        _key_fields(key, seek, self._seek, sizes, cycle, file._datime)
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
    ) -> None:
        self._start(handle, owned, algorithm, level)
        label = _checked(name.rstrip("/").rpartition("/")[2] or name, "file name")
        # The key in front of the file's own record: the fixed part, then
        # "TFile", the name, and an empty title; the name and title again.
        keylen = _keylen("TFile", label, "")
        nname = keylen + 2 + len(label.encode())
        #: The header, then the room the file's own record takes; both are
        #: filled in at close, when everything they point at has a place.
        self._out = _Output(handle, BEGIN + nname + DIRECTORY_BYTES)
        super().__init__(self, "", "", BEGIN, 0, nname)
        #: What the file calls itself: the last segment of where it is going.
        self.name = name
        self._listed_as = ("TFile", label, "")
        #: The file's identity, which its top directory shares, as in ROOT.
        self._file_uuid = self._uuid
        head = WBuffer()  # the fBEGIN block: the file is a key like any other
        sizes = (nname + DIRECTORY_BYTES, nname - keylen + DIRECTORY_BYTES, keylen)
        _key_fields(head, BEGIN, 0, sizes, 1, self._datime)
        for text in ("TFile", label, "", label, ""):
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

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"{len(self._keys)} keys so far"
        return f"<WritableFile {self.name!r}, {state}>"

    def _squeeze(self, payload: bytes) -> bytes:
        """The payload compressed with the file's setting, if that made it smaller."""
        if self._algorithm is None:
            return payload
        squeezed = compress(payload, self._algorithm, self._level)
        return squeezed if len(squeezed) < len(payload) else payload

    def _release(self, region: tuple[int, int] | None) -> None:
        """Give a replaced record's bytes, where and how many, to the free list."""
        if region is not None:
            seek, nbytes = region
            self._freed.append((seek, seek + nbytes - 1))

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

    def _assemble(self) -> None:
        """Everything that could only be placed once the objects were in."""
        for tree in self._trees:
            tree._finish()  # its baskets are in; now it can say where they went
        for ntuple in self._rntuples:
            ntuple._finish()  # likewise its pages; now its header, footer and anchor
        info = self._write_streamers()
        for directory in self._walk():
            if directory._dirty:
                directory._write_keys()
        self._write_header(*self._write_free(), info)

    def _write_streamers(self) -> tuple[int, int]:
        """The file's ``StreamerInfo`` record, and where it went."""
        info = _streamers(self._used)
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
    ``compression`` is ``zlib`` unless said otherwise - ``lzma``, ``lz4``,
    ``zstd``, or ``None`` to store everything as it is - and each object is
    stored raw anyway when compressing it did not make it smaller.
    """
    _check_compression(compression)
    if hasattr(target, "write"):
        return WritableFile(target, getattr(target, "name", "<file>"), False, compression, level)
    url = parse(target)
    if url.is_local:
        handle: IO[bytes] = open(url.path, "wb")
    else:
        from xrdclient.io import open_url

        # Persist on successful close: a process that dies part way through
        # leaves the server nothing, where cutting back could not be done.
        handle = open_url(url, "wb", config=config, posc=True)
    return WritableFile(handle, str(target), True, compression, level)


def _check_compression(compression: str | None) -> None:
    if compression is not None and compression not in CODES:
        raise ValueError(
            f"compression must be one of {', '.join(CODES)} or None, not {compression!r}"
        )
