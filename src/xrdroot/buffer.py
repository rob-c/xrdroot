"""The ROOT on-disk encoding, read one field at a time.

Everything ROOT writes is big-endian and self-describing: a record starts with
its byte count and version, so a reader that does not know a class can still
step over it. That is what makes a partial reader like this one honest - it
never has to guess a length.

Positions are counted the way ROOT counts them, from the start of the key
rather than the start of the decompressed payload, because the class and
object back-references written into the stream are absolute in those terms.
"""

from __future__ import annotations

import datetime
import struct
from collections.abc import Mapping
from functools import cache
from typing import Any

import numpy as np

from .errors import FormatError, UnsupportedFeatureError

__all__ = [
    "Buffer",
    "Clones",
    "Listed",
    "Named",
    "Written",
    "as_datetime",
    "gather",
    "numbers",
    "on_disk",
]


@cache
def on_disk(typename: str) -> np.dtype[Any]:
    """How a number of this type is laid out in a ROOT file: big-endian.

    ``typename`` is this library's word for it - ``'float32'``, ``'uint16'``
    - which is also NumPy's. A ``bool`` is a byte, as C++ has it.
    """
    if typename == "bool":
        return np.dtype("u1")
    return np.dtype(typename).newbyteorder(">")


class Named(dict[str, Any]):
    """One object's members, and the class the stream said they were the members of.

    An object written where it stands names its class in front of itself, and a
    pointer read by the type it was declared with - ``BaseABC *fABC`` holding a
    ``Derived`` - can only be made into the object ROOT would give back if that
    name survives the walk over the members, which are laid out apart from it.
    """

    __slots__ = ("classname",)

    def __init__(self, members: Mapping[str, Any], classname: str) -> None:
        super().__init__(members)
        self.classname = classname


def gather(data: bytes, starts: Any, lengths: Any) -> bytes:
    """The runs ``data[start:start + length]`` for every pair, end to end.

    This is how a basket's entries come out when they do not sit one after
    another at a fixed stride: one index built in C and one take, rather than
    a slice and a copy per entry.
    """
    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    kept = lengths > 0
    starts, lengths = starts[kept], lengths[kept]
    if not len(starts):
        return b""
    raw = np.frombuffer(data, np.uint8)
    ends = starts + lengths
    if np.all(starts[1:] >= ends[:-1]):
        # In order and apart, which is every basket's case: a byte mask that
        # steps up where a run starts and down where it ends costs a byte per
        # byte of basket, where an index would cost eight.
        steps = np.zeros(len(raw) + 1, np.int8)
        steps[starts] += 1
        steps[ends] -= 1
        return bytes(raw[np.cumsum(steps[:-1], dtype=np.int8).view(bool)].tobytes())
    before = np.cumsum(lengths) - lengths
    index = np.repeat(starts - before, lengths) + np.arange(int(lengths.sum()))
    return bytes(raw[index].tobytes())


def numbers(raw: bytes, typename: str) -> np.ndarray[Any, Any]:
    """Big-endian bytes as a NumPy array in this machine's own order.

    One pass in C, whatever the length, and a copy of its own rather than a
    view: the bytes usually belong to a basket that is about to be dropped.
    """
    values = np.frombuffer(raw, on_disk(typename))
    if typename == "bool":
        return values != 0
    return values.astype(typename)


def as_datetime(packed: int) -> datetime.datetime:
    """ROOT's packed date word, which counts its years from 1995.

    >>> as_datetime(0x2C44F105)
    datetime.datetime(2006, 1, 2, 15, 4, 5)
    """
    return datetime.datetime(
        (packed >> 26) + 1995,
        (packed >> 22) & 0xF,
        (packed >> 17) & 0x1F,
        (packed >> 12) & 0x1F,
        (packed >> 6) & 0x3F,
        packed & 0x3F,
    )


BYTE_COUNT_MASK = 0x40000000
NEW_CLASS_TAG = 0xFFFFFFFF
CLASS_MASK = 0x80000000
MAP_OFFSET = 2
IS_REFERENCED = 1 << 4
#: The bits a ``TClonesArray`` carries that say how it wrote what it holds:
#: the first asks ROOT to write the objects field by field rather than one
#: after another, and the second takes that back for a class that cannot be.
BYPASS_STREAMER = 1 << 12
NO_MEMBER_WISE = 1 << 17

#: A ``TObject``'s version and the two words every object carries: what :meth:`Buffer.referenced`
#: reads in one go, every object of every tree being one.
_TOBJECT = struct.Struct(">HII")

_U8 = struct.Struct(">B")
_I8 = struct.Struct(">b")
_U16 = struct.Struct(">H")
_I16 = struct.Struct(">h")
_U32 = struct.Struct(">I")
_I32 = struct.Struct(">i")
_I64 = struct.Struct(">q")
_F32 = struct.Struct(">f")
_F64 = struct.Struct(">d")


class Listed(list[Any]):
    """What a ``TList`` held, with the option each entry was added under.

    The options are drawing rather than data - ``"hist same"`` beside a
    histogram in a pad, ``"lp"`` beside a legend's entry - so a list reads as
    the plain list it is, and only what draws a canvas looks at
    :attr:`options`, one string per entry and in the same order.
    """

    #: The option each entry was added under, ``""`` for none.
    options: list[str]
    #: The class each entry named itself as, ``""`` for a null or one pointed back at.
    classes: list[str]

    def __init__(self, items: Any = (), options: Any = (), classes: Any = ()) -> None:
        super().__init__(items)
        self.options = list(options)
        self.classes = list(classes)


class Clones(list[Any]):
    """What a ``TClonesArray`` held - each object's members - and the one class they are of,
    which the array names once at its front."""

    #: The class every object in the array is.
    classname: str

    def __init__(self, items: Any = (), classname: str = "TObject") -> None:
        super().__init__(items)
        self.classname = classname


class Written(list[Any]):
    """What a collection that streams itself held, and the name it was written with.

    Such a collection is not described by the file's streamer information: its
    own record begins with the name ROOT reads into the object it was handed,
    so handing the entries back as a plain list throws the name away - and an
    array read into a fresh object, a member a constructor left null, comes
    back nameless where ROOT's would be named.
    """

    #: The name on file, empty for a collection written with none.
    name: str

    def __init__(self, items: Any = (), name: str = "") -> None:
        super().__init__(items)
        self.name = str(name)


class Buffer:
    """A cursor over one decompressed ROOT record.

        >>> buf = Buffer(b"\\x00\\x2a")
        >>> buf.u16()
        42

    ``offset`` is the length of the key this payload came from, and ``refs``
    is the class and object map shared by everything read out of it.
    """

    __slots__ = ("data", "pos", "offset", "refs")

    def __init__(self, data: bytes, offset: int = 0) -> None:
        self.data = data
        self.offset = offset
        self.pos = offset
        self.refs: dict[int, object] = {}

    def __repr__(self) -> str:
        return f"<Buffer at {self.pos} of {self.offset + len(self.data)}>"

    @property
    def remaining(self) -> int:
        """Bytes left, in case a caller wants to stop before running out."""
        return len(self.data) - (self.pos - self.offset)

    def _take(self, form: struct.Struct) -> Any:
        """The one value ``form`` covers, read where the cursor is.

        Not the tuple :meth:`scalars` gives, which is for a run of members: a member read on
        its own is the commonest read there is - a whole object walked member by member
        takes a million of them a tree - and the tuple it would make costs as much again.
        """
        index = self.pos - self.offset
        if index + form.size > len(self.data):
            raise FormatError(f"the record ends {form.size} bytes before it said it would")
        self.pos += form.size
        return form.unpack_from(self.data, index)[0]

    def scalars(self, form: struct.Struct) -> tuple[Any, ...]:
        """Every value ``form`` covers, for a reader that knew them all when it was built.

        A member of a whole object is a number of a type the reader knew before it
        read anything, and a run of such members is written out one number after
        another; one struct over the run reads them all. So this gives the values
        back as the tuple the struct unpacks, and :meth:`_take` is the one value
        the typed readers above take out of it.
        """
        index = self.pos - self.offset
        if index + form.size > len(self.data):
            raise FormatError(f"the record ends {form.size} bytes before it said it would")
        self.pos += form.size
        return form.unpack_from(self.data, index)

    def u8(self) -> int:
        return int(self._take(_U8))

    def i8(self) -> int:
        return int(self._take(_I8))

    def u16(self) -> int:
        return int(self._take(_U16))

    def i16(self) -> int:
        return int(self._take(_I16))

    def u32(self) -> int:
        return int(self._take(_U32))

    def i32(self) -> int:
        return int(self._take(_I32))

    def i64(self) -> int:
        return int(self._take(_I64))

    def f32(self) -> float:
        return float(self._take(_F32))

    def f64(self) -> float:
        return float(self._take(_F64))

    def bool(self) -> bool:
        return self.u8() != 0

    def scalar(self, form: struct.Struct) -> Any:
        """One value of ``form``'s type, for a reader that knew the type when it was built.

        The typed readers above name a type each and say what to call the value
        they give back; a member of a whole object knows both already, and a
        struct it built once is then the whole of reading it - which for a
        member as small as one ``Float_t`` is most of what reading it costs.
        """
        return self._take(form)

    def run(self, count: int, typename: str) -> np.ndarray[Any, Any]:
        """``count`` numbers of one type, read where they are rather than copied out first."""
        dtype = on_disk(typename)
        index = self._span(count * dtype.itemsize)
        values = np.frombuffer(self.data, dtype, count, index)
        if typename == "bool":
            return values != 0
        return values.astype(typename)

    def take(self, count: int) -> bytes:
        """``count`` raw bytes."""
        index = self.pos - self.offset
        if index + count > len(self.data):
            raise FormatError(f"asked for {count} bytes with {self.remaining} left")
        self.pos += count
        return self.data[index : index + count]

    def string(self) -> str:
        """A ``TString``: one length byte, or ``255`` and then four."""
        length = self.u8()
        if length == 255:
            length = self.i32()
        return self.take(length).decode("utf-8", "surrogateescape")

    def cstring(self) -> str:
        """A NUL-terminated name, which is how class names are written."""
        index = self.pos - self.offset
        end = self.data.find(b"\x00", index)
        if end < 0:
            raise FormatError("a class name ran off the end of the record")
        self.pos += end - index + 1
        return self.data[index:end].decode("utf-8", "surrogateescape")

    # -- records ----------------------------------------------------------

    def header(self) -> tuple[int, int | None]:
        """The version of the record starting here, and where it ends.

        The end is ``None`` for the rare record written without a byte count,
        which is legal and means only that it cannot be skipped over.
        """
        start = self.pos
        count = self.u32()
        if count & BYTE_COUNT_MASK:
            end = self.pos + (count & ~BYTE_COUNT_MASK)
        else:
            self.pos = start
            end = None
        return self.u16(), end

    def resume(self, end: int | None) -> None:
        """Carry on after a record, wherever the caller stopped inside it.

        A record whose byte count was written says where it ends and the
        cursor goes there; the rare one written without a byte count leaves
        the cursor where the fields ran out, which is all that can be done.
        """
        if end is not None:
            self.pos = end

    def check_count(self, start: int, end: int | None) -> tuple[int, int] | None:
        """``TBufferFile::CheckByteCount``: the bytes a record took against its own count.

        ROOT checks, once an object has been streamed, that it moved the cursor
        exactly as far as the record's byte count said it would, and says so when
        it did not: a record read by a layout that does not describe its bytes
        takes the wrong number of them, and every object after it is read from the
        wrong place. Both numbers here count the byte-count word itself, as ROOT's
        message does. ``start`` is where that word began, which is what
        :meth:`header` was called at; the answer is ``None`` when the two agree, or
        when the record was written without a byte count and has no length to be
        checked against.
        """
        if end is None:
            return None
        consumed, recorded = self.pos - start, end - start
        return None if consumed == recorded else (consumed, recorded)

    def skip_record(self) -> int:
        """Step over a whole record, and say which version it was.

        This is how a reader that only cares about a tree walks past the
        drawing attributes nobody streaming data will ever look at.
        """
        version, end = self.header()
        if end is None:
            raise FormatError("a record without a byte count cannot be skipped")
        self.pos = end
        return version

    def tobject(self) -> tuple[int, int]:
        """A ``TObject``: its identifier and bits, once the version is past."""
        unique, bits, _process = self.referenced()
        return unique, bits

    def referenced(self) -> tuple[int, int, int | None]:
        """A ``TObject`` and, when a ``TRef`` may point at it - ``kIsReferenced`` - which of the
        file's ``ProcessID`` records numbered it; ``None`` for an object no reference names.

        The two words come back exactly as the file wrote them. The bits are not the reader's to
        change here: these members are read back out and written again as they came in, so a
        word with ROOT's ``kIsOnHeap|kNotDeleted`` in it would be a word the file never held.
        ROOT's reader sets those two bits on the *object* it fills (``TObject::Streamer``), which
        is :meth:`xrdroot.pyroot.core.objects.TObject._bitword` here.

        The identifier is the number alone for the same reason the process is handed back
        beside it rather than folded into its top byte: the index means nothing until the file
        it was written in is known, and it is that file's records that name the process.
        """
        _version, unique, bits = self.scalars(_TOBJECT)
        process = self.u16() if bits & IS_REFERENCED else None
        return unique, bits, process

    def named(self) -> tuple[str, str]:
        """A ``TNamed``: the name and title every ROOT object carries."""
        _version, end = self.header()
        self.tobject()
        name, title = self.string(), self.string()
        self.resume(end)
        return name, title

    # -- arrays -----------------------------------------------------------

    def i32s(self, count: int) -> list[int]:
        return list(struct.unpack_from(f">{count}i", self.data, self._span(count * 4)))

    def i64s(self, count: int) -> list[int]:
        return list(struct.unpack_from(f">{count}q", self.data, self._span(count * 8)))

    def _span(self, size: int) -> int:
        index = self.pos - self.offset
        if index + size > len(self.data):
            raise FormatError(f"an array of {size} bytes does not fit in the record")
        self.pos += size
        return index

    # -- references -------------------------------------------------------

    def any(self, classes: dict[str, Any]) -> Any:
        """Read whatever object comes next, by the class name in the stream.

        ``classes`` maps a ROOT class name to a callable taking this buffer.
        A name that is not in it is stepped over and comes back as its own
        string, so an unknown branch in a tree costs a skip rather than a
        failure - the caller decides whether it needed that one.
        """
        start = self.pos
        count = self.u32()
        if not count & BYTE_COUNT_MASK or count == NEW_CLASS_TAG:
            # No byte count: what was read is the tag itself, and there is no
            # length to step over if the class turns out to be unknown.
            tag, after, end = count, self.pos, None
        else:
            after = self.pos
            end = start + 4 + (count & ~BYTE_COUNT_MASK)
            tag = self.u32()

        if not tag & CLASS_MASK:
            if tag == 0:
                return None
            return self.refs.get(tag)  # an object written once and pointed at since

        if tag == NEW_CLASS_TAG:
            name = self.cstring()
            self.refs[after + MAP_OFFSET] = name
        else:
            name = str(self.refs.get(tag & ~CLASS_MASK, ""))
            if not name:
                raise FormatError(f"a class reference at {start} points nowhere")

        obj = self._classed(classes, name, start, end)
        self.refs[start + MAP_OFFSET] = obj
        return obj

    def _classed(
        self, classes: dict[str, Any], name: str, start: int, end: int | None
    ) -> Any:
        """The object ``name`` names, read by ``classes`` or stepped over if it is not there.

        A record that does come back is handed on with the name in front of it kept, because
        the members alone do not say what class they are the members of, and a pointer read
        by the type it was declared with may point at a class derived from that one.
        """
        reader = classes.get(name)
        if reader is None:
            if end is None:
                raise FormatError(f"a {name} at {start} was written with no length to skip")
            self.pos = end
            return name
        obj = reader(self)
        if isinstance(obj, dict) and not isinstance(obj, Named):
            return Named(obj, name)
        return obj

    def tlist(self, classes: dict[str, Any]) -> Listed:
        """A ``TList``: like a ``TObjArray``, but each entry carries an option.

        The option is a string written after the object it belongs to - how
        a pad was told to draw it - which is kept beside the list, as
        :attr:`Listed.options`, for the one reader that wants it.
        """
        version, end = self.header()
        if version <= 3:
            raise FormatError(f"a TList of version {version} is older than any that names itself")
        self.tobject()
        self.string()
        items = Listed()
        for _ in range(self.i32()):
            items.classes.append(self._next_class())
            items.append(self.any(classes))
            items.options.append(self.string())
        self.resume(end)
        return items

    def _next_class(self) -> str:
        """The class the object that comes next names, read ahead without moving on - so
        an entry read as its members still says what it was."""
        at = self.pos
        word = self.u32()
        tag = word if not word & BYTE_COUNT_MASK or word == NEW_CLASS_TAG else self.u32()
        name = ""
        if tag == NEW_CLASS_TAG:
            name = self.cstring()
        elif tag & CLASS_MASK:
            name = str(self.refs.get(tag & ~CLASS_MASK, ""))
        self.pos = at
        return name

    def objarray(self, classes: dict[str, Any]) -> Written:
        """A ``TObjArray``: the container ROOT keeps branches and leaves in.

        ROOT 2 wrote version 2 of it, with no ``TObject`` in front and every
        slot of its capacity written out, the empty ones as null pointers.
        """
        version, end = self.header()
        if version > 2:
            self.tobject()
        name = self.string() if version > 1 else ""
        size, _low = self.i32(), self.i32()
        items = [self.any(classes) for _ in range(size)]
        self.resume(end)
        return Written(items, name)

    def clones(self, classes: dict[str, Any], fields: Any = None) -> list[Any]:
        """A ``TClonesArray``: many objects of one class, the class written once.

        ROOT keeps a pool of objects of a single class and reuses it entry
        after entry, which is the whole point of the thing, so the class is
        named at the front of the array rather than in front of every object.
        A byte before each object says whether that slot was ever filled; an
        empty one comes back as ``None`` rather than shifting the rest along.

        An array told to bypass its streamer is written field by field instead
        - every object's first member, then every object's second - with no
        byte per slot, since an array written that way cannot have holes.
        ``fields`` is asked for the members of the held class, in order, when
        the array turns out to be one of those.
        """
        version, end = self.header()
        bits = self.tobject()[1] if version > 2 else 0
        if version > 1:
            self.string()  # the name the array was given
        held = self.string().partition(";")[0]  # the class, and the version of it
        count, _low = abs(self.i32()), self.i32()
        if bits & BYPASS_STREAMER and not bits & NO_MEMBER_WISE:
            return Clones(self._memberwise_clones(held, count, end, fields), held)
        return Clones(self._streamed_clones(held, count, end, classes), held)

    def _memberwise_clones(self, held: str, count: int, end: int | None, fields: Any) -> list[Any]:
        steps = fields(held) if fields is not None else None
        if steps is None:
            raise UnsupportedFeatureError(
                f"a TClonesArray of {held} was written field by field, and this "
                f"file's streamer information does not describe {held} well "
                f"enough to read it that way"
            )
        rows: list[Any] = [{} for _ in range(count)]
        for label, step in steps:
            for row in rows:
                row[label] = step(self, row)
        self._require_clone_end(held, count, end)
        self.resume(end)
        return rows

    def _require_clone_end(self, held: str, count: int, end: int | None) -> None:
        if end is not None and self.pos != end:
            raise FormatError(
                f"a TClonesArray of {count} {held} written field by field ended "
                f"{abs(end - self.pos)} bytes from where it said it would, so "
                f"what was read out of it cannot be trusted"
            )

    def _streamed_clones(
        self, held: str, count: int, end: int | None, classes: dict[str, Any]
    ) -> list[Any]:
        one = classes.get(held)
        if one is None:
            raise UnsupportedFeatureError(
                f"a TClonesArray holds {held}, which this file's streamer information "
                f"does not describe well enough to read one"
            )
        items = [one(self) if self.u8() else None for _ in range(count)]
        self.resume(end)
        return items
