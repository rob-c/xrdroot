"""Another file's description of its classes, carried into the file being written.

A record copied from one file to another as it was - a class this library
has no layout for, or one it would rather not decode and encode again - is
readable in its new file only if that file describes the class the way the
old one did, version and all. So the old file's ``TStreamerInfo`` entries go
with it.

They cannot simply be cut out of the old list. ROOT writes a class's name in
full the first time it streams one and refers back to that place after, so
the tenth ``TStreamerBase`` in a list is a pointer to the first; cut out, it
points at nothing. Each entry is therefore walked the way ``Buffer.any``
would read it and written again with every class named in full and every
byte count made anew, which is what :func:`~xrdroot.writer._info_entries`
writes for the classes this library carries, and so can go into any list
anywhere. An entry holding a reference to an *object* met earlier - which
ROOT does not write in a streamer list - cannot be made to stand alone, and
is left behind rather than guessed at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..buffer import BYTE_COUNT_MASK, CLASS_MASK, MAP_OFFSET, NEW_CLASS_TAG, Buffer
from ..errors import FormatError
from ..file import Key, Source
from ..writer import WBuffer

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from ..writer import WritableFile

__all__ = ["carry", "read_entries"]

#: Enough of the record at ``fSeekInfo`` to read its key.
KEY_WINDOW = 512


class _Loose(Exception):
    """An entry that points back at an object, and so cannot stand on its own."""


class _Relocator:
    """One walk of a streamer list, writing every entry again self-contained.

    The walk has to be the whole list in order, because a class named in
    full in one entry is referred to by place in the entries after it.
    """

    def __init__(self, buf: Buffer) -> None:
        self.buf = buf
        #: The class names written in full so far, by the place they stand for.
        self.names: dict[int, str] = {}

    def slot(self) -> bytes:
        """One place an object may be: nothing, or an object naming its class."""
        buf = self.buf
        start = buf.pos
        count = buf.u32()
        if count == 0:
            return bytes(4)
        if not count & BYTE_COUNT_MASK or count == NEW_CLASS_TAG:
            raise _Loose(f"an object at {start} is a reference to one met before")
        end = start + 4 + (count & ~BYTE_COUNT_MASK)
        name = self._classname()
        out = WBuffer()
        index = out.tag(name)
        out.raw(self._body(name, end))
        out.end(index)
        buf.pos = end
        return bytes(out.data)

    def _classname(self) -> str:
        buf = self.buf
        at = buf.pos
        tag = buf.u32()
        if tag == NEW_CLASS_TAG:
            name = buf.cstring()
            self.names[at + MAP_OFFSET] = name
            return name
        name = self.names.get(tag & ~CLASS_MASK, "") if tag & CLASS_MASK else ""
        if not name:
            raise FormatError(f"the class of the object at {at} is a reference to nowhere")
        return name

    def _body(self, name: str, end: int) -> bytes:
        """What follows an object's class: walked where objects nest, else as it is."""
        walk = {"TStreamerInfo": self._info, "TObjArray": self._array, "TList": self._list}
        if name in walk:
            return walk[name]()
        return self.buf.take(end - self.buf.pos)

    def _info(self) -> bytes:
        """A ``TStreamerInfo``: its name, checksum and version, then its elements."""
        buf = self.buf
        version, end = buf.header()
        out = WBuffer()
        index = out.start(version)
        out.raw(self._verbatim_record())  # the TNamed, which holds no objects
        out.raw(buf.take(8))  # the checksum and the class version
        out.raw(self.slot())
        out.raw(self._rest(end))
        out.end(index)
        return bytes(out.data)

    def _array(self) -> bytes:
        """A ``TObjArray``: its object part, name, size and lower bound, then each slot."""
        buf = self.buf
        version, end = buf.header()
        out = WBuffer()
        index = out.start(version)
        start = buf.pos
        buf.tobject()
        buf.string()
        size = buf.i32()
        buf.i32()
        out.raw(buf.data[start - buf.offset : buf.pos - buf.offset])
        for _ in range(size):
            out.raw(self.slot())
        out.raw(self._rest(end))
        out.end(index)
        return bytes(out.data)

    def _list(self) -> bytes:
        """A ``TList``: like an array, but each slot followed by its option string."""
        buf = self.buf
        version, end = buf.header()
        out = WBuffer()
        index = out.start(version)
        start = buf.pos
        buf.tobject()
        buf.string()
        count = buf.i32()
        out.raw(buf.data[start - buf.offset : buf.pos - buf.offset])
        for _ in range(count):
            out.raw(self.slot())
            out.raw(self._option())
        out.raw(self._rest(end))
        out.end(index)
        return bytes(out.data)

    def _option(self) -> bytes:
        buf = self.buf
        start = buf.pos
        buf.string()
        return buf.data[start - buf.offset : buf.pos - buf.offset]

    def _verbatim_record(self) -> bytes:
        buf = self.buf
        start = buf.pos
        buf.skip_record()
        return buf.data[start - buf.offset : buf.pos - buf.offset]

    def _rest(self, end: int | None) -> bytes:
        """Whatever a record holds after the fields walked, up to its end."""
        if end is None:
            return b""
        return self.buf.take(end - self.buf.pos)


def _info_key(entry: bytes) -> tuple[str, int] | None:
    """Which class and version a self-contained entry describes, if it is a ``TStreamerInfo``."""
    buf = Buffer(entry)
    buf.u32()
    buf.u32()
    if buf.cstring() != "TStreamerInfo":
        return None
    buf.header()
    name, _title = buf.named()
    buf.u32()  # the checksum
    return name, buf.i32()


def read_entries(source: Source) -> dict[tuple[str, int], bytes]:
    """Every ``TStreamerInfo`` a file carries, by class and version, each self-contained.

    Each comes back as a list entry - the object naming its class in full,
    then the empty option every entry carries - ready to go into a streamer
    list of any file. An entry that cannot be made to stand alone is left
    out, with every entry after it, which may lean on it; so is anything in
    the list that is not a class description, such as the schema evolution
    rules ROOT keeps at its end.
    """
    seek, nbytes = source.info
    if seek <= 0 or nbytes <= 0:
        return {}
    key = Key(Buffer(source.read(seek, min(nbytes, KEY_WINDOW))))
    buf = Buffer(key.payload(source), key.keylen)
    walker = _Relocator(buf)
    _version, end = buf.header()
    buf.tobject()
    buf.string()
    found: dict[tuple[str, int], bytes] = {}
    for _ in range(buf.i32()):
        try:
            entry = walker.slot() + b"\x00"
        except _Loose:
            break  # everything after a loose entry may lean on it, so stop there
        buf.string()  # the option it was added under
        described = _info_key(entry)
        if described is not None:
            found.setdefault(described, entry)
    buf.resume(end)
    return found


def carry(
    file: WritableFile, source: Source, cache: dict[str, dict[tuple[str, int], bytes]]
) -> None:
    """Have ``file`` describe every class ``source`` describes, as ``source`` does.

    ``cache`` keeps what each source's list came to, by the source's name, so
    a file many records are copied from is walked once. What the file being
    written already carries for a class and version is kept: the first file
    to describe one decides how.
    """
    entries = cache.get(source.name)
    if entries is None:
        entries = cache[source.name] = read_entries(source)
    for described, entry in entries.items():
        file._carried.setdefault(described, entry)
