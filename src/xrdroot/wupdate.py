"""Adding to a ROOT file that is already there.

An update is a file being written that happens to start out full. What was
there stays exactly where it was: new objects, trees and directories go on
the end, a name written again becomes its next cycle, and at the close the
directories that gained something get new key lists, the streamer
information gains whatever classes the new objects need, the free list gains
the records those replaced, and the header is written last of all to point
at the lot. Until that last write the file still says what it said before,
and a ``with`` block that raises cuts it back to its old length - so a failed
update leaves the file byte for byte as it was.

Appending rather than fitting records into the file's gaps is deliberate.
Filling gaps would save a little space, and would mean writing over bytes the
old header still counts as free while the update is unfinished; an update
that then failed could not give them back.
"""

from __future__ import annotations

import struct
from functools import partial
from typing import IO, Any, NamedTuple

from xrdclient.url import parse

from .buffer import BYTE_COUNT_MASK, CLASS_MASK, MAP_OFFSET, NEW_CLASS_TAG, Buffer
from .compression import CODES
from .errors import FormatError, UnsupportedFeatureError
from .file import Key, Source
from .winfo import INFOS
from .writer import (
    DIRECTORY_BYTES,
    MAGIC,
    WIDE,
    WIDE_FILE,
    WritableDirectory,
    WritableFile,
    _check_compression,
    _closure,
    _info_entries,
    _Output,
    _seekable,
    _streamers,
)

__all__ = ["update"]

#: What ``compression`` says when it is left alone: carry on as the file does.
AS_FILED = "file"
#: The name of each algorithm by the number the header's fCompress holds.
#: ROOT's 0 means its global default, and 3 the zlib-alike it wrote before
#: 6.08; zlib is what both come to for anything new.
ALGORITHMS = {code: name for name, code in CODES.items()}
#: The classes a key names a directory by, old and new.
DIRECTORIES = ("TDirectory", "TDirectoryFile")
#: Enough of the record at ``fSeekInfo`` to read its key.
KEY_WINDOW = 512


class _Header(NamedTuple):
    """The fields of a file's header that an update reads or carries on."""

    version: int
    begin: int
    end: int
    seek_free: int
    nbytes_free: int
    nfree: int
    nbytes_name: int
    codes: int
    seek_info: int
    nbytes_info: int
    uuid: bytes


def _read_header(source: Source) -> _Header:
    """The header, small or wide, of the file being updated."""
    raw = source.read(0, 100)
    if not raw.startswith(MAGIC):
        raise FormatError(
            f"{source.name} does not start with {MAGIC!r}: it is not a ROOT file, "
            f"and an update will not make one of it"
        )
    version, begin = struct.unpack_from(">ii", raw, 4)
    form = ">qqiiiBiqi" if version >= WIDE_FILE else ">iiiiiBiii"
    end, seek_free, nbytes_free, nfree, nbytes_name, _units, codes, seek_info, nbytes_info = (
        struct.unpack_from(form, raw, 12)
    )
    at = 12 + struct.calcsize(form) + 2  # past the UUID's version
    fields = (end, seek_free, nbytes_free, nfree, nbytes_name, codes, seek_info, nbytes_info)
    return _Header(version, begin, *fields, raw[at : at + 16])


class _Record(NamedTuple):
    """A directory's record: what an update keeps of it, and what it rewrites."""

    created: int
    nbytes_keys: int
    nbytes_name: int
    seek_dir: int
    seek_parent: int
    seek_keys: int
    uuid: bytes


def _read_record(source: Source, seek: int) -> _Record:
    """The ``TDirectory`` record at ``seek``, small or wide."""
    buf = Buffer(source.read(seek, DIRECTORY_BYTES))
    wide = buf.u16() > WIDE
    created = buf.u32()
    buf.u32()  # when it was last changed, which this update is about to be
    nbytes_keys, nbytes_name = buf.i32(), buf.i32()
    seek_dir, seek_parent, seek_keys = (buf.i64() if wide else buf.i32() for _ in range(3))
    buf.u16()  # the version of the UUID
    record = _Record(
        created, nbytes_keys, nbytes_name, seek_dir, seek_parent, seek_keys, buf.take(16)
    )
    if record.seek_keys <= 0 or record.nbytes_keys <= 0:
        raise FormatError("a directory in this file has no key list to read")
    return record


def _key_list(source: Source, seek: int, nbytes: int) -> tuple[Key, list[tuple[Key, bytes]]]:
    """A directory's key list: the key in front of it, then each key and its bytes.

    The bytes are kept as they were written, so the keys already in a
    directory go into its new key list exactly as they came out of the old.
    """
    buf = Buffer(source.read(seek, nbytes))
    listed = Key(buf)
    found = []
    for _ in range(buf.i32()):
        start = buf.pos
        key = Key(buf)
        found.append((key, buf.data[start : buf.pos]))
    return listed, found


def _adopt(directory: WritableDirectory, source: Source, record: _Record) -> None:
    """Make ``directory`` the one this record describes, with all it holds.

    Its key list is only rewritten if something is put in it; the
    directories inside it are only read if something is put in them.
    """
    directory._ctime = record.created
    directory._uuid = record.uuid
    directory._seek = record.seek_dir
    directory._parent_seek = record.seek_parent
    directory._nname = record.nbytes_name
    listed, keys = _key_list(source, record.seek_keys, record.nbytes_keys)
    directory._listed_as = (listed.classname, listed.name, listed.title)
    directory._replaced = (record.seek_keys, record.nbytes_keys)
    directory._dirty = False
    for key, raw in keys:
        directory._keys.append(raw)
        if key.classname in DIRECTORIES:
            directory._found[key.name] = partial(_open_directory, source, directory, key)
        else:
            directory._cycles[key.name] = max(directory._cycles.get(key.name, 0), key.cycle)


def _open_directory(source: Source, parent: WritableDirectory, key: Key) -> WritableDirectory:
    """A directory already in the file, ready for more, the first time it is asked for."""
    record = _read_record(source, key.seek_key + key.keylen)
    path = f"{parent.path}/{key.name}" if parent.path else key.name
    found = WritableDirectory(
        parent._file, path, key.title, record.seek_dir, record.seek_parent, record.nbytes_name
    )
    _adopt(found, source, record)
    return found


def _read_free(
    source: Source, header: _Header
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    """The gaps the file already has, and where its free list itself is.

    The last entry, the open end of the file, is left out: the update's own
    close says where that now starts.
    """
    if header.seek_free <= 0 or header.nbytes_free <= 0:
        return [], None
    buf = Buffer(source.read(header.seek_free, header.nbytes_free))
    Key(buf)
    gaps = []
    for _ in range(header.nfree):
        wide = buf.u16() > WIDE
        first, last = (buf.i64(), buf.i64()) if wide else (buf.i32(), buf.i32())
        if first < header.end:
            gaps.append((first, last))
    return gaps, (header.seek_free, header.nbytes_free)


class _Infos:
    """The streamer information a file already has, kept as the bytes it came in.

    New classes are added after the old ones rather than the old ones being
    rewritten, so everything the file said about its classes it goes on
    saying exactly. The one thing that can change under them is the length
    of the key in front: every class reference in the list counts from the
    start of that key, so when the list moves past 2 GB and its key grows,
    each reference is found here and moved along with it.
    """

    def __init__(self, payload: bytes, keylen: int, title: str) -> None:
        self.data = payload
        self.keylen = keylen
        self.title = title
        #: Every class and version the list already describes.
        self.known: set[tuple[str, int]] = set()
        self._refs: list[int] = []
        self._names: dict[int, str] = {}
        buf = Buffer(payload, keylen)
        self._count_at, self._count = self._list(buf)
        self._end = buf.pos

    def merged(self, entries: bytes, count: int, keylen: int) -> bytes:
        """The list with ``count`` more entries, for a key ``keylen`` long."""
        data = bytearray(self.data[: self._end - self.keylen])
        for at in self._refs:
            index = at - self.keylen
            (tag,) = struct.unpack_from(">I", data, index)
            struct.pack_into(">I", data, index, tag + keylen - self.keylen)
        struct.pack_into(">i", data, self._count_at - self.keylen, self._count + count)
        data += entries
        struct.pack_into(">I", data, 0, (len(data) - 4) | BYTE_COUNT_MASK)
        return bytes(data)

    def _list(self, buf: Buffer) -> tuple[int, int]:
        """A ``TList``: where its count is, and the count, having been through it."""
        _version, end = buf.header()
        buf.tobject()
        buf.string()
        at = buf.pos
        count = buf.i32()
        for _ in range(count):
            self._slot(buf)
            buf.string()  # the option the entry was added under
        buf.resume(end)
        return at, count

    def _array(self, buf: Buffer) -> None:
        """A ``TObjArray``, which is where a ``TStreamerInfo`` keeps its elements."""
        _version, end = buf.header()
        buf.tobject()
        buf.string()
        size = buf.i32()
        buf.i32()  # the lower bound
        for _ in range(size):
            self._slot(buf)
        buf.resume(end)

    def _info(self, buf: Buffer) -> None:
        """A ``TStreamerInfo``: which class and version, then its elements."""
        _version, end = buf.header()
        name, _title = buf.named()
        buf.u32()  # the checksum
        self.known.add((name, buf.i32()))
        self._slot(buf)
        buf.resume(end)

    def _slot(self, buf: Buffer) -> None:
        """One place any object may be: nothing, an earlier object, or a new one.

        This follows ``Buffer.any`` step for step, noting where every tag
        that refers back to an earlier place is, rather than reading objects.
        """
        start = buf.pos
        count = buf.u32()
        if count & BYTE_COUNT_MASK and count != NEW_CLASS_TAG:
            self._counted(buf, start, count)
        elif count & CLASS_MASK:
            raise FormatError(
                f"an object at {start} in this file's streamer information was written "
                f"with no length, which ROOT never does there, so it cannot be stepped over"
            )
        elif count:
            self._refs.append(start)  # an object met before, named by where it was

    def _counted(self, buf: Buffer, start: int, count: int) -> None:
        """An object with its length in front: its class, then inside it if need be."""
        end = start + 4 + (count & ~BYTE_COUNT_MASK)
        at = buf.pos
        tag = buf.u32()
        if tag == NEW_CLASS_TAG:
            name = buf.cstring()
            self._names[at + MAP_OFFSET] = name
        else:
            self._refs.append(at)
            name = self._names.get(tag & ~CLASS_MASK, "") if tag & CLASS_MASK else ""
            if not name:
                raise FormatError(f"the class of the object at {start} is a reference to nowhere")
        walk = {"TStreamerInfo": self._info, "TObjArray": self._array, "TList": self._list}
        if name in walk:
            walk[name](buf)
        buf.resume(end)


def _read_infos(source: Source, header: _Header) -> tuple[_Infos, tuple[int, int] | None]:
    """The file's streamer information, and where it is, if it has any."""
    if header.seek_info <= 0 or header.nbytes_info <= 0:
        return _Infos(_streamers({}), 0, "Doubly linked list"), None
    key = Key(Buffer(source.read(header.seek_info, min(header.nbytes_info, KEY_WINDOW))))
    infos = _Infos(key.payload(source), key.keylen, key.title)
    return infos, (header.seek_info, header.nbytes_info)


def _setting(
    compression: str | None, level: int | None, codes: int
) -> tuple[str | None, int | None]:
    """The algorithm and level new records use: as asked, or as the file does."""
    if compression != AS_FILED:
        _check_compression(compression)
        return compression, level
    if not codes % 100:
        return None, None
    return ALGORITHMS.get(codes // 100, "zlib"), codes % 100 if level is None else level


class UpdatedFile(WritableFile):
    """A ROOT file that was already there, being added to.

    It is a :class:`~.writer.WritableFile` in every way that matters to
    whoever is writing: the same mapping, the same trees, the same
    directories. What differs is only what it starts from and how it ends.
    """

    def __init__(
        self,
        handle: IO[bytes],
        name: str,
        owned: bool,
        compression: str | None,
        level: int | None,
    ) -> None:
        if not _seekable(handle):
            raise ValueError(
                f"{name} cannot seek, and an update has to read what is there and "
                f"write over the header last; open it as a file that can"
            )
        source = Source(handle, name, owned=False)
        header = _read_header(source)
        _require_whole(handle, header, name)
        self._start(handle, owned, *_setting(compression, level, header.codes))
        at = header.begin + header.nbytes_name
        _require_room(source, header)
        record = _read_record(source, at)
        WritableDirectory.__init__(self, self, "", "", header.begin, 0, header.nbytes_name)
        _adopt(self, source, record)
        self.name = name
        self._version = header.version % WIDE_FILE
        self._file_uuid = header.uuid
        self._gaps, free = _read_free(source, header)
        self._release(free)
        self._infos, self._info_at = _read_infos(source, header)
        self._out = _Output.resume(handle, source.read(0, at + DIRECTORY_BYTES), header.end)

    def _write_streamers(self) -> tuple[int, int]:
        """The streamer information, rewritten only if new classes need describing."""
        known = self._infos.known
        names = [name for name in _closure(self._used) if (name, INFOS[name][1]) not in known]
        if not names:
            return self._info_at or (0, 0)
        title = self._infos.title
        keylen = self._key_length("TList", "StreamerInfo", title)
        payload = self._infos.merged(_info_entries(names), len(names), keylen)
        self._release(self._info_at)
        return self._put("TList", "StreamerInfo", title, payload, 1, listed=False)


def _require_whole(handle: IO[bytes], header: _Header, name: str) -> None:
    """Refuse a file whose length is not where its header says it ends."""
    length = handle.seek(0, 2)
    if length != header.end:
        raise FormatError(
            f"{name} is {length} bytes long, and its header says it ends at "
            f"{header.end}: it is still being written, or was cut short or added "
            f"to by something else, and an update will not guess which"
        )


def _require_room(source: Source, header: _Header) -> None:
    """Refuse a file whose top directory has no room for the wide record."""
    key = Key(Buffer(source.read(header.begin, header.nbytes_name)))
    room = key.nbytes - header.nbytes_name
    if room < DIRECTORY_BYTES:
        raise UnsupportedFeatureError(
            f"{source.name} keeps its top directory's record in {room} bytes, where "
            f"ROOT since version 4 leaves {DIRECTORY_BYTES} - the room an update needs "
            f"to rewrite it in; copy what the file holds into a new one instead"
        )


def update(
    target: Any,
    *,
    compression: str | None = AS_FILED,
    level: int | None = None,
    config: Any = None,
) -> WritableFile:
    """A ROOT file that is already at ``target``, opened to be added to.

        >>> with xrdroot.update("counts.root") as f:          # doctest: +SKIP
        ...     f["signal"] = newer        # the next cycle of what was there
        ...     f["runs/4712/signal"] = h  # into a directory, made if need be

    It is written to exactly as :func:`~.writer.create`'s file is, and what
    was there stays readable: new records go on the end, the directories
    that gained something get new key lists, and the header is written last.
    A ``with`` block that raises leaves the file byte for byte as it was.

    ``target`` is a local path, a URL of any scheme this library writes, or
    an already-open binary file that can read, write and seek, used as it is
    and left open. ``compression`` carries on with whatever the file says
    unless told otherwise, and then takes what :func:`~.writer.create` takes.
    """
    handle, name, owned = _open(target, config)
    try:
        return UpdatedFile(handle, name, owned, compression, level)
    except BaseException:
        if owned:
            handle.close()  # a file that will not update should not leak the handle
        raise


def _open(target: Any, config: Any) -> tuple[IO[bytes], str, bool]:
    """A handle on ``target`` to read and write, what to call it, and whose it is."""
    if hasattr(target, "write"):
        return target, getattr(target, "name", "<file>"), False
    url = parse(target)
    if url.is_local:
        return open(url.path, "r+b"), str(target), True
    from xrdclient.io import open_url

    return open_url(url, "r+b", config=config), str(target), True
