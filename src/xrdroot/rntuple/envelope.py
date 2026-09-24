"""The framing RNTuple wraps its metadata in: envelopes, frames and locators.

Nothing here is the ROOT streamer format. An RNTuple keeps its description in
*envelopes* of its own - little-endian throughout, a type and a length at the
front, an XXH3 at the back - and inside them *frames*: a signed size, negative
for a list, followed by what the frame holds. A reader is meant to step over a
frame by the size it declares rather than by what it understood of it, which
is what lets a file written by a newer ROOT, with fields this reader has never
heard of, still be read.

The one big-endian part is the anchor, the ``ROOT::RNTuple`` object a
directory lists: it is an ordinary ROOT key, and says where the header and
footer envelopes are.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

from ..compression import decompress
from ..errors import FormatError, UnsupportedFeatureError
from .checksum import checksum, verifying

if TYPE_CHECKING:
    from ..file import Source

__all__ = ["Cursor", "Builder", "Anchor", "Link", "envelope", "open_envelope", "read_blob"]

#: What each envelope type is called, by the number it carries.
KINDS = {1: "header", 2: "footer", 3: "page list"}
#: The type ID in the low sixteen bits of an envelope's first word.
HEADER, FOOTER, PAGE_LIST = 1, 2, 3
#: The first word and the XXH3 at the end: what every envelope costs.
ENVELOPE_OVERHEAD = 16
#: The anchor's fields: four versions of two bytes, seven sizes of eight.
ANCHOR_FIELDS = struct.Struct(">4H7Q")
#: The class version ROOT writes the anchor with, and this reader reads.
ANCHOR_VERSION = 2
#: The width of each offset a payload split across keys ends its first with.
CHUNK_OFFSET = 8
#: The feature flags this reader knows, of which it supports none: each is
#: something a file does that would be misread if it were ignored.
FEATURES = {0: "deferred columns inside a collection, which merging RNTuples makes"}


class Link:
    """Where a block is and how big it is either way: an envelope link.

    ``size`` is what is on disk, ``length`` what it holds once decompressed;
    the two are equal exactly when it was stored as it is.
    """

    __slots__ = ("offset", "size", "length")

    def __init__(self, offset: int, size: int, length: int) -> None:
        self.offset = offset
        self.size = size
        self.length = length

    def __repr__(self) -> str:
        return f"<Link {self.size} bytes at {self.offset}, {self.length} unpacked>"


class Cursor:
    """A walk through one envelope's payload, little-endian as RNTuple writes it.

    ``end`` is the end of whatever is being read - the envelope before its
    checksum, or a frame - and nothing is read past it: a frame that runs
    into its neighbour is a damaged file, and says so.
    """

    __slots__ = ("data", "pos", "end", "what")

    def __init__(self, data: bytes, pos: int, end: int, what: str) -> None:
        self.data = data
        self.pos = pos
        self.end = end
        self.what = what

    def take(self, size: int) -> bytes:
        if size < 0 or self.pos + size > self.end:
            raise FormatError(
                f"the RNTuple {self.what} envelope ends at byte {self.end}, where {size} "
                f"bytes were wanted from byte {self.pos}: it is damaged or truncated"
            )
        chunk = self.data[self.pos : self.pos + size]
        self.pos += size
        return chunk

    def unpack(self, form: str) -> Any:
        return struct.unpack("<" + form, self.take(struct.calcsize("<" + form)))[0]

    def u16(self) -> int:
        return int(self.unpack("H"))

    def u32(self) -> int:
        return int(self.unpack("I"))

    def u64(self) -> int:
        return int(self.unpack("Q"))

    def i32(self) -> int:
        return int(self.unpack("i"))

    def i64(self) -> int:
        return int(self.unpack("q"))

    def f64(self) -> float:
        return float(self.unpack("d"))

    def string(self) -> str:
        return self.take(self.u32()).decode("utf-8", "surrogateescape")

    def frame(self, listed: bool) -> tuple[int, int]:
        """Open a frame: where it ends, and for a list how many items it holds."""
        start = self.pos
        size = self.i64()
        if (size < 0) != listed:
            kinds = ("record", "list")
            raise FormatError(
                f"the RNTuple {self.what} envelope has a {kinds[size < 0]} frame at byte "
                f"{start} where a {kinds[listed]} frame belongs"
            )
        size = abs(size)
        least = 12 if listed else 8
        if size < least or start + size > self.end:
            raise FormatError(
                f"the RNTuple {self.what} envelope has a frame at byte {start} declaring "
                f"{size} bytes, which does not fit where it is"
            )
        count = self.u32() if listed else 0
        return start + size, count

    def seek(self, end: int) -> None:
        """Step to the end of a frame, whatever in it went unread."""
        if end < self.pos:
            raise FormatError(
                f"the RNTuple {self.what} envelope has a frame ending at byte {end}, "
                f"before what it holds was over at byte {self.pos}"
            )
        self.pos = end

    def locator(self) -> tuple[int, int]:
        """Where a block is on disk: its size and its offset from the start of the file.

        A negative size is a locator of another kind, negated: its type in the
        top byte and its own length in the low sixteen bits, as ROOT's
        ``SerializeLocator`` writes one. The only kind a file holds is the
        large one, a 64-bit size and offset for a block past two gigabytes.
        """
        size = self.i32()
        if size >= 0:
            return size, self.u64()
        start, head = self.pos - 4, -size
        kind = head >> 24
        if kind != 1:
            raise UnsupportedFeatureError(
                f"the RNTuple {self.what} envelope points at a block with a locator of "
                f"type {kind}, which is not the file kind; only ROOT files are read here"
            )
        found = self.u64(), self.u64()
        self.seek(start + (head & 0xFFFF))
        return found

    def link(self) -> Link:
        length = self.u64()
        size, offset = self.locator()
        return Link(offset, size, length)

    def flags(self) -> list[int]:
        """The feature flags, refusing any this reader does not know by name."""
        words = [self.u64()]
        while words[-1] >> 63:
            words.append(self.u64())
        for index, word in enumerate(words):
            _refuse_features(self.what, index, word & ~(1 << 63))
        return words


def _refuse_features(what: str, index: int, word: int) -> None:
    for bit in range(63):
        if word >> bit & 1:
            number = index * 63 + bit
            meaning = FEATURES.get(number, "a feature newer than this reader")
            raise UnsupportedFeatureError(
                f"the RNTuple {what} sets feature flag {number} - {meaning} - and a "
                f"reader that does not do a feature must not read past it"
            )


class Builder:
    """An envelope's payload being written: the other half of :class:`Cursor`.

    A frame's size is known only once what it holds is written, so
    :meth:`record` and :meth:`list` leave room and :meth:`close` fills it in.
    """

    __slots__ = ("data", "_lists")

    def __init__(self) -> None:
        self.data = bytearray()
        self._lists: set[int] = set()

    def pack(self, form: str, *values: Any) -> None:
        self.data += struct.pack("<" + form, *values)

    def string(self, text: str) -> None:
        raw = text.encode("utf-8", "surrogateescape")
        self.pack("I", len(raw))
        self.data += raw

    def record(self) -> int:
        mark = len(self.data)
        self.pack("q", 0)
        return mark

    def list(self, count: int) -> int:
        mark = self.record()
        self.pack("I", count)
        self._lists.add(mark)
        return mark

    def close(self, mark: int) -> None:
        size = len(self.data) - mark
        struct.pack_into("<q", self.data, mark, -size if mark in self._lists else size)

    def empty_list(self) -> None:
        self.close(self.list(0))

    def locator(self, size: int, offset: int) -> None:
        self.pack("iQ", size, offset)

    def link(self, link: Link) -> None:
        self.pack("Q", link.length)
        self.locator(link.size, link.offset)


def envelope(kind: int, payload: bytes) -> bytes:
    """A payload wrapped as an envelope: type and length first, XXH3 last."""
    body = struct.pack("<Q", kind | (len(payload) + ENVELOPE_OVERHEAD) << 16) + payload
    return body + struct.pack("<Q", checksum(body))


def open_envelope(raw: bytes, kind: int) -> Cursor:
    """Check an envelope is the kind it should be and whole, and walk into it."""
    what = KINDS[kind]
    if len(raw) < ENVELOPE_OVERHEAD:
        raise FormatError(f"the RNTuple {what} envelope is {len(raw)} bytes, too few to be one")
    word = struct.unpack_from("<Q", raw)[0]
    if word & 0xFFFF != kind or word >> 16 != len(raw):
        raise FormatError(
            f"the RNTuple {what} envelope is labelled as type {word & 0xFFFF} of "
            f"{word >> 16} bytes, where a {what} of {len(raw)} bytes was expected"
        )
    stored = struct.unpack_from("<Q", raw, len(raw) - 8)[0]
    if verifying() and checksum(raw[:-8]) != stored:
        raise FormatError(
            f"the RNTuple {what} envelope does not match the checksum stored with it, "
            f"so the file is damaged there"
        )
    return Cursor(raw, 8, len(raw) - 8, what)


def read_blob(source: Source, offset: int, size: int, max_key: int) -> bytes:
    """``size`` bytes of a block at ``offset``, following it across keys if it was split.

    ROOT keeps no key bigger than the anchor's maximum key size, and writes a
    bigger block as several: the first holds as much of the block as fits
    with, at its end, the offsets of the keys holding the rest.
    """
    if max_key == 0 or size <= max_key:
        return source.read(offset, size)
    chunks = _chunks(size, max_key)
    head = max_key - (chunks - 1) * CHUNK_OFFSET
    first = source.read(offset, max_key)
    places = struct.unpack(f"<{chunks - 1}Q", first[head:])
    pieces = [first[:head]]
    got = head
    for place in places:
        take = min(max_key, size - got)
        pieces.append(source.read(place, take))
        got += take
    return b"".join(pieces)


def _chunks(size: int, max_key: int) -> int:
    """How many keys ROOT splits a block of ``size`` across: ``ComputeNumChunks``."""
    count = -(-size // max_key)
    spare = (max_key - size % max_key) % max_key
    if (count - 1) * CHUNK_OFFSET > spare:
        count += 1
    return count


def read_block(source: Source, link: Link, max_key: int) -> bytes:
    """A block as it was before it was stored: decompressed if it was compressed."""
    raw = read_blob(source, link.offset, link.size, max_key)
    return raw if link.size == link.length else decompress(raw, link.length)


class Anchor:
    """The ``ROOT::RNTuple`` object: which version wrote it, and where the rest is.

    >>> anchor.header, anchor.footer          # doctest: +SKIP
    (<Link 283 bytes at 290, 623 unpacked>, <Link 176 bytes at 1066, 316 unpacked>)
    """

    __slots__ = ("version", "header", "footer", "max_key")

    def __init__(self, version: tuple[int, ...], header: Link, footer: Link, max_key: int) -> None:
        self.version = version
        self.header = header
        self.footer = footer
        self.max_key = max_key

    @classmethod
    def parse(cls, payload: bytes, name: str) -> Anchor:
        """Read one out of its key's payload: a counted record, then its checksum."""
        if len(payload) < 6 + ANCHOR_FIELDS.size + 8:
            raise FormatError(f"{name!r} is too short to be an RNTuple anchor")
        count, version = struct.unpack_from(">IH", payload)
        end = 4 + (count & ~0x40000000)
        if version != ANCHOR_VERSION or end + 8 > len(payload) or end < 6 + ANCHOR_FIELDS.size:
            raise UnsupportedFeatureError(
                f"{name!r} is an RNTuple anchor of class version {version} and "
                f"{end - 6} bytes, where version {ANCHOR_VERSION} of 64 is what this reads"
            )
        fields = ANCHOR_FIELDS.unpack_from(payload, 6)
        stored = struct.unpack_from(">Q", payload, end)[0]
        if verifying() and checksum(payload[6:end]) != stored:
            raise FormatError(
                f"{name!r} is an RNTuple anchor that does not match its checksum, "
                f"so the file is damaged there"
            )
        return cls._checked(fields, name)

    @classmethod
    def _checked(cls, fields: tuple[int, ...], name: str) -> Anchor:
        if fields[0] != 1:
            raise UnsupportedFeatureError(
                f"{name!r} is an RNTuple of format epoch {fields[0]}; this reader "
                f"reads epoch 1, the first public release, and nothing else"
            )
        header = Link(fields[4], fields[5], fields[6])
        footer = Link(fields[7], fields[8], fields[9])
        return cls(tuple(fields[:4]), header, footer, fields[10])

    def payload(self) -> bytes:
        """The anchor as its key holds it: a counted record, then its checksum."""
        fields = ANCHOR_FIELDS.pack(
            *self.version,
            self.header.offset,
            self.header.size,
            self.header.length,
            self.footer.offset,
            self.footer.size,
            self.footer.length,
            self.max_key,
        )
        head = struct.pack(">IH", (len(fields) + 2) | 0x40000000, ANCHOR_VERSION)
        return head + fields + struct.pack(">Q", checksum(fields))
