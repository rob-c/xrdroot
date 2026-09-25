"""Baskets moved from one tree into another as they are: what makes ``hadd`` fast.

A tree is mostly its baskets, and a basket is a compressed block of entries
that does not know which file it is in. So a tree merged from many files
need not decode a single value: each basket's record is read, and written
again unchanged behind a new key, and only the branch's tables of where its
baskets are and which entries each holds are new. That is ROOT's "fast
clone", and it costs a read and a write per basket rather than a decode, a
fill and a compression per entry.

Two things can stop a basket going across byte for byte. The first is
compression: a file told to hold its trees in another algorithm or level
than the input's gets its baskets decompressed and compressed again - still
without decoding an entry. The second is the key. A basket of entries of
different sizes carries, behind its entries, a table of where each begins,
counted from the start of its *key*; a key of another length would move
every one of them. ROOT writes every basket's key in its wide form so that
this never happens, and so is a basket copied here: in the width its key
had, which keeps its length. Only a key that has to grow - a basket from a
small-keyed file landing past 2 GB, or a tree renamed on the way - has its
table moved along with it, the payload unpacked for that and packed again.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ..buffer import Buffer
from ..compression import decompress
from ..errors import UnsupportedFeatureError
from ..file import Key, Source
from ..writer import WIDE

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from ..objects import BranchRecord
    from ..tree import Basket
    from ..wtree import WritableTree, _Column

__all__ = ["Moved", "move_baskets"]

#: What a basket keeps in its own key after the parts every key has: its
#: version, the size of buffer it was built with, the size of one entry (or
#: of its table of entries), how many entries it holds, where they end, and
#: a flag saying the entries are behind the key rather than in it.
HEAD = struct.Struct(">hiiiiB")
#: Enough of a basket's record to read its key, for one left where it is.
KEY_WINDOW = 256


class Moved(NamedTuple):
    """How a tree's baskets went across: as they were, packed again, or left in place."""

    verbatim: int
    repacked: int
    in_place: int


class _Record(NamedTuple):
    """One basket's record, read raw: its key, its own fields, and its payload."""

    key: Key
    extra: bytes
    fields: tuple[int, int, int, int, int]
    body: bytes


def _read(source: Source, seek: int, nbytes: int) -> _Record:
    """A basket's record as it is on file, its key's own fields picked out."""
    raw = source.read(seek, nbytes)
    buf = Buffer(raw)
    key = Key(buf)
    at = buf.pos
    version, bufsize, nevsize = buf.i16(), buf.i32(), buf.i32()
    if nevsize < 0:
        raise UnsupportedFeatureError(
            f"a basket of {key.name!r} in {source.name} was written with ROOT's I/O "
            f"features, which change what its key and its table of entries mean; "
            f"merge with fast=False and its entries are read and written anew"
        )
    nevbuf, last = buf.i32(), buf.i32()
    return _Record(
        key, raw[at : key.keylen], (version, bufsize, nevsize, nevbuf, last), raw[key.keylen :]
    )


def _shifted(table: bytes, shift: int) -> bytes:
    """A basket's table of entry offsets, each moved ``shift`` bytes along.

    The table is its count and then the offsets; a slot ROOT left at zero,
    the one past the last entry, stays zero, and anything after the table
    goes along as it was.
    """
    if not table:
        return b""
    (count,) = struct.unpack_from(">i", table)
    offsets = np.frombuffer(table, ">i4", count, 4).astype(np.int64)
    moved = np.where(offsets != 0, offsets + shift, 0).astype(">i4")
    return table[:4] + bytes(moved.tobytes()) + table[4 + 4 * count :]


class _Mover:
    """Where one column's baskets are going, and how they are to get there."""

    def __init__(self, tree: WritableTree, column: _Column, verbatim: bool) -> None:
        self.tree = tree
        self.column = column
        self.verbatim = verbatim
        self.out = tree._file
        self.counts = [0, 0, 0]

    def _key_length(self, extra: int, wide: bool) -> int:
        return self.out._key_length("TBasket", self.column.name, self.tree.name, extra, wide)

    def keyed(self, source: Source, seek: int, nbytes: int, entries: int, tabled: bool) -> None:
        """A basket with a record of its own: across as it is, if nothing stops it."""
        record = _read(source, seek, nbytes)
        wide = record.key.version > WIDE
        shift = self._key_length(len(record.extra), wide) - record.key.keylen
        self._size(record.fields[1])
        if self.verbatim and not (shift and tabled):
            extra = record.extra
            if shift:
                version, bufsize, nevsize, nevbuf, last = record.fields
                extra = HEAD.pack(version, bufsize, nevsize, nevbuf, last + shift, 0)
            place, size = self._write(record.body, extra, wide, record.key.objlen)
            self.counts[0] += 1
        else:
            data = record.body
            if record.key.compressed:
                data = decompress(record.body, record.key.objlen)
            place, size = self._repack(data, record.key.keylen, record.fields, tabled, wide)
            self.counts[1] += 1
        self.tree._adopt(self.column, place, entries, size)

    def inline(self, basket: Basket, tabled: bool) -> None:
        """A basket kept inside its branch's record, which has to become one of its own."""
        payload = basket.data
        if tabled:
            table = struct.pack(">i", len(basket.offsets))
            table += np.asarray(basket.offsets, ">i4").tobytes()
            payload = payload + table
        fields = (3, self.column.basket_size, basket.nevsize, basket.nevbuf, basket.last)
        place, size = self._repack(payload, basket.keylen, fields, tabled, False)
        self.counts[1] += 1
        self.tree._adopt(self.column, place, basket.nevbuf, size)

    def in_place(self, source: Source, seek: int, nbytes: int, entries: int) -> None:
        """A basket already in the file being written: pointed at, not copied."""
        key = Key(Buffer(source.read(seek, min(nbytes, KEY_WINDOW))))
        self.counts[2] += 1
        self.tree._adopt(self.column, (seek, nbytes), entries, key.keylen + key.objlen)

    def _size(self, bufsize: int) -> None:
        """The branch's basket size is the size its baskets were built with, if they say one."""
        self.column.basket_size = bufsize if bufsize > 0 else self.column.basket_size

    def _write(
        self, payload: bytes, extra: bytes, wide: bool, objlen: int | None
    ) -> tuple[tuple[int, int], int]:
        keylen = self._key_length(len(extra), wide)
        place = self.out._put(
            "TBasket",
            self.column.name,
            self.tree.name,
            payload,
            0,
            listed=False,
            extra=extra,
            objlen=objlen,
            wide=wide,
        )
        return place, keylen + (len(payload) if objlen is None else objlen)

    def _repack(
        self,
        data: bytes,
        keylen: int,
        fields: tuple[int, int, int, int, int],
        tabled: bool,
        wide: bool,
    ) -> tuple[tuple[int, int], int]:
        """A basket's unpacked payload, its table moved to fit its new key, packed again."""
        version, bufsize, nevsize, nevbuf, last = fields
        shift = self._key_length(HEAD.size, wide) - keylen
        entries = data[: last - keylen]
        table = _shifted(data[last - keylen :], shift) if tabled else data[last - keylen :]
        extra = HEAD.pack(version, bufsize, nevsize, nevbuf, last + shift, 0)
        return self._write(entries + table, extra, wide, None)


def move_baskets(
    tree: WritableTree,
    column: _Column,
    record: BranchRecord,
    source: Source,
    *,
    verbatim: bool,
    in_place: bool = False,
) -> Moved:
    """Every basket of one branch, into the column of ``tree`` it is to be part of.

    ``verbatim`` says the baskets may go across byte for byte, their
    compression being what the file being written wants - or being kept as
    it is, whatever that is; otherwise each is packed again. ``in_place``
    says the source is the very file being written, whose baskets need only
    be pointed at.
    """
    mover = _Mover(tree, column, verbatim)
    tabled = record.entry_offset_len > 0
    bounds = record.basket_entry
    held = len(record.baskets)
    for index in range(max(len(record.basket_seek), held)):
        entries = bounds[index + 1] - bounds[index]
        if index < held:
            mover.inline(record.baskets[index], tabled)
        elif in_place:
            mover.in_place(source, record.basket_seek[index], record.basket_bytes[index], entries)
        else:
            seek, nbytes = record.basket_seek[index], record.basket_bytes[index]
            mover.keyed(source, seek, nbytes, entries, tabled)
    return Moved(*mover.counts)

