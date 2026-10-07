"""The columns an object branch fills: members, counts, rows and whole objects.

Each is one of :mod:`.wtree`'s columns - it gathers entries into baskets and
sends them out by the same rule, with or without a table of where each
entry begins - whose entries are what a ``TBranchElement`` or
``TBranchObject`` holds rather than a leaf list's numbers. See
:mod:`.wbranch` for the branches they are the baskets of.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from .wpacking import BASIC, pack_member, packed_size
from .wtree import _Column, _rows_of, _Variable

__all__ = ["MemberColumn", "MemberRows", "CountColumn", "StreamedColumn", "VectorColumn"]

#: The bits ROOT sets in a byte count to say that is what it is.
BYTE_COUNT = 0x40000000
#: The version a ``std::vector`` writes in front of its values: ``TStreamerInfo``'s own
#: version, which every ROOT reads past, and which is 10 in the ROOT 6.40 this writer's
#: baskets are measured against - so they compress to the bytes ROOT's do.
VECTOR_VERSION = 10
#: The ``TLeafElement`` of a whole object or a count says its values are this wide: nothing.
NO_WIDTH = 0


class _ElementColumn:
    """What every column of an object branch says of itself, whatever its entries are."""

    __slots__ = ()

    def _leaf_class(self) -> tuple[str, str, int, bool]:
        return "TLeafElement", "", NO_WIDTH, False

    @property
    def typename(self) -> str:
        return "object"

    @property
    def described(self) -> str:
        return "object"


class MemberColumn(_ElementColumn, _Column):
    """One number of a split object per entry, packed as its streamer type packs it."""

    __slots__ = ("stype", "comment")

    def __init__(self, name: str, stype: int, comment: str, basket_size: int) -> None:
        self.stype, self.comment = stype, comment
        super().__init__(name, "d", 1, basket_size)
        self.size = packed_size(stype, comment)

    def _leaf_class(self) -> tuple[str, str, int, bool]:
        return "TLeafElement", "", BASIC[self.stype][1], False

    @property
    def typename(self) -> str:
        return BASIC[self.stype][2]

    @property
    def described(self) -> str:
        return self.typename

    def pack(self, value: Any) -> bytes:
        return pack_member(self.stype, self.comment, np.asarray([value], dtype=np.float64))

    def pack_many(self, values: Any) -> tuple[bytes, None]:
        given = np.asarray(values)
        if given.ndim != 1:
            raise ValueError(
                f"{self.name!r} takes one number per entry, and these are shaped {given.shape}"
            )
        return pack_member(self.stype, self.comment, given), None


class MemberRows(_ElementColumn, _Variable):
    """One member of every object in a split collection: a row of numbers per entry."""

    __slots__ = ("stype", "comment", "width")

    def __init__(self, name: str, stype: int, comment: str, basket_size: int) -> None:
        self.stype, self.comment = stype, comment
        super().__init__(name, "d", basket_size)
        self.width = packed_size(stype, comment)

    def _leaf_class(self) -> tuple[str, str, int, bool]:
        return "TLeafElement", "", BASIC[self.stype][1], False

    def pack(self, value: Any) -> bytes:
        return pack_member(self.stype, self.comment, np.asarray(value).reshape(-1))

    def pack_many(self, values: Any) -> tuple[bytes, np.ndarray[Any, Any]]:
        content, counts = _rows_of(self.name, values)
        return pack_member(self.stype, self.comment, content), counts * self.width


class CountColumn(_ElementColumn, _Variable):
    """How many objects a split collection holds each entry, the most of them kept."""

    __slots__ = ("maximum",)

    def __init__(self, name: str, basket_size: int) -> None:
        super().__init__(name, "i", basket_size)
        self.maximum = 0

    def pack(self, value: Any) -> bytes:
        return struct.pack(">i", int(value))

    def pack_many(self, values: Any) -> tuple[bytes, np.ndarray[Any, Any]]:
        counts = np.asarray(values, dtype=np.int64).reshape(-1)
        return bytes(counts.astype(">i4").tobytes()), np.full(len(counts), 4, np.int64)

    def note(self, raw: bytes, sizes: np.ndarray[Any, Any] | None) -> None:
        found = np.frombuffer(raw, ">i4")
        self.maximum = max(self.maximum, int(found.max(initial=0)))


class StreamedColumn(_ElementColumn, _Variable):
    """Whole objects, each entry the bytes their class's streamer made of one."""

    __slots__ = ()

    def __init__(self, name: str, basket_size: int) -> None:
        super().__init__(name, "B", basket_size)

    def pack(self, value: Any) -> bytes:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise ValueError(
                f"{self.name!r} holds objects already streamed, and this entry is a "
                f"{type(value).__name__} rather than their bytes"
            )
        return bytes(value)

    def pack_many(self, values: Any) -> tuple[bytes, np.ndarray[Any, Any]]:
        pieces = [self.pack(value) for value in values]
        return b"".join(pieces), np.asarray([len(piece) for piece in pieces], dtype=np.int64)


class VectorColumn(StreamedColumn):
    """A ``std::vector`` of numbers per entry: byte count, version, length, then the values."""

    __slots__ = ("dtype",)

    def __init__(self, name: str, code: str, basket_size: int) -> None:
        super().__init__(name, basket_size)
        self.dtype = np.dtype(code).newbyteorder(">")

    def pack(self, value: Any) -> bytes:
        raw, _sizes = self.pack_many([value])
        return raw

    def pack_many(self, values: Any) -> tuple[bytes, np.ndarray[Any, Any]]:
        content, counts = _rows_of(self.name, values)
        data = content.astype(self.dtype).view(np.uint8)
        sizes = 10 + counts * self.dtype.itemsize
        heads = np.zeros((len(counts), 10), dtype=np.uint8)
        heads[:, :4] = ((sizes - 4) | BYTE_COUNT).astype(">u4").view(np.uint8).reshape(-1, 4)
        heads[:, 4:6] = np.full(len(counts), VECTOR_VERSION, ">u2").view(np.uint8).reshape(-1, 2)
        heads[:, 6:] = counts.astype(">u4").view(np.uint8).reshape(-1, 4)
        return _interleaved(heads, data, counts * self.dtype.itemsize), sizes


def _interleaved(heads: np.ndarray[Any, Any], data: np.ndarray[Any, Any], lengths: Any) -> bytes:
    """Each entry's header and then its values, for every entry, in one pass."""
    starts = np.cumsum(lengths) - lengths
    out = np.empty(heads.size + data.size, dtype=np.uint8)
    begins = np.arange(len(lengths)) * heads.shape[1] + starts
    out[(begins[:, None] + np.arange(heads.shape[1])).reshape(-1)] = heads.reshape(-1)
    mask = np.ones(len(out), dtype=bool)
    mask[(begins[:, None] + np.arange(heads.shape[1])).reshape(-1)] = False
    out[mask] = data
    return bytes(out.tobytes())
