"""Column encodings: how a page's bytes become values, and back.

Every column is a run of elements of one simple type, little-endian, cut into
pages. The *split* encodings rearrange a page so that every element's first
byte comes first, then every second byte, and so on - nothing is gained by
itself, but the high bytes of similar numbers sit together and compress far
better. On top of that, offsets are stored as differences from the one
before (*delta*), and signed integers with their sign folded into the lowest
bit (*zigzag*), so that small numbers stay small either side of zero. Each
of these is undone a page at a time, because each is applied a page at a
time.

A boolean is one bit, least significant first. ``Real32Trunc`` keeps the top
bits of a float and drops the rest; ``Real32Quant`` spreads the range the
column declares over the integers its bits can count. Both are bit-packed end
to end with nothing between the elements, and both come back as ``float32``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import FormatError, UnsupportedFeatureError

__all__ = ["Encoding", "ENCODINGS", "decode", "encode", "stored_size"]


class Encoding:
    """One column type: its name, what its elements are, and how they are packed."""

    __slots__ = ("code", "name", "dtype", "split", "delta", "zigzag")

    def __init__(
        self,
        code: int,
        name: str,
        dtype: str,
        split: bool = False,
        delta: bool = False,
        zigzag: bool = False,
    ) -> None:
        self.code = code
        self.name = name
        #: What one element is once decoded, little-endian.
        self.dtype: np.dtype[Any] = np.dtype(dtype)
        self.split = split
        self.delta = delta
        self.zigzag = zigzag

    def __repr__(self) -> str:
        return f"<Encoding {self.name}>"

    @property
    def index(self) -> bool:
        """Whether this is a column of collection offsets."""
        return "Index" in self.name

    @property
    def packed(self) -> bool:
        """Whether the elements are bit-packed rather than whole bytes."""
        return self.name in ("Bit", "Real32Trunc", "Real32Quant")


#: The dtype of the Switch column: a variant's element index, then its tag.
SWITCH = np.dtype([("index", "<u8"), ("tag", "<u4")])

#: Every column type of the specification, by the number the file gives it.
ENCODINGS: dict[int, Encoding] = {
    code: Encoding(code, *spec)  # type: ignore[arg-type]
    for code, spec in {
        0x00: ("Bit", "?"),
        0x01: ("Byte", "u1"),
        0x02: ("Char", "i1"),
        0x03: ("Int8", "i1"),
        0x04: ("UInt8", "u1"),
        0x05: ("Int16", "<i2"),
        0x06: ("UInt16", "<u2"),
        0x07: ("Int32", "<i4"),
        0x08: ("UInt32", "<u4"),
        0x09: ("Int64", "<i8"),
        0x0A: ("UInt64", "<u8"),
        0x0B: ("Real16", "<f2"),
        0x0C: ("Real32", "<f4"),
        0x0D: ("Real64", "<f8"),
        0x0E: ("Index32", "<u4"),
        0x0F: ("Index64", "<u8"),
        0x10: ("Switch", SWITCH.str),
        0x11: ("SplitInt16", "<i2", True, False, True),
        0x12: ("SplitUInt16", "<u2", True),
        0x13: ("SplitInt32", "<i4", True, False, True),
        0x14: ("SplitUInt32", "<u4", True),
        0x15: ("SplitInt64", "<i8", True, False, True),
        0x16: ("SplitUInt64", "<u8", True),
        0x17: ("SplitReal16", "<f2", True),
        0x18: ("SplitReal32", "<f4", True),
        0x19: ("SplitReal64", "<f8", True),
        0x1A: ("SplitIndex32", "<u4", True, True),
        0x1B: ("SplitIndex64", "<u8", True, True),
        0x1C: ("Real32Trunc", "<f4"),
        0x1D: ("Real32Quant", "<f4"),
    }.items()
}
ENCODINGS[0x10].dtype = SWITCH  # a structured type has no string that round-trips


def encoding(code: int) -> Encoding:
    """The column type numbered ``code``, or a refusal naming the number."""
    found = ENCODINGS.get(code)
    if found is None:
        raise UnsupportedFeatureError(
            f"column type {code:#04x} is newer than the RNTuple specification this "
            f"reader follows (1.0), which ends at 0x1d"
        )
    return found


def stored_size(kind: Encoding, bits: int, count: int) -> int:
    """How many bytes ``count`` elements take in a page before compression."""
    if kind.packed:
        return (count * bits + 7) // 8
    return count * int(kind.dtype.itemsize)


def _bitfield(raw: bytes, bits: int, count: int) -> np.ndarray[Any, Any]:
    """``count`` unsigned integers of ``bits`` bits each, packed end to end."""
    flat = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little")[: count * bits]
    weights = np.left_shift(np.uint64(1), np.arange(bits, dtype=np.uint64))
    value: np.ndarray[Any, Any] = flat.reshape(count, bits).astype(np.uint64) @ weights
    return value


def _packbits(values: np.ndarray[Any, Any], bits: int) -> bytes:
    """The other way: integers of ``bits`` bits each, packed end to end."""
    shifts = np.arange(bits, dtype=np.uint64)
    flat = ((values.astype(np.uint64)[:, None] >> shifts) & np.uint64(1)).astype(np.uint8)
    return bytes(np.packbits(flat.reshape(-1), bitorder="little").tobytes())


def _real32(kind: Encoding, bits: int, low: float, high: float, raw: bytes, count: int) -> Any:
    """The two lossy float encodings, widened back into float32."""
    if kind.name == "Real32Trunc":
        if not 10 <= bits <= 31:
            raise FormatError(f"a Real32Trunc column says it keeps {bits} bits, not 10 to 31")
        kept = _bitfield(raw, bits, count).astype(np.uint32) << np.uint32(32 - bits)
        return kept.view(np.float32)
    if not 1 <= bits <= 32:
        raise FormatError(f"a Real32Quant column says it keeps {bits} bits, not 1 to 32")
    steps = _bitfield(raw, bits, count).astype(np.float64)
    return (low + (high - low) * steps / float((1 << bits) - 1)).astype(np.float32)


def _unsplit(raw: bytes, dtype: np.dtype[Any], count: int) -> np.ndarray[Any, Any]:
    width = dtype.itemsize
    planes = np.frombuffer(raw, np.uint8, count=width * count).reshape(width, count)
    return np.ascontiguousarray(planes.T).view(dtype).reshape(count)


def _unzigzag(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    unsigned = values.view(values.dtype.str.replace("i", "u"))
    back = (unsigned >> 1) ^ (np.zeros_like(unsigned) - (unsigned & 1))
    return back.view(values.dtype)


def decode(column: Any, raw: bytes, count: int) -> np.ndarray[Any, Any]:
    """One page's ``count`` elements, undone from however the column packed them.

    ``column`` is the :class:`~.schema.Column` describing the page, whose
    bits and range the packed encodings need.
    """
    kind = encoding(column.type)
    if len(raw) < stored_size(kind, column.bits, count):
        raise FormatError(
            f"a page of RNTuple column {column.id} holds {len(raw)} bytes, too few "
            f"for the {count} {kind.name} elements it is said to"
        )
    if kind.name == "Bit":
        flat = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little")
        return flat[:count].astype(bool)
    if kind.packed:
        return np.asarray(_real32(kind, column.bits, column.low, column.high, raw, count))
    if kind.split:
        values = _unsplit(raw, kind.dtype, count)
    else:
        values = np.frombuffer(raw, kind.dtype, count=count)
    if kind.delta:
        return np.cumsum(values, dtype=values.dtype)
    if kind.zigzag:
        return _unzigzag(values)
    return values


def encode(kind: Encoding, values: np.ndarray[Any, Any]) -> bytes:
    """One page's elements, packed the way ``kind`` stores them.

    The writer here only ever asks for the encodings it writes - booleans,
    whole bytes, plain little-endian numbers and the split ones - which is
    :func:`decode` backwards and in the other order: the differences and the
    zigzag go first, and the bytes are rearranged last.
    """
    if kind.name == "Bit":
        return _packbits(values.astype(np.uint8), 1)
    data = np.ascontiguousarray(values, dtype=kind.dtype)
    if kind.delta:
        data = np.diff(data, prepend=data.dtype.type(0))
    elif kind.zigzag:
        unsigned = data.dtype.str.replace("i", "u")
        data = ((data << 1) ^ (data >> (8 * data.dtype.itemsize - 1))).view(unsigned)
    if not kind.split:
        return bytes(data.tobytes())
    planes = data.view(np.uint8).reshape(len(data), data.dtype.itemsize)
    return bytes(np.ascontiguousarray(planes.T).tobytes())
