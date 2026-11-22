"""A C++ member's numbers as ``TStreamerInfo`` writes them: by its streamer type.

Most members are a big-endian copy of what the class holds. Two are not:
a ``Double32_t`` and a ``Float16_t`` are squeezed by the recipe in their
declaration's trailing comment, which :func:`xrdroot.interp._range` reads
the same way for writing as for reading - ``TStreamerElement::GetRange``'s
rules. A range, ``//[xmin,xmax,nbits]``, makes each value a whole number of
steps across it, four bytes; a bit count alone, ``//[0,0,nbits]``, keeps the
float's exponent byte and that many bits of its mantissa, three bytes; and
nothing at all is a ``float`` for a ``Double32_t``, or twelve bits of mantissa
for a ``Float16_t``. The arithmetic is ``TBufferFile::WriteDouble32``'s and
``WriteFloat16``'s, done on a whole column at once.
"""

from __future__ import annotations

import struct
from functools import cache
from typing import Any

import numpy as np

from .errors import UnsupportedFeatureError
from .interp import _range

__all__ = ["BASIC", "DOUBLE32", "FLOAT16", "pack_member", "pack_one", "packed_size"]

#: ``TStreamerInfo``'s ``kDouble32`` and ``kFloat16``: the two packed float types.
DOUBLE32, FLOAT16 = 9, 19

#: Each plain streamer type: the NumPy type it is on file, the ``fLenType`` its
#: leaf carries, and the name ``TLeaf::GetTypeName`` gives it.
BASIC: dict[int, tuple[str, int, str]] = {
    1: ("i1", 1, "Char_t"),
    2: (">i2", 2, "Short_t"),
    3: (">i4", 4, "Int_t"),
    4: (">i8", 8, "Long_t"),
    5: (">f4", 4, "Float_t"),
    6: (">i4", 4, "Int_t"),
    8: (">f8", 8, "Double_t"),
    11: ("u1", 1, "UChar_t"),
    12: (">u2", 2, "UShort_t"),
    13: (">u4", 4, "UInt_t"),
    14: (">u8", 8, "ULong_t"),
    15: (">u4", 4, "UInt_t"),
    16: (">i8", 8, "Long64_t"),
    17: (">u8", 8, "ULong64_t"),
    18: ("u1", 1, "Bool_t"),
    DOUBLE32: (">f4", 4, "Double32_t"),
    FLOAT16: (">f4", 2, "Float16_t"),
}

#: The streamer types that are one plain number on file, which a run of members can be
#: gathered into one ``struct`` over: every type ``BASIC`` describes but the two packed floats,
#: whose recipe is their declaration's comment and not their streamer type.
_PLAIN_STYPES = frozenset(BASIC) - {DOUBLE32, FLOAT16}

#: A float written as its exponent byte and a cut-down mantissa, three bytes.
TRUNCATED = np.dtype([("exp", "u1"), ("man", ">u2")])

#: ``struct``'s code for one value of a plain streamer type, by its NumPy kind and width.
#: The two agree on the bytes - both are big-endian, width for width - so a member that is
#: a single number can skip NumPy, whose per-call cost dwarfs the four bytes it produces.
CODES: dict[str, dict[int, str]] = {
    "i": {1: "b", 2: "h", 4: "i", 8: "q"},
    "u": {1: "B", 2: "H", 4: "I", 8: "Q"},
    "f": {4: "f", 8: "d"},
}

#: One ``struct.Struct`` per plain streamer type, made the first time a member needs it.
PACKERS: dict[int, struct.Struct] = {}


@cache
def _recipe(title: str) -> tuple[float, float, float]:
    """``xmin``, ``xmax`` and the factor a packed member's comment asks for."""
    found = _range(title)
    if found is None:
        raise UnsupportedFeatureError(
            f"a packed float's comment says {title!r}, which is not a range ROOT's "
            f"GetRange reads, so there is no knowing how to squeeze its values"
        )
    return found


def packed_size(stype: int, title: str) -> int:
    """How many bytes one value of a member of streamer type ``stype`` takes on file."""
    if stype not in (DOUBLE32, FLOAT16):
        return int(np.dtype(BASIC[stype][0]).itemsize)
    xmin, _xmax, factor = _recipe(title)
    if factor or (stype == DOUBLE32 and not int(xmin)):
        return 4
    return 3


def _packer(stype: int) -> struct.Struct:
    """The ``struct`` that writes one value of the plain streamer type ``stype``."""
    found = PACKERS.get(stype)
    if found is None:
        dtype = np.dtype(BASIC[stype][0])
        order = ">" if dtype.itemsize > 1 else ""
        found = PACKERS[stype] = struct.Struct(order + CODES[dtype.kind][dtype.itemsize])
    return found


def _wrapped(values: Any, target: str) -> Any:
    """``values`` as ``target``, wrapping a number too wide or negative for it, as C does.

    NumPy 2 wraps such a number where NumPy 1 refuses it outright, and what ROOT's file holds is
    the wrapped one either way - so the mask is written here rather than taken from whichever
    NumPy is installed.
    """
    given = np.asarray(values)
    kind = np.dtype(target)
    if kind.kind not in "iu":
        return given.astype(target)
    wrapped = np.asarray([_c_integer(value, kind) for value in given.reshape(-1)], dtype=kind)
    return wrapped.reshape(given.shape)


def _c_integer(value: Any, kind: np.dtype) -> int:
    """``value`` as C holds it in an integer of ``kind``'s width: truncated, then wrapped."""
    bits = kind.itemsize * 8
    whole = int(value) & ((1 << bits) - 1)
    low = whole - (1 << bits) if whole >= 1 << (bits - 1) else whole
    return int(low) if kind.kind == "i" else int(whole)


def _whole(stype: int, values: Any) -> bytes | None:
    """One value written through ``struct``, or ``None`` where NumPy must convert it
    instead: a float into an integer member truncates there and raises here, and a number
    too wide for the member wraps there, which is what ROOT on file really holds."""
    if not isinstance(values, np.ndarray) or values.ndim != 1 or len(values) != 1:
        return None
    try:
        return _packer(stype).pack(values[0])
    except (struct.error, TypeError, OverflowError):
        # Python 3.14 raises OverflowError where 3.13 raised struct.error, and both mean
        # the same thing here: this number does not fit that member, so NumPy converts it.
        return None


def pack_one(stype: int, title: str, value: Any) -> bytes:
    """One value of a member, as :func:`pack_member` writes a column holding one.

    A member that is a single number is nearly every member of a class of numbers -
    nine of them in each of the ten million records ``TClass::Streamer`` writes in
    roottest's own benchmark - so the number goes to a ``struct`` as it comes and
    never through NumPy, whose per-call cost is several times the four bytes it
    produces. A number that does not fit its member is written as it always was,
    truncated or wrapped as ROOT on file holds it.
    """
    if stype not in (DOUBLE32, FLOAT16):
        try:
            return _packer(stype).pack(value)
        except (struct.error, TypeError, OverflowError):
            pass
    return pack_member(stype, title, np.asarray([value]))


def pack_member(stype: int, title: str, values: Any) -> bytes:
    """A column of one member's values, as its streamer type writes them."""
    if stype not in (DOUBLE32, FLOAT16):
        one = _whole(stype, values)
        if one is not None:
            return one
        return bytes(_wrapped(values, BASIC[stype][0]).tobytes())
    given = np.asarray(values)
    xmin, xmax, factor = _recipe(title)
    if factor:
        clipped = np.clip(given.astype(np.float64), xmin, xmax)
        steps = np.trunc(0.5 + factor * (clipped - xmin))
        return bytes(steps.astype(np.uint64).astype(">u4").tobytes())
    nbits = int(xmin) if int(xmin) else (0 if stype == DOUBLE32 else 12)
    if not nbits:
        return bytes(given.astype(">f4").tobytes())
    return bytes(_truncated(given, nbits).tobytes())


def _truncated(values: np.ndarray[Any, Any], nbits: int) -> np.ndarray[Any, Any]:
    """The exponent byte and ``nbits`` of mantissa, rounded, with the sign above them."""
    floats = values.astype(np.float32)
    bits = floats.view(np.int32).astype(np.int64)
    out = np.zeros(len(floats), dtype=TRUNCATED)
    out["exp"] = (bits >> 23) & 0xFF
    man = ((bits >> (23 - nbits - 1)) & ((1 << (nbits + 1)) - 1)) + 1
    man = man >> 1
    man = np.where(man & (1 << nbits), (1 << nbits) - 1, man)
    out["man"] = np.where(floats < 0, man | (1 << (nbits + 1)), man)
    return out
