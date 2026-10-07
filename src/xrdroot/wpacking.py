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

from typing import Any

import numpy as np

from .errors import UnsupportedFeatureError
from .interp import _range

__all__ = ["BASIC", "DOUBLE32", "FLOAT16", "pack_member", "packed_size"]

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

#: A float written as its exponent byte and a cut-down mantissa, three bytes.
TRUNCATED = np.dtype([("exp", "u1"), ("man", ">u2")])


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


def pack_member(stype: int, title: str, values: Any) -> bytes:
    """A column of one member's values, as its streamer type writes them."""
    given = np.asarray(values)
    if stype not in (DOUBLE32, FLOAT16):
        return bytes(given.astype(BASIC[stype][0]).tobytes())
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
