"""``RooNaNPacker``: a NaN that carries how bad the evaluation that made it was.

Where a density cannot be computed - a negative normalisation, a negative
value - RooFit returns a NaN with a ``float`` in its low 32 bits, tagged in
the bits above, so that the badness travels through the arithmetic that
follows (a NaN's payload survives ``+`` and ``*``) up to the likelihood,
which hands Minuit the worst value seen so far plus ten times the badness.
This is that packing, on NumPy arrays, bit for bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["pack", "unpack", "tagged"]

#: The tag in the bits above the payload, and the mask that finds it.
MAGIC_TAG = 0x321AB00000000
MAGIC_MASK = 0x3FFFF00000000
#: A quiet NaN's bits.
QUIET_NAN = 0x7FF8000000000000


def pack(payload: Any) -> Any:
    """``packFloatIntoNaN``: NaNs carrying ``payload`` as ``float``s."""
    floats = np.asarray(payload, dtype=np.float32)
    low = floats.view(np.uint32).astype(np.uint64)
    bits = np.uint64(QUIET_NAN | MAGIC_TAG) | low
    found = np.asarray(bits, dtype=np.uint64).view(np.float64)
    return found if found.ndim else float(found)


def tagged(values: Any) -> Any:
    """``isNaNWithPayload``: which values are NaNs made by :func:`pack`."""
    array = np.asarray(values, dtype=np.float64)
    bits = array.view(np.uint64)
    return np.isnan(array) & ((bits & np.uint64(MAGIC_MASK)) == np.uint64(MAGIC_TAG))


def unpack(values: Any) -> Any:
    """``unpackNaN``: each value's payload, or zero for a value that carries none."""
    if np.ndim(values) == 0:
        return float(unpack(np.array([values], dtype=np.float64))[0])
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    low = (array.view(np.uint64) & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.float32)
    with np.errstate(invalid="ignore"):
        return np.where(tagged(array), low.astype(np.float64), 0.0)
