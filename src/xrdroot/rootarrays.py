"""Primitive storage descriptions shared by ROOT arrays and histogram cells.

These describe the numerical layout itself. Streamer interpretation and file
access belong to their consumers, so a histogram needs neither to know the
width and NumPy type of its cells.
"""

from __future__ import annotations

from .cxx import Prim

__all__ = ["ARRAYS"]

#: The arrays ROOT's own kit keeps numbers in - the contents of a histogram is
#: one - each of which streams itself as a count and then that many values,
#: with no record round it and nothing of the base class it inherits.
ARRAYS = {
    "TArrayC": Prim("int8", "b", 1),
    "TArrayS": Prim("int16", "h", 2),
    "TArrayI": Prim("int32", "i", 4),
    "TArrayL": Prim("int64", "q", 8),
    "TArrayL64": Prim("int64", "q", 8),
    "TArrayF": Prim("float32", "f", 4),
    "TArrayD": Prim("float64", "d", 8),
}
