"""The columns of a FITS table, binary or ASCII, read as ``TFITSHDU`` keeps them.

``TFITSHDU`` reads every column into one of four kinds: strings, numbers,
arrays of a fixed length, and arrays whose length changes from row to row
(the ``P`` and ``Q`` columns, whose values lie in the heap after the rows).
Every number is a double, ``TSCALn`` and ``TZEROn`` applied, a logical
being 1 or 0; a column of a type it does not read keeps no values and no
kind, as ROOT's keeps none.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from .cards import number
from .hdus import HDU

__all__ = ["ARRAY", "Column", "NUMBER", "STRING", "VECTOR", "columns"]

#: The kinds of column, named as ``TFITSHDU::PrintColumnInfo`` names them.
STRING, NUMBER, ARRAY, VECTOR = ("STRING", "REAL NUMBER", "FIXED-LENGTH ARRAY",
                                 "VARIABLE-LENGTH ARRAY")  # fmt: skip

#: A binary column's type code (CFITSIO's), width in bytes and element type, by letter.
BINARY = {"L": (14, 1, "u1"), "X": (1, 1, "u1"), "B": (11, 1, ">u1"), "I": (21, 2, ">i2"),
          "J": (41, 4, ">i4"), "K": (81, 8, ">i8"), "A": (16, 1, "u1"), "E": (42, 4, ">f4"),
          "D": (82, 8, ">f8"), "C": (83, 8, ">f4"), "M": (163, 16, ">f8"),
          "P": (0, 8, ">i4"), "Q": (0, 16, ">i8")}  # fmt: skip

#: The type codes ``TFITSHDU`` reads as numbers or strings; any other it warns of.
READ = frozenset({82, 21, 41, 42, 14, 1, 11, 16})

#: The element types of a variable-length array ``TFITSHDU`` reads; any other it reports.
VECTOR_TYPES = frozenset({21, 41, 42, 82})

#: ``rTa(max)``: a binary column's repeat, its letter, and - for P and Q - its elements'.
TFORM = re.compile(r"\s*(\d*)([A-Z])([A-Z]?)")


@dataclass
class Column:
    """One column: its name, its kind, CFITSIO's type code and a value for every row."""

    name: str
    kind: str
    typecode: int
    cells: Any = field(repr=False, default=None)


def _scaled(unit: HDU, n: int, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return values.astype(np.float64) * unit.real(f"TSCAL{n}", 1.0) + unit.real(f"TZERO{n}", 0.0)


def _strings(field_bytes: np.ndarray[Any, Any]) -> list[str]:
    return [bytes(row).split(b"\0")[0].decode("latin-1").rstrip() for row in field_bytes]


def _numbers(unit: HDU, n: int, letter: str, field_bytes: np.ndarray[Any, Any],
             repeat: int) -> tuple[str, Any]:  # fmt: skip
    """A column of numbers, one or ``repeat`` of them to a row."""
    if letter == "L":
        values = (field_bytes == ord("T")).astype(np.float64)
    else:
        raw = np.ascontiguousarray(field_bytes).view(BINARY[letter][2])
        values = _scaled(unit, n, raw)
    rows = values.reshape(len(field_bytes), repeat) if repeat else np.zeros((len(field_bytes), 1))
    return (NUMBER, rows[:, 0].copy()) if repeat <= 1 else (ARRAY, rows)


def _vectors(unit: HDU, n: int, element: str, field_bytes: np.ndarray[Any, Any],
             pointer: str) -> list[np.ndarray[Any, Any]]:  # fmt: skip
    """The arrays of a ``P`` or ``Q`` column: a count and an offset into the heap, each."""
    descriptors = np.ascontiguousarray(field_bytes).view(pointer).reshape(-1, 2)
    heap = unit.integer("THEAP", unit.integer("NAXIS1") * unit.integer("NAXIS2"))
    code, _, kind = BINARY[element]
    if code not in VECTOR_TYPES:
        return [np.zeros(int(count)) for count, _ in descriptors]
    return [_scaled(unit, n, np.frombuffer(unit.data, kind, int(count), heap + int(offset)))
            for count, offset in descriptors]  # fmt: skip


def _binary_column(unit: HDU, n: int, field_bytes: np.ndarray[Any, Any], form: Any) -> Column:
    """Column ``n`` of a binary table, from the bytes it takes in every row."""
    name, (repeat, letter, element) = unit.string(f"TTYPE{n}"), form
    code = BINARY[letter][0]
    if letter in "PQ":
        vectors = _vectors(unit, n, element, field_bytes, BINARY[letter][2])
        return Column(name, VECTOR, -BINARY[element][0], vectors)
    if letter == "X":
        raise UnsupportedFeatureError(
            f"Column {n} of this FITS table holds bits (TFORM X), which xrdroot does not read yet."
        )
    if code not in READ:
        return Column(name, "", code)
    if letter == "A":
        texts = _strings(field_bytes) if repeat else ["-"] * len(field_bytes)
        return Column(name, STRING, code, texts)
    kind, cells = _numbers(unit, n, letter, field_bytes, repeat)
    return Column(name, kind, code, cells)


def _form(unit: HDU, n: int) -> tuple[int, str, str]:
    """Column ``n``'s repeat, letter and - for a P or Q column - its elements' letter."""
    match = TFORM.match(unit.string(f"TFORM{n}"))
    if match is None or match.group(2) not in BINARY:
        raise UnsupportedFeatureError(
            f"Column {n} of this FITS table has a TFORM xrdroot does not know."
        )
    return int(match.group(1) or 1), match.group(2), match.group(3) or "J"


def _width(repeat: int, letter: str) -> int:
    if letter in "PQ":
        return BINARY[letter][1]
    return (repeat + 7) // 8 if letter == "X" else repeat * BINARY[letter][1]


def _binary(unit: HDU, rows: np.ndarray[Any, Any]) -> list[Column]:
    found, at = [], 0
    for n in range(1, unit.integer("TFIELDS") + 1):
        form = _form(unit, n)
        width = _width(form[0], form[1])
        found.append(_binary_column(unit, n, rows[:, at:at + width], form))
        at += width
    return found


def _ascii_number(text: str, decimals: int) -> float:
    """A number of an ASCII table: blank is 0, and one with no point has ``decimals`` implied."""
    text = text.strip()
    if not text:
        return 0.0
    value = number(text)
    implied = "." not in text and not re.search("[EeDd]", text)
    return value / 10**decimals if implied else value


def _ascii_column(unit: HDU, n: int, rows: np.ndarray[Any, Any]) -> Column:
    form = unit.string(f"TFORM{n}").strip()
    letter, sizes = form[0], form[1:].split(".")
    width, decimals = int(sizes[0]), int(sizes[1]) if len(sizes) > 1 else 0
    start = unit.integer(f"TBCOL{n}") - 1
    texts = _strings(rows[:, start:start + width])
    name = unit.string(f"TTYPE{n}")
    if letter == "A":
        return Column(name, STRING, 16, texts)
    values = np.array([_ascii_number(t, decimals) for t in texts], dtype=np.float64)
    code = {"I": 41, "F": 42, "E": 42}.get(letter, 82)
    return Column(name, NUMBER, code, _scaled(unit, n, values))


def columns(unit: HDU) -> list[Column]:
    """Every column of a table unit, in order."""
    width, count = unit.integer("NAXIS1"), unit.integer("NAXIS2")
    rows = np.frombuffer(unit.data, np.uint8, width * count).reshape(count, width)
    if unit.kind == "ASCII TABLE":
        return [_ascii_column(unit, n, rows) for n in range(1, unit.integer("TFIELDS") + 1)]
    return _binary(unit, rows)
