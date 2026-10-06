"""``RDF::FromSqlite``: the rows an SQL query gives, as columns typed as ROOT's ``RSqliteDS``
types them.

A column's type is its declared type - ``INTEGER`` a Long64_t, ``FLOAT`` a
double, ``TEXT`` a std::string, ``BLOB`` a std::vector<unsigned char>, any
other declared type an error - and, for a column declaring none (an
expression's), the type of its value in the first row, ``NULL`` making it
a ``void*``. A value is read as its column's type, as SQLite converts it:
``NULL`` is 0, 0.0, "" or no bytes. The declared types are found as SQLite
finds them, through a temporary view of the query.
"""

from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from collections.abc import Callable
from contextlib import closing
from typing import Any

import numpy as np

from .tabular import REMOTE, Columns, fetch

__all__ = ["DECLARED", "from_sqlite"]

#: The C++ type of each declared SQL type ``RSqliteDS`` knows.
DECLARED = {"INTEGER": "Long64_t", "FLOAT": "double", "TEXT": "std::string",
            "BLOB": "std::vector<unsigned char>"}  # fmt: skip

#: The C++ type of a first-row value, for a column declaring no type.
VALUED = {int: "Long64_t", float: "double", str: "std::string", bytes: "std::vector<unsigned char>"}


#: The number text starts with, as SQLite reads text as a number.
LEADING = re.compile(r"\s*[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?")


def _declared(db: sqlite3.Connection, query: str) -> list[str]:
    """Every result column's declared type, "" for one an expression makes."""
    db.execute(f"CREATE TEMP VIEW xrdroot_query AS {query.strip().rstrip(';')}")
    try:
        return [str(row[2]) for row in db.execute("PRAGMA temp.table_info(xrdroot_query)")]
    finally:
        db.execute("DROP VIEW temp.xrdroot_query")


def _cxx(declared: str, first: Any) -> str:
    if declared:
        if declared.upper() not in DECLARED:
            raise ValueError(f"Unexpected column decl type {declared!r} in this SQLite query.")
        return DECLARED[declared.upper()]
    return VALUED.get(type(first), "void*")


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else int(_leading(value, int))


def _real(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else float(_leading(value, float))


def _text(value: Any) -> str:
    return "" if value is None else value.decode() if isinstance(value, bytes) else str(value)


def _blob(value: Any) -> Any:
    raw = value.encode() if isinstance(value, str) else bytes(value or b"")
    return np.frombuffer(raw, np.uint8).copy()


#: How a value is read as each C++ type, as SQLite converts it.
READERS: dict[str, Callable[[Any], Any]] = {
    "Long64_t": _integer, "double": _real, "std::string": _text,
    "std::vector<unsigned char>": _blob, "void*": lambda value: None,
}  # fmt: skip


def _as(value: Any, kind: str) -> Any:
    """A value as SQLite gives one of the column's type."""
    return READERS[kind](value)


def _leading(value: Any, kind: type) -> Any:
    """What SQLite makes of text as a number: the number it starts with, or 0."""
    text = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value or "")
    found = LEADING.match(text)
    return kind(float(found.group())) if found else kind(0)


def _column(values: list[Any], kind: str) -> Any:
    converted = [_as(value, kind) for value in values]
    numeric = {"Long64_t": np.int64, "double": np.float64}
    if kind in numeric:
        return np.array(converted, dtype=numeric[kind])
    made = np.empty(len(converted), dtype=object)
    made[:] = converted
    return made


def _query(path: str, query: str) -> Columns:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        cursor = db.execute(query)
        names = [str(column[0]) for column in cursor.description or []]
        rows = cursor.fetchall()
        declared = _declared(db, query)
    first = rows[0] if rows else [None] * len(names)
    kinds = [_cxx(d, value) for d, value in zip(declared, first, strict=False)]
    columns = {name: _column([row[i] for row in rows], kind)
               for i, (name, kind) in enumerate(zip(names, kinds, strict=False))}  # fmt: skip
    return Columns(columns, dict(zip(names, kinds, strict=False)), "RSqliteDS")


def from_sqlite(name: str, query: str) -> Columns:
    """The rows ``query`` gives from the SQLite file (or URL) ``name``, as columns."""
    if not name.startswith(REMOTE):
        return _query(name, query)
    handle, path = tempfile.mkstemp(suffix=".sqlite")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(fetch(name))
        return _query(path, query)
    finally:
        os.unlink(path)
