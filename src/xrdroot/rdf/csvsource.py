"""``RDF::FromCSV``: a CSV file's columns, split, typed and read as ROOT's ``RCsvDS`` does.

A field ends at the delimiter unless quoted; a doubled quote is one quote;
spaces are part of a field. An empty field, ``nan`` and ``NaN`` are all
"nan". A column's type is the user's letter for it - ``O`` bool, ``D``
double, ``L`` Long64_t, ``T`` std::string - or else what its first value
looks like (the next ten rows are looked at while that is "nan"): an
integer, a number with a point, ``true`` or ``false``, anything else a
string. A "nan" is NaN in a double, 0 or false in an integer or a bool -
which ROOT warns of - and the letters "nan" in a string.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .tabular import Columns, fetch

__all__ = ["CsvOptions", "TYPES", "from_csv", "split"]

#: The C++ type of each of ``RCsvDS``'s type letters.
TYPES = {"O": "bool", "D": "double", "L": "Long64_t", "T": "std::string"}

#: ``RCsvDS``'s patterns for an integer and for a floating-point number.
INTEGER = re.compile(r"[-+]?[0-9]+")
DOUBLE = re.compile(r"[-+]?(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+|[0-9]*\.[0-9]+[eEdDqQ][-+]?[0-9]+)")

#: What ``std::stod`` and ``std::stoll`` read from the start of a string.
STOD = re.compile(r"\s*[-+]?(?:inf(?:inity)?|nan|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)",
                  re.IGNORECASE)  # fmt: skip
STOLL = re.compile(r"\s*[-+]?[0-9]+")


@dataclass
class CsvOptions:
    """``RCsvDS::ROptions``, with ROOT's defaults."""

    fHeaders: bool = True
    fDelimiter: str = ","
    fLeftTrim: bool = False
    fRightTrim: bool = False
    fSkipBlankLines: bool = True
    fSkipFirstNLines: int = 0
    fSkipLastNLines: int = 0
    fLinesChunkSize: int = -1
    fComment: str = "\0"
    fColumnNames: list[str] = field(default_factory=list)
    fColumnTypes: dict[str, str] = field(default_factory=dict)


def _value(line: str, at: int, delimiter: str) -> tuple[str, int]:
    """One field from ``at``: its text, and where the delimiter after it is."""
    text, quoted, start = [], False, at
    while at < len(line) and (line[at] != delimiter or quoted):
        if line[at] == '"' and line[at + 1:at + 2] == '"':
            text.append('"')
            at += 1
        elif line[at] == '"':
            quoted = not quoted
        else:
            text.append(line[at])
        at += 1
    value = "".join(text)
    return ("nan" if at == start or value in ("nan", "NaN") else value), at


def split(line: str, delimiter: str = ",") -> list[str]:
    """``RCsvDS::ParseColumns``: a line's fields, a delimiter ending the line adding a "nan"."""
    fields, at = [], 0
    while at < len(line):
        value, at = _value(line, at, delimiter)
        fields.append(value)
        if at == len(line) - 1 and line[at] == delimiter:
            fields.append("nan")
        at += 1
    return fields


def _kept_line(line: str, options: CsvOptions) -> str | None:
    """``RCsvDS::Readln``'s trimming and comments: the line as read, or ``None`` to skip it."""
    line = line.lstrip() if options.fLeftTrim else line
    if options.fComment != "\0" and options.fComment in line:
        if line.startswith(options.fComment):
            return None
        line = line[:line.index(options.fComment)]
    line = line.rstrip() if options.fRightTrim else line
    return None if options.fSkipBlankLines and not line else line


def _lines(text: str, options: CsvOptions) -> list[str]:
    """The lines a CSV file's header and records are read from, in order."""
    lines = text.splitlines()
    if options.fSkipLastNLines:
        if len(lines) < options.fSkipLastNLines:
            raise ValueError("Error: too many footer lines to skip in this CSV file.")
        lines = lines[:len(lines) - options.fSkipLastNLines]
    kept = (_kept_line(line, options) for line in lines[options.fSkipFirstNLines:])
    return [line for line in kept if line is not None]


def _inferred(value: str) -> str:
    """``RCsvDS::InferType``: the letter a first value's look gives its column."""
    if INTEGER.fullmatch(value):
        return "L"
    if DOUBLE.fullmatch(value):
        return "D"
    return "O" if value in ("true", "false") else "T"


def _letter(fields: list[list[str]], column: int) -> str:
    """A column's type from its first value, or the first of the next ten that is not "nan"."""
    for row in fields[:11]:
        if column < len(row) and row[column] != "nan":
            return _inferred(row[column])
    return "D"


def _number(pattern: re.Pattern[str], text: str, what: str) -> str:
    """The number ``std::stod`` or ``std::stoll`` reads from the start of a field."""
    found = pattern.match(text)
    if found is None:
        raise ValueError(f"A CSV field {text!r} of a {what} column is not a number.")
    return found.group().strip()


def _doubles(values: list[str]) -> Any:
    return np.array([np.nan if v == "nan" else float(_number(STOD, v, "double")) for v in values])


def _integers(values: list[str]) -> Any:
    return np.array([0 if v == "nan" else int(_number(STOLL, v, "Long64_t")) for v in values],
                    dtype=np.int64)  # fmt: skip


#: How the fields of a column of each type letter become its values.
CONVERTERS: dict[str, Callable[[list[str]], Any]] = {
    "T": lambda values: np.array(values, dtype=object), "D": _doubles, "L": _integers,
    "O": lambda values: np.array([v == "true" for v in values], dtype=bool),
}  # fmt: skip


def _converted(values: list[str], letter: str, name: str, empty: list[str]) -> Any:
    """A column's fields as the values its type makes of them; an integer or a bool column
    with an empty field is noted in ``empty``, for ROOT's warning."""
    if letter in "LO" and "nan" in values:
        empty.append(name)
    return CONVERTERS[letter](values)


def _warning(empty: list[str], letters: dict[str, str]) -> str | None:
    """ROOT's warning about integer and bool columns that had empty or NaN fields."""
    if not empty:
        return None
    said = ""
    for name in empty:
        kind = TYPES[letters[name]]
        said += f'Column "{name}" of type {kind} contains empty cell(s) or NaN(s).\n'
        said += (f"There is no `nan` equivalent for type {kind}, hence "
                 f"{'`0`' if kind == 'Long64_t' else '`false`'} is stored.\n")  # fmt: skip
    return said + ("Please manually set the column type to `double` (with `D`) in `FromCSV` to "
                   "read NaNs instead.\n")  # fmt: skip


def _names(first: list[str], options: CsvOptions) -> list[str]:
    """The columns' names: the user's, the header's, or ``Col0``, ``Col1`` ..."""
    if options.fColumnNames:
        if len(options.fColumnNames) != len(first):
            raise ValueError(f"Error: passed {len(options.fColumnNames)} column names for a CSV "
                             f"file containing {len(first)} columns!")  # fmt: skip
        return list(options.fColumnNames)
    return first if options.fHeaders else [f"Col{i}" for i in range(len(first))]


def _letters(names: list[str], fields: list[list[str]], options: CsvOptions) -> dict[str, str]:
    for name, letter in options.fColumnTypes.items():
        if name not in names:
            raise ValueError(f'There is no column with name "{name}".')
        if letter not in TYPES:
            raise ValueError(f"Type alias '{letter}' is not supported. Supported type aliases are "
                             "'O' for boolean, 'D' for double, 'L' for Long64_t, 'T' for "
                             "std::string.")  # fmt: skip
    return {name: options.fColumnTypes.get(name) or _letter(fields, i)
            for i, name in enumerate(names)}  # fmt: skip


def _records(name: str, options: CsvOptions) -> tuple[list[str] | None, list[list[str]]]:
    """A CSV file's header, when it has one, and its records, each split into its fields."""
    lines = _lines(fetch(name).decode("utf-8", "replace"), options)
    header = split(lines[0], options.fDelimiter) if options.fHeaders and lines else None
    records = [split(line, options.fDelimiter) for line in lines[1 if header else 0:]]
    if (options.fHeaders and header is None) or not records:
        raise ValueError(f"Could not read the header and the column types of the CSV file {name}.")
    return header, records


def _check_widths(name: str, names: list[str], records: list[list[str]]) -> None:
    short = next((n for n, row in enumerate(records) if len(row) != len(names)), None)
    if short is not None:
        raise ValueError(f"Record {short} of the CSV file {name} has {len(records[short])} fields "
                         f"where the file has {len(names)} columns.")  # fmt: skip


def from_csv(name: str, options: CsvOptions) -> Columns:
    """A CSV file's records as columns, the file (or URL) read whole."""
    header, records = _records(name, options)
    names = _names(header if header is not None else records[0], options)
    _check_widths(name, names, records)
    letters = _letters(names, records, options)
    empty: list[str] = []
    columns = {n: _converted([row[i] for row in records], letters[n], n, empty)
               for i, n in enumerate(names)}  # fmt: skip
    source = Columns(columns, {n: TYPES[letters[n]] for n in names}, "CSV data source")
    source.warning = _warning(empty, letters)
    return source
