"""What ``TFITSHDU::Print`` prints, character for character as ROOT prints it.

Two of ROOT's habits are kept. Listing a file's units, the name shown for
a unit with no ``EXTNAME`` is the last one found before it, ``PRIMARY``
until one is; and whether each record is printed with its comment is
decided by the unit's last record alone, having a comment or not.
"""

from __future__ import annotations

from typing import Any

from ..fitsio import ARRAY, NUMBER, STRING, VECTOR, Record, read_units
from .core.messages import Warning

__all__ = ["print_columns", "print_file", "print_records", "print_table"]


def _record(record: Record, commented: bool, indent: str = "") -> str:
    line = f"{indent}{record.keyword:<10} = {record.value}"
    return f"{line} / {record.comment}" if commented else line


def print_records(records: list[Record]) -> None:
    """``PrintHDUMetadata``: every record, with its comment when it has one."""
    for record in records:
        print(_record(record, bool(record.comment)))


def print_file(path: str, verbose: bool) -> None:
    """``PrintFileMetadata``: every unit of the file - its number, kind and name - and, with
    ``+``, its records."""
    with open(path, "rb") as stream:
        units = read_units(stream.read())
    print(f"Total: {len(units)} HDUs")
    name, commented = "PRIMARY", False
    for unit in units:
        name = unit.value("EXTNAME") or name
        commented = bool(unit.records[-1].comment) if unit.records else commented
        print(f"   [{unit.index}] {unit.kind} ({name})")
        for record in unit.records if verbose else []:
            print(_record(record, commented, "      "))
        print()


def print_columns(hdu: Any) -> None:
    """``PrintColumnInfo``: every column's name and kind."""
    if hdu._kind == "IMAGE":
        Warning("PrintColumnInfo", "this is not a table HDU.")
        return
    for column in hdu._columns:
        if column.kind:
            print(f"{column.name:<20} : {column.kind}")


def _cell(column: Any, row: int) -> str:
    if column.kind == STRING:
        return f"{column.cells[row]:<10}"
    return f"{column.cells[row]:<10.2g}" if column.kind == NUMBER else ""


def print_table(hdu: Any) -> None:
    """``PrintFullTable``: a header of the columns' names, a rule, then every row."""
    if hdu._kind == "IMAGE":
        Warning("PrintColumnInfo", "this is not a table HDU.")
        return
    for column in hdu._columns:
        if column.kind in (ARRAY, VECTOR):
            which = "fixed" if column.kind == ARRAY else "variable"
            Warning("PrintColumnInfo", f"The table contains column with {which}-length arrays "
                    "and cannot be flattened for printing.")  # fmt: skip
            return
    heading = "".join(f"{column.name:<10}| " for column in hdu._columns)
    print(f"\n{heading}\n{'-' * len(heading)}")
    for row in range(hdu._rows):
        print("".join(_cell(column, row) + "| " for column in hdu._columns))
