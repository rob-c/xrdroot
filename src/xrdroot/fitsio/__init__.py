"""FITS files read with NumPy alone: headers, images, and binary and ASCII tables.

A file is named as CFITSIO names one - ``sample.fits``, ``sample.fits[1]``,
``sample.fits[EVENTS]``, ``sample.fits[1][DATAMAX > 2e-15]`` - the first
bracket choosing a unit by its number or its ``EXTNAME`` and the second a
table's rows. :func:`open_unit` gives back the unit chosen, its columns
with only the rows kept, and every unit of the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..errors import FormatError, UnsupportedFeatureError
from .cards import Record, record
from .hdus import HDU, image, read_units
from .rowfilter import keep
from .tables import ARRAY, NUMBER, STRING, VECTOR, Column, columns

__all__ = ["ARRAY", "HDU", "NUMBER", "STRING", "VECTOR", "Column", "Opened", "Record",
           "base_path", "image", "open_unit", "read_units", "record"]  # fmt: skip

#: The brackets after a file's name.
BRACKET = re.compile(r"\[([^\]]*)\]")


@dataclass
class Opened:
    """A unit chosen from a file: the unit, its columns and rows, and all the file's units."""

    unit: HDU
    units: list[HDU] = field(repr=False)
    columns: list[Column] = field(default_factory=list, repr=False)
    rows: int = 0


def base_path(spec: str) -> str:
    """A file's name without its brackets, as ``TFITSHDU::CleanFilePath`` cuts it."""
    at = spec.find("[", 1)
    return spec if at < 0 else spec[:at]


def _chosen(units: list[HDU], which: str) -> HDU:
    which = which.strip()
    if which.isdigit():
        if int(which) >= len(units):
            raise OSError(f"This FITS file has no unit {which}: it has {len(units)}.")
        return units[int(which)]
    named = [u for u in units if u.string("EXTNAME").upper() == which.upper()]
    if not named:
        raise OSError(f"This FITS file has no unit named {which!r}.")
    return named[0]


def _kept_cells(cells: Any, kept: np.ndarray[Any, Any]) -> Any:
    """A column's cells in the rows kept: an array's by mask, a list's one by one."""
    if isinstance(cells, np.ndarray):
        return cells[kept]
    return [cell for cell, k in zip(cells or [], kept, strict=False) if k]


def _counted(unit: HDU, rows: int) -> HDU:
    """The unit with ``NAXIS2`` saying how many rows it now has."""
    return replace(unit, records=[
        Record(r.keyword, str(rows), r.comment) if r.keyword == "NAXIS2" else r
        for r in unit.records])  # fmt: skip


def _filtered(opened: Opened, text: str) -> None:
    """Only the rows a filter keeps, and ``NAXIS2`` counting them, as CFITSIO's copy has."""
    if re.match(r"\s*(col|bin)\s", text, re.IGNORECASE):
        raise UnsupportedFeatureError(f"CFITSIO's column and binning filters ([{text}]) are "
                                      "not read by xrdroot; a row filter is.")  # fmt: skip
    values = {c.name.upper(): np.asarray(c.cells) for c in opened.columns
              if c.kind in (NUMBER, STRING)}  # fmt: skip
    kept = keep(text, values, opened.rows)
    for column in opened.columns:
        column.cells = _kept_cells(column.cells, kept)
    opened.rows = int(kept.sum())
    opened.unit = _counted(opened.unit, opened.rows)


def open_unit(spec: str) -> Opened:
    """The unit ``spec`` names, read, with a table's rows filtered as it asks."""
    path = base_path(spec)
    with open(path, "rb") as stream:
        units = read_units(stream.read())
    if not units:
        raise FormatError(f"{path} does not start with a FITS header.")
    brackets = BRACKET.findall(spec[len(path):])
    opened = Opened(_chosen(units, brackets[0]) if brackets else units[0], units)
    if opened.unit.kind != "IMAGE":
        opened.columns = columns(opened.unit)
        opened.rows = opened.unit.integer("NAXIS2")
    for text in brackets[1:]:
        _filtered(opened, text)
    return opened
