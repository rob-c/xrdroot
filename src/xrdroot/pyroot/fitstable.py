"""What ``TFITSHDU`` reads of a table unit: its columns, whole, and its cells, one by one.

A column is named by its number from 0 or by its name. Every accessor warns
as ROOT's does - not a table, no such column, a cell out of bounds, a column
of the wrong kind - and gives ``nullptr``, which is ``None``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fitsio import ARRAY, STRING, VECTOR, Column
from .core.collections import TObjArray, TObjString
from .core.messages import Error, Info, Warning
from .core.strings import TString
from .vectors import TArrayD, TVectorD

__all__ = ["TableReads"]


class TableReads:
    """The table accessors of ``TFITSHDU``."""

    _kind: str
    _columns: list[Column]
    _rows: int

    def _column(self, where: str, col: Any, error: bool = False) -> Column | None:
        """The column ``col`` names - a number or a name - or ``None``, warned of."""
        complain = Error if error else Warning
        if self._kind == "IMAGE":
            complain(where, "this is not a table HDU.")
            return None
        if isinstance(col, (int, np.integer)):
            if 0 <= col < len(self._columns):
                return self._columns[int(col)]
            complain(where, "column index out of bounds.")
            return None
        found = next((c for c in self._columns if c.name == str(col)), None)
        if found is None:
            complain(where, "column not found.")
        return found

    def _row(self, where: str, row: int) -> bool:
        if 0 <= row < self._rows:
            return True
        Warning(where, "row index out of bounds.")
        return False

    def GetTabNColumns(self) -> int:
        return len(self._columns)

    def GetTabNRows(self) -> int:
        return self._rows

    def GetColumnNumber(self, colname: str) -> int:
        return next((n for n, c in enumerate(self._columns) if c.name == str(colname)), -1)

    def GetColumnName(self, colnum: int) -> TString:
        found = self._column("GetColumnName", int(colnum), error=True)
        return TString(found.name if found else "")

    def GetTabStringColumn(self, col: Any) -> TObjArray | None:
        found = self._column("GetTabStringColumn", col)
        if found is None or found.kind != STRING:
            if found is not None:
                Warning("GetTabStringColumn",
                        "attempting to read a column that is not of type 'kString'.")
            return None
        return _array([TObjString(text) for text in found.cells])

    def GetTabRealVectorColumn(self, col: Any) -> TVectorD | None:
        where = "GetTabRealVectorColumn"
        found = self._column(where, col)
        if found is not None and found.kind in (ARRAY, VECTOR):
            fixed = found.kind == ARRAY
            Warning(where, "attempting to read a column whose cells have embedded "
                    f"{'fixed' if fixed else 'variable'}-length arrays")  # fmt: skip
            Info(where, "Use GetTabRealVectorCells() or GetTabRealVectorCell() instead." if fixed
                 else "Use GetTabVarLengthCell() instead.")  # fmt: skip
            return None
        return None if found is None else TVectorD(np.asarray(found.cells, dtype=np.float64))

    def _not_vector(self, found: Column | None) -> bool:
        if found is not None and found.kind == VECTOR:
            Warning("GetTabRealVectorCells", "attempting to read a column whose cells have "
                    "embedded variable-length arrays")  # fmt: skip
            Info("GetTabRealVectorCells", "Use GetTabVarLengthCell() instead.")
            return False
        return found is not None

    def GetTabRealVectorCells(self, col: Any) -> TObjArray | None:
        found = self._column("GetTabRealVectorCells", col)
        if not self._not_vector(found):
            return None
        return _array([TVectorD(np.atleast_1d(cell)) for cell in found.cells])  # type: ignore

    def GetTabRealVectorCell(self, row: int, col: Any) -> TVectorD | None:
        found = self._column("GetTabRealVectorCell", col)
        if found is None or not self._row("GetTabRealVectorCell", int(row)):
            return None
        return TVectorD(np.atleast_1d(found.cells[int(row)])) if self._not_vector(found) else None

    def GetTabVarLengthVectorCell(self, row: int, col: Any) -> TArrayD | None:
        found = self._column("GetTabVarLengthVectorCell", col)
        if found is None or not self._row("GetTabVarLengthVectorCell", int(row)):
            return None
        return TArrayD(np.atleast_1d(found.cells[int(row)]))


def _array(items: list[Any]) -> TObjArray:
    made = TObjArray()
    for item in items:
        made.Add(item)
    return made
