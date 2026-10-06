"""Views of part of a matrix - a row, a column, the diagonal - that read and write the matrix.

``TMatrixDRow(m, i)`` is row ``i`` of ``m`` itself, not a copy: indexed
from the matrix's own lower bound, set whole from a number or a vector -
``TMatrixDColumn(A, 0) = 1.0`` - and scaled in place; arithmetic on one
gives the values, which another view can then be set to.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .matrices import TMatrixT, values_of

__all__ = ["TMatrixDRow", "TMatrixDColumn", "TMatrixDDiag", "TMatrixFRow", "TMatrixFColumn",
           "TMatrixFDiag", "TMatrixDRow_const", "TMatrixDColumn_const", "TMatrixDDiag_const",
           "TMatrixTRow", "TMatrixTColumn", "TMatrixTDiag"]  # fmt: skip


class _View:
    """Part of a matrix, by the indices of its elements, read and written in the matrix."""

    def __init__(self, matrix: TMatrixT, where: tuple[Any, Any], lwb: int) -> None:
        self._matrix, self._where, self._lwb = matrix, where, lwb

    def values(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._matrix.values[self._where])

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray[Any, Any]:
        """A copy of the elements, as the matrix holds them now."""
        return np.array(self.values(), dtype=dtype)

    def __len__(self) -> int:
        return len(self.values())

    def GetNdim(self) -> int:
        return len(self)

    def __call__(self, i: Any) -> float:
        return float(self.values()[int(i) - self._lwb])

    __getitem__ = __call__

    def __setitem__(self, i: Any, value: Any) -> None:
        held = self.values().copy()
        held[int(i) - self._lwb] = value
        self._matrix.values[self._where] = held

    __setcall__ = __setitem__

    def _assign(self, value: Any) -> None:
        """``view = value``: every element set to the number, or to the vector's elements."""
        self._matrix.values[self._where] = values_of(value) if np.ndim(value) else float(value)

    def __imul__(self, factor: Any) -> _View:
        self._matrix.values[self._where] = self.values() * values_of(factor)
        return self

    def __iadd__(self, other: Any) -> _View:
        self._matrix.values[self._where] = self.values() + values_of(other)
        return self

    def __mul__(self, other: Any) -> np.ndarray[Any, Any]:
        return self.values() * values_of(other)

    __rmul__ = __mul__

    def __add__(self, other: Any) -> np.ndarray[Any, Any]:
        return self.values() + values_of(other)

    def Sum(self) -> float:
        return float(self.values().sum())


class TMatrixDRow(_View):
    """``TMatrixDRow(m, i)``: row ``i`` of ``m``."""

    def __init__(self, matrix: TMatrixT, row: Any) -> None:
        super().__init__(matrix, (int(row) - matrix.rowlwb, slice(None)), matrix.collwb)


class TMatrixDColumn(_View):
    """``TMatrixDColumn(m, j)``: column ``j`` of ``m``."""

    def __init__(self, matrix: TMatrixT, column: Any) -> None:
        super().__init__(matrix, (slice(None), int(column) - matrix.collwb), matrix.rowlwb)


class TMatrixDDiag(_View):
    """``TMatrixDDiag(m)``: the diagonal of ``m``."""

    def __init__(self, matrix: TMatrixT) -> None:
        count = min(matrix.values.shape)
        super().__init__(matrix, (np.arange(count), np.arange(count)), 0)


#: The other spellings: of floats, read-only, and the class templates themselves.
TMatrixFRow = TMatrixDRow_const = TMatrixTRow = TMatrixDRow
TMatrixFColumn = TMatrixDColumn_const = TMatrixTColumn = TMatrixDColumn
TMatrixFDiag = TMatrixDDiag_const = TMatrixTDiag = TMatrixDDiag
