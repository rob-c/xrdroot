"""``TMatrixDSym``: the covariance and correlation matrices a fit result hands back.

A fit result's matrices are ROOT's symmetric matrices, which a macro reads
by ``m(i, j)`` or ``m[i][j]`` and prints with ``Print()`` in
``TMatrixTBase::Print``'s layout - five columns a sheet, ``%11.4g`` each.
This is that much of the class, over a NumPy array.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import cout

__all__ = ["TMatrixDSym", "TVectorD"]


class TMatrixDSym:
    """A symmetric matrix of doubles."""

    def __init__(self, n: Any = 0, values: Any = None) -> None:
        if isinstance(n, TMatrixDSym):
            values, n = n.values.copy(), n.values.shape[0]
        self.values = (
            np.zeros((int(n), int(n))) if values is None else np.array(values, dtype=np.float64)
        )

    def __call__(self, i: int, j: int) -> float:
        return float(self.values[i, j])

    def __getitem__(self, i: int) -> Any:
        return self.values[i]

    def GetNrows(self) -> int:
        return int(self.values.shape[0])

    def GetNcols(self) -> int:
        return int(self.values.shape[1])

    def Determinant(self) -> float:
        return float(np.linalg.det(self.values))

    def Invert(self) -> TMatrixDSym:
        self.values = np.linalg.inv(self.values)
        return self

    def ClassName(self) -> str:
        return "TMatrixTSym<double>"

    def GetName(self) -> str:
        return ""

    def Print(self, option: str = "") -> None:
        cout.write(matrix_text(self.values))

    def __array__(self, dtype: Any = None) -> np.ndarray[Any, Any]:
        return self.values if dtype is None else self.values.astype(dtype)


class TVectorD:
    """A vector of doubles, as ``TVectorT<double>``."""

    def __init__(self, values: Any = ()) -> None:
        self.values = np.array(values, dtype=np.float64)

    def __getitem__(self, i: int) -> float:
        return float(self.values[i])

    def __call__(self, i: int) -> float:
        return float(self.values[i])

    def GetNrows(self) -> int:
        return len(self.values)


def matrix_text(values: np.ndarray[Any, Any], fmt: str = "%11.4g ") -> str:
    """``TMatrixTBase::Print``: sheets of five columns under a numbered bar."""
    nrows, ncols = values.shape
    nch = min(len(fmt % 123.456789) + 1, 18)
    digits = 1 + int(np.log10(ncols)) if ncols else 1
    column = (" " * (nch // 2) + f"%{digits}d").ljust(nch)[:nch] + "|"
    per_sheet = 10 if nch <= 8 else 5
    bar = "-" * (5 + nch * min(per_sheet, ncols))
    text = f"\n{nrows}x{ncols} matrix is as follows"
    for start in range(0, ncols, per_sheet):
        cols = range(start, min(start + per_sheet, ncols))
        text += "\n\n     |" + "".join(column % j for j in cols) + f"\n{bar}\n"
        for i in range(nrows):
            text += f"{i:4d} |" + "".join(fmt % values[i, j] for j in cols) + "\n"
    return text + "\n"
