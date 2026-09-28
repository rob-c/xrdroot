"""The cells of a PDE-Foam: a binary tree of cuts, each cell half of its parent in one variable.

A cell knows its parent, its two daughters, the variable it was cut in
(``best``) and where (``xdiv``, a fraction of its width); its position and
size in the unit hypercube are worked out from its ancestors' cuts as
``PDEFoamCell::GetHcub`` works them out, from the cell up. The active
cells - the leaves - hold the values an event is given: the elements.
:meth:`Cells.columns` and :meth:`Cells.from_columns` are how a foam is
written to its ``_foams.root`` file and read back.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["Cells"]

f32 = np.float32


class Cells:
    """The cells of a foam of ``dim`` variables, room for ``ncells`` of them."""

    def __init__(self, dim: int, ncells: int) -> None:
        self.dim, self.ncells, self.last = dim, ncells, -1
        self.status = np.zeros(ncells, dtype=np.int32)
        self.parent = np.full(ncells, -1, dtype=np.int32)
        self.dau0 = np.full(ncells, -1, dtype=np.int32)
        self.dau1 = np.full(ncells, -1, dtype=np.int32)
        self.best = np.full(ncells, -1, dtype=np.int32)
        self.xdiv = np.full(ncells, 0.5)
        self.intg = np.zeros(ncells)
        self.driv = np.zeros(ncells)
        self.elements = np.zeros((ncells, 2))
        self._hcub: dict[int, tuple[Any, Any]] = {}

    def fill(self, parent: int | None) -> int:
        """``CellFill``: the next cell, active, with half its parent's integral and driver."""
        self.last += 1
        cell = self.last
        self.status[cell], self.best[cell], self.xdiv[cell] = 1, -1, 0.5
        if parent is None:
            self.intg[cell] = self.driv[cell] = 0.0
        else:
            self.parent[cell] = parent
            self.intg[cell], self.driv[cell] = 0.5 * self.intg[parent], 0.5 * self.driv[parent]
        return cell

    def hcub(self, cell: int) -> tuple[Any, Any]:
        """``GetHcub``: the cell's corner and size, composed from the cell up to the root."""
        if cell not in self._hcub:
            posi, size = np.zeros(self.dim), np.ones(self.dim)
            child, parent = cell, int(self.parent[cell])
            while parent >= 0:
                k, x = int(self.best[parent]), float(self.xdiv[parent])
                if child == self.dau0[parent]:
                    size[k] *= x
                    posi[k] *= x
                else:
                    size[k] *= 1.0 - x
                    posi[k] = posi[k] * (1.0 - x) + x
                child, parent = parent, int(self.parent[parent])
            self._hcub[cell] = (posi, size)
        return self._hcub[cell]

    def volume(self, cell: int) -> float:
        """``CalcVolume``: the product of the cell's sizes."""
        return float(np.prod(self.hcub(cell)[1]))

    def depth(self, cell: int) -> int:
        """``GetDepth``: 1 for the root, one more for each generation below it."""
        depth = 1
        while self.parent[cell] >= 0:
            cell, depth = int(self.parent[cell]), depth + 1
        return depth

    def find(self, points: Any) -> Any:
        """``FindCell`` of every point of the unit hypercube: the active cell holding it."""
        edges = np.zeros(self.last + 1)
        for cell in np.flatnonzero(self.status[: self.last + 1] != 1):
            k = int(self.best[cell])
            posi, size = self.hcub(int(self.dau0[cell]))
            edges[cell] = posi[k] + size[k]
        cells = np.zeros(len(points), dtype=np.intp)
        rows = np.arange(len(points))
        while True:
            inner = np.flatnonzero(self.status[cells] != 1)
            if not len(inner):
                return cells
            here = cells[inner]
            left = points[rows[inner], self.best[here]] <= edges[here]
            cells[inner] = np.where(left, self.dau0[here], self.dau1[here])

    def to_unit(self, values: Any, xmin: Any, xmax: Any) -> Any:
        """``VarTransform``: values mapped from ``[xmin, xmax]`` onto ``[0, 1]``, in single precision."""
        return ((np.asarray(values, dtype=np.float64) - xmin) / (xmax - xmin)).astype(f32)

    def cut_counts(self) -> Any:
        """How many inactive cells are cut in each variable: ``GetNCuts``."""
        inactive = np.flatnonzero(
            (self.status[: self.last + 1] == 0) & (self.best[: self.last + 1] >= 0)
        )
        return np.bincount(self.best[inactive], minlength=self.dim)

    # -- the foam's file ------------------------------------------------------------------

    def columns(self) -> dict[str, Any]:
        """The cells as the columns of a tree, one entry per cell."""
        n = self.last + 1
        return {
            "status": self.status[:n],
            "parent": self.parent[:n],
            "dau0": self.dau0[:n],
            "dau1": self.dau1[:n],
            "best": self.best[:n],
            "xdiv": self.xdiv[:n],
            "intg": self.intg[:n],
            "driv": self.driv[:n],
            "element0": self.elements[:n, 0],
            "element1": self.elements[:n, 1],
        }

    @classmethod
    def from_columns(cls, dim: int, columns: dict[str, Any]) -> Cells:
        n = len(columns["status"])
        made = cls(dim, n)
        made.last = n - 1
        for name in ("status", "parent", "dau0", "dau1", "best", "xdiv", "intg", "driv"):
            getattr(made, name)[:] = np.asarray(columns[name])
        made.elements[:, 0] = np.asarray(columns["element0"])
        made.elements[:, 1] = np.asarray(columns["element1"])
        return made
