"""``TFoamCell``: the hyper-rectangles of a foam, and where each one is.

A foam is a binary tree of cells over the unit hypercube. ROOT does not store
where a cell is: a cell knows its parent, and a parent knows which edge it
was cut along and at what fraction, so ``TFoamCell::GetHcub`` finds a cell's
corner and size by walking from the cell up to the root, rescaling as it goes.
The walk is repeated here in ROOT's order with ROOT's arithmetic, because the
corner it arrives at is not the same number, to the last bit, as one found by
cutting down from the root - and every event is that corner plus a fraction
of that size.

A cell's place never changes once the cell exists (its ancestors were cut
before it was made), so the walk is made once, when the foam first looks at
the cell, and kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["Cell"]


class Cell:
    """One cell of the tree: its place in the list, its links, how it would be cut, its integrals.

    ``best`` and ``xdiv`` are the edge and fraction ``TFoam::Explore`` chose
    for cutting it; ``intg``, ``driv`` and ``prim`` are its estimates of the
    true integral, the driver integral that decides which cell is cut next,
    and the primary integral that decides how often events land in it.
    """

    __slots__ = (
        "serial",
        "active",
        "parent",
        "dau0",
        "dau1",
        "best",
        "xdiv",
        "intg",
        "driv",
        "prim",
        "volume",
        "posi",
        "size",
    )

    def __init__(self, parent: Cell | None, serial: int) -> None:
        self.serial = serial
        self.active = True
        self.parent = parent
        self.dau0: Cell | None = None
        self.dau1: Cell | None = None
        self.best = -1
        self.xdiv = 0.5
        # ``CellFill``: a daughter starts with half its parent's integrals,
        # which ``Explore`` takes back out of every ancestor when it replaces them.
        self.intg = 0.0
        self.driv = 0.0
        if parent is not None:
            self.intg = 0.5 * parent.intg
            self.driv = 0.5 * parent.driv
        self.prim = 0.0
        self.volume = 0.0
        self.posi: Any = None
        self.size: Any = None

    def place(self, dim: int) -> None:
        """Find the cell's corner, size and volume as ``GetHcub`` and ``CalcVolume`` do."""
        posi = [0.0] * dim
        size = [1.0] * dim
        child = self
        parent = self.parent
        while parent is not None:
            k, x = parent.best, parent.xdiv
            if child is parent.dau0:
                size[k] *= x
                posi[k] *= x
            else:
                size[k] *= 1.0 - x
                posi[k] = posi[k] * (1.0 - x) + x
            child, parent = parent, parent.parent
        volume = 1.0
        for edge in size:
            volume *= edge
        self.posi = np.array(posi)
        self.size = np.array(size)
        self.volume = volume
