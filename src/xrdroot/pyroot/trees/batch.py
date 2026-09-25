"""``GetEntry`` one entry at a time, read a block of entries at a time.

C++ reads a basket and hands out its entries one by one; reading an entry at
a time through NumPy would cost a whole read for every entry. So the first
entry asked for reads the block of :data:`STEP` entries around it, for each
branch that is wanted, and the entries after it are served from that until
one falls outside - which for the loop every tutorial writes, entry after
entry, is one read per branch per block.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any



__all__ = ["Batch", "STEP"]

#: How many entries are read at once.
STEP = 1000


def row_of(values: Any, at: int) -> Any:
    """Entry ``at`` of a block: a number, one row of an array, or one row of a Jagged."""
    return values[at]


class Batch:
    """The block of entries last read, branch by branch, from ``source()``."""

    def __init__(self, source: Callable[[], Any]) -> None:
        self._source = source
        self.start = self.stop = 0
        self._values: dict[str, Any] = {}

    def reset(self) -> None:
        """Forget what was read, because what it was read from has changed."""
        self.start = self.stop = 0
        self._values = {}

    def value(self, name: str, entry: int) -> Any:
        """Entry ``entry`` of the branch ``name``."""
        if not self.start <= entry < self.stop:
            self._move(entry)
        if name not in self._values:
            self._values[name] = self._source()[name].array(self.start, self.stop)
        return row_of(self._values[name], entry - self.start)

    def _move(self, entry: int) -> None:
        self.start = entry - entry % STEP
        self.stop = min(self.start + STEP, len(self._source()))
        self._values = {}
