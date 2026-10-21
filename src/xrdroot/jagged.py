"""Rows and entry selections shared by column readers and numerical analysis.

The same NumPy-backed values serve tree I/O, histogram formulas and library
conversion. Keeping their shape and selection rules here lets those consumers
share one representation without importing the file reader.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["Jagged", "concatenate", "scattered", "take"]


class Jagged(Sequence[Any]):
    """Rows of different lengths: one flat array, and where each row starts.

        >>> jets.tolist()                          # doctest: +SKIP
        [[12.5, 3.5], [], [88.0, 1.25, 0.5]]

    This is the shape a physics file is usually in - a variable number of
    particles per collision - and it is kept flat because that is how it
    arrives and how a tensor wants it back. ``content`` is every value of
    every row in one NumPy array, and ``offsets`` the ``len + 1`` places the
    rows start and stop; row ``i`` is ``content[offsets[i]:offsets[i + 1]]``,
    which is exactly the layout Awkward Array and Arrow keep a list in.
    """

    __slots__ = ("content", "offsets")

    def __init__(self, content: Any, offsets: Any) -> None:
        self.content: np.ndarray[Any, Any] = np.asarray(content)
        self.offsets: np.ndarray[Any, Any] = np.asarray(offsets, dtype=np.int64)

    def __repr__(self) -> str:
        return f"<Jagged {len(self)} rows of {len(self.content)} {self.content.dtype} values>"

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            stop = max(start, stop)
            low, high = self.offsets[start], self.offsets[stop]
            return Jagged(self.content[low:high], self.offsets[start : stop + 1] - low)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("row out of range")
        return self.content[self.offsets[index] : self.offsets[index + 1]]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Jagged):
            return NotImplemented
        return bool(
            np.array_equal(self.offsets - self.offsets[0], other.offsets - other.offsets[0])
            and np.array_equal(self.flat, other.flat)
        )

    __hash__ = None  # type: ignore[assignment]

    @property
    def flat(self) -> np.ndarray[Any, Any]:
        """Every value of every row, in order, with nothing either side."""
        return self.content[self.offsets[0] : self.offsets[-1]]

    def lengths(self) -> np.ndarray[Any, Any]:
        """How long each row is."""
        return np.diff(self.offsets)

    def tolist(self) -> list[list[Any]]:
        return [row.tolist() for row in self]

    def padded(
        self, width: int | None = None, fill: float = 0.0
    ) -> tuple[np.ndarray[Any, Any], int]:
        """A rectangle: every row cut or filled to the same width.

        Gives back the flat values and the width, which is what a tensor of
        shape ``(rows, width)`` is made of - ``.reshape(-1, width)`` makes it
        that. ``width`` defaults to the longest row, so nothing is lost unless
        a number is asked for.
        """
        lengths = self.lengths()
        if width is None:
            width = int(lengths.max(initial=0))
        out = np.full((len(self), width), fill, dtype=self.content.dtype)
        kept = np.minimum(lengths, width)
        rows = np.repeat(np.arange(len(self)), kept)
        columns = np.arange(int(kept.sum())) - np.repeat(np.cumsum(kept) - kept, kept)
        out[rows, columns] = self.content[self.offsets[:-1][rows] + columns]
        return out.reshape(-1), width

    def to_awkward(self) -> Any:
        """The same rows as an Awkward Array, sharing the same memory."""
        from .library import awkward_list

        return awkward_list(self)

    def to_arrow(self) -> Any:
        """The same rows as an Arrow list array, which pandas and Polars take."""
        from .library import arrow_list

        return arrow_list(self)

    def take(self, rows: Any) -> Jagged:
        """The rows at the given places, in the order given, as rows of their own."""
        rows = np.asarray(rows, dtype=np.int64)
        starts = self.offsets[:-1][rows]
        lengths = self.offsets[1:][rows] - starts
        offsets = np.zeros(len(rows) + 1, np.int64)
        np.cumsum(lengths, out=offsets[1:])
        index = np.repeat(starts - offsets[:-1], lengths) + np.arange(offsets[-1])
        return Jagged(self.content[index], offsets)

    @staticmethod
    def join(pieces: Sequence[Jagged]) -> Jagged:
        """Rows from several places, one after another, with the offsets carried on."""
        lengths = np.concatenate([piece.lengths() for piece in pieces])
        offsets = np.zeros(len(lengths) + 1, np.int64)
        np.cumsum(lengths, out=offsets[1:])
        return Jagged(np.concatenate([piece.flat for piece in pieces]), offsets)


def concatenate(pieces: Sequence[Any]) -> Any:
    """The values of several reads of one column, as one read of them all.

    What one read gives, several give joined: NumPy arrays end to end,
    :class:`Jagged` rows with each piece's offsets carried on from the last,
    and lists of values one list. This is how a range read across baskets,
    files of a chain, or a scattered set of entries comes back whole.
    """
    first = pieces[0]
    if isinstance(first, Jagged):
        return Jagged.join(pieces)
    if isinstance(first, np.ndarray):
        return np.concatenate(pieces)
    joined: list[Any] = []
    for piece in pieces:
        joined.extend(piece)
    return joined


def scattered(rows: Any, bounds: Any, read: Any, empty: Any) -> Any:
    """Entries picked from anywhere, read a block at a time and put back in order.

    ``bounds`` are where each block - a basket, a file of a chain - starts,
    and one past where the last ends. The entries are sorted into their
    blocks, ``read(block, entries)`` gives the values of each block's share,
    and the pieces are joined and put back in the order ``rows`` asked for
    them in; ``empty()`` is what asking for none at all gives.
    """
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    ordered = rows[order]
    homes = np.searchsorted(np.asarray(bounds, dtype=np.int64), ordered, side="right") - 1
    pieces = [read(int(home), ordered[homes == home]) for home in np.unique(homes)]
    if not pieces:
        return empty()
    joined = concatenate(pieces)
    if np.array_equal(order, np.arange(len(order))):
        return joined
    return take(joined, np.argsort(order, kind="stable"))


def take(values: Any, rows: Any) -> Any:
    """The entries at the given places of what one read gave, whatever its shape."""
    if isinstance(values, (np.ndarray, Jagged)):
        return values[rows] if isinstance(values, np.ndarray) else values.take(rows)
    return [values[row] for row in np.asarray(rows).tolist()]


def _bounds(total: int, entry_start: int, entry_stop: int | None) -> tuple[int, int]:
    """A range of entries, with negative ends counted from the last, as in Python."""
    start = total + entry_start if entry_start < 0 else entry_start
    stop = total if entry_stop is None else entry_stop
    if stop < 0:
        stop += total
    return max(start, 0), min(stop, total)
