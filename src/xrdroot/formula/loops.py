"""The implicit loop of a ``TTree::Draw`` expression, worked out for every entry at once.

ROOT's rule, from its own documentation, is that every dimension of every
branch not given an index is looped over; that the dimensions looped over
are matched up left to right across the branches, ignoring those given an
index; and that dimensions matched together share one index and run to the
*smallest* of their sizes. ``pt - eta`` with three of one and two of the
other is two values; ``m[][2] - v`` pairs the rows of ``m`` with the
elements of ``v``. A branch with no dimension left to loop over - a number
per entry, or ``pt[0]`` - goes with every element of the loop, whatever it
is.

Here that loop is run level by level, over every entry at once. A *space* is
one level of it: an array saying which entry each of its items belongs to,
and which item of the level above it came from. The first space is the
entries themselves. Each level after that is made by asking every branch
still looping how many elements it has at each item, taking the smallest, and
repeating each item that many times - one ``np.repeat`` for the whole tree
rather than a Python loop per entry. A *cursor* follows one branch down
through its dimensions as the levels are made: the node it is at for each
item, and whether that node is really there.

A branch inside ``Alt$``'s first argument is weak: it is looped over with the
rest but does not cut the loop short, and where it has run out its value is
missing and ``Alt$`` gives the alternative - which is how ROOT's
``arr1 + Alt$(arr2, 0)`` runs the length of ``arr1``.

Running out of dimensions is not the same as running out of loop, though: when
the loop has levels left that the weak branch has no dimension for, ROOT reads
that branch *flat*, at ``index * count + local`` - the last dimension's index
becomes a mixed-radix number whose digits are the remaining levels. So
``fMatrix - Alt$(fClosestDistance, 0)`` of a ``fMatrix[4][4]`` and a
``fClosestDistance`` of ``fNvertex`` elements subtracts ``fClosestDistance[i0*4
+ i1]``, and gives the alternative once that runs past the entry's elements.
A strong branch in the same place is simply repeated instead, which is where
ROOT's ``m - v`` of a ``m[2][3]`` and a ``v[6]`` reads ``v[i0]``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np

from .columns import Layout
from .nodes import Index, Node

__all__ = ["Space", "Cursor", "Scope", "both"]

Array = Any


def both(first: Array | None, second: Array | None) -> Array | None:
    """Two validity masks as one, ``None`` standing for all valid."""
    if first is None:
        return second
    if second is None:
        return first
    return first & second


class Space:
    """One level of the loop: which entry each item is in, and which item above it came from.

    What a level is made of is worked out when first asked for, since the
    commonest loop - every element of every row, of branches that all have
    the same number - needs none of it but the row lengths.
    """

    __slots__ = ("entries", "above", "count", "_parent", "_local", "_entry")

    def __init__(self, entries: int, above: Space | None = None, count: Array = None) -> None:
        self.entries = entries
        #: The level this one repeats the items of, and how many times each.
        self.above = above
        self.count = count
        self._parent: Array | None = None
        self._local: Array | None = None
        self._entry: Array | None = None if above else np.arange(entries, dtype=np.int64)

    @classmethod
    def of_entries(cls, entries: int) -> Space:
        return cls(entries)

    @property
    def looped(self) -> bool:
        """Is this more than the entries themselves - are there elements?"""
        return self.above is not None

    def __len__(self) -> int:
        return self.entries if self.above is None else int(self.count.sum())

    @property
    def parent(self) -> Array:
        """The item of the level above that each item was repeated from."""
        if self._parent is None:
            self._parent = np.repeat(np.arange(len(self.count), dtype=np.int64), self.count)
        return self._parent

    @property
    def local(self) -> Array:
        """Which repetition of its parent each item is: the loop's index at this level."""
        if self._local is None:
            starts = np.cumsum(self.count) - self.count
            self._local = np.arange(len(self.parent), dtype=np.int64) - starts[self.parent]
        return self._local

    @property
    def entry(self) -> Array:
        """The entry each item belongs to."""
        if self._entry is None:
            assert self.above is not None
            self._entry = self.above.entry[self.parent]
        return self._entry

    def expand(self, count: Array) -> Space:
        """The next level: each item repeated ``count`` times, one per element of the loop."""
        return Space(self.entries, self, count)

    def offsets(self) -> Array:
        """Where each entry's items start and stop, as :class:`~xrdroot.tree.Jagged` keeps them."""
        offsets = np.zeros(self.entries + 1, np.int64)
        if self.above is None:
            offsets[1:] = 1
        elif self.above.above is None:
            offsets[1:] = self.count
        else:
            offsets[1:] = np.bincount(self.entry, minlength=self.entries)
        return np.cumsum(offsets, out=offsets)


class Cursor:
    """One branch followed down its dimensions: the node it is at for every item.

    While it has taken every child of every node, in order, the nodes it is
    at are one unbroken run, and it keeps only where that run starts and
    stops: its values are then a slice of the column rather than a gather.
    """

    __slots__ = (
        "node",
        "layout",
        "specs",
        "weak",
        "at",
        "span",
        "last",
        "index",
        "_place",
        "valid",
    )

    def __init__(self, node: Node, layout: Layout, specs: tuple[Index, ...], weak: bool) -> None:
        #: The expression node this is the value of.
        self.node = node
        self.layout = layout
        self.specs = specs
        self.weak = weak
        #: How many of its dimensions it has gone down.
        self.at = 0
        #: The run of nodes it is at, while they are one.
        self.span: tuple[int, int] | None = (0, layout.rows)
        #: For a weak one, the nodes its last dimension was read from, and at what index.
        self.last: Array | None = None
        self.index: Array | None = None
        self._place: Array | None = None
        self.valid: Array | None = None

    @property
    def place(self) -> Array:
        """The node it is at, for every item."""
        if self._place is None:
            assert self.span is not None
            self._place = np.arange(*self.span, dtype=np.int64)
        return self._place

    def move(self, place: Array) -> None:
        self._place = place
        self.span = None

    @property
    def finished(self) -> bool:
        return self.at == len(self.specs)

    @property
    def looping(self) -> bool:
        """Is the next dimension one to loop over, rather than one given an index?"""
        return not self.finished and self.specs[self.at] is None

    def sizes(self) -> Array:
        """How many children each node it is at has, in the dimension it is at."""
        if self.span is None:
            return self.layout.sizes(self.at, self.place)
        offsets = self.layout.offsets[self.at]
        start, stop = self.span
        return offsets[start + 1 : stop + 1] - offsets[start:stop]

    @property
    def bottom(self) -> bool:
        """Is the dimension it is at the last one, below which only values are?"""
        return self.at == self.layout.dims - 1

    def take(self, node: Array, index: Array) -> None:
        """Remember where the last dimension was read, in case the loop goes on past it."""
        if self.weak and self.bottom:
            self.last = node
            self.index = index

    def spread(self, space: Space) -> None:
        """Read one level past the last dimension: its contents taken flat, as ROOT does.

        ROOT pairs a loop's levels with a branch's dimensions left to right; a
        branch that has no dimension for a level is read at ``index * count +
        local`` there, so the elements of its last dimension line up with the
        items of the levels below it. Past the elements the entry has, the
        value is missing and whatever weakens it - ``Alt$`` - takes over.
        """
        assert self.last is not None and self.index is not None
        parent = space.parent
        offsets = self.layout.offsets[self.layout.dims - 1]
        node = self.last[parent]
        if self.valid is not None:  # where a dimension ran out, its last node is a placeholder
            node = np.minimum(node, len(offsets) - 2)
        index = self.index[parent] * space.count[parent] + space.local
        room = offsets[node + 1] - offsets[node]
        valid = both(index < room, None if self.valid is None else self.valid[parent])
        self.last = node
        self.index = index if valid is None else np.where(valid, index, 0)
        self.valid = valid
        self.move(offsets[node] + self.index)

    def step(self, index: Array, ok: Array | None) -> None:
        """Go down one dimension, to the ``index``-th child of each node."""
        sizes = self.sizes()
        good = both(both((index >= 0) & (index < sizes), ok), self.valid)
        self.take(self.place, index)
        self.move(self.layout.child(self.at, self.place, index, good))
        self.valid = good
        self.at += 1

    def follow(self, space: Space, sizes: Array, full: bool) -> None:
        """Go down the dimension looped over, to the child the loop is at for each new item."""
        if full and self.span is not None and self.valid is None:
            self.take(self.place[space.parent], space.local)
            offsets = self.layout.offsets[self.at]
            self.span = (int(offsets[self.span[0]]), int(offsets[self.span[1]]))
            self._place = None
            self.at += 1
            return
        place = self.place[space.parent]
        self.take(place, space.local)
        valid = None if self.valid is None else self.valid[space.parent]
        good = both(space.local < sizes[space.parent], valid) if self.weak else valid
        self.move(self.layout.child(self.at, place, space.local, True if good is None else good))
        self.valid = good
        self.at += 1

    def result(self) -> tuple[Array, Array | None]:
        """What this branch gives where it stopped: its values, or with ``size()`` its sizes."""
        if len(self.specs) != self.layout.dims:
            return self.sizes(), self.valid
        if self.span is not None:
            return self.layout.content[self.span[0] : self.span[1]], self.valid
        return self.layout.values(self.place), self.valid


#: How a cursor learns the value of an index expression, one per entry: values and
#: validity, ``None`` for all valid.
Indexer = Callable[[Node], tuple[Array, Array]]


class Scope:
    """The loop one expression runs, and the value of every branch in it at the last level.

    ``spaces`` are the levels, entries first; ``done`` has, for each branch's
    node, its values and validity and the level at which it stopped going
    down, from where they are carried to the last level when asked for.
    """

    __slots__ = ("spaces", "done", "flat")

    def __init__(self, entries: int) -> None:
        self.spaces = [Space.of_entries(entries)]
        self.done: dict[int, tuple[Array, Array | None, int]] = {}
        #: The weak cursors that ran out of dimensions, read flat if a level follows.
        self.flat: list[Cursor] = []

    @property
    def space(self) -> Space:
        return self.spaces[-1]

    def run(self, cursors: list[Cursor], indexer: Indexer) -> None:
        """Make every level of the loop, until no branch has a dimension left to loop over."""
        active = self._settle(cursors, indexer)
        while active:
            self.spaces.append(self._expand(active))
            self._flatten()
            active = self._settle(active, indexer)

    def _settle(self, cursors: list[Cursor], indexer: Indexer) -> list[Cursor]:
        """Take each cursor through the indexed dimensions in front; keep those still looping."""
        active = []
        for cursor in cursors:
            self._index(cursor, indexer)
            if cursor.finished:
                self._record(cursor)
                if cursor.weak and cursor.at and len(cursor.specs) == cursor.layout.dims:
                    self.flat.append(cursor)  # it may yet be read flat, one level further down
            else:
                active.append(cursor)
        return active

    def _record(self, cursor: Cursor) -> None:
        """Note what a cursor gives at the level it stopped at."""
        values, valid = cursor.result()
        self.done[id(cursor.node)] = (values, valid, len(self.spaces) - 1)

    def _flatten(self) -> None:
        """Let the weak cursors read their last dimension flat at the level just made."""
        for cursor in self.flat:
            cursor.spread(self.space)
            self._record(cursor)

    def _index(self, cursor: Cursor, indexer: Indexer) -> None:
        while not cursor.finished and not cursor.looping:
            spec = cursor.specs[cursor.at]
            assert spec is not None
            index, ok = indexer(spec)
            if self.space.looped and np.ndim(index):
                index = index[self.space.entry]
                ok = None if ok is None else ok[self.space.entry]
            cursor.step(index, ok)

    def _expand(self, active: list[Cursor]) -> Space:
        """One level more: as many items per item as the shortest strong branch has elements."""
        sizes = [cursor.sizes() for cursor in active]
        strong = [
            size if cursor.valid is None else np.where(cursor.valid, size, 0)
            for cursor, size in zip(active, sizes, strict=False)
            if not cursor.weak
        ]
        if not strong:  # only weak ones: nothing to cut short, so the longest runs
            strong = [functools.reduce(np.maximum, sizes)]
        space = self.space.expand(functools.reduce(np.minimum, strong))
        for cursor, size in zip(active, sizes, strict=False):
            cursor.follow(space, size, size is space.count or np.array_equal(size, space.count))
        return space

    def reduce(self, values: Array, level: int) -> tuple[Array, Array | None]:
        """Values per item of the last level, taken down to one per item of ``level``.

        Every level of the loop is made by repeating the items of the one above
        it, so a level's values are found again in the first item under each of
        its items. An item with nothing under it has none, which is marked.
        """
        if level == len(self.spaces) - 1:
            return values, None
        width = len(self.spaces[level])
        if not len(values):  # the loop stops short of this level: nothing has a value
            return np.zeros(width, dtype=values.dtype), np.zeros(width, np.bool_)
        below, ok = self.under(level)
        return values[below], ok

    def under(self, level: int) -> tuple[Array, Array | None]:
        """For each item of one level, the first item of the last level that came from it."""
        below = self.spaces[level + 1].parent
        for space in self.spaces[level + 2 :]:
            below = below[space.parent]
        row = np.arange(len(self.spaces[level]), dtype=np.int64)
        found = np.minimum(np.searchsorted(below, row), len(below) - 1)
        ok = below[found] == row
        return found, None if ok.all() else ok

    def value(self, node: Node) -> tuple[Array, Array | None, bool]:
        """A branch's values and validity at the last level, and whether they are per item."""
        values, valid, level = self.done[id(node)]
        if level == 0:
            return values, valid, False
        for space in self.spaces[level + 1 :]:
            assert space.parent is not None
            values = values[space.parent]
            valid = None if valid is None else valid[space.parent]
        return values, valid, True
