"""``CloneTree`` and ``CopyTree``: a tree's branches, and some of its entries, as a new tree.

A clone has a slot for every active leaf of the tree it came from, each
reading - as ROOT's clone shares its branches' buffers - the value that tree
last read into it, so ``CloneTree(0)`` followed by a loop of ``GetEntry`` and
``Fill`` copies the entries the loop chooses. Whatever the clone is to start
with - every entry, the first ``n``, or those a selection keeps - is copied
across whole columns at a time rather than an entry at a time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...tree import Jagged
from .addresses import Address
from .store import Slot, memory_tree

__all__ = ["Follow", "clone", "subset"]

#: Type codes whose width is the platform's, and the ones that are not.
PLATFORM = {"l": "q", "L": "Q"}


class Follow(Address):
    """A clone's address: whatever the tree it was cloned from holds for one leaf now."""

    room = None

    def __init__(self, tree: Any, column: str, sized: bool, text: bool) -> None:
        self.tree = tree
        self.column = column
        self.sized = sized
        self.text = text

    def get(self, count: int | None = None) -> Any:
        bound = self.tree._addresses.get(self.column)
        if bound is not None:
            return bound.get(count)
        value = self.tree._current(self.column)
        if value is None:
            raise ValueError(
                f"the clone reads {self.column!r} from the entry {self.tree.GetName()!r} last "
                f"read, and it has read none yet; call GetEntry on it before Fill"
            )
        if count is None:
            return value.item() if isinstance(value, np.generic) else value
        return np.asarray(value).reshape(-1)[:count]

    def put(self, value: Any) -> None:
        """Nothing: a clone's leaves are the other tree's, which ``GetEntry`` fills there."""


def _code(sample: Any) -> str:
    dtype = sample.content.dtype if isinstance(sample, Jagged) else np.asarray(sample).dtype
    return str(PLATFORM.get(dtype.char, dtype.char))


def _wanted(tree: Any) -> list[Any]:
    """The leaves a clone takes: every active one, and the counters those need."""
    leaves = tree._leaves()
    active = {leaf.column for leaf in leaves if tree._active(leaf.column)}
    active |= {leaf.counter for leaf in leaves if leaf.column in active and leaf.counter}
    return [leaf for leaf in leaves if leaf.column in active]


def _slot(tree: Any, leaf: Any, backing: Any) -> Slot:
    sample = backing[leaf.column].array(0, 0)
    code = "C" if leaf.text else _code(sample)
    follow = Follow(tree, leaf.column, leaf.vector, leaf.text)
    name = leaf.column if "." not in leaf.column else leaf.name
    return Slot(
        name,
        leaf.branch if "." not in leaf.branch else name,
        code,
        follow,
        size=leaf.size,
        counter=leaf.counter,
        title=leaf.title,
    )


def clone(tree: Any, made: Any, start: int, stop: int, selection: str | None) -> Any:
    """Give ``made`` a slot for each leaf ``tree`` has, and the entries asked for."""
    backing = tree._backing()
    leaves = _wanted(tree)
    slots = [_slot(tree, leaf, backing) for leaf in leaves]
    for slot in slots:
        made._store.add(slot)
    if stop > start:
        columns = [leaf.column for leaf in leaves]
        read = backing.arrays(columns, start, stop, cut=selection or None)
        count = len(read[columns[0]]) if columns else 0
        made._store.extend({s.name: read[c] for s, c in zip(slots, columns)}, count)
    made._changed()
    return made


def subset(backing: Any, rows: Any, name: str) -> Any:
    """The entries ``rows`` names of a tree, as a tree of their own, read from memory."""
    own = [label for label in backing.readable() if "." not in label]
    read = backing.arrays(own, entries=np.asarray(rows, dtype=np.int64))
    counted = {
        label: backing[label].leaf.count.name
        for label in own
        if backing[label].is_jagged and backing[label].leaf.count is not None
    }
    counted = {label: counter for label, counter in counted.items() if counter in read}

    def write(out: Any) -> None:
        from ...wtree import spec_of

        specs = {
            label: (_code(values), counted[label]) if label in counted else spec_of(label, values)
            for label, values in read.items()
            if label not in counted.values()
        }
        types = {counter: _code(read[counter]) for counter in counted.values()}
        tree = out.tree(name, specs, counters=types)
        tree.extend({label: read[label] for label in specs})

    return memory_tree(name, write)
