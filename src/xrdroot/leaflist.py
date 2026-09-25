"""A branch of several leaves, some of them of a length that changes per entry.

``TTree::Branch("orange", &event, "n/I:px[n]/F:py[n]/F:q2/F")`` writes every
leaf of an entry one after another into the same bytes, so where ``py``
starts depends on how many ``px`` there were, and so does where ``q2`` is -
the offset its leaf records is right only for an entry with no tracks at all.
ROOT reads such a branch leaf by leaf, from the front of each entry, and so
does this: the counters it meets on the way say how long each array after
them is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .buffer import numbers

if TYPE_CHECKING:
    from .interp import Numeric
    from .objects import BranchRecord, LeafRecord
    from .tree import Branch

__all__ = ["walkable", "walked"]


def walkable(record: BranchRecord) -> bool:
    """Is this a branch of several plain leaves, one of variable length, counted inside it?"""
    from .objects import LEAF_TYPES

    leaves = record.leaves
    if len(leaves) < 2 or all(leaf.count is None for leaf in leaves):
        return False
    seen: set[str] = set()
    for leaf in leaves:
        if leaf.classname not in LEAF_TYPES or leaf.classname == "TLeafC":
            return False
        if leaf.count is not None and leaf.count.name not in seen:
            return False
        seen.add(leaf.name)
    return True


def _spans(data: bytes, at: int, leaves: list[LeafRecord]) -> list[tuple[int, int]]:
    """Where each leaf's values start in one entry, and how many there are."""
    counts: dict[str, int] = {}
    spans = []
    for leaf in leaves:
        many = leaf.length * (counts[leaf.count.name] if leaf.count is not None else 1)
        spans.append((at, many))
        if leaf.length == 1 and leaf.count is None and leaf.typecode not in "fd":
            counts[leaf.name] = int(numbers(data[at : at + leaf.itemsize], str(leaf.typename))[0])
        at += many * leaf.itemsize
    return spans


def walked(branch: Branch, start: int, stop: int) -> Any:
    """One leaf's values over ``[start, stop)``, found by walking each entry from its front."""
    from .tree import Jagged

    leaves = branch.record.leaves
    which = leaves.index(branch.leaf)
    size = branch.leaf.itemsize
    pieces, lengths = [], []
    for index, low, high in branch._spans(start, stop):
        basket = branch.basket(index)
        for entry in range(low, high):
            at, many = _spans(basket.data, basket.start_of(entry, 0), leaves)[which]
            pieces.append(basket.data[at : at + many * size])
            lengths.append(many)
    values = cast("Numeric", branch.column).decode(b"".join(pieces))
    if branch.leaf.count is None:
        return values.reshape(-1, branch.leaf.length) if branch.leaf.length > 1 else values
    offsets = np.zeros(len(lengths) + 1, np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return Jagged(values, offsets)
