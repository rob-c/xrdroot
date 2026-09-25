"""How many bytes ROOT counts for a branch's own record, which ``Print`` adds to its baskets'.

``TBranch::GetTotalSize`` is the bytes of the branch's baskets plus the
length of the branch streamed on its own - its name, title, counters, basket
tables and leaf, and the leaf counting it, streamed whole since nothing
before it in that buffer has. This writer streams a branch exactly that way,
so the length is found by streaming one like it, taking away the class tag in
front that ``GetTotalSize`` does not stream, and allowing for names and
titles of other lengths than the ones this writer would give. A leaf of a
class this writer does not write - an object, a ``std::vector`` - has no such
twin, and counts its baskets alone.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from ...writer import BASKET_BYTES, WBuffer
from ...wtree import LEAVES, _branch, _Column, _Counter, _leaf, _Rows, _Text

__all__ = ["Streamed", "streamed"]

#: The bytes in front of a branch streamed as an object - byte count, new-class
#: tag and ``TBranch``'s name - that ``TBranch::Streamer`` alone does not write.
TAG_BYTES = 13

#: The byte count in front of an object streamed as an object of its own.
COUNT_BYTES = 4

#: The array type code of each plain leaf class, signed and unsigned.
CODES = {(leaf, unsigned): code for code, (leaf, _, _, unsigned) in LEAVES.items()}


class Streamed(NamedTuple):
    """What a one-leaf branch's record is made of, as far as its length goes."""

    name: str
    title: str
    leaf_name: str
    leaf_title: str
    classname: str
    unsigned: bool
    size: int
    counter: str | None
    baskets: int


def _column(branch: Streamed) -> Any:
    if branch.classname == "TLeafC":
        return _Text(branch.name, BASKET_BYTES)
    code = CODES.get((branch.classname, branch.unsigned))
    code = code or CODES.get((branch.classname, not branch.unsigned))
    if code is None:
        return None
    if branch.counter is not None:
        return _Rows(branch.name, code, BASKET_BYTES, _Counter(branch.counter, BASKET_BYTES))
    return _Column(branch.name, code, branch.size, BASKET_BYTES)


def _counter_leaf(column: Any) -> int:
    """The counter's leaf, streamed whole where the counted leaf points at it.

    Streamed as the object a pointer leads to, it has no byte count of its
    own in front, and its class is named only if the counted leaf's is
    another: a class already streamed in the buffer is a reference to it.
    """
    if not isinstance(column, _Rows):
        return 0
    buf = WBuffer()
    _leaf(buf, column.counter, 0)
    named = len(column.counter.classname) + 1
    return (
        len(buf.data) - COUNT_BYTES - (named if column.counter.classname == column.classname else 0)
    )


def streamed(branch: Streamed) -> int:
    """The length of a one-leaf branch streamed alone, or 0 for a leaf of another class."""
    column = _column(branch)
    if column is None:
        return 0
    column.seeks, column.sizes = [0] * branch.baskets, [0] * branch.baskets
    column.starts = [0] * (branch.baskets + 1)
    buf = WBuffer()
    _branch(buf, column, 0, 1, 0)
    renamed = (
        len(branch.title)
        - len(column.title)
        + len(branch.leaf_name)
        - len(branch.name)
        + len(branch.leaf_title)
        - len(column.leaf_title)
    )
    return len(buf.data) - TAG_BYTES + renamed + _counter_leaf(column)
