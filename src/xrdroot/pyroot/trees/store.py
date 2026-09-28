"""A tree being filled: what each ``Fill`` read, kept until it is written.

ROOT's ``Fill`` reads every branch's address and puts what it finds in the
branch's basket. Here each leaf is a :class:`Slot` that reads its address
into a list, and the lists are turned into columns - NumPy arrays, rows of
:class:`~xrdroot.Jagged`, lists of strings - only when the tree is written or
read back. Writing hands the columns to :class:`xrdroot.WritableTree`, which
lays them out as ROOT does; reading back writes them into a ROOT file in
memory and opens it with :func:`xrdroot.open_root`, so a tree still being
filled is drawn, scanned and read by exactly the code that reads one from a
file.

What a leaf is follows from how it was declared: a number, a fixed array
(``x[3]/F``), an array counted by another leaf of the same tree
(``x[n]/F``, written as ROOT writes it: the counter's branch, then the
array's, pointing at it), a ``std::vector`` (written as rows with a counter
of their own, ``n`` and the branch's name, since this writer lays out rows
the leaf-list way) or text (``/C`` or a ``std::string``).
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

import numpy as np

from ...tree import Jagged, concatenate
from ..stl import cpp_name
from .addresses import Address
from .leaflist import LEAF_CLASSES, TYPE_NAMES

__all__ = ["Slot", "Store", "memory_tree"]

#: The kinds of leaf a slot can be, which say how it is read and written.
SCALAR, FIXED, COUNTED, VECTOR, TEXT = "scalar", "fixed", "counted", "vector", "text"

#: The type codes a counter can be: the integers.
INTEGERS = "bBhHiIqQ"


class Slot:
    """One leaf being filled: where its values come from, and what they have been.

    ``branch`` is the name of the branch it is a leaf of, which is its own
    name unless the branch was given a list of several leaves.
    """

    def __init__(
        self,
        name: str,
        branch: str,
        code: str,
        address: Address,
        *,
        size: int = 1,
        counter: str | None = None,
        title: str | None = None,
    ) -> None:
        self.name = name
        self.branch = branch
        self.code = code
        self.address = address
        self.size = size
        self.counter = counter
        self.title = name if title is None else title
        self.kind = _kind(code, address, size, counter)
        self.pending: list[Any] = []
        self.chunks: list[Any] = []

    def __repr__(self) -> str:
        return f"<Slot {self.name!r} of {self.typename} ({self.kind})>"

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self.code)

    @property
    def typename(self) -> str:
        """What ``TLeaf::GetTypeName`` says: ``Float_t``, or the vector's class."""
        if self.kind == VECTOR:
            return f"vector<{cpp_name(self.dtype)}>"
        if self.kind == TEXT and not self.title.endswith("/C"):
            return "string"
        return TYPE_NAMES[self.code]

    @property
    def classname(self) -> str:
        """The leaf's class: ``TLeafF`` and its kin, or ``TLeafElement`` for a vector."""
        return "TLeafElement" if self.kind == VECTOR else LEAF_CLASSES[self.code]

    def read(self, counts: Mapping[str, int]) -> Any:
        """This entry's value, from the address: a number, an array or a string."""
        if self.kind in (SCALAR, TEXT, VECTOR):
            return self.address.get()
        count = self.size if self.counter is None else counts[self.counter] * self.size
        room = self.address.room
        if room is not None and count > room:
            raise ValueError(
                f"{self.name!r} is counted by {self.counter!r}, which says {count} this entry, "
                f"and its address has room for {room}"
            )
        return self.address.get(count)

    def keep(self, value: Any) -> None:
        self.pending.append(value)

    def seal(self) -> None:
        """Turn what has been filled so far into a column, joined to what went before."""
        if self.pending:
            self.chunks.append(self._assembled(self.pending))
            self.pending = []

    def _assembled(self, values: list[Any]) -> Any:
        if self.kind == TEXT:
            return [str(value) for value in values]
        if self.kind in (COUNTED, VECTOR):
            return _jagged(values, self.dtype)
        array = np.asarray(values, dtype=self.dtype)
        return array.reshape(len(values), self.size) if self.kind == FIXED else array

    def column(self) -> Any:
        """Every entry filled, as one column."""
        self.seal()
        if len(self.chunks) > 1:
            self.chunks = [concatenate(self.chunks)]
        return self.chunks[0]

    def spec(self) -> Any:
        """What :meth:`xrdroot.WritableDirectory.tree` is told this column is."""
        if self.kind == TEXT:
            return str
        if self.kind == VECTOR:
            return (self.code, None)
        if self.kind == COUNTED:
            return (self.code, self.counter)
        return (self.code, self.size) if self.kind == FIXED else self.code


def _kind(code: str, address: Address, size: int, counter: str | None) -> str:
    if code == "C" or address.text:
        return TEXT
    if address.sized:
        return VECTOR
    if counter is not None:
        return COUNTED
    return FIXED if size > 1 else SCALAR


def _jagged(rows: list[Any], dtype: np.dtype[Any]) -> Jagged:
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = [np.asarray(row, dtype=dtype).reshape(-1) for row in rows]
    content = np.concatenate(flat) if flat else np.zeros(0, dtype=dtype)
    return Jagged(content.astype(dtype, copy=False), offsets)


class Store:
    """Every slot of a tree being filled, and how many entries they hold."""

    def __init__(self) -> None:
        self.slots: dict[str, Slot] = {}
        #: The title of each branch of several leaves: the leaf list it was given.
        self.titles: dict[str, str] = {}
        self.entries = 0

    def add(self, slot: Slot) -> None:
        self.slots[slot.name] = slot

    def counters(self) -> dict[str, str]:
        """The leaves that count another's values, with the integer type each is."""
        named = {slot.counter for slot in self.slots.values() if slot.counter is not None}
        return {name: self.slots[name].code for name in named}

    def fill(self) -> int:
        """Read every slot's address once, as one entry; the bytes it took come back."""
        counts: dict[str, int] = {}
        read: list[tuple[Slot, Any]] = []
        for slot in self.slots.values():
            value = slot.read(counts)
            if slot.kind == SCALAR and slot.code in INTEGERS:
                counts[slot.name] = int(value)
            read.append((slot, value))
        for slot, value in read:  # only once every address has been read without complaint
            slot.keep(value)
        self.entries += 1
        return sum(_nbytes(slot, value) for slot, value in read)

    def extend(self, columns: Mapping[str, Any], count: int) -> None:
        """Take in many entries at once, a column for every slot, as a copy of a tree does."""
        for slot in self.slots.values():
            slot.seal()
            slot.chunks.append(columns[slot.name])
        self.entries += count

    def columns(self) -> dict[str, Any]:
        """Every column to write: all but the counters, which are written from the rows."""
        counted = self.counters()
        return {name: slot.column() for name, slot in self.slots.items() if name not in counted}

    def specs(self) -> dict[str, Any]:
        counted = self.counters()
        return {name: slot.spec() for name, slot in self.slots.items() if name not in counted}

    def write(self, directory: Any, name: str, title: str, classname: str = "TTree") -> Any:
        """Write every entry into ``directory``, as a tree of this name, title and class."""
        tree = directory.tree(
            name, self.specs(), title=title, counters=self.counters(), classname=classname
        )
        if self.entries:
            tree.extend(self.columns())
        return tree

    def reset(self) -> None:
        """Forget every entry, keeping the branches: ``TTree::Reset``."""
        for slot in self.slots.values():
            slot.pending, slot.chunks = [], []  # the branches and their titles stay
        self.entries = 0


def _nbytes(slot: Slot, value: Any) -> int:
    if slot.kind == TEXT:
        return len(str(value)) + 1
    return int(np.size(value)) * int(slot.dtype.itemsize)


def memory_tree(name: str, write: Any) -> Any:
    """A tree written by ``write(directory)`` into a ROOT file in memory, and read back.

    This is how anything here that is not in a file yet - a tree being
    filled, the entries an entry list picks - is read by the same reader as
    everything that is.
    """
    from ... import create, open_root

    buffer = io.BytesIO()
    with create(buffer) as out:
        write(out)
    return open_root(io.BytesIO(buffer.getvalue()))[name]
