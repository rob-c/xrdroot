"""``TEntryList``: the entry numbers of a tree that a selection kept.

``tree.Draw(">>elist", cut, "entrylist")`` makes one, and
``tree.SetEntryList(elist)`` makes a loop, a ``Draw`` or a ``Scan`` go
through those entries alone. Its entries are kept in order and once each,
as ROOT keeps them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from ._base import _TObjectLike

__all__ = ["TEntryList"]


class TEntryList(_TObjectLike):
    """``TEntryList``: entry numbers, in order, each once."""

    _classname = "TEntryList"

    def __init__(self, name: str = "", title: str = "", tree: Any = None) -> None:
        super().__init__(name, title)
        self._set: set[int] = set()
        self._sorted: np.ndarray[Any, Any] | None = None
        self._tree_name = tree.GetName() if tree is not None else ""
        self._file_name = ""
        self._next = 0

    def __repr__(self) -> str:
        return f"<TEntryList {self._name!r} of {self.GetN()} entries>"

    def __len__(self) -> int:
        return len(self._set)

    def __iter__(self) -> Iterator[int]:
        return iter(self._entries().tolist())

    def _entries(self) -> np.ndarray[Any, Any]:
        if self._sorted is None:
            self._sorted = np.array(sorted(self._set), dtype=np.int64)
        return self._sorted

    def _enter_many(self, entries: Iterable[int], tree: Any = None) -> None:
        self._set.update(int(entry) for entry in entries)
        self._sorted = None
        if tree is not None and not self._tree_name:
            self._tree_name = tree.GetName()

    def Enter(self, entry: int, tree: Any = None) -> bool:
        """Add ``entry``; whether it was not there already."""
        new = int(entry) not in self._set
        self._enter_many([entry], tree)
        return new

    def Remove(self, entry: int, tree: Any = None) -> bool:
        found = int(entry) in self._set
        self._set.discard(int(entry))
        self._sorted = None
        return found

    def Contains(self, entry: int, tree: Any = None) -> int:
        return int(int(entry) in self._set)

    def GetN(self) -> int:
        return len(self._set)

    def GetEntry(self, index: int) -> int:
        """The ``index``-th entry number, in order, or ``-1`` past the end."""
        entries = self._entries()
        return int(entries[index]) if 0 <= index < len(entries) else -1

    def Next(self) -> int:
        """The entry after the one ``Next`` gave last, or ``-1`` when there are no more."""
        found = self.GetEntry(self._next)
        self._next += 1
        return found

    def Add(self, other: TEntryList) -> None:
        self._enter_many(other._set)

    def Reset(self) -> None:
        self._set.clear()
        self._sorted = None
        self._next = 0

    def SetTree(self, tree: Any, filename: str = "") -> None:
        self._tree_name = tree if isinstance(tree, str) else tree.GetName()
        self._file_name = str(filename)

    def GetTreeName(self) -> str:
        return self._tree_name

    def Print(self, option: str = "") -> None:
        """What ``TEntryList::Print`` shows: tree, file and count, and with ``all`` each entry."""
        print(f"{self._tree_name} {self._file_name} {self.GetN()}")
        if "all" in str(option).lower():
            for entry in self._entries().tolist():
                print(entry)
