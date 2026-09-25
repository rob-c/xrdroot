"""``TTreeReader``, ``TTreeReaderValue`` and ``TTreeReaderArray``: a tree walked entry by entry.

    >>> reader = TTreeReader("ntuple", f)                          # doctest: +SKIP
    >>> px = TTreeReaderValue['float'](reader, "px")               # doctest: +SKIP
    >>> while reader.Next():                                       # doctest: +SKIP
    ...     h.Fill(px.__deref__())          # *px in C++; px.Get()[0] as PyROOT has it

A value's ``Get()`` is what C++'s ``Get()`` points at: a one-element array
for a number, so that ``px.Get()[0]`` is the number, and the vector or
string itself for one of those; ``__deref__()`` - C++'s ``*px`` - is the
number itself. An array reads as a sequence: ``len``, ``[i]``, iteration,
``GetSize()``. Each is read from the tree's own block of entries, so a loop
over a reader costs what a loop of ``GetEntry`` does.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ._base import _TObjectLike
from .core import python_value

__all__ = ["TTreeReader", "TTreeReaderValue", "TTreeReaderArray"]

#: ``TTreeReader::EEntryStatus``: what ``SetEntry`` says.
ENTRY_VALID, ENTRY_NO_TREE, ENTRY_BEYOND_END = 0, 2, 7


def _tree_of(tree: Any, where: Any) -> Any:
    """The tree a reader walks: a tree, or a tree's name and the file or directory it is in."""
    from . import wrap

    if not isinstance(tree, str):
        return tree if hasattr(tree, "_batch") else wrap(tree)
    if where is None:
        raise ValueError(
            f"a TTreeReader of {tree!r} needs the file it is in, as in TTreeReader({tree!r}, f)"
        )
    if isinstance(where, str):
        from ... import open_root

        where = open_root(where)
    found = where.Get(tree) if hasattr(where, "Get") else where[tree]
    return found if hasattr(found, "_batch") else wrap(found)


class TTreeReader(_TObjectLike):
    """``TTreeReader``: see the module's docstring."""

    _classname = "TTreeReader"

    def __init__(self, tree: Any = None, where: Any = None, entrylist: Any = None) -> None:
        super().__init__()
        self._tree = None if tree is None else _tree_of(tree, where)
        self._entry = -1
        self._begin, self._end = 0, -1
        self._list = entrylist

    def __iter__(self) -> Iterator[int]:
        while self.Next():
            yield self._entry

    def _stop(self) -> int:
        total = self.GetEntries()
        return total if self._end < 0 else min(self._end, total)

    def Next(self) -> bool:
        """Move to the next entry; whether there was one."""
        return self.SetEntry(max(self._entry + 1, self._begin)) == ENTRY_VALID

    def SetEntry(self, entry: int) -> int:
        """Make ``entry`` the current one; ``0``, ``kEntryValid``, if it is there."""
        self._entry = int(entry)
        if self._tree is None:
            return ENTRY_NO_TREE
        if not 0 <= self._entry < self._stop():
            return ENTRY_BEYOND_END
        number = self._entry if self._list is None else self._list.GetEntry(self._entry)
        self._tree._read_entry = int(number)
        return ENTRY_VALID

    def SetEntriesRange(self, begin: int, end: int = -1) -> int:
        """Walk only entries ``begin`` up to ``end``, ``-1`` meaning to the last."""
        self._begin, self._end = int(begin), int(end)
        self._entry = int(begin) - 1
        return ENTRY_VALID

    def Restart(self) -> None:
        self._entry = self._begin - 1

    def GetCurrentEntry(self) -> int:
        return self._entry

    def GetEntries(self, force: bool = False) -> int:
        if self._tree is None:
            return 0
        return int(self._tree.GetEntries() if self._list is None else self._list.GetN())

    def GetTree(self) -> Any:
        return self._tree

    def SetTree(self, tree: Any, where: Any = None) -> None:
        self._tree = _tree_of(tree, where)
        self._entry = self._begin - 1

    def IsChain(self) -> bool:
        return self._tree is not None and self._tree.ClassName() == "TChain"


class _Read:
    """What a value and an array share: the reader, the branch, and its value now."""

    #: The C++ type asked for, when the class was made for one.
    _type: Any = None

    def __class_getitem__(cls, kind: Any) -> type:
        return type(f"{cls.__name__}<{kind}>", (cls,), {"_type": kind})

    def __init__(self, reader: TTreeReader, name: str) -> None:
        self._reader = reader
        self._name = str(name)
        tree = reader._tree
        self._leaf: Any = tree._leaf_info(self._name) if tree is not None else None
        if tree is not None and self._leaf is None:
            raise KeyError(
                f"the tree {tree.GetName()!r} has no branch called {self._name!r} to read"
            )

    def _value(self) -> Any:
        tree = self._reader._tree
        if tree is None or tree._read_entry < 0:
            raise ValueError(
                f"{self._name!r} has no value until the reader is at an entry; call Next first"
            )
        return tree._current(self._leaf.column)

    def IsValid(self) -> bool:
        return self._leaf is not None

    def GetBranchName(self) -> str:
        return self._name


class TTreeReaderValue(_Read):
    """``TTreeReaderValue<T>``: one branch's value at the reader's entry."""

    def Get(self) -> Any:
        """What C++'s ``Get()`` points at: a one-element array of a number, or the object."""
        value = python_value(self._value(), self._leaf.vector)
        if isinstance(value, (int, float, bool, np.number)):
            return np.array([value])
        return value

    def __deref__(self) -> Any:
        """C++'s ``*value``: the number itself, or the object."""
        return python_value(self._value(), self._leaf.vector)

    def __float__(self) -> float:
        return float(self.__deref__())

    def __int__(self) -> int:
        return int(self.__deref__())


class TTreeReaderArray(_Read):
    """``TTreeReaderArray<T>``: one branch's collection at the reader's entry, as a sequence."""

    def _items(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._value()).reshape(-1)

    def __len__(self) -> int:
        return len(self._items())

    def __getitem__(self, index: Any) -> Any:
        found = self._items()[index]
        return found.item() if isinstance(found, np.generic) else found

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items().tolist())

    def GetSize(self) -> int:
        return len(self)

    size = GetSize

    def IsEmpty(self) -> bool:
        return len(self) == 0

    def At(self, index: int) -> Any:
        return self[index]
