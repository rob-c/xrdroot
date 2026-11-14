"""``TEntryList``: the entry numbers of a tree that a selection kept - or of several trees.

``tree.Draw(">>elist", cut, "entrylist")`` makes one, and
``tree.SetEntryList(elist)`` makes a loop, a ``Draw`` or a ``Scan`` go
through those entries alone. Its entries are kept in order and once each,
as ROOT keeps them, and as runs - ``EnterRange(0, 10**9)`` costs two numbers,
not a billion.

A list of entries of more than one tree holds a sub-list per tree, in the
order the trees were met (``GetLists``): entering an entry of a second tree
turns the list into one of sub-lists, and ``AddSubList`` adds one. Its
entries, ``GetN`` and ``GetEntry``, run through the sub-lists in order;
``TChain::SetEntryList(list, "sync")`` gives each sub-list the number of the
chain's tree it is of, which ``GetEntryAndTree`` hands back.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from ._base import _TObjectLike

__all__ = ["TEntryList"]

#: An empty run of entries.
_NONE = np.zeros(0, dtype=np.int64)


class _Runs:
    """Entry numbers as sorted, disjoint runs ``[start, end)``."""

    def __init__(self) -> None:
        self.starts, self.ends = _NONE, _NONE

    def add(self, starts: Any, ends: Any) -> None:
        """The runs ``[starts[i], ends[i])`` added, merged with those held."""
        low = np.concatenate([self.starts, np.asarray(starts, dtype=np.int64)])
        high = np.concatenate([self.ends, np.asarray(ends, dtype=np.int64)])
        keep = high > low
        low, high = low[keep], high[keep]
        if not len(low):
            return
        order = np.argsort(low, kind="stable")
        low, high = low[order], high[order]
        reach = np.maximum.accumulate(high)
        breaks = np.concatenate([[True], low[1:] > reach[:-1]])
        firsts = np.flatnonzero(breaks)
        self.starts, self.ends = low[firsts], np.maximum.reduceat(high, firsts)

    def remove(self, entry: int) -> None:
        """One entry taken out, splitting the run it is in."""
        at = self.find(entry)
        start, end = int(self.starts[at]), int(self.ends[at])
        pieces_low = [s for s, e in ((start, entry), (entry + 1, end)) if e > s]
        pieces_high = [e for s, e in ((start, entry), (entry + 1, end)) if e > s]
        self.starts = np.concatenate([self.starts[:at], pieces_low, self.starts[at + 1:]])
        self.ends = np.concatenate([self.ends[:at], pieces_high, self.ends[at + 1:]])
        self.starts, self.ends = self.starts.astype(np.int64), self.ends.astype(np.int64)

    def find(self, entry: int) -> int:
        """The run holding ``entry``, or -1."""
        at = int(np.searchsorted(self.starts, entry, side="right")) - 1
        return at if at >= 0 and entry < self.ends[at] else -1

    def count(self) -> int:
        return int((self.ends - self.starts).sum())

    def at(self, index: int) -> int:
        """The ``index``-th entry, in order, or -1."""
        lengths = self.ends - self.starts
        totals = np.cumsum(lengths)
        if not 0 <= index < (int(totals[-1]) if len(totals) else 0):
            return -1
        run = int(np.searchsorted(totals, index, side="right"))
        return int(self.starts[run] + index - (totals[run] - lengths[run]))

    def array(self) -> np.ndarray[Any, Any]:
        if not len(self.starts):
            return _NONE
        return np.concatenate([np.arange(s, e, dtype=np.int64)
                               for s, e in zip(self.starts, self.ends, strict=True)])  # fmt: skip


def _names_of(tree: Any, entering: bool) -> tuple[str, str]:
    """The tree name and file name a list records of ``tree``: its path in its file and the
    file's, or - a tree in memory - its name, which ``Enter`` prefixes with ``gROOT``'s."""
    found = getattr(tree, "GetCurrentFile", lambda: None)()
    if found is not None:
        from pathlib import Path

        return str(tree.GetName()), str(Path(str(found.GetName())).resolve())
    if not entering:
        return str(tree.GetName()), ""
    from ..core.troot import gROOT

    return f"{gROOT.GetName()}/{tree.GetName()}", ""


class TEntryList(_TObjectLike):
    """``TEntryList([name, title[, tree | treename, filename]])``: entry numbers, in order, each
    once - of one tree, or of several in sub-lists."""

    _classname = "TEntryList"

    def __init__(self, name: str = "", title: str = "", tree: Any = None,
                 filename: Any = None) -> None:  # fmt: skip
        super().__init__(name, title)
        self._runs = _Runs()
        self._lists: list[TEntryList] | None = None
        self._tree_name, self._file_name = "", ""
        if isinstance(tree, str):
            self._tree_name, self._file_name = tree, str(filename or "")
        elif tree is not None:
            self._tree_name, self._file_name = _names_of(tree, False)
        self._tree_number = -1
        self._next = 0

    def __repr__(self) -> str:
        return f"<TEntryList {self._name!r} of {self.GetN()} entries>"

    def __len__(self) -> int:
        return self.GetN()

    def __iter__(self) -> Iterator[int]:
        return iter(self._entries().tolist())

    def _entries(self) -> np.ndarray[Any, Any]:
        if self._lists is not None:
            return np.concatenate([_NONE, *(each._entries() for each in self._lists)])
        return self._runs.array()

    # -- which list an entry goes in ------------------------------------------------------

    def _target(self, tree: Any, filename: Any = None, entering: bool = True) -> TEntryList:
        """The list an entry of ``tree`` (or of the tree named so, in ``filename``) goes in:
        this one, or - once a second tree is met - the sub-list for it. A selection drawn
        into the list (``entering`` false) names a tree in memory as the tree does."""
        if tree is None:
            return self
        names = (tree, str(filename or "")) if isinstance(tree, str) else \
            _names_of(tree, entering)
        if self._lists is None and (not self._tree_name or names == self._names()):
            self._tree_name, self._file_name = names
            return self
        return self._sub_list(names)

    def _names(self) -> tuple[str, str]:
        return self._tree_name, self._file_name

    def _sub_list(self, names: tuple[str, str]) -> TEntryList:
        if self._lists is None:  # this list's entries become the first sub-list
            first = TEntryList("", "", *self._names())
            first._runs, self._runs = self._runs, _Runs()
            self._lists = [first]
        found = next((each for each in self._lists if each._names() == names), None)
        if found is None:
            found = TEntryList("", "", *names)
            self._lists.append(found)
        return found

    # -- entries -----------------------------------------------------------------------------

    def _enter_many(self, entries: Iterable[int], tree: Any = None) -> None:
        values = np.unique(np.fromiter((int(e) for e in entries), dtype=np.int64))
        self._target(tree, entering=False)._runs.add(values, values + 1)

    def Enter(self, entry: int, tree: Any = None, filename: Any = None) -> bool:
        """``Enter(entry[, tree])`` or ``Enter(entry, treename, filename)``: whether it is new."""
        target = self._target(tree, filename)
        new = target._runs.find(int(entry)) < 0
        target._runs.add([int(entry)], [int(entry) + 1])
        return new

    def EnterRange(self, start: Any, end: Any, tree: Any = None, step: Any = 1) -> int:
        """``EnterRange(start, end[, tree])``: the entries from ``start`` up to ``end``; how many
        were added."""
        target = self._target(tree)
        before = target._runs.count()
        if int(step) == 1:
            target._runs.add([int(start)], [int(end)])
        else:
            values = np.arange(int(start), int(end), int(step), dtype=np.int64)
            target._runs.add(values, values + 1)
        return target._runs.count() - before

    def Remove(self, entry: int, tree: Any = None) -> bool:
        target = self._target(tree)
        found = target._runs.find(int(entry)) >= 0
        if found:
            target._runs.remove(int(entry))
        return found

    def Contains(self, entry: int, tree: Any = None) -> int:
        return int(self._target(tree)._runs.find(int(entry)) >= 0)

    def GetN(self) -> int:
        if self._lists is not None:
            return sum(each.GetN() for each in self._lists)
        return self._runs.count()

    def GetEntry(self, index: int) -> int:
        """The ``index``-th entry number, in order - through the sub-lists - or ``-1`` past the
        end."""
        return self.GetEntryAndTree(index)[0] if self._lists is not None else \
            self._runs.at(int(index))

    def GetEntryAndTree(self, index: Any, treenum: Any = None) -> Any:
        """``GetEntryAndTree(index, treenum)``: the entry in its tree, and - into ``treenum`` -
        which of the chain's trees it is in, as ``SetEntryList(list, "sync")`` numbered them."""
        from ..core.refs import store

        found, number = self._located(int(index))
        if treenum is not None:
            store(treenum, number)
        return found if treenum is not None else (found, number)

    def _located(self, index: int) -> tuple[int, int]:
        if self._lists is None:
            found = self._runs.at(index)
            return found, (self._tree_number if found >= 0 else -1)
        for each in self._lists:
            size = each.GetN()
            if index < size:
                return each._runs.at(index), each._tree_number
            index -= size
        return -1, -1

    def Next(self) -> int:
        """The entry after the one ``Next`` gave last, or ``-1`` when there are no more."""
        found = self.GetEntry(self._next)
        self._next += 1
        return found

    def Add(self, other: TEntryList) -> None:
        """``Add``: another list's entries - of its own tree, or of each of its sub-lists."""
        for each in other._lists if other._lists is not None else [other]:
            target = self._target(each._tree_name or None, each._file_name)
            target._runs.add(each._runs.starts, each._runs.ends)

    def AddSubList(self, other: TEntryList) -> None:
        """``AddSubList``: ``other`` as the next sub-list."""
        if self._lists is None:
            self._lists = []
        self._lists.append(other)

    def GetLists(self) -> Any:
        """``GetLists``: the sub-lists, a ``TList``, or ``None`` when this list is of one tree."""
        if self._lists is None:
            return None
        from ..core.collections import TList

        made = TList()
        for each in self._lists:
            made.Add(each)
        return made

    def GetEntryList(self, treename: Any, filename: Any = "", option: Any = "") -> Any:
        """``GetEntryList(treename, filename)``: the sub-list of that tree, or ``None``."""
        wanted = (str(treename), str(filename or ""))
        lists = self._lists if self._lists is not None else [self]
        return next((each for each in lists if each._names() == wanted), None)

    def Reset(self) -> None:
        self._runs, self._lists, self._next = _Runs(), None, 0

    def SetTree(self, tree: Any, filename: str = "") -> None:
        if isinstance(tree, str):
            self._tree_name, self._file_name = tree, str(filename)
        else:
            self._tree_name, self._file_name = _names_of(tree, False)

    def SetTreeName(self, name: Any) -> None:
        self._tree_name = str(name)

    def SetFileName(self, name: Any) -> None:
        self._file_name = str(name)

    def SetTreeNumber(self, number: Any) -> None:
        self._tree_number = int(number)

    def GetTreeName(self) -> str:
        return self._tree_name

    def GetFileName(self) -> str:
        return self._file_name

    def GetTreeNumber(self) -> int:
        return self._tree_number

    def Print(self, option: str = "") -> None:
        """What ``TEntryList::Print`` shows: tree, file and count, and with ``all`` each entry."""
        print(f"{self._tree_name} {self._file_name} {self.GetN()}")
        if "all" in str(option).lower():
            for entry in self._entries().tolist():
                print(entry)
