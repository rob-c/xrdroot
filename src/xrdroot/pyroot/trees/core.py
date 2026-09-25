"""What every ``TTree`` here is underneath: a store being filled, or a tree that was read.

A tree made by ``TTree(name, title)`` fills a :class:`~.store.Store`; one a
file handed back reads an :class:`xrdroot.TTree` or :class:`xrdroot.Chain`.
Either way everything that reads - ``GetEntry``, ``Draw``, ``Scan`` - reads
:meth:`_TreeCore._backing`, which for a tree being filled is what it holds so
far, written into memory and read back, made again only once more has been
filled.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..stl import std
from ._base import _TObjectLike, hooks
from .addresses import Address
from .batch import Batch
from .layout import BranchInfo, from_store, from_tree
from .store import Store, memory_tree

__all__ = ["_TreeCore", "python_value"]


def python_value(value: Any, vector: bool) -> Any:
    """A value as a PyROOT script is handed it: a number, an array, or a ``std::vector``."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    if vector and isinstance(value, np.ndarray):
        return std.vector[value.dtype](value)
    return value


class _TreeCore(_TObjectLike):
    """A tree's name, where its entries are, and what its branches are."""

    _classname = "TTree"

    def __init__(self, name: str = "", title: str = "", splitlevel: int = 99, dir: Any = None):
        super().__init__(name, title)
        self._store: Store | None = Store()
        self._source: Any = None
        self._snapshot: Any = None
        self._snapshot_entries = -1
        #: Has ``Write`` put the entries in a file, so that they are baskets on file?
        self._written = False
        self._layout_cache: list[BranchInfo] | None = None
        self._addresses: dict[str, Address] = {}
        self._disabled: set[str] = set()
        self._read_entry = -1
        self._batch = Batch(self._backing)
        self._friends: list[tuple[str, Any]] = []
        self._entry_list: Any = None
        self._aliases: dict[str, str] = {}
        self._index: Any = None
        self._weight = 1.0
        self._estimate = 1_000_000
        self._directory = hooks.directory() if dir is None else dir
        append = getattr(self._directory, "Append", None)
        if append is not None:
            append(self)

    @classmethod
    def _over(cls, source: Any, classname: str | None = None) -> Any:
        """One of these over a tree or chain that was read, as ``TFile::Get`` hands it back."""
        made = cls.__new__(cls)
        _TreeCore.__init__(made, source.name, getattr(source, "title", ""), dir=False)
        made._directory = None
        made._store = None
        made._source = source
        if classname:
            made._classname = classname
        return made

    def __repr__(self) -> str:
        return f"<{self._classname} {self._name!r} with {self.GetEntries()} entries>"

    def __len__(self) -> int:
        return self.GetEntries()

    def __bool__(self) -> bool:
        return True

    @property
    def _xrd(self) -> Any:
        """The :class:`xrdroot.TTree` or :class:`xrdroot.Chain` this reads, per the contract."""
        return self._backing()

    def _backing(self) -> Any:
        """What is read: the tree from the file, or what has been filled, read back."""
        if self._store is None:
            return self._source
        if self._snapshot is None or self._snapshot_entries != self._store.entries:
            store, name, title = self._store, self._name or "tree", self._title
            self._snapshot = memory_tree(name, lambda out: store.write(out, name, title))
            self._snapshot_entries = store.entries
            self._befriend(self._snapshot)
        return self._snapshot

    def _changed(self) -> None:
        """Forget what was worked out from the entries, because there are more of them."""
        self._snapshot = None
        self._layout_cache = None
        self._batch.reset()

    def _befriend(self, backing: Any) -> None:
        """Give what is read the friends that can be read beside it, entry for entry."""
        for alias, friend in self._friends:
            other = friend._backing()
            known = getattr(backing, "friends", None)
            if known is not None and alias not in known and len(other) == len(backing):
                backing.add_friend(other, alias)

    def _layout(self) -> list[BranchInfo]:
        if self._layout_cache is None:
            if self._store is not None:
                stats = self._backing() if self._store.slots else None
                self._layout_cache = from_store(self._store, stats, self._written)
            else:
                self._layout_cache = from_tree(_first_tree(self._source))
        return self._layout_cache

    def _leaves(self) -> list[Any]:
        return [leaf for branch in self._layout() for leaf in branch.leaves]

    def _columns(self) -> list[str]:
        return [leaf.column for leaf in self._leaves()]

    def GetEntries(self, selection: str | None = None) -> int:
        """How many entries there are - or, given a selection, how many it keeps."""
        if selection:
            return self._selected(selection)
        return self._store.entries if self._store is not None else len(self._source)

    def _selected(self, selection: str) -> int:
        kept = 0
        for batch in self._backing().iterate(["Entry$"], cut=selection, aliases=self._aliases):
            kept += len(batch["Entry$"])
        return kept

    def GetEntriesFast(self) -> int:
        return self.GetEntries()

    def GetReadEntry(self) -> int:
        return self._read_entry

    def _current(self, column: str) -> Any:
        """The value of ``column`` in the entry last read, or ``None`` before any was."""
        if self._read_entry < 0:
            return None
        return self._batch.value(column, self._read_entry)

    def GetDirectory(self) -> Any:
        return self._directory

    GetCurrentFile = GetDirectory


def _first_tree(source: Any) -> Any:
    """The tree whose branches stand for all of them: the first file's, for a chain."""
    return source.tree(0) if hasattr(source, "spans") else source
