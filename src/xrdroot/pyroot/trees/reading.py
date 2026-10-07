"""``SetBranchAddress`` and ``GetEntry``: a tree read an entry at a time, into addresses.

The loop every ROOT tutorial writes - bind an address to each branch, then
``GetEntry(i)`` for each entry and look in the addresses - is served from
blocks of entries read a branch at a time (:mod:`.batch`), each value put
into the address it was bound to (:mod:`.addresses`). A PyROOT script may
instead read ``tree.px`` after ``GetEntry``, or walk ``for event in tree``,
and gets the same values.
"""

from __future__ import annotations

import fnmatch
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np

from ._base import ListOf
from .addresses import address_of, members_of
from .bind import bind_object, is_object
from .branches import TBranch, TLeaf
from .core import _TreeCore, python_value

__all__ = ["_Reading"]

#: What ``SetBranchAddress`` gives back: ``kMatch``, and ``kMissingBranch``.
MATCH, MISSING_BRANCH = 0, -5


def _bound(address: Any, leaves: list[Any], what: str) -> list[Any]:
    """The address each leaf reads into: the one given, or each member of it, in order."""
    if len(leaves) == 1:
        return [address_of(address, what)]
    sizes = [each.size for each in leaves]
    return members_of(address, [each.name for each in leaves], sizes, what)


def _vector_for(address: Any, leaves: list[Any]) -> Any:
    """``std::vector<float> *v = nullptr`` bound to a vector's branch: a vector made for it.

    ROOT makes the vector a null pointer is pointed at; here it is made and
    put in the pointer's cell, and the vector is what each entry is read into.
    """
    held = getattr(address, "value", address)
    empty = held is None or (isinstance(held, int) and not isinstance(held, bool) and held == 0)
    if len(leaves) != 1 or not leaves[0].vector or not empty or held is address:
        return address
    from ..stl import std

    inner = leaves[0].typename.removeprefix("vector<").removesuffix(">").strip()
    address.value = std.vector[inner]()
    return address.value


def _columns_bound(name: str, branch: Any, leaf: Any, address: Any) -> dict[str, Any]:
    """What each column of a branch - or the one leaf asked for - is read into."""
    if branch is not None and is_object(branch):
        return bind_object(branch, address)
    leaves: list[Any] = branch.leaves if branch is not None else [leaf]
    address = _vector_for(address, leaves)
    found = _bound(address, leaves, f"the branch {name!r}")
    return {each.column: one for each, one in zip(leaves, found, strict=False)}


class _Reading(_TreeCore):
    """Addresses bound to branches, and entries read into them."""

    def _branch_info(self, name: str) -> Any:
        return next((branch for branch in self._every_branch() if branch.name == name), None)

    def _leaf_info(self, name: str) -> Any:
        leaves = self._leaves()
        found = next((leaf for leaf in leaves if leaf.column == name), None)
        return found or next((leaf for leaf in leaves if leaf.name == name), None)

    def SetBranchAddress(self, name: str, address: Any, ptr: Any = None) -> int:
        """Read branch ``name`` into ``address`` from now on; ``-5`` if there is no such branch."""
        branch = self._branch_info(name)
        leaf = self._leaf_info(name) if branch is None else None
        if branch is None and leaf is None:
            print(f"Error in <TTree::SetBranchAddress>: unknown branch -> {name}", file=sys.stderr)
            return MISSING_BRANCH
        for column, one in _columns_bound(name, branch, leaf, address).items():
            self._addresses[column] = one
            self._rebind(column, one)
        if hasattr(ptr, "value"):  # SetBranchAddress(name, &x, &branch): the branch handed back
            ptr.value = self.GetBranch(name)
        return MATCH

    def _rebind(self, column: str, address: Any) -> None:
        """A tree being filled reads the same address it is read into, as ROOT's does."""
        if self._store is not None and column in self._store.slots:
            self._store.slots[column].address = address

    def ResetBranchAddresses(self) -> None:
        self._addresses.clear()

    def ResetBranchAddress(self, branch: Any) -> None:
        name = branch.GetName() if hasattr(branch, "GetName") else str(branch)
        for leaf in self._leaves():
            if name in (leaf.branch, leaf.column):
                self._addresses.pop(leaf.column, None)

    def SetBranchStatus(self, pattern: str, status: Any = 1, found: Any = None) -> None:
        """Turn the branches ``pattern`` matches - a name, or a wildcard - on or off."""
        for leaf in self._leaves():
            names = (leaf.branch, leaf.column, leaf.name)
            if any(fnmatch.fnmatchcase(each, pattern) for each in names):
                (self._disabled.discard if status else self._disabled.add)(leaf.column)

    def GetBranchStatus(self, name: str) -> bool:
        return any(
            name in (leaf.branch, leaf.column) and leaf.column not in self._disabled
            for leaf in self._leaves()
        )

    def _active(self, column: str) -> bool:
        return column not in self._disabled

    def GetEntry(self, entry: int = 0, getall: int = 0) -> int:
        """Read entry ``entry`` into every bound address; the bytes it took, or 0 if none."""
        return self._load(int(entry), None)

    def _load(self, entry: int, columns: list[str] | None) -> int:
        if not 0 <= entry < self.GetEntries():
            return 0
        self._read_entry = entry
        wanted = self._columns() if columns is None else columns
        nbytes = 0
        for column in wanted:
            if not self._active(column):
                continue
            nbytes += self._put(column, entry)
        if columns is None:
            self._load_friends(entry)
        return max(nbytes, 1)

    def _put(self, column: str, entry: int) -> int:
        address = self._addresses.get(column)
        if address is None:
            return 4
        value = self._batch.value(column, entry)
        address.put(value.item() if isinstance(value, np.generic) else value)
        return max(int(np.asarray(value).nbytes) if not isinstance(value, str) else len(value), 1)

    def _load_friends(self, entry: int) -> None:
        """Read the same entry of every friend, or the entry its index matches."""

    def LoadTree(self, entry: int) -> int:
        """Make ``entry`` the one being read; which it is in its own tree comes back."""
        if not 0 <= entry < self.GetEntries():
            return -2
        self._read_entry = int(entry)
        return int(entry)

    def GetTree(self) -> Any:
        return self

    def GetTreeNumber(self) -> int:
        return 0

    def GetEntryNumber(self, entry: int) -> int:
        """The tree's entry number of the ``entry``-th entry read, through any entry list."""
        if self._entry_list is not None:
            return int(self._entry_list.GetEntry(entry))
        return int(entry) if 0 <= entry < self.GetEntries() else -1

    def _entry_numbers(self) -> Any:
        if self._entry_list is not None:
            return self._entry_list._entries()
        return range(self.GetEntries())

    def __iter__(self) -> Iterator[Any]:
        """Every entry, read in turn, as PyROOT's ``for event in tree`` gives them."""
        for entry in self._entry_numbers():
            self.GetEntry(int(entry))
            yield self

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        leaf = self._leaf_info(name)
        if leaf is None:
            return super().__getattr__(name)
        return python_value(self._current(leaf.column), leaf.vector)

    def GetBranch(self, name: str) -> TBranch | None:
        """The branch ``name``, at the top of the tree or under a branch of objects."""
        found = self._branch_info(name)
        return None if found is None else TBranch(self, found)

    def GetListOfBranches(self) -> ListOf:
        return ListOf(TBranch(self, branch) for branch in self._layout())

    def GetListOfLeaves(self) -> ListOf:
        return ListOf(TLeaf(self, leaf) for leaf in self._leaves())

    def GetLeaf(self, name: str, leafname: str | None = None) -> TLeaf | None:
        """The leaf ``name`` - or ``leafname`` of the branch ``name`` - or ``None``."""
        if leafname is not None:
            branch = self.GetBranch(name)
            return None if branch is None else branch.GetLeaf(leafname)
        leaf = self._leaf_info(name.replace("/", "."))
        return None if leaf is None else TLeaf(self, leaf)
