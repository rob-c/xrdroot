"""``TBranch`` and ``TLeaf``: a tree's columns, by ROOT's names for them.

A branch and its leaves are views onto the tree they belong to: a leaf's
``GetValue`` is the value of the entry the tree last read, and a branch's
``GetEntry`` reads that one branch of an entry into its address. What each
is - name, title, type, length - comes from :mod:`.layout`, the same for a
tree being filled as for one read from a file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._base import ListOf, _TObjectLike
from .layout import BranchInfo, LeafInfo

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .tree import TTree

__all__ = ["TBranch", "TLeaf"]


class TLeaf(_TObjectLike):
    """``TLeaf``: one leaf of a branch, and its value in the entry last read."""

    _classname = "TLeaf"

    def __init__(self, tree: TTree, info: LeafInfo) -> None:
        super().__init__(info.name, info.title)
        self._tree = tree
        self._info = info
        self._classname = info.classname

    def __repr__(self) -> str:
        return f"<{self._classname} {self._name!r} of {self._info.typename}>"

    def GetTypeName(self) -> str:
        return self._info.typename

    def GetBranch(self) -> TBranch | None:
        return self._tree.GetBranch(self._info.branch)

    def GetLeafCount(self) -> TLeaf | None:
        """The leaf counting this one's values, or ``None`` when it counts itself."""
        counter = self._info.counter
        return None if counter is None else self._tree.GetLeaf(counter)

    def GetLenStatic(self) -> int:
        return self._info.size

    def IsUnsigned(self) -> bool:
        return self._info.typename.startswith("U")

    def _current(self) -> Any:
        return self._tree._current(self._info.column)

    def GetLen(self) -> int:
        """How many values this leaf holds in the entry last read."""
        value = self._current()
        if self._info.text:
            return 1
        return int(np.size(value)) if value is not None else self._info.size

    GetNdata = GetLen

    def GetValue(self, index: int = 0) -> float:
        """Value ``index`` of the entry last read, as a number: zero before any is read."""
        value = self._current()
        if value is None:
            return 0.0
        if self._info.text:
            return float(len(str(value)))
        flat = np.asarray(value).reshape(-1)
        return float(flat[index]) if index < len(flat) else 0.0

    def GetValueLong64(self, index: int = 0) -> int:
        return int(self.GetValue(index))

    def GetValuePointer(self) -> Any:
        """What the entry last read holds for this leaf, as ROOT's buffer would."""
        return self._current()


class TBranch(_TObjectLike):
    """``TBranch``: a column of a tree, its leaves, and what its baskets hold."""

    _classname = "TBranch"

    def __init__(self, tree: TTree, info: BranchInfo) -> None:
        super().__init__(info.name, info.title)
        self._tree = tree
        self._info = info
        if info.classname:
            self._classname = "TBranchElement"
        self._leaves = ListOf(TLeaf(tree, leaf) for leaf in info.leaves)

    def __repr__(self) -> str:
        return f"<{self._classname} {self._name!r} with {len(self._leaves)} leaves>"

    def GetClassName(self) -> str:
        return self._info.classname

    def GetEntries(self) -> int:
        return self._tree.GetEntries()

    def GetListOfLeaves(self) -> ListOf:
        return self._leaves

    def GetListOfBranches(self) -> ListOf:
        return ListOf()

    def GetLeaf(self, name: str) -> TLeaf | None:
        return self._leaves.FindObject(name)  # type: ignore[no-any-return]

    def GetTree(self) -> TTree:
        return self._tree

    def GetMother(self) -> TBranch:
        return self

    def GetEntry(self, entry: int = 0, getall: int = 0) -> int:
        """Read this branch alone of entry ``entry`` into its address; the bytes come back."""
        return self._tree._load(entry, [leaf.column for leaf in self._info.leaves])

    def SetAddress(self, address: Any) -> None:
        self._tree.SetBranchAddress(self._name, address)

    def GetAddress(self) -> Any:
        return self._tree._addresses.get(self._name)

    def GetReadEntry(self) -> int:
        return self._tree.GetReadEntry()

    def GetBasketSize(self) -> int:
        return self._info.basket_size

    def GetTotBytes(self) -> int:
        return self._info.tot_bytes

    def GetZipBytes(self) -> int:
        return self._info.zip_bytes

    def GetWriteBasket(self) -> int:
        return self._info.baskets

    def Print(self, option: str = "") -> None:
        from .printing import branch_lines

        print("\n".join(branch_lines(self._info, 0)))
