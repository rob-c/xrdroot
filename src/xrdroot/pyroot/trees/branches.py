"""``TBranch`` and ``TLeaf``: a tree's columns, by ROOT's names for them.

A branch and its leaves are views onto the tree they belong to: a leaf's
``GetValue`` is the value of the entry the tree last read, and a branch's
``GetEntry`` reads that one branch of an entry into its address. What each
is - name, title, type, length - comes from :mod:`.layout`, the same for a
tree being filled as for one read from a file.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ...compression import CODES
from ..core.references import IS_REFERENCED, NUMBER_MASK
from ._base import ListOf, _TObjectLike
from .layout import BranchInfo, LeafInfo
from .store import OBJECT

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .reading import _Reading as TTree

__all__ = ["TBranch", "TBranchRef", "TLeaf", "TRefTable"]

#: ``RCompressionSetting::EDefaults::kUseCompiledDefault``: what a branch is set to
#: when ``SetCompressionSettings`` is given nothing - zlib at level 1.
DEFAULT_COMPRESSION = 101


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
        self._made = info
        if info.classname:
            self._classname = "TBranchElement"
        self._leaves = ListOf(TLeaf(tree, leaf) for leaf in info.leaves)
        self._compress: int | None = None

    @property
    def _info(self) -> BranchInfo:
        """What the branch is now - its baskets once the tree is written, say - which the
        tree knows; as it was made, if the tree no longer has a branch of its name."""
        found = self._tree._branch_info(self._name)
        return found if found is not None else self._made

    def __repr__(self) -> str:
        return f"<{self._classname} {self._name!r} with {len(self._leaves)} leaves>"

    def GetClassName(self) -> str:
        return self._info.classname

    def GetEntries(self) -> int:
        return self._tree.GetEntries()

    def GetListOfLeaves(self) -> ListOf:
        return self._leaves

    def GetListOfBranches(self) -> ListOf:
        """The branches under this one: one per member, for a branch of objects split."""
        return ListOf(TBranch(self._tree, child) for child in self._info.children)

    def GetSplitLevel(self) -> int:
        return 99 if self._info.children else 0

    def SetAutoDelete(self, autodel: bool = True) -> None:
        """Whether ROOT deletes the object read into at each entry; here it is replaced."""

    def GetLeaf(self, name: str) -> TLeaf | None:
        return self._leaves.FindObject(name)  # type: ignore[no-any-return]

    def GetTree(self) -> TTree:
        return self._tree

    def GetMother(self) -> TBranch:
        return self

    def GetEntry(self, entry: int = 0, getall: int = 0) -> int:
        """Read this branch alone of entry ``entry`` into its address; the bytes come back."""
        leaves = [leaf for branch in self._info.walk() for leaf in branch.leaves]
        return self._tree._load(entry, [leaf.column for leaf in leaves])

    def SetAddress(self, address: Any) -> None:
        self._tree.SetBranchAddress(self._name, address)

    def GetAddress(self) -> Any:
        held = self._tree._reader_buffers.get(self._name)
        return held.reference() if held is not None else self._tree._addresses.get(self._name)

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

    def SetCompressionSettings(self, settings: int = DEFAULT_COMPRESSION) -> None:
        """``SetCompressionSettings``: this branch's baskets compressed otherwise than its
        file's, ``algorithm * 100 + level`` as ROOT spells it; a tree read keeps it to say."""
        self._compress = int(settings)
        store = getattr(self._tree, "_store", None)
        if store is not None:
            store.compressions[self._name] = int(settings)
            self._tree._changed()

    def SetCompressionLevel(self, level: int = 1) -> None:
        algorithm = self.GetCompressionAlgorithm()
        self.SetCompressionSettings(algorithm * 100 + max(0, min(int(level), 99)))

    def SetCompressionAlgorithm(self, algorithm: int = 0) -> None:
        self.SetCompressionSettings(int(algorithm) * 100 + self.GetCompressionLevel())

    def GetCompressionSettings(self) -> int:
        """What the branch's baskets are compressed with: what it was told, what its
        record says, or else its file's, which ROOT gives a branch as it is made."""
        if self._compress is not None:
            return self._compress
        recorded = self._info.compress
        if recorded >= 0:
            return recorded
        algorithm, level = self._tree._compression()
        return CODES[algorithm] * 100 + int(level) if algorithm and level else 0

    def GetCompressionLevel(self) -> int:
        return self.GetCompressionSettings() % 100

    def GetCompressionAlgorithm(self) -> int:
        return self.GetCompressionSettings() // 100

    def Print(self, option: str = "") -> None:
        from .printing import branch_lines

        print("\n".join(branch_lines(self._info, 0)))


def _referenced(value: Any) -> Iterator[Any]:
    """Every object in ``value`` that a ``TRef`` may point at, as a branch holds it."""
    from ..core.collections import TCollection
    from ..core.objects import TObject

    if isinstance(value, TCollection):
        for item in value:
            yield from _referenced(item)
    elif isinstance(value, TObject):
        if value.TestBit(IS_REFERENCED):
            yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _referenced(item)


class TRefTable(_TObjectLike):
    """``TRefTable``: which branch holds each object a ``TRef`` points at.

    The table is the tree's, kept by its :class:`TBranchRef`, and it is what
    lets a reference be resolved when the branch holding its object has not
    been read: the object's identifier - and the process that numbered it -
    names the branch to read. ROOT records a row per process, because an
    identifier is unique only within the process that gave it, and a file may
    hold references made by several.
    """

    _classname = "TRefTable"

    #: ``fgRefTable``: the table of the tree an entry is being read or written for.
    _current: ClassVar[TRefTable | None] = None

    def __init__(self, owner: Any = None, size: int = 100) -> None:
        super().__init__("", "")
        self._owner = owner
        self._parents = self._made_parents()
        #: One row of parent numbers per process, by the process's identifier: 0 is none.
        self._rows: list[list[int]] = []
        #: The processes whose rows these are, in ``fProcessGUIDs`` order.
        self._processes: list[Any] = []
        self._parent = -1
        self._uid = 0
        self._uid_context: Any = None
        TRefTable._current = self

    @staticmethod
    def GetRefTable() -> TRefTable | None:
        """``GetRefTable``: the table of the tree an entry is being read or written for."""
        return TRefTable._current

    @staticmethod
    def SetRefTable(table: TRefTable | None) -> None:
        TRefTable._current = table

    @staticmethod
    def _made_parents() -> Any:
        from ..core.collections import TObjArray

        return TObjArray(1)

    def _index(self, context: Any) -> int:
        """The row of ``context``'s process, made at the end if the table has none."""
        from ..core.references import TProcessID

        held = context if context is not None else TProcessID.GetSessionProcessID()
        for at, known in enumerate(self._processes):
            if known is held:
                return at
        self._processes.append(held)
        self._rows.append([])
        return len(self._rows) - 1

    def Add(self, uid: int, context: Any = None) -> int:
        """``Add``: the current parent holds object ``uid`` of ``context``; -1 if none is set."""
        row = self._rows[self._index(context)]
        number = int(uid) & NUMBER_MASK
        if number >= len(row):
            row.extend([0] * (number + 1 - len(row)))
        if self._parent < 0:
            return -1
        row[number] = self._parent + 1
        return number

    def SetParent(self, parent: Any, branchID: int = -1) -> int:
        """``SetParent``: the branch being read or written now, and the number it has here."""
        if branchID >= 0:
            self._parents.AddAtAndExpand(parent, int(branchID))
            self._parent = int(branchID)
            return self._parent
        found = int(self._parents.IndexOf(parent))
        if found < 0:  # a branch asked for again comes back as another object over the same
            found = next(
                (at for at in range(self._parents.GetEntriesFast())
                 if getattr(self._parents.At(at), "GetName", lambda: None)()
                 == getattr(parent, "GetName", lambda: None)()),
                -1,
            )
        if found < 0:
            found = int(self._parents.GetEntriesFast())
            self._parents.AddAtAndExpand(parent, found)
        self._parent = found
        return found

    def GetParent(self, uid: int, context: Any = None) -> Any:
        """``GetParent``: the branch holding object ``uid`` of ``context``, or ``None``."""
        row = self._rows[self._index(context)]
        number = int(uid) & NUMBER_MASK
        if number >= len(row) or not row[number] or not self._parents:
            return None
        return self._parents.At(row[number] - 1)

    def GetParents(self) -> Any:
        return self._parents

    def GetSize(self, context: Any = None) -> int:
        """``GetSize``: how many identifiers the row of ``context``'s process has room for."""
        return len(self._rows[self._index(context)])

    def GetN(self, context: Any = None) -> int:
        """``GetN``: how many identifiers the row of ``context``'s process holds, last first."""
        row = self._rows[self._index(context)]
        end = len(row)
        while end and not row[end - 1]:
            end -= 1
        return end

    def GetOwner(self) -> Any:
        return self._owner

    def GetUID(self) -> int:
        return self._uid

    def GetUIDContext(self) -> Any:
        return self._uid_context

    def SetUID(self, uid: int, context: Any = None) -> None:
        """``SetUID``: the identifier - and its process - a reference is asking about now."""
        self._uid = int(uid) & NUMBER_MASK
        self._uid_context = context

    def Clear(self, option: str = "") -> None:
        """``Clear``: forget every row's entries, keeping the rows and their room."""
        for row in self._rows:
            row[:] = [0] * len(row)
        self._parent = -1

    def Reset(self, option: str = "") -> None:
        """``Reset``: clear the table, and forget which branches it knew."""
        self.Clear(option)
        self._parents.Clear()

    def Notify(self) -> bool:
        """``Notify``: what the owner does about the identifier last set."""
        return bool(self._owner is not None and self._owner.Notify())

    def FillBuffer(self, buf: Any) -> None:
        """``FillBuffer``: every process's row, as ROOT writes one entry of the table."""
        buf.i32(-len(self._rows))
        for row in self._rows:
            buf.i32(len(row))
            for value in row:
                buf.i32(value)

    def ReadBuffer(self, buf: Any) -> None:
        """``ReadBuffer``: the rows ROOT wrote, one process at a time."""
        first = buf.i32()
        from ..core.references import TProcessID

        rows: list[list[int]] = []
        if first < 0:
            for _ in range(-first):
                count = buf.i32()
                rows.append(list(buf.i32s(count)))
        else:  # the older layout, one process only: its row written without a count pair
            rows = [list(buf.i32s(first))]
        self._rows = rows
        #: The process of each row is ``fProcessGUIDs``'s business, read by the streamer;
        #: until it is, a read table's rows are the session's, which is what a file of one
        #: process has.
        self._processes = [TProcessID.GetSessionProcessID()] * len(rows)

    def GetProcessGUIDs(self) -> list[str]:
        """``fProcessGUIDs``: the processes whose rows this table holds, by UUID."""
        return [str(process.GetTitle()) for process in self._processes]

    def _described(self) -> dict[str, Any]:
        """The rows and their processes as plain data, for the writer to put on file."""
        rows: list[list[int]] = []
        for row in self._rows:
            end = len(row)
            while end and not row[end - 1]:
                end -= 1
            rows.append(row[:end])
        return {"guids": self.GetProcessGUIDs(), "rows": rows}


#: What ``TBranchRef`` is called and titled, as ``TBranchRef::TBranchRef(TTree*)`` sets them.
REF_NAME = "TRefTable"
REF_TITLE = "List of branch numbers with referenced objects"


class TBranchRef(TBranch):
    """``TBranchRef``: the branch carrying a tree's :class:`TRefTable`.

    ``TTree::BranchRef`` makes one. It is not among the tree's branches - the
    tree record's ``fBranchRef`` slot holds it - and it has no leaves: its one
    basket holds the table's rows, one per entry, so a reader knows which
    branch to load to find the object a ``TRef`` names.
    """

    _classname = "TBranchRef"

    def __init__(self, tree: TTree) -> None:
        info = BranchInfo(
            name=REF_NAME,
            title=REF_TITLE,
            classname="TBranchRef",
            leaves=[],
            entries=0,
            tot_bytes=0,
            zip_bytes=0,
            baskets=0,
            basket_size=32000,
            compress=1,
        )
        super().__init__(tree, info)
        self._ref_table = TRefTable(self, 100)
        self._requested = -1

    def GetRefTable(self) -> TRefTable:
        return self._ref_table

    def SetParent(self, obj: Any, branchID: int = -1) -> int:
        """``SetParent``: the branch the table's next references belong to."""
        return self._ref_table.SetParent(obj, int(branchID))

    def SetRequestedEntry(self, entry: int) -> None:
        self._requested = int(entry)

    def Clear(self, option: str = "") -> None:
        """``Clear``: no references have been read or written for the entry starting now."""
        self._ref_table.Clear(option)

    def Reset(self, option: str = "") -> None:
        self._ref_table.Reset(option)
        self._made = self._made._replace(entries=0)

    def Notify(self) -> bool:
        """``Notify``: read the branch holding the object a reference asked for."""
        table = self._ref_table
        branch = table.GetParent(table.GetUID(), table.GetUIDContext())
        if branch is not None and branch.GetReadEntry() != self._requested:
            branch.GetEntry(self._requested)
        return True

    def Print(self, option: str = "") -> None:
        from .printing import branch_lines

        print("\n".join(branch_lines(self._info, 0)))

    def _account(self, store: Any) -> None:
        """``Add`` every referenced object the fill just read, under the branch holding it."""
        from ..core.references import TProcessID
        from .objects import _held

        for slot in store.slots.values():
            if slot.kind != OBJECT:
                continue
            branch = self._tree.GetBranch(slot.branch)
            if branch is None:
                continue
            numbered = False
            for obj in _referenced(_held(slot.address)):
                number = int(obj.GetUniqueID()) & NUMBER_MASK
                if not numbered:  # the branch takes its number here, before its objects
                    self._ref_table.SetParent(branch, -1)
                    numbered = True
                self._ref_table.Add(number, TProcessID.GetProcessWithUID(obj, obj))
        self._made = self._made._replace(entries=int(store.entries))

    def _has_rows(self) -> bool:
        """Whether any referenced object has been recorded: is there a table to write?"""
        return any(any(row) for row in self._ref_table._rows)

    def _described(self) -> dict[str, Any]:
        """The table as plain data, which the writer lays out beside the tree."""
        return {
            "name": REF_NAME,
            "title": REF_TITLE,
            "entries": int(self.GetEntries()),
            **self._ref_table._described(),
        }
