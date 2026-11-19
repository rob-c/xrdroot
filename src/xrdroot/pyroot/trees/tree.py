"""``TTree``: made, given branches and filled as ROOT's is - or read, as one from a file.

    >>> x = np.zeros(1, dtype=np.float64)
    >>> t = TTree("t", "a tree")
    >>> t.Branch("x", x, "x/D")
    >>> for i in range(10):
    ...     x[0] = i * 0.5
    ...     t.Fill()
    >>> t.Draw("x", "x > 2", "goff")
    5

``Fill`` reads the value every branch's address holds at that moment -
ROOT's pointer semantics, kept - and ``Write`` writes every entry so far
into the directory the tree is in. See :mod:`.addresses` for what an
address can be, and :mod:`.leaflist` for the leaf lists ``Branch`` takes.
"""

from __future__ import annotations

from typing import Any, NoReturn

from .addresses import address_of, members_of
from .branches import TBranch, TBranchRef
from .copying import clone
from .leaflist import Leaf, parse
from .objects import object_slot
from .player import MAX_ENTRIES, _Player
from .store import INTEGERS, Slot

__all__ = ["TTree"]

#: Type codes whose width is the platform's, and the ones that are not.
PLATFORM = {"l": "q", "L": "Q"}
#: The type codes a leaf can be.
CODES = "bBhHiIqQfd?"


def _undeclared(address: Any, name: str) -> NoReturn:
    """What a branch given no leaf list is told of an address that cannot say its type.

    A collection says its type when its elements are numbers or text and not when they
    are objects: one level of counting is all a branch here lays out, so a collection of
    collections is refused by name rather than handed a leaf list that cannot express it.
    """
    from ...errors import UnsupportedFeatureError

    if address.sized:
        raise UnsupportedFeatureError(
            f"the branch {name!r} holds a std::vector of {address.vector.value_type}, whose "
            f"elements carry sizes of their own, and this writer lays out one count per "
            f"entry: a branch of collections of collections would be written with the "
            f"inner sizes lost. Give each inner collection a branch of its own, or write "
            f"the values into a leaf list that counts them"
        )
    raise TypeError(
        f"the branch {name!r} was given no leaf list, and what its address holds does "
        f"not say its type; give one, as in Branch({name!r}, x, '{name}/D')"
    )


def _declared(address: Any, name: str) -> list[Leaf]:
    """The leaf a branch declared without a leaf list is: what its address holds."""
    if address.text:
        return [Leaf(name, "C", 1, None, name)]
    code = "" if address.dtype is None else PLATFORM.get(address.dtype.char, address.dtype.char)
    if not code or code not in CODES:
        _undeclared(address, name)
    size = 1 if address.sized or address.room is None else address.room
    return [Leaf(name, str(code), size, None, name if size == 1 else f"{name}[{size}]")]


def _slot(branch: str, leaf: Leaf, address: Any, title: str | None) -> Slot:
    """The slot for one leaf: the branch's own name and title if it is the only leaf."""
    name = branch if title is not None else leaf.name
    return Slot(
        name,
        branch,
        leaf.code,
        address,
        size=leaf.size,
        counter=leaf.counter,
        title=leaf.title if title is None else title,
    )


def _arranged(address: Any, leaflist: Any, bufsize: int, splitlevel: int) -> tuple[Any, ...]:
    """``Branch``'s arguments in their places, whichever of ROOT's forms they came in.

    ``Branch(name, "std::vector<float>", &object)`` names the class before
    the object; ``Branch(name, &object, bufsize, splitlevel)`` gives an
    object, not a leaf list, and the sizes one place early.
    """
    if isinstance(address, str) and leaflist is not None and not isinstance(leaflist, str):
        return leaflist, None, bufsize, splitlevel
    if isinstance(leaflist, int) and not isinstance(leaflist, bool):
        return address, None, leaflist, bufsize
    return address, leaflist, bufsize, splitlevel


class TTree(_Player):
    """``TTree``: see the module's docstring."""

    def _writable(self, what: str) -> Any:
        if self._store is None:
            raise TypeError(
                f"{self._name!r} was read from a file, and {what} adds to a tree being "
                f"filled; CloneTree(0) makes one with its branches to fill"
            )
        return self._store

    def Branch(
        self,
        name: str,
        address: Any = None,
        leaflist: Any = None,
        bufsize: int = 32000,
        splitlevel: int = 99,
    ) -> TBranch:
        """A new branch, read from ``address`` at every ``Fill``: see the module's docstring."""
        store = self._writable("Branch")
        self._require_new(name)
        what = f"the branch {name!r}"
        address, leaflist, bufsize, splitlevel = _arranged(address, leaflist, bufsize, splitlevel)
        made = object_slot(name, address, int(splitlevel), bufsize) if leaflist is None else None
        if made is not None:
            store.add(made)
            return self._added(name)
        if isinstance(leaflist, str):
            leaves = parse(leaflist)
            addresses = self._addressed(address, leaves, what)
        else:
            addresses = [address_of(address, what)]
            leaves = _declared(addresses[0], name)
        title = leaflist if isinstance(leaflist, str) else name
        for leaf, one in zip(leaves, addresses, strict=False):
            self._check_counter(leaf, name)
            slot = _slot(name, leaf, one, title if len(leaves) == 1 else None)
            slot.split, slot.basket_size = int(splitlevel), bufsize
            store.add(slot)
        if len(leaves) > 1:
            store.titles[name] = title
        return self._added(name)

    def Bronch(
        self, name: str, classname: Any = None, address: Any = None,
        bufsize: int = 32000, splitlevel: int = 99,
    ) -> TBranch:  # fmt: skip
        """``Bronch(name, classname, address)``: ROOT's older spelling of ``Branch`` for a branch
        of objects, the class's name given before the object it is read from.

        ROOT keeps the name for the code that still calls it - roottest's ``io/bigevent``
        does - and it makes the same branch ``Branch`` does from the same arguments.
        """
        return self.Branch(name, classname, address, bufsize, splitlevel)

    def _added(self, name: str) -> TBranch:
        """The branch just added, now that the tree knows of it."""
        self._changed()
        branch = self.GetBranch(name)
        assert branch is not None
        return branch

    def _require_new(self, name: str) -> None:
        """A branch can be added before the first entry, under a name not taken."""
        store = self._writable("Branch")
        if store.entries:
            raise ValueError(
                f"{self._name!r} already has {store.entries} entries, and a branch added now "
                f"would have none for them; declare every branch before the first Fill"
            )
        if any(slot.branch == name for slot in store.slots.values()):
            raise ValueError(f"{self._name!r} already has a branch called {name!r}")

    def _addressed(self, address: Any, leaves: list[Leaf], what: str) -> list[Any]:
        if len(leaves) == 1:
            return [address_of(address, what)]
        taken = [slot.name for slot in self._writable("Branch").slots.values()]
        clash = [leaf.name for leaf in leaves if leaf.name in taken]
        if clash:
            raise ValueError(
                f"{what} has leaves {', '.join(clash)}, which this tree has already; a leaf "
                f"list is written a leaf to a column, so its leaves' names have to be new"
            )
        return members_of(address, [leaf.name for leaf in leaves], [l.size for l in leaves], what)

    def _check_counter(self, leaf: Leaf, name: str) -> None:
        if leaf.counter is None:
            return
        slots = self._writable("Branch").slots
        counter = slots.get(leaf.counter)
        if counter is None or counter.kind != "scalar" or counter.code not in INTEGERS:
            raise ValueError(
                f"{leaf.title!r} of the branch {name!r} is counted by {leaf.counter!r}, which "
                f"has to be an integer leaf of this tree declared before it"
            )
        if leaf.size > 1:
            raise ValueError(
                f"{leaf.title!r} of the branch {name!r} has a fixed size inside its count, "
                f"which this writer does not lay out; give it a leaf of its own per column"
            )

    def Fill(self) -> int:
        """Read every branch's address as one more entry; the bytes it took come back."""
        store = self._writable("Fill")
        if self._ref_branch is not None:
            self._ref_branch.Clear()  # ROOT clears the table before the branches are filled
        nbytes = int(store.fill())
        if self._ref_branch is not None:
            self._ref_branch._account(store)  # every referenced object's branch, this entry
        self._changed()
        return nbytes

    def Write(self, name: str | None = None, option: int = 0, bufsize: int = 0) -> int:
        """Write every entry so far into the tree's directory, as a new cycle of it.

        A tree given no branches has nothing a reader could make anything of,
        and is left out of the file - a macro stopped before its ``Branch`` call
        leaves its file closing over such a tree.
        """
        return self._write_to(self._directory, name or self._name)

    def _write_to(self, directory: Any, name: str) -> int:
        """The tree written into ``directory`` under ``name``, as ``Write`` writes it."""
        store = self._writable("Write")
        if not store.slots:
            return 0
        directory = getattr(directory, "_xrd", directory)
        if directory is None or not hasattr(directory, "tree"):
            raise ValueError(
                f"{self._name!r} is in memory, in no file; open one for writing before "
                f"making the tree, or give it one with SetDirectory, then Write"
            )
        self._check_ref(directory)
        self._wrote = store.write(directory, name, self._title, self._classname)
        if self._ref_branch is not None:
            self._wrote._ref_table = self._ref_branch._described()
        self._written = True
        self._layout_cache = None
        return max(sum(branch.tot_bytes for branch in self._layout()), 1)

    def _check_ref(self, directory: Any) -> None:
        """A tree whose references were recorded can be written only where the writer puts
        the table in the tree record: one that would lose it is refused, not written wrong."""
        refs = self._ref_branch
        if refs is None or not refs._has_rows() or getattr(directory, "writes_ref_table", False):
            return
        from ...errors import UnsupportedFeatureError

        raise UnsupportedFeatureError(
            f"TTree::Write cannot write {self._name!r}'s TBranchRef: this writer puts no "
            f"fBranchRef in the tree record, so the table saying which entry holds each "
            f"referenced object would be lost and a TRef read from the file would find "
            f"nothing; write the tree without BranchRef until the writer carries it"
        )

    def _write_into(self, directory: Any, name: str) -> int:
        """``WriteObject``/``WriteTObject``: the tree written into ``directory`` under ``name`` -
        with no branches at all, when it has none, as ROOT writes it asked to."""
        store = self._writable("WriteObject")
        held, self._directory = self._directory, directory
        try:
            if store.slots:
                return self.Write(name)
            target = getattr(directory, "_xrd", directory)
            self._wrote = store.write(target, name, self._title, self._classname, empty=True)
            self._written = True
            return 1
        finally:
            self._directory = held

    def FlushBaskets(self, create_cluster: bool = True) -> int:
        """``FlushBaskets``: the entries filled since the last end a cluster of their own."""
        self._writable("FlushBaskets").flush()
        return 0

    def AutoSave(self, option: str = "") -> int:
        """Write the tree as it is so far, as ROOT's ``AutoSave`` does."""
        return self.Write()

    def SetDirectory(self, directory: Any) -> None:
        """``SetDirectory``: kept in - and written into - ``directory`` from now on, out of
        the one it was in; given ``nullptr``, in none, so closing that file leaves it be."""
        held = self._directory
        if held is not None and hasattr(held, "Remove"):
            held.Remove(self)
        self._directory = None
        if directory is None or directory == 0:
            return
        real = directory.__real__() if hasattr(directory, "__real__") else directory
        if hasattr(real, "Append"):
            real.Append(self, True)
        self._directory = real

    def Reset(self, option: str = "") -> None:
        """Forget every entry, keeping the branches and their addresses."""
        self._writable("Reset").reset()
        self._changed()

    def CloneTree(self, nentries: int = -1, option: str = "") -> TTree:
        """A tree of the same branches, reading this one's, with its first ``nentries`` entries."""
        total = self.GetEntries()
        stop = total if nentries < 0 else min(int(nentries), total)
        return clone(self, TTree(self._name, self._title), 0, stop, None)  # type: ignore[no-any-return]

    def CopyTree(
        self,
        selection: str = "",
        option: str = "",
        nentries: int = MAX_ENTRIES,
        firstentry: int = 0,
    ) -> TTree:
        """A tree of the same branches, holding the entries ``selection`` keeps."""
        stop = min(self.GetEntries(), firstentry + nentries)
        made = TTree(self._name, self._title)
        return clone(self, made, int(firstentry), stop, selection)  # type: ignore[no-any-return]

    def ReadFile(self, filename: str, descriptor: str = "", delimiter: str = " ") -> int:
        """Read a text file of columns into new branches; how many lines were read."""
        from .text import read_file

        return read_file(self, str(filename), str(descriptor), str(delimiter))

    def SetAutoSave(self, autos: int = -300_000_000) -> None:
        """How often ROOT saves the header while filling; here it is saved by Write."""

    def SetAutoFlush(self, autof: int = -30_000_000) -> None:
        """How often ROOT flushes baskets; here baskets are laid out when written."""

    def SetMaxTreeSize(self, maxsize: int = 100_000_000_000) -> None:
        """How big ROOT lets a file grow before starting another; one file here."""

    #: ``TTree::SetBranchStyle``: 1 splits objects into ``TBranchElement``s, 0 into the
    #: ``TBranchObject``s of ROOT 3.
    _branch_style = 1

    @staticmethod
    def SetBranchStyle(style: int = 1) -> None:
        TTree._branch_style = int(style)

    @staticmethod
    def GetBranchStyle() -> int:
        return TTree._branch_style

    #: ``fBranchRef``: the branch holding the table of referenced objects, made by BranchRef.
    _ref_branch: TBranchRef | None = None

    def BranchRef(self) -> TBranchRef:
        """``BranchRef``: give the tree the branch that says which entry holds each object
        a ``TRef`` points at, and hand it back - the same one every time it is called."""
        if self._ref_branch is None:
            self._writable("BranchRef")
            self._ref_branch = TBranchRef(self)
        return self._ref_branch

    def GetBranchRef(self) -> TBranchRef | None:
        """``GetBranchRef``: the branch ``BranchRef`` made, or ``None`` if it never was."""
        return self._ref_branch

    def SetBasketSize(self, name: str, size: int = 16000) -> None:
        """A branch's basket size in ROOT; this writer sizes its own baskets."""

    def SetCacheSize(self, size: int = -1) -> int:
        """ROOT's read cache; reads here are a block of entries at a time anyway."""
        return 0

    def AddBranchToCache(self, name: str, subbranches: bool = False) -> int:
        return 0

    def SetCircular(self, maxentries: int) -> None:
        """Keep at most ``maxentries`` entries, the newest, in memory; 0 or less keeps all."""
        self._writable("SetCircular").circular = max(int(maxentries), 0)
        self._changed()

    def OptimizeBaskets(self, maxmemory: int = 10_000_000, minComp: float = 1.1) -> None:
        """What ROOT does to basket sizes; nothing to do here."""

    def StartViewer(self, *arguments: Any) -> None:
        """ROOT's tree viewer, which a batch session has no window for."""

    def Refresh(self) -> None:
        """Read the header again, for a tree another process writes; nothing to do here."""
