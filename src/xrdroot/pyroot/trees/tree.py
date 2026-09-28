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

from typing import Any

from .addresses import address_of, members_of
from .branches import TBranch
from .copying import clone
from .leaflist import Leaf, parse
from .player import MAX_ENTRIES, _Player
from .store import INTEGERS, Slot

__all__ = ["TTree"]

#: Type codes whose width is the platform's, and the ones that are not.
PLATFORM = {"l": "q", "L": "Q"}
#: The type codes a leaf can be.
CODES = "bBhHiIqQfd?"


def _declared(address: Any, name: str) -> list[Leaf]:
    """The leaf a branch declared without a leaf list is: what its address holds."""
    if address.text:
        return [Leaf(name, "C", 1, None, name)]
    code = "" if address.dtype is None else PLATFORM.get(address.dtype.char, address.dtype.char)
    if not code or code not in CODES:
        raise TypeError(
            f"the branch {name!r} was given no leaf list, and what its address holds does "
            f"not say its type; give one, as in Branch({name!r}, x, '{name}/D')"
        )
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
        if isinstance(address, str) and leaflist is not None and not isinstance(leaflist, str):
            # Branch(name, "std::vector<float>", &object): the class named, the object after it.
            address, leaflist = leaflist, None
        if isinstance(leaflist, str):
            leaves = parse(leaflist)
            addresses = self._addressed(address, leaves, what)
        else:
            addresses = [address_of(address, what)]
            leaves = _declared(addresses[0], name)
        title = leaflist if isinstance(leaflist, str) else name
        for leaf, one in zip(leaves, addresses):
            self._check_counter(leaf, name)
            store.add(_slot(name, leaf, one, title if len(leaves) == 1 else None))
        if len(leaves) > 1:
            store.titles[name] = title
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
        nbytes = int(self._writable("Fill").fill())
        self._changed()
        return nbytes

    def Write(self, name: str | None = None, option: int = 0, bufsize: int = 0) -> int:
        """Write every entry so far into the tree's directory, as a new cycle of it."""
        store = self._writable("Write")
        directory = self._directory
        directory = getattr(directory, "_xrd", directory)
        if directory is None or not hasattr(directory, "tree"):
            raise ValueError(
                f"{self._name!r} is in memory, in no file; open one for writing before "
                f"making the tree, or give it one with SetDirectory, then Write"
            )
        store.write(directory, name or self._name, self._title, self._classname)
        self._written = True
        self._layout_cache = None
        return max(sum(branch.tot_bytes for branch in self._layout()), 1)

    def AutoSave(self, option: str = "") -> int:
        """Write the tree as it is so far, as ROOT's ``AutoSave`` does."""
        return self.Write()

    def SetDirectory(self, directory: Any) -> None:
        self._directory = directory

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
