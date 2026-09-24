"""Entry lists: which entries of a tree a selection kept, saved to be used again.

A selection run over a big tree in ROOT can be kept as a ``TEntryList``
rather than as a smaller copy of the tree - just the numbers of the entries
that passed. It keeps them in blocks of four thousand, each either a bitmap
or a list of the entries that passed (or, when nearly all did, of the ones
that did not), and a list made over a chain keeps one list per tree of it.
``TEventList`` is the older form, a plain sorted array.

Either comes back as an :class:`EntryList`, whose :attr:`~EntryList.entries`
are the entry numbers as NumPy, ready to hand to ``tree.arrays(...,
entries=...)`` - which reads the baskets those entries are in and no others.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .errors import FormatError, UnsupportedFeatureError

__all__ = ["ENTRY_LISTS", "EntryList", "selected"]

#: The classes this reads.
ENTRY_LISTS = ("TEntryList", "TEventList")

#: How many entries one ``TEntryListBlock`` covers: ROOT's ``kBlockSize``.
BLOCK = 4000


def _block(block: dict[str, Any], index: int) -> np.ndarray[Any, Any]:
    """The entries one block says passed, counted from the start of the tree.

    A block of type 0 is a bitmap, a bit an entry and sixteen to a word; one
    of type 1 is a sorted list, of the entries that passed or - when
    ``fPassing`` is false, because nearly all did - of the ones that did not.
    """
    indices = np.asarray(block.get("fIndices", ()), dtype=np.int64)
    passed = int(block.get("fNPassed", 0))
    if int(block.get("fType", 0)) == 0:
        bits = (indices[:, None] >> np.arange(16)) & 1
        local = np.flatnonzero(bits.ravel())
    elif block.get("fPassing", True):
        local = indices[:passed]
    else:
        local = np.setdiff1d(np.arange(BLOCK), indices)[:passed]
    return local + index * BLOCK


class EntryList:
    """A ``TEntryList`` or ``TEventList``: the numbers of the entries a selection kept.

        >>> kept = f["passed_cuts"]                       # doctest: +SKIP
        >>> kept.entries[:5]
        array([ 3,  4, 17, 30, 31])
        >>> tree.arrays(["pt"], entries=kept)             # doctest: +SKIP

    A list made over a chain holds one list per tree, in :attr:`lists`, each
    naming the tree and file it belongs to; ``tree.arrays(entries=...)`` picks
    the one for the tree it is reading.
    """

    __slots__ = ("classname", "members", "lists", "_entries")

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        #: The class the file says this is.
        self.classname = classname
        #: Every member, as it was written.
        self.members = members
        held = members.get("fLists") or []
        #: One list per tree, for a list made over a chain; empty otherwise.
        self.lists: list[EntryList] = [
            item if isinstance(item, EntryList) else EntryList("TEntryList", item) for item in held
        ]
        self._entries = self._read()

    def _read(self) -> np.ndarray[Any, Any]:
        if self.classname == "TEventList":
            count = int(self.members.get("fN", 0))
            return np.asarray(self.members.get("fList", ()), dtype=np.int64)[:count]
        blocks = self.members.get("fBlocks") or []
        if not all(isinstance(block, dict) for block in blocks):
            raise FormatError(f"a {self.classname} holds blocks that are not TEntryListBlocks")
        found = [_block(block, index) for index, block in enumerate(blocks)]
        return np.concatenate(found) if found else np.zeros(0, dtype=np.int64)

    def __repr__(self) -> str:
        where = f" of {self.tree_name!r}" if self.tree_name else ""
        return f"<{self.classname} {self.name!r}{where}, {len(self)} entries>"

    def __len__(self) -> int:
        """How many entries it keeps, across every tree it has a list for."""
        if self.lists:
            return sum(len(one) for one in self.lists)
        return len(self._entries)

    @property
    def name(self) -> str:
        return str(self.members.get("TNamed", {}).get("fName", ""))

    @property
    def title(self) -> str:
        return str(self.members.get("TNamed", {}).get("fTitle", ""))

    @property
    def tree_name(self) -> str:
        """The tree the entries are of, which ROOT writes down when it knows."""
        return str(self.members.get("fTreeName", ""))

    @property
    def file_name(self) -> str:
        """The file that tree was in, when the list says."""
        return str(self.members.get("fFileName", ""))

    @property
    def entries(self) -> np.ndarray[Any, Any]:
        """The entry numbers, in order, as ``int64``.

        A list with a list per tree has no one set of numbers: they count from
        the start of each tree. :attr:`lists` has them, or :meth:`for_tree`
        finds the one that is wanted.
        """
        if self.lists:
            trees = ", ".join(f"{one.tree_name} in {one.file_name}" for one in self.lists)
            raise UnsupportedFeatureError(
                f"{self.name!r} keeps a list per tree ({trees}), each counting from its "
                f"own tree's first entry; take the one wanted from .lists, or with "
                f".for_tree(name)"
            )
        return self._entries.copy()

    def for_tree(self, tree_name: str, file_name: str | None = None) -> EntryList:
        """The list for one tree: this one, or the one of :attr:`lists` it names.

        ``file_name`` tells apart two trees of the same name in different
        files; it matches the file the list names in full or by its last
        part, since the list was usually made with the file somewhere else.
        """
        if not self.lists:
            return self
        found = self.lists_for(tree_name, file_name)
        if len(found) != 1:
            trees = ", ".join(f"{one.tree_name} in {one.file_name}" for one in self.lists)
            raise KeyError(
                f"{self.name!r} has {len(found) or 'no'} lists for {tree_name!r}"
                f"{f' in {file_name}' if file_name else ''} among its {len(self.lists)} "
                f"({trees}); name the file too, or pick from .lists"
            )
        return found[0]

    def lists_for(self, tree_name: str, file_name: str | None = None) -> list[EntryList]:
        """Every one of :attr:`lists` that is for this tree, in this file if one is named."""
        return [one for one in self.lists if _names(one, tree_name, file_name)]


def _names(one: EntryList, tree_name: str, file_name: str | None) -> bool:
    """Whether a list is the one for this tree in this file."""
    if one.tree_name.strip("/") != tree_name.strip("/"):
        return False
    if file_name is None:
        return True
    return one.file_name == file_name or os.path.basename(one.file_name) == os.path.basename(
        file_name
    )


def selected(
    entries: Any, total: int, tree_name: str = "", file_name: str | None = None
) -> np.ndarray[Any, Any]:
    """Which entries of a tree of ``total`` to read: the numbers, checked.

    ``entries`` is an :class:`EntryList` - the list for this tree, when it
    keeps one per tree - or anything NumPy makes a one-dimensional array of
    whole numbers of, including a boolean mask as long as the tree.
    """
    if isinstance(entries, EntryList):
        entries = (entries.for_tree(tree_name, file_name) if tree_name else entries).entries
    wanted = np.asarray(entries)
    if wanted.dtype == np.bool_:
        if wanted.shape != (total,):
            raise ValueError(
                f"a mask of {wanted.size} entries was given for a tree of {total}: a "
                f"mask says yes or no to every entry, so it is as long as the tree"
            )
        return np.flatnonzero(wanted)
    if wanted.ndim != 1 or (wanted.size and not np.issubdtype(wanted.dtype, np.integer)):
        raise ValueError(
            "entries are a one-dimensional run of whole entry numbers, an EntryList, "
            "or a mask of one bool per entry"
        )
    wanted = wanted.astype(np.int64)
    outside = wanted[(wanted < 0) | (wanted >= total)]
    if outside.size:
        raise IndexError(
            f"entry {int(outside[0])} is not in a tree of {total} entries, and "
            f"{outside.size} of those asked for are outside it"
        )
    return wanted
