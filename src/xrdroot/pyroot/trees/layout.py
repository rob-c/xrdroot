"""What a tree's branches and leaves are, whether it is being filled or was read.

``GetListOfBranches``, ``GetLeaf``, ``Print`` and ``Show`` all need the same
thing - each branch's name, title and leaves, each leaf's type and length -
and a tree has it from one of two places: the slots of a tree being filled,
or the records of one read from a file. :func:`from_store` and
:func:`from_tree` turn either into a list of :class:`BranchInfo`, in the
order ROOT lists them, so nothing above has to know which it was.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .store import TEXT, VECTOR, Store

__all__ = ["LeafInfo", "BranchInfo", "from_store", "from_tree"]

#: The type name each leaf class reads as, signed and then unsigned.
LEAF_TYPES: dict[str, tuple[str, str]] = {
    "TLeafB": ("Char_t", "UChar_t"),
    "TLeafS": ("Short_t", "UShort_t"),
    "TLeafI": ("Int_t", "UInt_t"),
    "TLeafL": ("Long64_t", "ULong64_t"),
    "TLeafG": ("Long_t", "ULong_t"),
    "TLeafF": ("Float_t", "Float_t"),
    "TLeafF16": ("Float16_t", "Float16_t"),
    "TLeafD": ("Double_t", "Double_t"),
    "TLeafD32": ("Double32_t", "Double32_t"),
    "TLeafO": ("Bool_t", "Bool_t"),
    "TLeafC": ("Char_t", "Char_t"),
}


class LeafInfo(NamedTuple):
    """One leaf: what it is called, which column holds it, and what it holds."""

    name: str
    #: The name its values are read by: its own, or ``branch.leaf``.
    column: str
    title: str
    typename: str
    classname: str
    #: How many values an entry holds when that is fixed: ``fLen``.
    size: int
    #: The leaf counting it, if one does.
    counter: str | None
    branch: str
    #: Does it hold text, or a ``std::vector``?
    text: bool
    vector: bool


class BranchInfo(NamedTuple):
    """One branch: its name, title, leaves, and what its baskets hold."""

    name: str
    title: str
    classname: str
    leaves: list[LeafInfo]
    entries: int
    tot_bytes: int
    zip_bytes: int
    baskets: int
    basket_size: int


def from_store(store: Store, stats: Any = None) -> list[BranchInfo]:
    """The branches of a tree being filled, with the byte counts of ``stats`` if it has them.

    ``stats`` is the tree as read back from memory, whose records hold what
    writing the baskets took; without it every count is zero.
    """
    grouped: dict[str, list[LeafInfo]] = {}
    for slot in store.slots.values():
        leaf = LeafInfo(
            slot.name,
            slot.name,
            slot.title,
            slot.typename,
            slot.classname,
            slot.size,
            slot.counter,
            slot.branch,
            slot.kind == TEXT,
            slot.kind == VECTOR,
        )
        grouped.setdefault(slot.branch, []).append(leaf)
    return [_stored(name, leaves, store, stats) for name, leaves in grouped.items()]


def _stored(name: str, leaves: list[LeafInfo], store: Store, stats: Any) -> BranchInfo:
    title = store.titles.get(name, leaves[0].title)
    classname = leaves[0].typename if leaves[0].vector else ""
    records = [stats.branches[leaf.name].record for leaf in leaves] if stats is not None else []
    return BranchInfo(name, title, classname, leaves, store.entries, *_sizes(records))


def _sizes(records: list[Any]) -> tuple[int, int, int, int]:
    """What a branch's records say its baskets hold: bytes, zipped bytes, count, size."""
    return (
        sum(record.tot_bytes for record in records),
        sum(record.zip_bytes for record in records),
        max((len(record.basket_seek) for record in records), default=0),
        max((record.basket_size for record in records), default=32000),
    )


def _read_leaf(label: str, branch: Any) -> LeafInfo:
    leaf = branch.leaf
    record = branch.record
    vector = leaf.classname == "TLeafElement" and record.classname.startswith("vector<")
    typename = record.classname if leaf.classname == "TLeafElement" else _typename(leaf)
    counter = leaf.count.name if leaf.count is not None else None
    return LeafInfo(
        leaf.name or label,
        label,
        leaf.title or label,
        typename,
        leaf.classname,
        leaf.length,
        counter,
        record.name,
        leaf.classname == "TLeafC" or typename in ("string", "TString"),
        vector,
    )


def _typename(leaf: Any) -> str:
    found = LEAF_TYPES.get(leaf.classname)
    if found is None:
        return str(leaf.classname)
    return found[1] if leaf.unsigned else found[0]


def from_tree(tree: Any) -> list[BranchInfo]:
    """The branches of a tree read from a file - the first file's, for a chain."""
    grouped: dict[int, tuple[Any, list[LeafInfo]]] = {}
    for label, branch in tree.branches.items():
        record = branch.record
        grouped.setdefault(id(record), (record, []))[1].append(_read_leaf(label, branch))
    return [_recorded(record, leaves) for record, leaves in grouped.values()]


def _recorded(record: Any, leaves: list[LeafInfo]) -> BranchInfo:
    return BranchInfo(
        record.name,
        record.title,
        record.classname,
        leaves,
        record.entries,
        record.tot_bytes,
        record.zip_bytes,
        len(record.basket_seek),
        record.basket_size,
    )
