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

from .sizes import Streamed, streamed
from .store import OBJECT, TEXT, VECTOR, Store

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
    #: The bytes of the branch's own record, which ROOT's total adds to its baskets'.
    streamed: int = 0
    #: The branches under it, for a branch of objects: one per member, or per member of
    #: every object in a collection.
    children: tuple[BranchInfo, ...] = ()
    #: ``TBranchElement``'s ``fID`` and ``fType``: what kind of branch of objects it is -
    #: ``fID`` -2 the top of a split object, ``fType`` 2 or more one holding others.
    fid: int = -1
    btype: int = 0

    def walk(self) -> list[BranchInfo]:
        """This branch and every branch under it, in the order ROOT lists them."""
        return [self, *(below for child in self.children for below in child.walk())]


def from_store(store: Store, stats: Any = None, written: bool = False) -> list[BranchInfo]:
    """The branches of a tree being filled, with the byte counts of ``stats`` if it has them.

    ``stats`` is the tree as read back from memory, whose records hold what
    writing the baskets took; without it every count is zero. Until the tree
    is ``written``, its baskets are ROOT's one basket in memory: none on file.
    """
    grouped: dict[str, list[LeafInfo]] = {}
    objects: dict[str, BranchInfo] = {}
    for slot in store.slots.values():
        if slot.kind in (OBJECT, VECTOR):  # a branch of objects: a vector is one too
            from .objectinfo import object_info

            objects[slot.branch] = object_info(slot, stats, store.entries, written)
            grouped[slot.branch] = []
            continue
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
    made = [
        objects.get(name) or _plain(_stored(name, leaves, store, stats), written)
        for name, leaves in grouped.items()
    ]
    return made


def _plain(info: BranchInfo, written: bool) -> BranchInfo:
    """A branch of numbers, with its record's length; none of its baskets on file until
    the tree is ``written``."""
    return _with_record(info if written else info._replace(zip_bytes=0, baskets=0))


def _with_record(info: BranchInfo) -> BranchInfo:
    """A branch with the length of its own record, when it is one leaf this writer knows."""
    if len(info.leaves) != 1:
        return info
    leaf = info.leaves[0]
    made = Streamed(
        info.name,
        info.title,
        leaf.name,
        leaf.title,
        leaf.classname,
        leaf.typename.startswith("U"),
        leaf.size,
        leaf.counter,
        info.baskets,
    )
    return info._replace(streamed=streamed(made))


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
    tops = getattr(tree, "records", None)
    if tops is None:  # a tree that keeps no records of its own: every branch, flat
        return [_with_record(_recorded(record, leaves)) for record, leaves in grouped.values()]
    return [_nested(record, grouped) for record in tops]


def _nested(record: Any, grouped: dict[int, tuple[Any, list[LeafInfo]]]) -> BranchInfo:
    """A branch read from a file and the branches under it, as ROOT nests them."""
    leaves = grouped.get(id(record), (record, []))[1]
    children = tuple(_nested(child, grouped) for child in record.branches)
    fid, btype = (record.element[4], record.element[5]) if record.element else (-1, 0)
    made = _with_record(_recorded(record, leaves))
    return made._replace(children=children, fid=fid, btype=btype)


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
