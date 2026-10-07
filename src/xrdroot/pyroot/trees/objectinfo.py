"""What a branch of objects is, for ``Print``, ``GetBranch`` and ``GetListOfBranches``.

The branch is laid out by :mod:`xrdroot.wbranch` - the same branches, the
same leaves, under the same names - and this turns that layout into the
:class:`~.layout.BranchInfo` everything that describes a tree reads, with
what writing it took: each column's baskets as the tree read back from
memory records them, and the length ROOT's ``TBranch::GetTotalSize``
streams the branch to (:func:`xrdroot.wbranch.streamed_size`).
"""

from __future__ import annotations

from typing import Any

from ...wbranch import WHOLE, Branch, NoBaskets, ObjectBranch, streamed_size
from ...wpacking import BASIC
from ...writer import BASKET_BYTES
from .layout import BranchInfo, LeafInfo

__all__ = ["object_info"]

#: What ``TLeafElement::GetTypeName`` says of a collection's count.
COUNT_TYPE = "Int_t"


def object_info(slot: Any, stats: Any, entries: int, written: bool) -> BranchInfo:
    """The branch an object slot fills, its baskets as ``stats`` - the tree read back - has
    them; until the tree is ``written``, none of them on file."""
    top = slot.spec().build(slot.name, BASKET_BYTES)
    records = getattr(stats, "branches", {})
    for branch in _walk(top):
        found = records.get(branch.name)
        if found is not None and not isinstance(branch.baskets, NoBaskets):
            _restore(branch.baskets, found.record)
    return _info(top, entries, written)


def _walk(branch: Branch) -> list[Branch]:
    return [branch, *(below for child in branch.children for below in _walk(child))]


def _restore(column: Any, record: Any) -> None:
    """A column given the baskets a record says it has, for measuring the branch."""
    column.seeks = list(record.basket_seek)
    column.sizes = list(record.basket_bytes)
    column.starts = list(record.basket_entry)
    column.tot_bytes, column.zip_bytes = record.tot_bytes, record.zip_bytes
    if hasattr(column, "offset_len"):
        column.offset_len = record.entry_offset_len


def _info(branch: Branch, entries: int, written: bool) -> BranchInfo:
    baskets = branch.baskets
    children = tuple(_info(child, entries, written) for child in branch.children)
    on_file = written and not isinstance(baskets, NoBaskets)
    return BranchInfo(
        branch.name,
        branch.title,
        branch.info.classname,
        [_leaf(branch, leaf) for leaf in branch.own],
        entries,
        baskets.tot_bytes,
        baskets.zip_bytes if on_file else 0,
        len(baskets.seeks) if on_file else 0,
        baskets.basket_size,
        streamed_size(branch, entries, held=written),
        children,
        branch.info.fid,
        branch.info.btype,
    )


def _leaf(branch: Branch, leaf: Any) -> LeafInfo:
    """One leaf of a branch of objects, read by its branch's name."""
    typename = _typename(branch)
    classname = "TLeafObject" if isinstance(branch, ObjectBranch) else "TLeafElement"
    return LeafInfo(
        leaf.name,
        branch.name,
        leaf.title if classname == "TLeafElement" else leaf.classname,
        typename,
        classname,
        1,
        None,
        branch.name,
        False,
        typename.startswith("vector<"),
    )


def _typename(branch: Branch) -> str:
    """What ``TLeafElement::GetTypeName`` says: a member's type, or the object's class."""
    info = branch.info
    if info.btype == 4:
        return COUNT_TYPE
    if info.fid == WHOLE or info.stype < 0:
        return info.classname
    return BASIC[info.stype][2]
