"""What ``TTree::Print`` counts for a circular tree: its one basket, never written.

``TTree::SetCircular`` keeps a tree's entries in memory and its baskets
there with them - ROOT never flushes a circular tree's basket, it moves the
oldest entries out instead - so ``TBranch::GetTotalSize``, which streams the
branch on its own, streams that one basket with it: its class tag, its key
header, its own fields, and its buffer, which holds the key header again
before the values. The key header names the branch and the tree, so each of
their names is counted twice.

``TTree::Print``'s total streams the whole tree in one buffer instead, so a
class seen once is a reference after: each branch after the first names
``TBranch`` in four bytes rather than twelve, a leaf of a class already
streamed saves its class's name, and a basket after the first saves
``TBasket``'s. Every branch in the tree's buffer also carries its own byte
count, and the tree's list of leaves a reference to its leaf.

The arithmetic holds for the branches this writer streams as ROOT does -
one leaf of a fixed number of values - and the counts were measured against
ROOT 6.40's; any other branch keeps the counts of the baskets this writer
would write.
"""

from __future__ import annotations

from collections.abc import Sequence

from .layout import BranchInfo
from .sizes import Streamed, streamed
from .store import Store

__all__ = ["circular", "branch_in_memory", "tree_record"]

#: A basket streamed with its branch, less its two copies of the branch's and the tree's
#: names: its byte count and class tag, key header and fields, and the header again.
BASKET_RECORD = 142
#: The tree's own record without its branches, less its name and title.
TREE_RECORD = 267
#: A branch in the tree's list: its byte count, and the tree's reference to its leaf.
IN_LIST = 8
#: The class tag naming ``TBranch`` the first time, and a reference to it after.
BRANCH_TAG, BRANCH_REFERENCE = 12, 4
#: What a basket after the first saves: ``TBasket``'s name, and its string's length.
BASKET_NAME = len("TBasket") + 1


def _plain(branch: BranchInfo) -> bool:
    """Whether the branch is one leaf of a fixed number of values, as ROOT's circular ones."""
    if len(branch.leaves) != 1:
        return False
    leaf = branch.leaves[0]
    return leaf.counter is None and not leaf.text and not leaf.vector


def circular(branches: Sequence[BranchInfo], tree: str, store: Store) -> list[BranchInfo]:
    """The branches of a circular tree as ROOT counts them, each with its basket in memory."""
    return [
        branch_in_memory(branch, tree, store.slots[branch.leaves[0].name].dtype.itemsize)
        for branch in branches
    ]


def branch_in_memory(branch: BranchInfo, tree: str, itemsize: int) -> BranchInfo:
    """The branch as a circular tree holds it: its values and one basket, all in memory."""
    if not _plain(branch):
        return branch
    leaf = branch.leaves[0]
    values = branch.entries * leaf.size * itemsize
    basket = BASKET_RECORD + 2 * len(branch.name) + 2 * len(tree) if branch.entries else 0
    record = Streamed(
        branch.name,
        branch.title,
        leaf.name,
        leaf.title.split("/")[0],  # ROOT's leaf is titled by its leaf list, less the type
        leaf.classname,
        leaf.typename.startswith("U"),
        leaf.size,
        None,
        1,
    )
    return branch._replace(tot_bytes=values + basket, streamed=streamed(record))


def tree_record(name: str, title: str, branches: Sequence[BranchInfo]) -> int | None:
    """What the tree adds to its branches' bytes, streamed in one buffer; None if unknown.

    ``branch_in_memory`` counts the basket in a branch's total bytes and its
    record apart, the two added being what ``GetTotalSize`` says; this is
    every record, the tree's own among them, as the tree's buffer holds them.
    """
    if not all(_plain(branch) for branch in branches):
        return None
    total = TREE_RECORD + len(name) + len(title)
    classes: set[str] = set()
    baskets = 0
    for at, branch in enumerate(branches):
        total += IN_LIST + (BRANCH_REFERENCE if at else BRANCH_TAG) + branch.streamed
        leaf = branch.leaves[0].classname
        total -= len(leaf) + 1 if leaf in classes else 0
        classes.add(leaf)
        if branch.entries:
            total -= BASKET_NAME if baskets else 0
            baskets += 1
    return total
