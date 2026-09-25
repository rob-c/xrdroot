"""``TTree::Print`` and ``TTree::Show``: the text ROOT prints, laid out as it lays it out.

The formats are ``TTree::Print``'s and ``TBranch::Print``'s ``printf``
strings, and a long branch title is broken where ``TBranch::Print`` breaks
it - at a colon, before the line would pass column 77, the next line
starting ``*         |``. ``Show`` is ``TTree::Show``'s: a leaf's name in
fifteen columns, at most twenty of its values, and the line broken after the
first value and then every five (for floating point and text) or ten (for
integers) - ROOT's own arithmetic, kept exactly.

The byte counts are this reader's: the baskets' sizes as the file records
them, which for a tree written here are the sizes this writer's baskets
took, and for ROOT's own trees are ROOT's, less the few hundred bytes ROOT
adds for the size of the branch record itself.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from .layout import BranchInfo, LeafInfo

__all__ = ["tree_lines", "branch_lines", "show_lines"]

#: The column a line of ``Print`` ends at, before its closing star.
LINE_END = 77
#: The line of stars around a tree's header.
STARS = "*" * 78
#: The line of dots after each branch.
DOTS = "*" + "." * 76 + "*"
#: How many values of one leaf ``Show`` prints at most: its ``lenmax``.
SHOW_MAX = 20
#: The leaf classes ``Show`` puts five values to a line for, rather than ten.
FIVE_A_LINE = ("TLeafF", "TLeafD", "TLeafC", "TLeafF16", "TLeafD32")


def _ratio(tot: int, zipped: int) -> float:
    return (tot + 0.00001) / zipped if zipped else 1.0


def tree_lines(name: str, title: str, entries: int, branches: Sequence[BranchInfo]) -> list[str]:
    """The header ``TTree::Print`` puts above the branches."""
    tot = sum(branch.tot_bytes for branch in branches)
    zipped = sum(branch.zip_bytes for branch in branches)
    return [
        STARS,
        f"*Tree    :{name:<10}: {title:<54} *",
        f"*Entries : {entries:8d} : Total = {tot:15d} bytes  File  Size = {zipped:10d} *",
        f"*        :          : Tree compression factor = {_ratio(tot, zipped):6.2f}"
        "                       *",
        STARS,
    ]


def _title(branch: BranchInfo) -> str:
    """What ``TBranch::Print`` shows for a title: the leaf's type when the title says nothing."""
    text = "" if branch.title == branch.name else branch.title
    if len(branch.leaves) != 1:
        return text
    if len(text) >= 2 and text[-2] == "/" and text[-1].isalpha():
        return text
    return branch.leaves[0].typename + (f" {text}" if text else "")


def _pieces(text: str) -> Iterator[str]:
    """A title in the pieces ``TBranch::Print`` wraps it by: up to and with each colon."""
    begin = 0
    while begin < len(text):
        end = begin + 1
        while end < len(text) - 1 and text[end] != ":":
            end += 1
        yield text[begin : end + 1]
        begin = end + 1


def _wrapped(head: str, text: str) -> str:
    line, column = head, len(head)
    for piece in _pieces(text):
        if column + len(piece) >= LINE_END + 1:
            line += " " * (LINE_END - column) + "*\n*" + " " * 9 + "| "
            column = 12
        line += piece
        column += len(piece)
    return line + " " * (LINE_END - column) + "*"


def branch_lines(branch: BranchInfo, count: int) -> list[str]:
    """The four lines ``TBranch::Print`` prints for one branch, numbered ``count``."""
    text = _title(branch)
    head = f"*Br{count:5d} :{branch.name:<9} : "
    first = head + f"{text or ' ':<54}"
    first = _wrapped(head, text) if len(first) > LINE_END else first + " *"
    ratio = _ratio(branch.tot_bytes, branch.zip_bytes)
    return [
        first,
        f"*Entries :{branch.entries:9d} : Total  Size={branch.tot_bytes:11d} bytes  "
        f"File Size  = {branch.zip_bytes:10d} *",
        f"*Baskets :{branch.baskets:9d} : Basket Size={branch.basket_size:11d} bytes  "
        f"Compression= {ratio:6.2f}     *",
        DOTS,
    ]


def _printed(leaf: LeafInfo, value: Any) -> str:
    """One value as ``TLeaf::PrintValue`` prints it: ``%g`` for floating point, else whole."""
    number = np.asarray(value).item()
    if isinstance(number, float):
        return f"{number:g}"
    return str(int(number))


def _values(leaf: LeafInfo, value: Any) -> list[Any]:
    if leaf.text:
        return [str(value)]
    return list(np.asarray(value).reshape(-1)[:SHOW_MAX])


def _joined(leaf: LeafInfo, value: Any) -> str:
    if leaf.text:
        return str(value)
    per_line = 5 if leaf.classname in FIVE_A_LINE or leaf.vector else 10
    shown = [_printed(leaf, each) for each in _values(leaf, value)]
    text = ""
    for at, each in enumerate(shown):
        text += each
        if at == len(shown) - 1:
            break
        text += ", "
        if at % per_line == 0:
            text += "\n" + " " * 18
    return text


def show_lines(entry: int, leaves: Sequence[tuple[LeafInfo, Any]]) -> list[str]:
    """What ``TTree::Show`` prints for one entry, given each leaf and its value in it."""
    lines = [f"======> EVENT:{entry}"]
    for leaf, value in leaves:
        if not leaf.text and np.size(value) == 0:
            continue
        lines.append(f" {leaf.name:<15} = {_joined(leaf, value)}")
    return lines
