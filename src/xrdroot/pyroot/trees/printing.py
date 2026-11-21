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

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from .layout import BranchInfo, LeafInfo
from .memory import tree_record

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


def tree_lines(
    name: str,
    title: str,
    entries: int,
    branches: Sequence[BranchInfo],
    key: Any = None,
    record: int | None = None,
    packed: int = 0,
) -> list[str]:
    """The header ``TTree::Print`` puts above the branches.

    ROOT counts the tree's own key and record in its totals; ``key`` is the
    :class:`xrdroot.Key` the tree was read from, when whoever read it says,
    ``record`` the length of the tree's key and record when it is known
    otherwise - in memory, or written in this session - and ``packed`` that
    record's bytes on file. Without a key the tree is one made here, and ROOT's
    total is then the baskets plus whatever the tree streams to, which
    :func:`tree_record` counts when every branch is one it can count.
    """
    every = [below for branch in branches for below in branch.walk()]
    tot = sum(branch.tot_bytes for branch in every)
    zipped = sum(branch.zip_bytes for branch in every)
    total = tot + _own_record(name, title, branches, key, record)
    on_file = zipped + packed if key is None else zipped + key.nbytes
    return [
        STARS,
        f"*Tree    :{name:<10}: {title:<54} *",
        f"*Entries : {entries:8d} : Total = {total:15d} bytes  File  Size = {on_file:10d} *",
        f"*        :          : Tree compression factor = {_ratio(tot, zipped):6.2f}"
        "                       *",
        STARS,
    ]


def _own_record(
    name: str, title: str, branches: Sequence[BranchInfo], key: Any, record: int | None
) -> int:
    """What the tree's own record adds to its branches' bytes, as ``TTree::Print`` counts it.

    ROOT streams the tree object itself - ``b.Length()`` - and adds the key and
    the record of the file's key only for a tree read from a file: a tree that is
    new has neither, so nothing but its streamed self is added, and the baskets
    count only once it has been written (``TBranch::GetTotalSize`` adds them only
    while ``fZipBytes`` is set).
    """
    if key is not None:
        return int(key.keylen) + int(key.objlen)
    return record if record is not None else (tree_record(name, title, branches) or 0)


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
    first = _wrapped(head, text) if len(first) > LINE_END else first.ljust(LINE_END) + "*"
    ratio = _ratio(branch.tot_bytes, branch.zip_bytes)
    where = f"File Size  = {branch.zip_bytes:10d} *"
    if not branch.zip_bytes and not branch.baskets:
        where = "One basket in memory    *"  # what ROOT says of a branch not yet written
    total = branch.tot_bytes + branch.streamed
    return [
        first,
        f"*Entries :{branch.entries:9d} : Total  Size={total:11d} bytes  {where}",
        f"*Baskets :{branch.baskets:9d} : Basket Size={branch.basket_size:11d} bytes  "
        f"Compression= {ratio:6.2f}     *",
        DOTS,
    ]


def element_lines(branch: BranchInfo, count: int) -> tuple[list[str], int]:
    """``TBranchElement::Print``: a split object's heading, then each branch under it.

    The top of a split object (``fID`` -2) prints a heading of its own; a
    branch holding others and a count (``fType`` 2 or more) prints as a
    branch too; then come the branches under it, numbered on from ``count``.
    What comes back is the lines and the number the next branch takes.
    """
    if not branch.children:
        return branch_lines(branch, count), count + 1
    lines: list[str] = []
    if branch.fid == -2:
        heading = f"{branch.name:<66}"
        if branch.title != branch.name:
            heading = f"{branch.name:<9} : {branch.title:<54}"
        lines += [
            f"*Branch  :{heading} *",
            f"*Entries : {branch.entries:8d} : BranchElement (see below)"
            "                              *",
            DOTS,
        ]
    if branch.btype >= 2:
        lines += branch_lines(branch, count)
        count += 1
    for child in branch.children:
        more, count = element_lines(child, count)
        lines += more
    return lines, count


def _printed(leaf: LeafInfo, value: Any) -> str:
    """One value as ``TLeaf::PrintValue`` prints it: ``%g`` for floating point, else whole.

    A value that is one object's members rather than a number is an object ROOT streams
    itself - a ``TRef`` - which it prints as the class the value names, or, when the record
    names no class, as the pointer ROOT would have shown.
    """
    if isinstance(value, Mapping):
        return str(value.get("classname") or "->")
    number = np.asarray(value).item()
    if isinstance(number, float):
        return f"{number:g}"
    return str(int(number))


def _values(value: Any) -> list[Any]:
    return list(np.asarray(value).reshape(-1)[:SHOW_MAX])


def _joined(leaf: LeafInfo, value: Any) -> str:
    if leaf.text:
        return str(value)
    if leaf.vector:  # ROOT prints where the vector is, not what is in it
        return f"({leaf.typename}*)0x{id(value):x}"
    per_line = 5 if leaf.classname in FIVE_A_LINE or leaf.vector else 10
    text = ""
    for at, each in enumerate(_values(value)):
        if at:  # after every value but the last, and a new line after the first and each run
            text += ", " + ("\n" + " " * 18 if (at - 1) % per_line == 0 else "")
        text += _printed(leaf, each)
    return text


def show_lines(entry: int, leaves: Sequence[tuple[LeafInfo, Any]]) -> list[str]:
    """What ``TTree::Show`` prints for one entry, given each leaf and its value in it."""
    lines = [f"======> EVENT:{entry}"]
    for leaf, value in leaves:
        if isinstance(value, Mapping):
            # A branch that holds a whole object with nothing split out of it hands back the
            # object's members, not a number: ROOT prints the object's members, one a line.
            lines += _record_lines(leaf, value)
            continue
        if not leaf.text and not leaf.vector and np.size(value) == 0:
            continue
        texts = _texts(value)
        printed = _joined(leaf, value) if texts is None else ", ".join(texts)
        lines.append(f" {leaf.name:<15} = {printed}")
    return lines


def _numbers_or_text(value: Any) -> bool:
    """Is this a value ``TLeaf::PrintValue`` can print as numbers - a ``char[]``'s text is not,
    which the caller has already taken out?"""
    flat = np.asarray(value).reshape(-1)
    return bool(flat.size) and flat.dtype.kind in "biufc"


def _texts(value: Any) -> list[str] | None:
    """The strings a member of a ``TString`` per object holds - one name per object of a split
    collection - or ``None`` when the member is not text."""
    flat = np.asarray(value).reshape(-1)
    if not flat.size or flat.dtype.kind not in "US":
        return None
    return [_text_of(each) for each in flat[:SHOW_MAX]]


def _is_characters(value: Any) -> bool:
    """Is this a ``char[N]`` as the reader hands one back - a run of ``int8``?"""
    return isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == np.int8


def _text_of(value: Any) -> str:
    """The C string a ``char[]`` holds: its bytes up to the first NUL, as ``TLeafC`` reads it."""
    if isinstance(value, np.ndarray):
        value = value.tobytes()
    if isinstance(value, bytes):
        value = value.split(b"\x00", 1)[0].decode("utf-8", "replace")
    return str(value)


def _record_lines(leaf: LeafInfo, value: Mapping[str, Any], depth: int = 1) -> list[str]:
    """One object's members as ``Show`` prints them, a base's members where the base is."""
    lines: list[str] = []
    for name, held in value.items():
        if isinstance(held, Mapping):
            lines += _record_lines(leaf, held, depth + 1)
            continue
        one = _one_member(leaf, name, held, depth)
        if one is not None:
            lines.append(one)
    return lines


def _one_member(leaf: LeafInfo, name: str, held: Any, depth: int) -> str | None:
    """One member as ``Show`` prints it: its text, its numbers, or ``->`` for a pointer.

    A collection of objects - a ``TClonesArray`` - is a pointer ROOT prints as ``->`` rather
    than expanding, and a member that is an object is printed by its own lines instead; Show
    is one entry's numbers, not the whole tree under it.
    """
    marker = f" {'  ' * (depth - 1)}{name:<15} ="
    if isinstance(held, (str, bytes)) or _is_characters(held):  # a char[] member: its text
        return f"{marker} {_text_of(held)}"
    texts = _texts(held)
    if texts is not None:  # a TString member: every object of the collection's own
        return f"{marker} {', '.join(texts)}"
    if _numbers_or_text(held):
        return f"{marker} {_joined(leaf, held)}"
    return None if held is None or np.size(held) == 0 else f"{marker} ->"
