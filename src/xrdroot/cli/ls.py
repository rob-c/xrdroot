"""``xrdroot ls``: what a file holds, as ROOT's ``rootls`` and go-hep's ``root-ls`` list it.

    $ xrdroot ls -t tests/data/small-flat-tree.root
    === [tests/data/small-flat-tree.root] ===
    version: 60806
    TTree  tree  my tree title  (cycle=1, entries=100)
      Int32    "Int32/I"    int32
      ...

A line per key - its class, name, title and cycle - with the directories
walked all the way down, each level indented under the one it is in. ``-t``
opens every tree and RNTuple and lists its columns with their types, and
``-l`` adds what each record costs: the bytes on disk, uncompressed, the
ratio between them and when it was written, and a tree's baskets.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from typing import Any, NamedTuple

from ..errors import ROOTError
from ..file import Directory, Key, ROOTFile
from .target import CONTAINERS, TABLES, key_of, opened, resolve

__all__ = ["add_parser", "run", "listing"]

#: Spaces a level of directory, or a tree's columns, is indented by.
INDENT = "  "


class Options(NamedTuple):
    """What ``-t`` and ``-l`` asked for."""

    trees: bool
    long: bool


class Row(NamedTuple):
    """One line of a listing, before its columns are lined up with the others'.

    ``kind`` says which lines line up with which: keys with keys, and a
    tree's columns with the columns.
    """

    kind: str
    depth: int
    cells: tuple[str, str, str]
    notes: str


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "ls",
        help="list the keys of files, as rootls does",
        description="List every key - class, name, title and cycle - directories and all.",
    )
    parser.add_argument("files", nargs="+", metavar="FILE[:path]", help="a file, or a path in one")
    parser.add_argument("-k", "--key", help="the path in the file, for a name without .root")
    parser.add_argument(
        "-t", "--trees", action="store_true", help="list the columns of trees and RNTuples"
    )
    parser.add_argument(
        "-l", "--long", action="store_true", help="sizes, compression and dates as well"
    )


def run(args: argparse.Namespace) -> int:
    options = Options(args.trees, args.long)
    for index, text in enumerate(args.files):
        if index:
            print()
        with opened(text, args.key) as (file, path):
            print("\n".join(listing(file, path, options)))
    return 0


def listing(file: ROOTFile, path: str, options: Options) -> list[str]:
    """The lines ``ls`` prints for one file, or one path in it."""
    head = [f"=== [{file.name}] ===", f"version: {file.version}"]
    target = resolve(file, path)
    if isinstance(target, Directory):
        rows = list(_directory_rows(target, options, 0))
    else:
        directory, key = key_of(file, path)
        rows = list(_key_rows(directory, key, options, 0))
    return head + _aligned(rows)


def _directory_rows(directory: Directory, options: Options, depth: int) -> Iterator[Row]:
    for key in directory.all_keys():
        yield from _key_rows(directory, key, options, depth)


def _key_rows(directory: Directory, key: Key, options: Options, depth: int) -> Iterator[Row]:
    """A key's line, then what is under it: a directory's keys, or a tree's columns."""
    notes = [f"cycle={key.cycle}"]
    if options.long:
        notes.append(_sizes(key))
    columns: list[Row] = []
    if options.trees and key.classname in TABLES:
        notes_more, columns = _table(directory, key, options, depth + 1)
        notes.extend(notes_more)
    yield Row("key", depth, (key.classname, key.name, key.title), f"({', '.join(notes)})")
    yield from columns
    if key.classname in CONTAINERS:
        yield from _directory_rows(directory[f"{key.name};{key.cycle}"], options, depth + 1)


def _sizes(key: Key) -> str:
    """What a record costs: on disk, uncompressed, the ratio, and when it was written."""
    stored = key.nbytes - key.keylen
    ratio = key.objlen / stored if stored > 0 else 1.0
    return (
        f"{key.nbytes} bytes on disk, {key.objlen} uncompressed, x{ratio:.2f}, "
        f"{key.time:%Y-%m-%d %H:%M:%S}"
    )


def _table(
    directory: Directory, key: Key, options: Options, depth: int
) -> tuple[list[str], list[Row]]:
    """A tree's or an RNTuple's entries, and a line per column - or why it would not open."""
    try:
        table = directory[f"{key.name};{key.cycle}"]
    except ROOTError as why:
        return [f"unreadable: {why}"], []
    return [f"entries={len(table)}"], list(_column_rows(table, options, depth))


def _column_rows(table: Any, options: Options, depth: int) -> Iterator[Row]:
    branches = getattr(table, "branches", None)
    if branches is None:  # an RNTuple: fields, with the C++ they were written from
        for name, cxx in table.cxx_types().items():
            yield Row("column", depth, (name, cxx, table.fields[name].typename), "")
        return
    for name, branch in branches.items():
        kind = table.typenames()[name]
        notes = ""
        if options.long:
            notes = f"({branch.num_baskets} baskets, {sum(branch.record.basket_bytes)} bytes)"
        yield Row("column", depth, (name, f'"{branch.title}"', kind), notes)


def _aligned(rows: list[Row]) -> list[str]:
    """The rows as lines, each kind's columns as wide as their widest."""
    widths: dict[str, list[int]] = {}
    for row in rows:
        width = widths.setdefault(row.kind, [0, 0, 0])
        sizes = (len(INDENT) * row.depth + len(row.cells[0]), len(row.cells[1]), len(row.cells[2]))
        widths[row.kind] = [max(pair) for pair in zip(width, sizes)]
    return [_line(row, widths[row.kind]) for row in rows]


def _line(row: Row, widths: list[int]) -> str:
    indent = INDENT * row.depth
    first = f"{indent}{row.cells[0]}".ljust(widths[0])
    text = f"{first} {row.cells[1]:<{widths[1]}} {row.cells[2]:<{widths[2]}} {row.notes}"
    return text.rstrip()
