"""``xrdroot scan``: ``TTree::Scan`` from the command line.

    $ xrdroot scan tests/data/simple.root:tree "one:two" "one > 2"
    ************************************
    *    Row   *       one *       two *
    ************************************
    *        2 *         3 *       3.3 *
    *        3 *         4 *       4.4 *
    ************************************
    ==> 2 selected entries

The expressions and the cut are ``TTree::Draw``'s, and the table is laid out
exactly as ROOT lays it; a file with one tree needs no ``:tree``.
"""

from __future__ import annotations

import argparse
from typing import Any

from .target import opened, resolve

__all__ = ["add_parser", "run", "table_of"]


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "scan",
        help="print a tree's values as TTree::Scan does",
        description="Print expressions over a tree's entries as ROOT's table of text.",
    )
    parser.add_argument("file", metavar="FILE[:tree]", help="a file, and the tree in it")
    parser.add_argument(
        "expressions", nargs="?", default="*", help='"expr:expr", or "*" for every column'
    )
    parser.add_argument("cut", nargs="?", default="", help="the entries to keep")
    parser.add_argument("-k", "--key", help="the tree, for a file name without .root")
    parser.add_argument("-n", "--entries", type=int, default=None, help="at most this many")
    parser.add_argument("--first", type=int, default=0, help="the entry to start from")
    parser.add_argument("--width", type=int, default=None, help="every column this wide")
    parser.add_argument("--precision", type=int, default=None, help="digits of a number")


def table_of(file: Any, path: str) -> Any:
    """The tree ``path`` names, or the file's only tree when it names none."""
    return resolve(file, path) if path else file.tree()


def run(args: argparse.Namespace) -> int:
    with opened(args.file, args.key) as (file, path):
        tree = table_of(file, path)
        table = tree.scan(
            args.expressions,
            args.cut,
            entries=args.entries,
            first_entry=args.first,
            width=args.width,
            precision=args.precision,
        )
    print(table.rstrip("\n"))
    return 0
