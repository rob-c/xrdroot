"""``xrdroot draw``: ``TTree::Draw`` from the command line, into a picture or characters.

    $ xrdroot draw tests/data/small-flat-tree.root:tree "Float64" "Int32 > 50" -o f.png
    $ xrdroot draw tests/data/small-flat-tree.root "Float64:Int32" --option prof

The histogram, profile or graph ``TTree::Draw`` fills - binned by ROOT's rules,
or by ``>>h(40, 0, 4)`` in the expression - is drawn into ``-o``'s file, or
without ``-o`` printed as its ``.text()`` picture, with the number of fills
after it as ROOT's ``Draw`` returns it.
"""

from __future__ import annotations

import argparse
from typing import Any

from .render import options, render
from .scan import table_of
from .target import opened

__all__ = ["add_parser", "run"]


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "draw",
        help="fill a histogram from a tree as TTree::Draw does, and draw it",
        description="Fill a histogram, profile or graph from expressions over a tree and "
        "draw it into a picture file, or in characters without -o.",
    )
    parser.add_argument("file", metavar="FILE[:tree]", help="a file, and the tree in it")
    parser.add_argument("expression", help='"x", "y:x" or "z:y:x", and >>h(bins) if wanted')
    parser.add_argument("cut", nargs="?", default="", help="the weight, or the entries to keep")
    parser.add_argument("-k", "--key", help="the tree, for a file name without .root")
    parser.add_argument("-o", "--output", help="the picture file: .png, .pdf, .svg...")
    parser.add_argument("--option", default="", help="Draw's option: prof, l, p...")
    parser.add_argument("-n", "--entries", type=int, default=None, help="at most this many")
    parser.add_argument("--first", type=int, default=0, help="the entry to start from")
    parser.add_argument(
        "--style",
        action="append",
        metavar="NAME=VALUE",
        help="a keyword for .plot(), such as color=red; again for more",
    )


def run(args: argparse.Namespace) -> int:
    style = options(args.style)
    with opened(args.file, args.key) as (file, path):
        drawn = table_of(file, path).draw(
            args.expression, args.cut, args.option, entries=args.entries, first_entry=args.first
        )
    if args.output is None:
        print(drawn.text())
    else:
        print(f"wrote {render(drawn, args.output, style, drawn.classname)}")
    print(f"==> {drawn.selected} selected")
    return 0
