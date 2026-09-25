"""``xrdroot print``: draw what a file holds into picture files - ROOT's ``rootprint``.

    $ xrdroot print tests/data/gauss-h1.root:h1d -o h1d.png
    $ xrdroot print tests/data/graphs.root -o graphs.pdf     # graphs_tg.pdf, graphs_tge.pdf...
    $ xrdroot print tests/data/gauss-h1.root:h1d             # no -o: drawn in characters

One object goes to the file ``-o`` names, in the format its suffix says. A
directory, or a whole file, draws everything in it that draws - each to
``-o``'s name with the object's path after an underscore - and skips what
does not. Without ``-o``, an object's ``.text()`` picture is printed, which
needs nothing installed. ``--option name=value`` is handed to ``.plot()``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..errors import UnsupportedFeatureError
from ..file import Directory
from .render import options, render
from .target import CONTAINERS, key_of, opened, resolve, walk

__all__ = ["add_parser", "run"]


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "print",
        help="draw objects into png, pdf or svg files, as rootprint does",
        description="Draw a histogram, graph or canvas - or everything in a directory - "
        "into picture files; without -o, draw one in characters.",
    )
    parser.add_argument("file", metavar="FILE[:path]", help="a file, or a path in one")
    parser.add_argument("-k", "--key", help="the path in the file, for a name without .root")
    parser.add_argument("-o", "--output", help="the picture file: .png, .pdf, .svg...")
    parser.add_argument(
        "--option",
        action="append",
        metavar="NAME=VALUE",
        help="a keyword for .plot(), such as color=red; again for more",
    )


def _named(output: str, path: str) -> str:
    """``out.png`` for the object at ``dir/h``: ``out_dir_h.png``."""
    where = Path(output)
    return str(where.with_name(f"{where.stem}_{path.replace('/', '_')}{where.suffix}"))


def _everything(directory: Directory, prefix: str, output: str, style: dict[str, Any]) -> int:
    """Draw everything in ``directory`` that draws; say what was skipped and why."""
    for path, key, parent in walk(directory, prefix):
        if key.classname in CONTAINERS:
            continue
        try:
            value = parent[key.name]
            written = render(value, _named(output, path), style, key.classname)
        except UnsupportedFeatureError as why:
            print(f"skipped {path} ({key.classname}): {why}")
            continue
        print(f"wrote {written}")
    return 0


def _text(value: Any, classname: str) -> int:
    drawn = getattr(value, "text", None)
    if not callable(drawn):
        raise UnsupportedFeatureError(
            f"a {classname} has no picture in characters; give -o a picture file to draw it into"
        )
    print(drawn())
    return 0


def run(args: argparse.Namespace) -> int:
    style = options(args.option)
    with opened(args.file, args.key) as (file, path):
        target = resolve(file, path)
        if isinstance(target, Directory):
            if args.output is None:
                raise ValueError("a whole directory is drawn into files: say where with -o")
            return _everything(target, f"{path}/" if path else "", args.output, style)
        classname = key_of(file, path)[1].classname
        if args.output is None:
            return _text(target, classname)
        print(f"wrote {render(target, args.output, style, classname)}")
    return 0
