"""``xrdroot cp``: ROOT's ``rootcp``, and ``TTree::CopyTree`` with ``--cut``.

    $ xrdroot cp in.root out.root                         # everything
    $ xrdroot cp 'in.root:hists/*' in.root:events out.root
    $ xrdroot cp in.root:events skim.root:skims --cut 'nMuon >= 2' --columns nMuon,Muon_pt

Each SOURCE is a file, and after it - as ``rootcp`` spells it - a colon
and the path or shell pattern of what to copy from it; with none, everything
in it. DEST is a file, added to if it is there unless ``--recreate`` says to
write it anew, and after it a colon and the directory to copy into. A file
is told from what follows it by the ``.root`` its name ends with, so that a
URL's own colons are left alone. See :func:`xrdroot.copy`.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from ..errors import ROOTError
from ..file import open_root
from ..merging import copy
from ..merging.files import _is_local_file, _local_path, resolve_compression
from ..writer import WritableFile, create
from ..wupdate import update

__all__ = ["add_parser", "run", "split"]

#: What a file's name ends with, before the colon and the path inside it.
SUFFIX = ".root:"


def split(spec: str) -> tuple[str, str | None]:
    """``file.root:path`` as the file and the path, or the file and ``None``."""
    at = spec.rfind(SUFFIX)
    if at < 0:
        return spec, None
    end = at + len(SUFFIX) - 1
    return spec[:end], spec[end + 1 :] or None


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """The ``cp`` subcommand, with ``rootcp``'s way of naming what to copy."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "cp",
        help="copy objects between ROOT files, as rootcp does",
        description="Copy objects from each SOURCE into DEST as rootcp does; a tree given "
        "--cut or --columns is copied as TTree::CopyTree copies it.",
    )
    parser.add_argument("sources", nargs="+", metavar="SOURCE", help="file.root[:pattern]")
    parser.add_argument("destination", metavar="DEST", help="file.root[:directory]")
    parser.add_argument(
        "-c", "--compress", type=int, help="ROOT's compression setting for DEST, as 505"
    )
    parser.add_argument("--recreate", action="store_true", help="write DEST anew")
    parser.add_argument("--cut", help="keep only the tree entries that pass this")
    parser.add_argument("--columns", help="the branches to keep, separated by commas")
    parser.add_argument(
        "--slow", action="store_true", help="write trees' entries afresh, never their baskets"
    )
    parser.add_argument("-r", "--recursive", action="store_true", help="taken: always so")
    return parser


def _open(destination: str, recreate: bool, compression: int | None, first: str) -> WritableFile:
    """The file the copies go into: added to if it is there, else made - compressed as asked,
    or, for a new file, as the first source is."""
    if _is_local_file(destination) and os.path.exists(_local_path(destination)) and not recreate:
        if compression is None:
            return update(destination)
        algorithm, level = resolve_compression(compression, 0)
        return update(destination, compression=algorithm, level=level)
    with open_root(first) as source:
        codes = source.compression
    algorithm, level = resolve_compression(compression, codes)
    return create(destination, compression=algorithm, level=level)


def run(args: argparse.Namespace) -> int:
    """Copy as asked; ``0`` when it is done, ``1`` with the reason on stderr when it is not."""
    target, directory = split(args.destination)
    columns = args.columns.split(",") if args.columns else None
    try:
        sources = [split(spec) for spec in args.sources]
        with _open(target, args.recreate, args.compress, sources[0][0]) as out:
            into = out.mkdir(directory) if directory else out
            for path, pattern in sources:
                copy(path, into, pattern, cut=args.cut, columns=columns, fast=not args.slow)
    except (ROOTError, ValueError, OSError, KeyError, TypeError) as why:
        print(f"xrdroot cp: {why}", file=sys.stderr)
        return 1
    return 0
