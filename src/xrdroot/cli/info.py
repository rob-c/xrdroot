"""``xrdroot info``: what a file's header says about it, and the classes it describes.

    $ xrdroot info tests/data/gauss-h1.root
    file:          tests/data/gauss-h1.root
    ROOT version:  60806 (6.08/06)
    ...

The header is the first hundred bytes of every ROOT file: which ROOT wrote
it, where it ends, where its free segments and its class descriptions are,
the compression it was written with and the UUID it was given. After it,
every class the file's streamer information describes, with the number of
members each has.
"""

from __future__ import annotations

import argparse
import uuid
from typing import Any

from ..file import ROOTFile
from ..wupdate import _read_header
from .target import opened

__all__ = ["add_parser", "run", "information"]

#: ROOT's compression algorithms, by the hundreds of the code a file keeps.
ALGORITHMS = {0: "zlib", 1: "zlib", 2: "lzma", 3: "zlib (old)", 4: "lz4", 5: "zstd"}


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "info",
        help="a file's header - version, compression, UUID, sizes - and its classes",
        description="Print what a ROOT file's header says, and the classes it describes.",
    )
    parser.add_argument("files", nargs="+", metavar="FILE", help="a file")


def run(args: argparse.Namespace) -> int:
    for index, text in enumerate(args.files):
        if index:
            print()
        with opened(text) as (file, _):
            print("\n".join(information(file)))
    return 0


def compression(code: int) -> str:
    """ROOT's compression code in words: ``101`` is zlib at level 1, ``0`` none at all."""
    algorithm, level = divmod(code, 100)
    if level == 0:
        return f"none ({code})"
    return f"{ALGORITHMS.get(algorithm, f'algorithm {algorithm}')}, level {level} ({code})"


def _release(version: int) -> str:
    """``62406`` as ROOT writes its releases: ``6.24/06``."""
    major, rest = divmod(version, 10000)
    minor, patch = divmod(rest, 100)
    return f"{major}.{minor:02d}/{patch:02d}"


def information(file: ROOTFile) -> list[str]:
    """The lines ``info`` prints for one file."""
    header = _read_header(file._source)
    version = header.version % 1000000
    classes = file._source.streamers()
    lines = [
        f"file:              {file.name}",
        f"ROOT version:      {version} ({_release(version)})",
        f"format:            {'64-bit' if header.version >= 1000000 else '32-bit'} seeks",
        f"size:              {header.end} bytes",
        f"compression:       {compression(header.codes)}",
        f"UUID:              {uuid.UUID(bytes=header.uuid)}",
        f"first record:      {header.begin}",
        f"free segments:     {header.nfree} at {header.seek_free} ({header.nbytes_free} bytes)",
        f"streamer info:     at {header.seek_info} ({header.nbytes_info} bytes)",
        f"keys at the top:   {len(file.all_keys())}",
        f"classes described: {len(classes)}",
    ]
    lines.extend(f"  {name} ({len(members)} members)" for name, members in classes.items())
    return lines
