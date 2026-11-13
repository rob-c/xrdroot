"""``xrdroot mkdir``: ROOT's ``rootmkdir`` - directories added to ROOT files, or new files.

    $ xrdroot mkdir example.root:dir            # the file must be there, dir not
    $ xrdroot mkdir -p example.root:a/b/c       # every level, and no complaint if there
    $ xrdroot mkdir new.root                    # an empty ROOT file, as touch makes one

Each FILE:path is checked before any file is touched - a ``;cycle`` in a
directory's path is refused - and then each is made in turn, the file
updated in place; what goes wrong with one is said as ``rootmkdir`` says it,
the rest still made, and the status is 1. ``-p`` makes the file and the
parent directories as well, and ``-v`` says what was made.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from ..errors import ROOTError
from ..writer import WritableDirectory, WritableFile, create
from ..wupdate import update
from .sources import DIRECTORIES, split_source
from .tooling import Flag, Log, OptionError, Options, raw_parser

__all__ = ["add_parser", "run", "USAGE"]

#: What ``rootmkdir`` says when its command line is wrong, on standard error.
USAGE = "usage: rootmkdir [-p|--parents] [-v|--verbose] FILE:path/to/dir [...]\n"

#: The rest of ``rootmkdir -h``, after the usage line.
HELP = """
Add directories in ROOT files

positional arguments:
  FILE:path             File(s) and path(s) of directories to create

options:
  -h, --help            get this help
  -p, --parents         create parent directories if needed, no error if directory already exists
  -v, --verbose         be verbose (can be repeated for increased verbosity)

Note that calling rootmkdir without a directory path can be used to create files (similar to the `touch` command).
You can also simultaneously create files and directories in it if you pass the -p flag.

examples:
- rootmkdir example.root:dir
  Add the directory 'dir' to the ROOT file 'example.root' (file must exists)

- rootmkdir example.root:dir1/dir2
  Add the directory 'dir2' in 'dir1' which exists in the ROOT file 'example.root'

- rootmkdir -p example.root:dir1/dir2/dir3
  Make parent directories of 'dir3' as needed, no error if target directory already exists

- rootmkdir non_existent.root
  Create an empty ROOT file named 'non_existent.root'

- rootmkdir -p non_existent.root:dir1
  Create a ROOT file named 'non_existent.root' and directory `dir1` in it
"""  # noqa: E501

#: ``rootmkdir``'s flags.
FLAGS = (
    Flag(("-h", "--help")),
    Flag(("-p", "--parents")),
    Flag(("-v", "--verbose"), repeatable=True),
)


def add_parser(subparsers: Any) -> None:
    raw_parser(subparsers, "mkdir", "add directories to ROOT files, as rootmkdir does")


def run(args: argparse.Namespace) -> int:
    log = Log("rootmkdir")
    try:
        options = Options(FLAGS).parse(args.argv)
    except OptionError as why:
        sys.stderr.write(f"{why}\n{USAGE}")
        return 1
    if options.has("h"):
        sys.stderr.write(USAGE + HELP)
        return 0
    log.verbosity = options.count("v")
    if not options.args:
        sys.stderr.write(USAGE)
        return 1
    targets = []
    for text in options.args:
        try:
            name, path = split_source(text)
        except ValueError as why:
            log.error(str(why))
            return 1
        if ";" in path:
            log.error("cannot specify cycle for the directory to create.")
            return 1
        targets.append((name, path))
    parents = options.has("p")
    failed = [not _made(name, path, parents, log) for name, path in targets]
    return 1 if any(failed) else 0


def _opened(name: str, exists: bool) -> WritableFile:
    return update(name) if exists else create(name)


def _refused(name: str, path: str, parents: bool, exists: bool, log: Log) -> bool:
    """Whether ``rootmkdir`` refuses before opening: no file without ``-p``, or one already."""
    if parents or exists == bool(path):
        return False
    if path:
        log.error(f"cannot create directory '{name}:{path}': file does not exist. Use the -p "
                  "flag if you want to create the file alongside the directories.")  # fmt: skip
    else:
        log.error(f"cannot create file '{name}': already exists.")
    return True


def _made(name: str, path: str, parents: bool, log: Log) -> bool:
    """One FILE:path made, as ``rootmkdir`` makes it; whether all went well."""
    exists = os.path.exists(name)
    if _refused(name, path, parents, exists, log):
        return False
    try:
        out = _opened(name, exists)
    except (OSError, ValueError, ROOTError) as why:
        log.error(f"failed to open '{name}' for writing.")
        log.info(2, str(why))
        return False
    with out:
        if not exists:
            log.info(1, f"created file '{name}'")
        return not path or _directory(out, name, path, parents, log)


def _directory(top: WritableDirectory, name: str, path: str, parents: bool, log: Log) -> bool:
    """The directory at ``path`` in ``top``, its parents made only with ``parents``."""
    parts = path.split("/")
    here: WritableDirectory | None = top
    for depth in range(len(parts) - 1):
        assert here is not None
        here = _parent(here, name, path, "/".join(parts[: depth + 1]), parents, log)
        if here is None:
            return False
    assert here is not None
    kind = _kind(here, parts[-1])
    if kind is not None and kind not in DIRECTORIES:
        log.error(f"error adding '{path}': path is already taken by an object of type '{kind}'")
        return False
    if kind is not None:
        if not parents:
            log.error(f"error adding '{path}': a directory already exists at that path.")
        return parents
    here.mkdir(parts[-1])
    log.info(1, f"created directory '{name}:{path}'")
    return True


def _parent(
    here: WritableDirectory, name: str, path: str, spelled: str, parents: bool, log: Log
) -> WritableDirectory | None:
    """One directory on the way to ``path``: there, or made with ``-p`` - else ``None``."""
    part = spelled.rpartition("/")[2]
    kind = _kind(here, part)
    if kind is not None and kind not in DIRECTORIES:
        log.error(f"error adding '{path}': path '{spelled}' is already taken by an object "
                  f"of type '{kind}'")  # fmt: skip
        return None
    if kind is None and not parents:
        log.error(f"cannot create directory '{path}': parent directory '{spelled}' does "
                  "not exist. If you want to create the entire hierarchy, use the -p "
                  "flag.")  # fmt: skip
        return None
    made = here.mkdir(part)
    if kind is None:
        log.info(1, f"created directory '{name}:{spelled}'")
    return made


def _kind(directory: WritableDirectory, name: str) -> str | None:
    """The class of what ``name`` holds in a directory being written, or ``None``."""
    return directory.classname_of(name)
