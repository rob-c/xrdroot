"""``xrdroot rm``: ROOT's ``rootrm`` - objects, directories or whole files taken away.

    $ xrdroot rm example.root:hist          # the key hist, every cycle of it
    $ xrdroot rm -r example.root:dir        # a directory, and what it held
    $ xrdroot rm -r example.root            # the file itself
    $ xrdroot rm -i 'example.root:h*'       # asking first, for each

Each FILE is a file and a pattern in it, read as :mod:`.sources` reads one;
everything it matches is removed - each cycle a key of its own - and the
file updated in place, the records' bytes going to its free list. A
directory, or the file itself, needs ``-r``, and without it is passed over
with ``rootrm``'s word on why. ``-v`` says what went, ``-n`` says it and
removes nothing, ``-f`` passes silently over what is not there. The first
pattern that matches nothing stops the run, with status 1.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import ExitStack
from typing import Any

from ..wupdate import update
from .sources import Matches, matching
from .tooling import Flag, Log, OptionError, Options, raw_parser

__all__ = ["add_parser", "run", "USAGE"]

#: What ``rootrm`` says when its command line is wrong, on standard error.
USAGE = (
    "usage: rootrm [-h] [-i|--interactive] [-r|--recursive] [-v|--verbose] "
    "FILE[:path/to/obj] [...]\n"
)

#: The rest of ``rootrm -h``, after the usage line.
HELP = """
Remove objects from ROOT files

positional arguments:
  FILE:path             File(s) and path of objects to remove

options:
  -f, --force           ignore nonexistent files and arguments, never prompt
  -h, --help            show this help message and exit
  -i, --interactive     prompt before deleting each object
  -n, --dry-run         show what would be removed, but don't actually remove anything. Implies `-v`.
  -r, --recursive       recurse inside directories
  -v, --verbose         be verbose (can be repeated for increased verbosity)

examples:
- rootrm example.root:hist
  Remove the object 'hist' from the ROOT file 'example.root'

- rootrm example.root:dir/hist
  Remove the object 'hist' from the directory 'dir' inside the ROOT file 'example.root'

- rootrm example.root
  Remove the ROOT file 'example.root'

- rootrm -i example.root:hist
  Display a confirmation request before deleting: 'remove 'hist' from 'example.root' ? (y/n) :'
"""  # noqa: E501

#: ``rootrm``'s flags.
FLAGS = (
    Flag(("-f", "--force")),
    Flag(("-h", "--help")),
    Flag(("-i", "--interactive")),
    Flag(("-n", "--dry-run")),
    Flag(("-r", "--recursive")),
    Flag(("-v", "--verbose"), repeatable=True),
)


def add_parser(subparsers: Any) -> None:
    raw_parser(subparsers, "rm", "remove objects or files, as rootrm does")


def _usage(options: Options, log: Log) -> int | None:
    """``rootrm``'s answer to a command line it will not run - or ``None`` to go on."""
    if options.has("h"):
        sys.stderr.write(USAGE + HELP)
        return 0
    if options.has("f") and options.has("i"):
        log.error("-i and -f flags are mutually exclusive.")
    elif options.args:
        return None
    sys.stderr.write(USAGE)
    return 1


def run(args: argparse.Namespace) -> int:
    log = Log("rootrm")
    try:
        options = Options(FLAGS).parse(args.argv)
    except OptionError as why:
        sys.stderr.write(f"{why}\n{USAGE}")
        return 1
    log.verbosity = max(options.count("v"), int(options.has("n"))) + 1
    refused = _usage(options, log)
    if refused is not None:
        return refused
    return _Remover(options, log).run(options.args)


class _Remover:
    """One ``rootrm`` run: what it was told, and the files it has already removed."""

    def __init__(self, options: Options, log: Log) -> None:
        self.force = options.has("f")
        self.interactive = options.has("i")
        self.recursive = options.has("r")
        self.dry = options.has("n")
        self.log = log
        self.removed: set[str] = set()

    def run(self, sources: list[str]) -> int:
        for source in sources:
            failed = self._source(source)
            if failed is not None:
                return failed
        return 0

    def _source(self, source: str) -> int | None:
        """Remove what one source names; the status to stop with, if it is the last."""
        try:
            matched = matching(source, self.recursive)
        except ValueError as why:
            self.log.error(str(why))
            return 1
        if matched.filename in self.removed:
            matched.close()
            return None
        if not os.access(matched.filename, os.W_OK) and "://" not in matched.filename:
            matched.close()
            matched.errors = [f"File '{matched.filename}' does not exist or is not writable."]
        if matched.errors:
            matched.close()
            if self.force:
                return None
            for message in matched.errors:
                self.log.error(message)
            return 1
        self._remove(matched)
        return None

    def _remove(self, matched: Matches) -> None:
        """Every leaf, then every directory - or the file, after which nothing else.

        The file is opened to be updated only once something in it is to go.
        """
        matched.close()
        with ExitStack() as stack:
            out = None
            for index in matched.leaves + matched.dirs:
                if not self._allowed(matched, index):
                    continue
                if index == 0:
                    self._file(matched.filename)
                    return
                if out is None and not self.dry:
                    out = stack.enter_context(update(matched.filename))
                self._key(matched, index, out)

    def _allowed(self, matched: Matches, index: int) -> bool:
        """Whether a node may go: ``-r`` for a directory, and yes to ``-i``'s question."""
        node = matched.nodes[index]
        if node.is_directory and not self.recursive:
            what = "a ROOT file" if index == 0 else "a directory"
            self.log.error(f"cannot remove '{node.name}': is {what}. Use -r to remove it.")
            return False
        if not self.interactive:
            return True
        if index == 0:
            question = f"remove '{matched.filename}' ? (y/n) "
        else:
            question = f"remove '{matched.path(index)}' from '{matched.filename}' ? (y/n) "
        sys.stdout.write(question)
        sys.stdout.flush()
        answer = sys.stdin.readline().split()
        return bool(answer) and answer[0] in ("y", "Y")

    def _file(self, filename: str) -> None:
        """The file itself unlinked - unless this is ``-n``'s dry run."""
        if self.dry:
            return
        try:
            os.unlink(filename)
        except OSError as why:
            if not self.force:
                self.log.error(f"failed to remove '{filename}': {why.strerror}")
            return
        self.removed.add(filename)

    def _key(self, matched: Matches, index: int, out: Any) -> None:
        if out is not None:
            self._delete(matched, index, out)
        self.log.info(2, f"removed '{matched.filename}:{matched.path(index)}'")

    def _delete(self, matched: Matches, index: int, out: Any) -> None:
        node = matched.nodes[index]
        assert node.key is not None
        parent = matched.path(node.parent)
        directory = out.mkdir(parent) if parent else out
        if self.log.verbosity > 2:
            sys.stdout.write(f"Deleting key: {node.name} at address {node.key.seek_key}, "
                             f"nbytes = {node.key.nbytes}\n")  # fmt: skip
        directory.delete(node.name, node.cycle)
