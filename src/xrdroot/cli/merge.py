"""``xrdroot merge``: ROOT's ``hadd``, flag for flag.

    $ xrdroot merge all.root run1.root run2.root run3.root
    $ xrdroot merge -f505 all.root run*.root        # overwrite, as zstd level 5
    $ xrdroot merge -a all.root run4.root           # merge into what is there
    $ xrdroot merge -L list.txt -Ltype SkipListed all.root run*.root

The flags are ``hadd``'s and mean what they mean there, with one
difference: with no ``-f`` setting the output is compressed as the first
input is (``hadd -ff``) and trees' baskets go across as they are
(``hadd -fk``), where ``hadd`` would write ROOT's 101 - ``-f101`` asks for
that. The flags this merge has no use for are taken and ignored rather than
refused, so a ``hadd`` command line runs as it is: ``-j``, since the inputs
are merged in one process one at a time, and ``-n``, since only one input is
ever open. See :func:`xrdroot.merge`.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence
from typing import Any

from ..errors import ROOTError
from ..merging import Merged, merge

__all__ = ["add_parser", "run"]

#: The compression settings ``hadd`` takes after ``-f``: none, a bare level
#: meaning zlib, or ``algorithm * 100 + level``.
SETTINGS = [0, *range(1, 10), *(a * 100 + level for a in range(1, 6) for level in range(10))]
#: Every spelling of the ``-f`` family beyond ``-f`` itself.
SPELLINGS = [
    "-ff",
    "-fk",
    "-ffk",
    "-fkf",
    *(f"-f{setting}" for setting in SETTINGS),
    *(f"-fk{setting}" for setting in SETTINGS),
]
#: The verbosity ``hadd`` has unless told: errors, warnings and a little news.
VERBOSE = 2


class _FFlag(argparse.Action):
    """``-f``, ``-ff``, ``-fk``, ``-f505`` and ``-fk505``: all of them force, and some say more."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        letters = (option_string or "-f")[2:]
        digits = letters.lstrip("fk")
        flags = letters[: len(letters) - len(digits)]
        namespace.force = True
        namespace.keep_compression |= "k" in flags
        namespace.first_compression |= "f" in flags
        if digits:
            namespace.compression = int(digits)


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """The ``merge`` subcommand, with ``hadd``'s flags."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "merge",
        help="merge ROOT files into one, as hadd does",
        description="Merge SOURCES into TARGET as ROOT's hadd does: histograms added up, "
        "trees concatenated with their baskets copied as they are.",
    )
    parser.add_argument("target", help="the file to write")
    parser.add_argument("sources", nargs="+", help="the files to merge, in order")
    parser.add_argument(
        "-f",
        action=_FFlag,
        nargs=0,
        help="write over TARGET; -f[0-509] also sets its compression (as -f505), -fk keeps "
        "each input's baskets compressed as they are, -ff compresses as the first input",
    )
    parser.add_argument(*SPELLINGS, action=_FFlag, nargs=0, help=argparse.SUPPRESS)
    parser.add_argument("-a", dest="append", action="store_true", help="merge into TARGET")
    parser.add_argument(
        "-k", dest="skip_errors", action="store_true", help="pass over inputs that will not open"
    )
    parser.add_argument(
        "-O", dest="reoptimize", action="store_true", help="write trees' entries afresh"
    )
    parser.add_argument("-T", dest="no_trees", action="store_true", help="leave the trees out")
    parser.add_argument("-L", dest="listed", metavar="FILE", help="a file of object names")
    parser.add_argument(
        "-Ltype",
        dest="list_type",
        choices=("SkipListed", "OnlyListed"),
        help="whether -L's objects are skipped or are the only ones merged",
    )
    parser.add_argument(
        "-v", dest="verbosity", nargs="?", type=int, const=99, default=VERBOSE, help="0 to 3"
    )
    parser.add_argument("-j", dest="jobs", nargs="?", type=int, const=0, help="taken, ignored")
    parser.add_argument("-n", dest="max_open", type=int, help="taken, ignored")
    parser.set_defaults(
        force=False, compression=None, keep_compression=False, first_compression=False
    )
    return parser


def _listed(path: str | None, kind: str | None) -> tuple[list[str] | None, list[str] | None]:
    """The names ``-L`` lists, as the skip and only lists :func:`xrdroot.merge` takes."""
    if (path is None) != (kind is None):
        raise ValueError("-L and -Ltype go together: the list, and what to do with it")
    if path is None:
        return None, None
    with open(path) as lines:
        names = [line.split()[0] for line in lines if line.split() and line[0] != "#"]
    return (names, None) if kind == "SkipListed" else (None, names)


def _keep(args: argparse.Namespace) -> bool | None:
    """``hadd``'s ``-fk`` and ``-ff`` as :func:`xrdroot.merge`'s ``keep_compression``."""
    if args.keep_compression:
        return True
    if args.first_compression:
        return False
    return None


def run(args: argparse.Namespace) -> int:
    """Merge as asked; ``0`` when it is done, ``1`` with the reason on stderr when it is not."""
    try:
        skip, only = _listed(args.listed, args.list_type)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            merged = merge(
                args.target,
                args.sources,
                compression=args.compression,
                force=args.force,
                fast=not args.reoptimize,
                skip_keys=skip,
                only_keys=only,
                append=args.append,
                keep_compression=_keep(args),
                skip_errors=args.skip_errors,
                trees=not args.no_trees,
            )
    except (ROOTError, ValueError, OSError, KeyError, TypeError) as why:
        print(f"xrdroot merge: {why}", file=sys.stderr)
        return 1
    _report(args, merged, [str(warning.message) for warning in caught])
    return 0


def _report(args: argparse.Namespace, merged: Merged, warned: Sequence[str]) -> None:
    """What ``hadd`` would say at this verbosity: warnings, then what was made."""
    if args.verbosity >= 1:
        for message in warned:
            print(f"xrdroot merge: warning: {message}", file=sys.stderr)
    if args.verbosity < VERBOSE:
        return
    verbatim, repacked, in_place = merged.baskets
    print(f"xrdroot merge: target {args.target}, from {len(args.sources)} sources")
    print(
        f"xrdroot merge: {len(merged.objects)} objects; tree baskets {verbatim} copied as "
        f"they were, {repacked} packed again, {in_place} left in place; "
        f"{merged.entries} entries written afresh"
    )
    if args.verbosity > VERBOSE:
        for path, how in merged.objects.items():
            print(f"xrdroot merge:   {path}: {how}")
