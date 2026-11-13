"""``xrdroot root`` (and ``xrdroot cint``): ROOT's C++ prompt, as ``root -l`` gives it.

    $ xrdroot root -l f.root              # f.root attached as _file0, then root [0]
    $ xrdroot root -l -b -q 'fit.C(1000)' # the macro run, then out with its value
    $ xrdroot root -l -e 'gRandom->Gaus()'
    $ echo 'TH1F h("h","h",10,0,1); h.Fill(.5); h.GetEntries()' | xrdroot root -l -b
    (double) 1.0000000

The options are ``root``'s. The files given are taken in order: a ``.root``
file is opened as ``_file0``, ``_file1``... - ``Attaching file f.root as
_file0...`` - and a macro, ``x.C``, ``x.C(args)`` or ``x.C+``, is run as
``.x`` runs it, after ``Processing x.C...``. Lines given with ``-e`` run
first. ``-q`` leaves once they are done, with the exit status of the last
macro's value as ``root -q`` gives it. ``rootlogon.C`` in the working
directory runs as the prompt starts and ``rootlogoff.C`` as it ends, unless
``-n``. There are never graphics windows here, so ``-b`` is always so.

The Python prompt with ROOT's dot commands is ``xrdroot shell``; this one is
C++, translated into Python by :mod:`xrdroot.cint` as it is typed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..pycint.interactive import Interpreter, interact
from ..pycint.interactive.commands import dispatch
from ..pycint.interactive.terminal import say
from .run import LOGOFF, LOGON

__all__ = ["add_parser", "run", "banner"]


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "root",
        help="ROOT's C++ prompt, as `root -l` is: C++ typed, translated and run",
        description="A C++ prompt in the style of ROOT's cling: open the ROOT files given "
        "as _file0, _file1..., run the macros given, then prompt for C++.",
    )
    parser.add_argument(
        "files", nargs="*", metavar="FILE", help="ROOT files to attach, and macros: x.C(args)"
    )
    parser.add_argument("-l", dest="no_banner", action="store_true", help="no banner")
    parser.add_argument("-b", dest="batch", action="store_true",
                        help="batch mode, no graphics windows: always so here")  # fmt: skip
    parser.add_argument("-q", dest="quit", action="store_true",
                        help="leave once the files, macros and -e lines are done")  # fmt: skip
    parser.add_argument("-x", dest="exit_on_error", action="store_true",
                        help="leave at the first error")  # fmt: skip
    parser.add_argument("-n", dest="no_logon", action="store_true",
                        help="run no rootlogon.C or rootlogoff.C")  # fmt: skip
    parser.add_argument("-e", dest="lines", action="append", default=[], metavar="LINE",
                        help="a line to run, as if typed; may be given again")  # fmt: skip


def banner() -> str:
    """What the prompt says first without ``-l``: what it is - xrdroot, and not ROOT."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        release = version("xrdroot")
    except PackageNotFoundError:
        release = "(not installed)"
    return (
        f"xrdroot {release}: a C++ prompt in the style of ROOT's, translated into Python.\n"
        "  This is not ROOT; ROOT's names are xrdroot.pyroot's. .help lists the commands, "
        ".q leaves."
    )


def _logon(prompt: Interpreter, name: str) -> None:
    """``rootlogon.C`` or ``rootlogoff.C``, run as ``.x`` runs it - but, as in ROOT, not
    typed at the prompt, so no prompt number is used up and its value is no exit status."""
    if Path(name).is_file():
        status = prompt.last_status
        say(dispatch(prompt, f".x {name}") or [])
        prompt.last_status = status


def _start(prompt: Interpreter, args: argparse.Namespace) -> None:
    """The ``-e`` lines, then the files in their order, until one ends the session."""
    for line in args.lines:
        if not prompt.finished:
            say(prompt.feed(line))
    for given in args.files:
        if not prompt.finished:
            say(prompt.given(given))


def run(args: argparse.Namespace) -> int:
    if not args.no_banner:
        sys.stdout.write(banner() + "\n")
    prompt = Interpreter(exit_on_error=args.exit_on_error)
    if not args.no_logon:
        _logon(prompt, LOGON)
    _start(prompt, args)
    status = prompt.status if args.quit or prompt.finished else interact(prompt)
    if not args.no_logon:
        _logon(prompt, LOGOFF)
    prompt.close()
    sys.stdout.flush()
    return status
