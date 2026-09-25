"""``xrdroot``: ROOT's prompt and command-line kit, over any URL this library opens.

    $ xrdroot f.root                      # root -l f.root: a shell, with _file0
    $ xrdroot ls -t root://host//f.root   # rootls -t
    $ xrdroot dump f.root:dir/h           # every bin, point and entry, as text
    $ xrdroot diff a.root b.root          # exit status 0 the same, 1 different
    $ xrdroot print f.root:h -o h.png     # rootprint
    $ xrdroot scan f.root:events "pt:eta" "pt > 30"
    $ xrdroot draw f.root:events pt -o pt.png
    $ xrdroot info f.root                 # the header, the compression, the classes

The subcommand registry
-----------------------

Every subcommand is a module of this package named after it, and
:data:`COMMANDS` lists them; adding one is writing the module and adding its
name to that list - one line, nothing else. The contract each module keeps:

``xrdroot.cli.<name>`` exposes

* ``add_parser(subparsers)``, which calls ``subparsers.add_parser("<name>",
  ...)`` - with the module's own name, which is how :func:`main` finds the
  module again - and adds that subcommand's arguments to the parser it gets
  back;
* ``run(args)``, which is handed the parsed :class:`argparse.Namespace` and
  returns the exit status as an ``int`` (``None`` is taken as ``0``). It
  writes its results to ``sys.stdout`` and refusals to ``sys.stderr``.

A :class:`~xrdroot.ROOTError`, ``KeyError``, ``OSError``, ``ValueError`` or
``TypeError`` escaping ``run`` is printed as one line, ``xrdroot <name>:
<why>``, and exits with status 2 - the status ``diff`` uses for trouble -
so a subcommand refuses by raising a full sentence rather than by printing
one. :mod:`xrdroot.cli.target` splits ``FILE[:path]`` the way every
subcommand should, and :func:`xrdroot.cli.target.opened` opens one.

``xrdroot`` alone starts the shell, and so does ``xrdroot f.root ...`` - as
``root -l f.root`` does - when the first argument is no subcommand.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from typing import Any

from ..errors import ROOTError

__all__ = ["COMMANDS", "build_parser", "main"]

#: The subcommands, in the order ``xrdroot --help`` lists them: each is the
#: module ``xrdroot.cli.<name>``, exposing ``add_parser(subparsers)`` and
#: ``run(args)``. Adding a subcommand is adding its name here.
COMMANDS = [
    "shell",
    "ls",
    "dump",
    "print",
    "diff",
    "scan",
    "draw",
    "info",
]

#: What a subcommand raises to refuse, printed as a line rather than a traceback.
REFUSALS = (ROOTError, KeyError, OSError, ValueError, TypeError)

#: The exit status of a subcommand that refused - ``diff``'s status for trouble.
TROUBLE = 2


def _module(name: str) -> Any:
    return importlib.import_module(f"{__name__}.{name}")


def build_parser() -> argparse.ArgumentParser:
    """The ``xrdroot`` parser, with a subparser from every module in :data:`COMMANDS`."""
    parser = argparse.ArgumentParser(
        prog="xrdroot",
        description="ROOT's prompt and command-line tools, over any URL xrdroot can open.",
        epilog="With no subcommand - or with files first, as `root -l f.root` - a shell starts.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name in COMMANDS:
        _module(name).add_parser(subparsers)
    return parser


def _as_shell(argv: list[str]) -> list[str]:
    """``xrdroot f.root`` and plain ``xrdroot`` mean the shell, as ``root -l f.root`` does."""
    if not argv:
        return ["shell"]
    first = argv[0]
    if first in COMMANDS or first.startswith("-"):
        return argv
    return ["shell", *argv]


def _message(why: BaseException) -> str:
    """A refusal as one line: a ``KeyError``'s sentence without the quotes ``str`` adds."""
    if isinstance(why, KeyError) and why.args:
        return str(why.args[0])
    return str(why)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``xrdroot`` with ``argv`` - ``sys.argv[1:]`` unless given - and return its status."""
    arguments = _as_shell(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(arguments)
    try:
        status = _module(args.command).run(args)
    except REFUSALS as why:
        # Not print(): importing the ``print`` subcommand makes that name the module here.
        sys.stderr.write(f"xrdroot {args.command}: {_message(why)}\n")
        return TROUBLE
    return 0 if status is None else int(status)
