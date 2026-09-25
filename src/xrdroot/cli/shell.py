"""``xrdroot`` and ``xrdroot shell``: ROOT's prompt, as a Python one.

    $ xrdroot tests/data/graphs.root
    xrdroot 0.1.0: ROOT files in Python, no ROOT required.
      _file0 = tests/data/graphs.root
    >>> .ls
    >>> _file0["tg"].points

``root -l f.root`` opens the files it is given as ``_file0``, ``_file1`` and so
on, and so does this; a ``.py`` among them is a macro, run after the files
are open, as ``root f.root macro.C`` runs one. The namespace is ``from
xrdroot import *`` with NumPy as ``np``, and the prompt takes ROOT's
dot-commands - ``.ls``, ``.cd``, ``.x`` and the rest in :mod:`.dot`.

IPython is the prompt when it is installed, with the commands and
``%root_ls``-style magics loaded as the ``xrdroot`` extension; otherwise it is
the standard library's :class:`code.InteractiveConsole`, which needs nothing.
"""

from __future__ import annotations

import argparse
import code
import importlib
import sys
from typing import Any

from ..session import gROOT, preloaded
from .dot import translate

__all__ = ["add_parser", "run", "Console", "banner", "namespace"]


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "shell",
        help="a Python prompt with ROOT's names and commands, as `root -l` is",
        description="Open the files given as _file0, _file1..., run any .py given as a "
        "macro, then prompt - with IPython if it is installed.",
    )
    parser.add_argument("files", nargs="*", metavar="FILE", help="ROOT files, and .py macros")
    parser.add_argument(
        "-q", "--quit", action="store_true", help="leave once the files and macros are done"
    )
    parser.add_argument(
        "--plain", action="store_true", help="the standard library's prompt, even with IPython"
    )


def namespace(files: list[str]) -> dict[str, Any]:
    """The names the prompt starts with: xrdroot's, ``np``, and each file as ``_fileN``."""
    names = preloaded()
    for index, target in enumerate(files):
        names[f"_file{index}"] = gROOT.open(target)
    return names


def banner(names: dict[str, Any]) -> str:
    """What the prompt says first: what it is, the commands, and the files it opened."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        release = version("xrdroot")
    except PackageNotFoundError:
        release = "(not installed)"
    lines = [
        f"xrdroot {release}: ROOT files in Python, no ROOT required.",
        "  Everything in xrdroot is here, with numpy as np; "
        ".ls .pwd .cd .x .help, and .q to leave.",
    ]
    lines.extend(
        f"  {name} = {value.name}" for name, value in names.items() if name.startswith("_file")
    )
    return "\n".join(lines)


class Console(code.InteractiveConsole):
    """The standard library's prompt, reading ROOT's dot-commands as the Python they mean."""

    def push(self, line: str, *args: Any, **kwargs: Any) -> bool:
        return super().push(translate(line), *args, **kwargs)


def _ipython(names: dict[str, Any], said: str) -> bool:
    """Prompt with IPython and the ``xrdroot`` extension; ``False`` if IPython is not there."""
    try:
        ipython = importlib.import_module("IPython")
    except ImportError:
        return False
    print(said)
    ipython.start_ipython(argv=["--no-banner", "--ext=xrdroot"], user_ns=names)
    return True


def interact(names: dict[str, Any], plain: bool = False) -> None:
    """Prompt until the user leaves, with IPython unless ``plain`` or it is not installed."""
    said = banner(names)
    if not plain and _ipython(names, said):
        return
    try:
        Console(names, filename="<xrdroot>").interact(banner=said, exitmsg="")
    except SystemExit:
        pass


def run(args: argparse.Namespace) -> int:
    files = [name for name in args.files if not name.endswith(".py")]
    macros = [name for name in args.files if name.endswith(".py")]
    try:
        names = namespace(files)
        for macro in macros:
            gROOT.macro(macro)
        if not args.quit:
            interact(names, plain=args.plain)
    finally:
        gROOT.close_all()
        sys.stdout.flush()
    return 0
