"""``xrdroot run``: ``root -b -q file.C`` - a ROOT macro run, C++ or Python.

    $ xrdroot run hsimple.C
    $ xrdroot run 'fit.C(1000, "gaus")'
    $ xrdroot run hsimple.C --python      # the translation, printed, nothing run
    $ xrdroot run script.py               # a PyROOT script, with ROOT as xrdroot.pyroot

A C++ macro is translated into Python by :mod:`xrdroot.cint` and run, and the
function named like its file called with the arguments given in brackets, as
``root`` does; ``file.C+`` runs the same way. A Python script is run as a
PyROOT script, with ``import ROOT`` giving :mod:`xrdroot.pyroot`.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Any

__all__ = ["add_parser", "run"]


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "run",
        help="run a ROOT macro: file.C[(args)], translated from C++, or a PyROOT script",
        description="Run a ROOT macro as root -b -q would: file.C, file.C(args) or file.py.",
    )
    parser.add_argument("macro", help='the macro, with any arguments: file.C or "file.C(1, 2)"')
    parser.add_argument(
        "--python",
        action="store_true",
        help="print the Python the macro translates into, and run nothing",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="translate afresh rather than use a translation kept from before",
    )
    parser.add_argument(
        "-n",
        dest="no_logon",
        action="store_true",
        help="as root -n: run no rootlogon.C or rootlogoff.C from the working directory",
    )


def _script(path: Path) -> int:
    """A PyROOT script, run with ``import ROOT`` meaning :mod:`xrdroot.pyroot`."""
    import importlib

    sys.modules["ROOT"] = importlib.import_module("xrdroot.pyroot")
    runpy.run_path(str(path), run_name="__main__")
    return 0


#: The macros ``root`` runs from the working directory as it starts and as it ends -
#: ``Rint.Logon`` and ``Rint.Logoff`` in ROOT's ``system.rootrc``.
LOGON, LOGOFF = "rootlogon.C", "rootlogoff.C"


def _session_macro(name: str, use_cache: bool) -> None:
    """Run ``rootlogon.C`` or ``rootlogoff.C`` if the working directory has one, as root does.

    ``root -b -q -l file.C`` still runs both (only ``-n`` stops them), so a
    macro run beside them prints what they print before and after its own.
    A Python script does not: PyROOT looks only for ``.rootlogon.C``.
    """
    from ..cint.execute import run as run_macro

    if Path(name).is_file():
        run_macro(Path(name), use_cache=use_cache)


def run(args: argparse.Namespace) -> int:
    from ..cint import translate_file
    from ..cint.execute import arguments, split_call
    from ..cint.execute import run as run_macro

    path_text, given = split_call(args.macro)
    path = Path(path_text)
    if path.suffix == ".py":
        return _script(path)
    if args.python:
        sys.stdout.write(translate_file(path))
        return 0
    logons = () if args.no_logon else (LOGON, LOGOFF)
    for name in logons[:1]:
        _session_macro(name, not args.no_cache)
    run_macro(path, arguments(given), use_cache=not args.no_cache)
    for name in logons[1:]:
        _session_macro(name, not args.no_cache)
    return 0
