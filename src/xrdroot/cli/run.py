"""``xrdroot run``: ``root -b -q file.C`` - a ROOT macro run, C++ or Python - or a C++ program.

    $ xrdroot run hsimple.C
    $ xrdroot run 'fit.C(1000, "gaus")'
    $ xrdroot run hsimple.C --python      # the translation, printed, nothing run
    $ xrdroot run script.py               # a PyROOT script, with ROOT as xrdroot.pyroot
    $ xrdroot run stressVector.cxx -- 500 # a program's main(argc, argv)
    $ xrdroot run testTH1.cxx -- --gtest_filter='TH1.*'   # a GoogleTest program
    $ xrdroot run a.cxx b.cxx             # several files, one program

A C++ macro is translated into Python by :mod:`xrdroot.pycint` and run, and the
function named like its file called with the arguments given in brackets, as
``root`` does; ``file.C+`` runs the same way. What the function returns is
printed as cling prints it, ``(TCanvas *) 0x...``, and is the exit status, as
``root -q`` makes it: ``hsimple.C``, returning its ``TFile *``, exits 255.

A file without that function is started as the program it is: its
``main(argc, argv)`` is called - ``argv`` the program's name, then whatever
follows ``--`` - and its value is the exit status; else the GoogleTest
tests it defines are run, as ``gtest_main`` runs them, exiting 0 if and only
if none failed; else an unnamed macro ``{ ... }`` runs its block. A file with
none of these is refused, rather than run as nothing: ROOT's ``.x`` would
call the missing function and fail too. Several files are one program: each
is translated and run in order, in one namespace, and the first says how
the program starts.

A Python script is run as a PyROOT script, with ``import ROOT`` giving
:mod:`xrdroot.pyroot` and ``sys.argv`` the script and what follows ``--``.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .shims import install_cppyy, install_root, python_warnings

__all__ = ["add_parser", "run"]


class _ProgramParser(argparse.ArgumentParser):
    """``run``'s parser: what follows ``--`` is the program's own arguments, kept apart.

    argparse would otherwise drop the ``--`` and add what follows it to the
    files, so a program's ``--gtest_filter=...`` could not be told from a
    file to run.
    """

    def parse_known_args(  # type: ignore[override]
        self, args: Sequence[str] | None = None, namespace: Any = None
    ) -> tuple[argparse.Namespace, list[str]]:
        given = list(sys.argv[1:] if args is None else args)
        cut = given.index("--") if "--" in given else len(given)
        parsed, extra = super().parse_known_args(given[:cut], namespace)
        parsed.argv = given[cut + 1 :]
        return parsed, extra


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "run",
        help="run a ROOT macro (file.C[(args)]) or C++ program, translated, or a PyROOT script",
        description=(
            "Run a ROOT macro as root -b -q would (file.C, file.C(args) or file.py), or a C++ "
            "program: its main(), or its GoogleTest tests, with the arguments after --."
        ),
    )
    parser.__class__ = _ProgramParser
    parser.add_argument(
        "macro",
        nargs="+",
        help='the macro, with any arguments: file.C or "file.C(1, 2)"; more files join it',
    )
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
    parser.add_argument(
        "--library",
        dest="libraries",
        action="append",
        default=[],
        metavar="NAME",
        help="a library to load first, as gSystem->Load does (libFoo, or its .xrdlib file)",
    )
    parser.add_argument("--argv0", help="the program's own name, argv[0]: its source's stem")


def _script(path: Path, argv: list[str]) -> int:
    """A PyROOT script, run with ``import ROOT`` meaning :mod:`xrdroot.pyroot`.

    Its standard output is :func:`xrdroot.stdio.split`, so that what it
    prints and what ROOT prints arrive in the order they would from PyROOT.
    ``sys.argv`` is the script and the arguments after ``--``, as ``python
    script.py`` makes it, so a script parsing its own options does not find
    ``run`` among them.
    """
    from ..stdio import split

    install_root()
    install_cppyy()
    before, warned = sys.argv, python_warnings(True)
    sys.argv = [str(path), *argv]
    try:
        with split():
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = before
        python_warnings(warned)
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
    from ..pycint.execute import run as run_macro

    if Path(name).is_file():
        run_macro(Path(name), use_cache=use_cache)


def ordered_output() -> None:
    """Write every line as it is made, as a C++ program's one ``std::cout`` does.

    A tutorial prints through Python's ``print`` and through whatever the kit itself logs,
    and a pipe - which is how a harness always runs one - makes ``print`` buffer its lines
    until the process ends, so the two streams leave in an order the program never wrote.
    ROOT has one stream and one order; line buffering gives the same here.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None and not sys.stdout.isatty():
        reconfigure(line_buffering=True)


def _preloaded(libraries: list[str]) -> None:
    """The libraries a program built with ``--library`` loads before it starts."""
    from ..pycint.library import load

    for library in libraries:
        load(library)


def run(args: argparse.Namespace) -> int:
    from ..pycint import translate_file
    from ..pycint.execute import called, split_call
    from ..pycint.library.aclic import split as split_aclic
    from ..pycint.runtime.application import started

    first, *more = args.macro
    argv = list(getattr(args, "argv", []))
    given = split_call(first)[1]
    path = Path(called(first))  # with ACLiC's ``+``, if it was given one
    if path.suffix == ".py":
        ordered_output()
        return _script(path, argv)
    paths = [path, *map(Path, more)]
    if args.python:
        for each in paths:
            sys.stdout.write(translate_file(Path(split_aclic(str(each))[0])))
        return 0
    ordered_output()
    with started(["root.exe", "-b", "-q", *args.macro]):
        return _program(args, paths, given, argv)


def _program(args: argparse.Namespace, paths: list[Path], given: str, argv: list[str]) -> int:
    """The macro or program run, as ``root -b -q`` runs it: logon, it, logoff."""
    from ..pycint.execute import arguments, run_program
    from ..pycint.library.aclic import split as split_aclic

    _preloaded(list(getattr(args, "libraries", [])))
    logons = () if args.no_logon else (LOGON, LOGOFF)
    for name in logons[:1]:
        _session_macro(name, not args.no_cache)
    program = [getattr(args, "argv0", None) or Path(split_aclic(str(paths[0]))[0]).stem, *argv]
    status = run_program(paths, arguments(given), program, use_cache=not args.no_cache)
    for name in logons[1:]:  # after the returned value's line, before root exits with it
        _session_macro(name, not args.no_cache)
    return status
