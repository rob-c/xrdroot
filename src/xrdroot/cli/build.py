"""``xrdroot build``: ROOT's build steps - rootcling, a linker, ACLiC - as translations.

    $ xrdroot build dictionary --name Event --linkdef EventLinkDef.h Event.h
    $ xrdroot build library --name Shapes Circle.cxx Square.cxx
    $ xrdroot build executable --name stressTest --library Event stressTest.cxx
    $ xrdroot build aclic macro.C+

A C++ build makes shared libraries and programs; xrdroot makes their
translations, and loads them the way ROOT loads the real ones
(:mod:`xrdroot.pycint.library`).

``dictionary`` is rootcling's (or genreflex's, with ``--selection``): the
headers translated into ``lib<NAME>.xrdlib``, with a rootmap so that a class
it provides is found by name, and a warning for each class the LinkDef or the
selection file asks for and the headers do not declare. ``library`` is a
library linked from sources, ``lib<NAME>.xrdlib``. ``executable`` writes a
program, ``NAME``, that runs as ``xrdroot run SOURCE... -- args`` with the
libraries loaded first, in the Python that built it, from any directory.
``aclic`` is ACLiC's ``macro_C.so``: ``macro_C.xrdlib`` beside the macro, or
in ``--out``. Each exits ``0`` when built, ``1`` with rootcling's reason on
standard error when a file is missing, and ``2`` with one line, ``xrdroot
build: file:line: why``, for C++ the translator refuses: a build that cannot be
done here.
"""

from __future__ import annotations

import argparse
import shlex
import stat
import sys
from pathlib import Path
from typing import Any

__all__ = ["add_parser", "run", "launcher"]


def _common(parser: argparse.ArgumentParser, inputs: str) -> None:
    parser.add_argument("--name", required=True, help="what the result is called")
    parser.add_argument("-I", dest="includes", action="append", default=[], metavar="DIR",
                        type=Path, help="a directory to find included headers in")  # fmt: skip
    parser.add_argument("--out", type=Path, default=None, metavar="DIR",
                        help="where to write it; the working directory if not given")  # fmt: skip
    parser.add_argument("--library", dest="libraries", action="append", default=[],
                        metavar="NAME", help="a library to load first; repeatable")  # fmt: skip
    parser.add_argument("files", nargs="+", type=Path, metavar=inputs)


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "build",
        help="build dictionaries, libraries, programs and ACLiC libraries, as translations",
        description="ROOT's build steps for xrdroot: rootcling's dictionary, a library, an "
        "executable, ACLiC's library - each the translation of its C++.",
    )
    steps = parser.add_subparsers(dest="step", metavar="STEP", required=True)
    dictionary = steps.add_parser("dictionary", help="rootcling's or genreflex's dictionary")
    _common(dictionary, "HEADER")
    dictionary.add_argument("--linkdef", type=Path, help="the LinkDef.h rootcling reads")
    dictionary.add_argument("--selection", type=Path, help="the selection XML genreflex reads")
    dictionary.add_argument("--no-rootmap", dest="rootmap", action="store_false",
                            help="no rootmap: found only once loaded by name")  # fmt: skip
    _common(steps.add_parser("library", help="a library from sources"), "SOURCE")
    _common(steps.add_parser("executable", help="a program from sources"), "SOURCE")
    aclic = steps.add_parser("aclic", help="ACLiC's library of a macro: macro.C+")
    aclic.add_argument("macro", metavar="MACRO.C[+[+]][g|O]")
    aclic.add_argument("--name", default=None, help="the library's name; macro_C if not given")
    aclic.add_argument("--out", type=Path, default=None, metavar="DIR",
                       help="where to write it; beside the macro if not given")  # fmt: skip


def run(args: argparse.Namespace) -> int:
    from ..pycint.library import BuildError

    try:
        STEPS[args.step](args)
    except BuildError as why:  # what the translator refuses goes on, said on one line, as 2
        sys.stderr.write(f"Error: {why}\n")
        return 1
    return 0


def _dictionary(args: argparse.Namespace) -> Path:
    from ..pycint.library import build_dictionary

    return build_dictionary(args.name, args.files, linkdef=args.linkdef,
                            selection=args.selection, includes=args.includes, out=args.out,
                            rootmap=args.rootmap, libraries=args.libraries)  # fmt: skip


def _library(args: argparse.Namespace) -> Path:
    from ..pycint.library import build_library

    return build_library(args.name, args.files, includes=args.includes, out=args.out,
                         libraries=args.libraries)  # fmt: skip


def _aclic(args: argparse.Namespace) -> Path | None:
    """ACLiC's library of the macro, built unless an up-to-date one is there."""
    from ..pycint.library import compiled
    from ..pycint.library.aclic import split

    path, options = split(args.macro)
    library = compiled(Path(path), force="++" in options or "f" in options, name=args.name,
                       out=args.out, explain=False)  # fmt: skip
    return library.path


def _executable(args: argparse.Namespace) -> Path:
    """The program ``NAME``: the sources translated now, so a refusal is the build's, and a
    launcher that runs them."""
    from ..pycint.dialect import ROOT as DIALECT
    from ..pycint.library import BuildError
    from ..pycxx.execute import translated

    for source in args.files:
        if not source.is_file():
            raise BuildError(f"{source}: no such file")
        translated(source.read_text(encoding="utf-8", errors="replace"),
                   str(source.resolve()), DIALECT, True)  # fmt: skip
    out: Path = args.out or Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    where = out / str(args.name)
    where.write_text(launcher(args.files, _linked(args.libraries, out)), encoding="utf-8")
    where.chmod(where.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return where


def _linked(names: list[str], out: Path) -> list[str]:
    """The libraries a program loads, each by its file if it is built already, else by name
    to be looked for when it runs; ROOT's own, and GoogleTest, need no loading."""
    return [_located(name.removeprefix("ROOT::"), out) for name in names if not _provided(name)]


def _provided(name: str) -> bool:
    """Whether ``name`` - ``Tree``, ``ROOT::Hist``, ``libRooFit``, ``gtest`` - is a library
    xrdroot has without loading it: ROOT's own, or GoogleTest."""
    from ..pycint.library.loading import ROOT_LIBRARIES
    from ..pycint.library.manifest import stem_of

    bare = stem_of(name.removeprefix("ROOT::"))
    return f"lib{bare.removeprefix('lib')}" in ROOT_LIBRARIES or bare.startswith(("gtest", "gmock"))


def _located(name: str, out: Path) -> str:
    """``name``'s library file, absolute, if one is built - beside the program or on the
    search path, as ``name`` or ``lib<name>`` - else ``name``, looked for when it runs."""
    from ..pycint.library import find

    for candidate in (name, f"lib{name}"):
        for place in (str(out / candidate), candidate):
            found = find(place)
            if found is not None:
                return str(found.resolve())
    sys.stderr.write(f"Warning in <xrdroot build>: library {name} is not built yet; the "
                     "program looks for it when it runs\n")  # fmt: skip
    return name


def launcher(sources: list[Path], libraries: list[str]) -> str:
    """The text of a program that runs ``sources`` as ``xrdroot run`` does, in this Python
    and this xrdroot, from any directory, with ``libraries`` loaded first."""
    import xrdroot

    package = Path(xrdroot.__file__).resolve().parent.parent
    words = ["exec", shlex.quote(sys.executable), "-m", "xrdroot", "run", '--argv0 "$0"']
    words += [f"--library {shlex.quote(library)}" for library in libraries]
    words += [shlex.quote(str(source.resolve())) for source in sources]
    return (
        "#!/bin/sh\n"
        "# Built by xrdroot build executable: the translated program, run as the compiled\n"
        "# one would be, with its arguments.\n"
        f"PYTHONPATH={shlex.quote(str(package))}${{PYTHONPATH:+:$PYTHONPATH}}\n"
        "export PYTHONPATH\n"
        f"{' '.join(words)} -- \"$@\"\n"
    )


#: What each step of ``xrdroot build`` does, by its name.
STEPS = {
    "dictionary": _dictionary,
    "library": _library,
    "executable": _executable,
    "aclic": _aclic,
}
