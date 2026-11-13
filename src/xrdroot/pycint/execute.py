"""Running a macro as ROOT does: ``root file.C``, ``.x file.C(args)``, ``gROOT->ProcessLine``.

A macro is translated in the ROOT dialect (or its translation found in the
cache), the Python run, and then the function named like the file called
with the arguments ``.x`` gave it - ``hsimple.C`` runs ``hsimple()``; an
unnamed ``{ ... }`` macro's block is that function. ``file.C+`` (ACLiC,
compile first) runs the same way: there is nothing to compile. What the
macro raises as it runs is raised again as a :class:`MacroError` at the C++
line it happened on. The running itself is :mod:`xrdroot.pycxx.execute`'s;
this is ROOT's face on it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..pycxx import execute as _cxx
from ..pycxx.errors import MacroError, Refusal
from ..pycxx.translation import translation
from .dialect import ROOT as DIALECT
from .runtime import ROOT

__all__ = [
    "run",
    "run_and_quit",
    "run_program",
    "run_source",
    "load",
    "process_line",
    "split_call",
    "arguments",
    "EXTENSIONS",
    "MacroError",
    "Refusal",
]

#: The file extensions of a C++ macro.
EXTENSIONS = (".C", ".c", ".cxx", ".cpp", ".cc", ".h", ".hxx")

#: ``file.C+(1, "a")``: a macro, ACLiC's ``+`` or ``++``, and its arguments.
CALL = re.compile(r"^\s*(?P<path>[^(]+?)(?P<plus>\+{0,2}[gO]?)\s*(?:\((?P<args>.*)\))?\s*$", re.S)


def split_call(text: str) -> tuple[str, str]:
    """``hsimple.C+(1)`` as the path ``hsimple.C`` and the argument text ``1``."""
    found = CALL.match(text)
    if found is None:
        raise ValueError(f"{text!r} does not name a macro, as file.C or file.C(arguments)")
    return found["path"].strip(), (found["args"] or "").strip()


def arguments(text: str) -> tuple[Any, ...]:
    """The values of ``.x file.C(1000, "gaus", 2.5)``'s argument list, read as C++."""
    if not text.strip():
        return ()
    source = f"void __arguments() {{ __take({text}); }}"
    python = translation(source, "<arguments>", dialect=DIALECT).python
    taken: list[Any] = []
    namespace: dict[str, Any] = {"__name__": "__cint_arguments__"}
    exec(compile(python, "<arguments>", "exec"), namespace)
    namespace["ROOT"] = _Taker(taken)
    namespace["__arguments"]()
    return tuple(taken[0]) if taken else ()


class _Taker:
    """Stands in for ROOT while arguments are read: ``__take`` is what collects them."""

    def __init__(self, into: list[Any]) -> None:
        self.into = into

    def __getattr__(self, name: str) -> Any:
        if name == "__take":
            return lambda *values: self.into.append(values)
        return getattr(ROOT, name)


def _path(path: str | Path) -> Path:
    """A macro's file, ACLiC's ``+`` taken off: there is nothing to compile."""
    return Path(str(path).rstrip("+"))


def run_source(
    source: str,
    file: str = "<macro>",
    args: tuple[Any, ...] = (),
    *,
    root: Any = None,
    use_cache: bool = False,
    call: bool = True,
) -> Any:
    """Translate and run a macro's text; call its function (or unnamed block) with ``args``.

    With ``root``, ROOT's names are looked up there rather than in
    :mod:`xrdroot.pyroot`; with ``call`` false, the macro's names are what is
    handed back.
    """
    return _cxx.run_source(
        source, file, args, dialect=DIALECT, namespace=root, use_cache=use_cache, call=call
    )


def run(
    path: str | Path, args: tuple[Any, ...] = (), *, root: Any = None, use_cache: bool = True
) -> Any:
    """``.x path(args)``: translate the macro at ``path`` and run it, as ROOT would."""
    where = _path(path)
    source = where.read_text(encoding="utf-8", errors="replace")
    return run_source(source, str(where), tuple(args), root=root, use_cache=use_cache)


def run_and_quit(path: str | Path, args: tuple[Any, ...] = (), *, use_cache: bool = True) -> int:
    """``root -b -q path(args)``: run the macro, print what it returned, give ROOT's exit status.

    Cling prints the returned value after the macro's own output, and ROOT
    quitting exits with it - see :mod:`xrdroot.pycint.returned`.
    """
    return run_program([path], args, use_cache=use_cache)


def run_program(
    paths: Sequence[str | Path],
    args: tuple[Any, ...] = (),
    argv: Sequence[str] = (),
    *,
    use_cache: bool = True,
) -> int:
    """``xrdroot run a.cxx b.cxx -- argv``: the files as one program, run from its entry point.

    Each file is translated and run, in order, in one namespace; then the
    program is started as the first file says. A function named like the
    first file is called with ``args``, as ``root -b -q`` calls it, its value
    printed and exited with; else ``main(argc, argv)``, its value the exit
    status, as a compiled program's; else the GoogleTest tests the files
    define, as ``gtest_main`` runs them; else - nothing to run - a refusal.
    ``argv`` is the program's whole command line, its name first.
    """
    wheres = [_path(path) for path in paths]
    return _cxx.run_program(wheres, args, argv, dialect=DIALECT, use_cache=use_cache)


def load(path: str | Path, *, root: Any = None) -> dict[str, Any]:
    """``.L path``: translate the macro and define what it declares, calling nothing."""
    where = _path(path)
    source = where.read_text(encoding="utf-8", errors="replace")
    namespace: dict[str, Any] = run_source(source, str(where), root=root, call=False)
    return namespace


def process_line(line: str, *, root: Any = None) -> Any:
    """``gROOT->ProcessLine(line)``: a ``.x``/``.L`` command, or C++ statements run at once."""
    text = line.strip()
    if text.startswith((".x", ".X")):
        path, given = split_call(text[2:])
        return run(path, arguments(given), root=root)
    if text.startswith(".L"):
        return load(split_call(text[2:])[0], root=root)
    if text.startswith("."):
        raise ValueError(f"{text.split()[0]} is not a command gROOT->ProcessLine runs here")
    return _cxx.run_line(text, file="<ProcessLine>", dialect=DIALECT, namespace=root)
