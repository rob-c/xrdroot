"""Running a macro as ROOT does: ``root file.C``, ``.x file.C(args)``, ``gROOT->ProcessLine``.

A macro is translated in the ROOT dialect (or its translation found in the
cache), the Python run, and then the function named like the file called
with the arguments ``.x`` gave it - ``hsimple.C`` runs ``hsimple()``; an
unnamed ``{ ... }`` macro's block is that function. ``file.C+`` (ACLiC,
compile first) builds the macro's library, ``file_C.xrdlib``, loads it and
calls the same function from it (:mod:`.library.aclic`); ``.L libFoo.so``
loads a library. Libraries a macro names with ``R__LOAD_LIBRARY(Foo)`` or
``#pragma cling load("libFoo")`` are loaded before it is translated, as
cling loads them when it reads the line. What the macro raises as it runs is
raised again as a :class:`MacroError` at the C++ line it happened on. The
running itself is :mod:`xrdroot.pycxx.execute`'s; this is ROOT's face on it.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..pycxx import execute as _cxx
from ..pycxx.errors import MacroError, Refusal
from ..pycxx.runtime.namespace import DECLARED
from ..pycxx.translation import translation
from .dialect import ROOT as DIALECT
from .library import aclic
from .library import loading as libraries
from .runtime import ROOT

__all__ = [
    "run",
    "run_and_quit",
    "run_program",
    "run_source",
    "load",
    "process_line",
    "split_call",
    "called",
    "exists",
    "define",
    "session_dialect",
    "arguments",
    "EXTENSIONS",
    "MacroError",
    "Refusal",
]

#: The file extensions of a C++ macro.
EXTENSIONS = (".C", ".c", ".cxx", ".cpp", ".cc", ".h", ".hxx")

#: ``file.C+(1, "a")``: a macro, ACLiC's ``+`` or ``++``, and its arguments.
CALL = re.compile(
    r"^\s*(?P<path>[^(]+?)(?P<plus>(?:\+{1,2}[gOkfcs]*)?)\s*(?:\((?P<args>.*)\))?\s*$", re.S
)


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
    """A macro's file, ACLiC's ``+`` and its options taken off."""
    return Path(aclic.split(str(path))[0])


#: ``R__LOAD_LIBRARY(Foo)`` and ``#pragma cling load("Foo")``: a library a macro loads as
#: it is read; ``R__ADD_LIBRARY_PATH(dir)`` and ``#pragma cling add_library_path("dir")``:
#: where libraries are looked for.
PRAGMAS = re.compile(
    r"""R__(?P<macro>LOAD_LIBRARY|ADD_LIBRARY_PATH)\s*\(\s*"?(?P<name>[^")]+?)"?\s*\)
    |^[ \t]*\#[ \t]*pragma[ \t]+cling[ \t]+(?P<pragma>load|add_library_path)
     [ \t]*\([ \t]*"(?P<argument>[^"]*)"[ \t]*\)""",
    re.M | re.X,
)

#: What each says to do, by its name: load a library, or look in a directory for them.
_LOADS = frozenset({"LOAD_LIBRARY", "load"})


def preload(source: str) -> None:
    """Load the libraries ``source`` names to cling as it is read, and add the directories
    it says to look in - before a word of it is translated, as cling does."""
    for found in PRAGMAS.finditer(source):
        what = found["macro"] or found["pragma"]
        value = os.path.expandvars((found["name"] or found["argument"]).strip())
        if what in _LOADS:
            libraries.load(value)
        else:
            _add_dynamic_path(value)


def _add_dynamic_path(directory: str) -> None:
    from ..pyroot.core.system import gSystem

    gSystem.AddDynamicPath(directory)


#: The ``#define`` and ``#undef`` lines run at the prompt or by ``ProcessLine``, in order:
#: cling keeps them for every macro read after - ``root -e '#define STAGE "first"' run.C``.
DEFINES: list[str] = []

#: A directive that defines or undefines a macro.
#: A line that declares - a namespace, a class, a template - which cling keeps at the top,
#: rather than statements run in a block of their own.
DECLARATION = re.compile(r"^(?:namespace|class|struct|union|template\s*<)\b")

DEFINITION = re.compile(r"^[ \t]*#[ \t]*(?:define|undef)\b.*$", re.M)


def define(text: str) -> None:
    """Keep the ``#define`` and ``#undef`` lines of ``text`` for the macros run after it."""
    DEFINES.extend(found.group(0).strip() for found in DEFINITION.finditer(text))


def session_dialect() -> Any:
    """ROOT's dialect, with the macros defined so far in the session predefined - a dialect
    of its own name, so a translation made with them is cached apart."""
    if not DEFINES:
        return DIALECT
    text = "\n".join(DEFINES) + "\n"
    return _defined(text)


@lru_cache(maxsize=16)
def _defined(text: str) -> Any:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return replace(DIALECT, name=f"{DIALECT.name}+{digest}", predefined=DIALECT.predefined + text)


def exists(spec: str) -> bool:
    """Whether what ``.x spec`` runs is there: the macro's file, or a library on the path."""
    path = aclic.split(spec)[0]
    return libraries.find(path) is not None if aclic.is_library(path) else Path(path).is_file()


def _preread(where: Path) -> None:
    """A macro's libraries loaded before its library is built or called - when it is a
    macro, not a library."""
    if where.is_file():
        _read(where)


def _read(where: Path) -> str:
    """A macro's text, the libraries it loads loaded."""
    source = where.read_text(encoding="utf-8", errors="replace")
    preload(source)
    return source


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
    preload(source)
    before = dict(DECLARED) if "namespace" in source else None
    done = _cxx.run_source(
        source, file, args, dialect=session_dialect(), namespace=root, use_cache=use_cache,
        call=call,
    )
    if before is None:
        return done
    spaces, hidden = libraries.declare_namespaces(source, before)
    if call:
        return done
    return {**{name: value for name, value in done.items() if name not in hidden}, **spaces}


def run(
    path: str | Path, args: tuple[Any, ...] = (), *, root: Any = None, use_cache: bool = True
) -> Any:
    """``.x path(args)``: translate the macro at ``path`` and run it, as ROOT would - or, for
    ``path+``, build its library, load it and call its function from there."""
    if aclic.is_compiled(str(path)):
        _preread(_path(path))
        return aclic.call(str(path), tuple(args))[0]
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
    for where in wheres:
        _preread(where)
    if aclic.is_compiled(str(paths[0])):
        return _compiled(str(paths[0]), tuple(args))
    return _cxx.run_program(wheres, args, argv, dialect=session_dialect(), use_cache=use_cache)


def _compiled(spec: str, args: tuple[Any, ...]) -> int:
    """``root -b -q file.C+(args)``: ACLiC's library's function called, its value printed as
    cling prints it, and the exit status it makes."""
    try:
        value, made = aclic.call(spec, args)
    finally:
        sys.stdout.flush()
    line = DIALECT.shown(value, made.returns)
    if line is not None:
        sys.stdout.write(line + "\n")
    return DIALECT.status(value, made.returns)


def load(path: str | Path, *, root: Any = None) -> dict[str, Any]:
    """``.L path``: translate the macro and define what it declares, calling nothing - or, for
    ``path+``, build and load its library; for ``libFoo.so``, load that library."""
    text = str(path).strip()
    if aclic.split(text)[1]:
        _read(_path(text))
        return aclic.load(text).module
    if aclic.is_library(text):
        return _library(text)
    where = _path(path)
    source = where.read_text(encoding="utf-8", errors="replace")
    namespace: dict[str, Any] = run_source(source, str(where), root=root, call=False)
    return namespace


def _library(name: str) -> dict[str, Any]:
    """``.L libFoo.so``: the library loaded; what it defined, or nothing if it was not found."""
    if libraries.load(name) < 0:
        return {}
    found = libraries.find(name)  # none for ROOT's own libraries, which are xrdroot's
    return libraries.LOADED[str(found.resolve())].module if found is not None else {}


def process_line(line: str, *, root: Any = None) -> Any:
    """``gROOT->ProcessLine(line)``: a ``.x``/``.L`` command, or C++ statements run at once."""
    text = line.strip()
    if text.startswith((".x", ".X")):
        return _executed(text[2:], root)
    if text.startswith(".L"):
        load(called(text[2:]), root=root)
        return 0
    if text.startswith("."):
        raise ValueError(f"{text.split()[0]} is not a command gROOT->ProcessLine runs here")
    preload(text)
    define(text)
    if text.startswith("#"):  # ``#include "Foo.h"``: declarations, at the top as cling has them
        run_source(text + "\n", "<ProcessLine>", root=root, call=False)
        return 0
    if DECLARATION.match(text) and _declared_at_top(text, root):
        return 0
    return _cxx.run_line(text, file="<ProcessLine>", dialect=session_dialect(), namespace=root)


def _declared_at_top(text: str, root: Any) -> bool:
    """``class A {};``, ``namespace N {...}``: declarations, run at the top as cling has them
    - unless statements follow them, which a block of their own runs instead."""
    try:
        run_source(text + "\n", "<ProcessLine>", root=root, call=False)
    except Refusal:
        return False
    return True


def called(text: str) -> str:
    """``file.C+g(1, 2)`` without its arguments: the file and ACLiC's options."""
    found = CALL.match(text)
    if found is None:
        raise ValueError(f"{text!r} does not name a macro, as file.C or file.C(arguments)")
    return found["path"].strip() + found["plus"]


def _executed(text: str, root: Any) -> Any:
    """``ProcessLine(".x file.C(args)")``: the macro run, and its value printed as cling's
    ``.x`` prints it - then handed back, as ``ProcessLine`` returns it."""
    given = split_call(text)[1]
    spec = called(text)
    if aclic.is_compiled(spec):
        _preread(_path(spec))
        value, made = aclic.call(spec, arguments(given))
        spelled = made.returns
    else:
        where = _path(spec)
        source = _read(where)
        spelled = _cxx.translated(source, str(where), session_dialect(), True).returns
        value = run_source(source, str(where), arguments(given), root=root, use_cache=True)
    line = DIALECT.shown(value, spelled)
    if line is not None:
        sys.stdout.write(line + "\n")
    return value
