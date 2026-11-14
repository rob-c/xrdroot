"""The prompt's dot commands: ``.x``, ``.L``, ``.ls``, ``.q`` and the rest of cling's and ROOT's.

Each is a function of the interpreter and the text after the command's
name, handing back what it has to say as :class:`Said` lines; what a macro
or ROOT itself prints goes to standard output as it happens. ``.ls``,
``.cd`` and ``.pwd`` are ROOT's ``gDirectory->ls()``, ``cd()`` and
``pwd()``, on the files the prompt opened through ``TFile``; ``.files``
lists every file the session (:mod:`xrdroot.session`) has open.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from ...pycxx.errors import CxxError as CintError
from ..declaring import forget
from ..execute import arguments, called, exists, load, run_program, split_call
from ..library.aclic import split as split_aclic
from ..library.build import BuildError
from ..library.manifest import stem_of

if TYPE_CHECKING:
    from .interpreter import Interpreter

__all__ = ["Said", "COMMAND", "COMMANDS", "HELP", "dispatch", "failure"]


class Said(NamedTuple):
    """A line the prompt says: on standard output, or - ``error`` - on standard error."""

    text: str
    error: bool = False


#: A command: a dot, a name or one of cling's punctuation commands, and what follows.
COMMAND = re.compile(r"^\.(?P<name>[A-Za-z]+|[!?@>])\s*(?P<rest>.*?)\s*$", re.S)

#: What ``.help`` prints: every command this prompt knows.
HELP = """\
This is xrdroot's C++ prompt: what is typed is translated into Python and run.
  .x file.C[(args)]  run a macro: its function named like the file, with the arguments
  .X file.C[(args)]  the same
  .L file.C[+]       load a macro: declare its functions and classes, run nothing
  .U file.C          unload what .L loaded
  .undo [n]          take back the last n inputs, one by default
  .I [dir]           look for #include files in dir too; .I alone lists the ones added
  .ls [option]       list the current directory, gDirectory->ls()
  .cd [dir]          go into a directory of the current file; .cd alone, to the top
  .pwd               where the session is, gDirectory->pwd()
  .files             the files open in the session, and the macros loaded
  .libraries         the libraries loaded
  .! command         run a shell command
  .> file            send standard output to the file; .> alone sends it back
  .g [name]          the variables declared at the prompt, and their types
  .class Name        a class: its bases and its methods
  .rawInput [0|1]    cling's raw input mode, kept only as a setting here
  .O [level]         cling's optimization level, kept only as a setting here
  .@                 cancel the lines of an unfinished input
  .help, .?          this list
  .q, .qqq, .exit    leave"""


def failure(why: CintError) -> Said:
    """A macro's failure as cling reports one, at the C++ line it happened on - or ACLiC's,
    whose compiler's errors were said already."""
    if isinstance(why, BuildError):
        return Said(f"Error in <ACLiC>: {why.why}", True)
    where = why.where
    spot = f"{where.file}:{where.line}:1: " if where is not None else ""
    return Said(f"{spot}error: {why.why}", True)


def _missing(path: str, doing: str) -> Said:
    return Said(f"Error in <{doing}>: macro {path} not found in path .", True)


def _run(interp: Interpreter, rest: str) -> list[Said]:
    """``.x file.C(args)``: the macro's function called, and the value it returns printed."""
    if not rest:
        return [Said("error: .x needs a macro to run: .x file.C, or .x file.C(arguments)", True)]
    path, given = split_call(rest)
    interp.last_status, interp.failed = 1, True  # until it has run: it ends the command line
    if not exists(called(rest)):
        return [_missing(path, "TApplication::ExecuteFile")]
    try:
        interp.last_status = run_program([called(rest)], arguments(given))
    except CintError as why:
        return [failure(why)]
    except TypeError as why:
        return [Said(f"error: {why}", True)]
    interp.failed = False
    return []


def _load(interp: Interpreter, rest: str) -> list[Said]:
    """``.L file.C``: what the macro declares, declared, and nothing called; ``.L file.C+``,
    ACLiC's library of it built and loaded; ``.L libFoo.so``, the library loaded."""
    spec = called(rest) if rest else ""
    path = split_aclic(spec)[0]
    library = stem_of(path) != os.path.basename(path)
    if not library and not Path(path).is_file():
        return [_missing(path or "(none)", "TRint::ProcessLine")]
    try:
        names = load(spec)
    except CintError as why:
        return [failure(why)]
    interp.loaded[str(Path(path))] = [
        name for name, value in names.items() if getattr(value, "__module__", "") == "__cint__"
    ]
    return []


def _unload(interp: Interpreter, rest: str) -> list[Said]:
    """``.U file.C``: what ``.L`` declared from the file, forgotten."""
    names = interp.loaded.pop(str(Path(split_call(rest)[0].rstrip("+"))) if rest else "", None)
    if names is None:
        return [Said(f"error: {rest or 'nothing'} was not loaded with .L, so there is "
                     "nothing to unload", True)]  # fmt: skip
    forget(names)
    return []


def _undo(interp: Interpreter, rest: str) -> list[Said]:
    """``.undo [n]``: the last ``n`` inputs - one, by default - taken back."""
    if rest and not rest.isdigit():
        return [Said(f"error: .undo takes a number of inputs to take back, not {rest!r}", True)]
    interp.unit.undo(int(rest or 1))
    return []


def _include(interp: Interpreter, rest: str) -> list[Said]:
    """``.I dir``: a directory ``#include`` looks in, for the prompt and for ACLiC; ``.I``
    alone, the directories added so far, as cling lists its ``-I`` flags."""
    if not rest:
        return [Said(line) for path in interp.unit.includes for line in ("-I", str(path))]
    interp.unit.includes.append(Path(rest))
    _root().gSystem.AddIncludePath(f"-I{rest}")
    return []


def _root() -> Any:
    from ... import pyroot

    return pyroot


def _ls(interp: Interpreter, rest: str) -> list[Said]:
    _root().gDirectory.ls(rest)
    return []


def _cd(interp: Interpreter, rest: str) -> list[Said]:
    root = _root()
    if rest:
        root.gDirectory.cd(rest)
    else:
        root.gROOT.cd()
    return []


def _pwd(interp: Interpreter, rest: str) -> list[Said]:
    """``.pwd``: the current directory and the current style, as ROOT says them."""
    root = _root()
    return [Said(f"Current directory: {root.gDirectory.GetPath()}"),
            Said(f"Current style:     {root.gStyle.GetName()}")]  # fmt: skip


def _files(interp: Interpreter, rest: str) -> list[Said]:
    """``.files``: the files open, then the macros loaded."""
    from ...session import gROOT

    opened = [Said(f"  {found.name}") for found in gROOT.files]
    loaded = [Said(f"  {path}") for path in interp.loaded]
    return [Said("Files open:"), *opened, Said("Macros loaded:"), *loaded]


def _shell(interp: Interpreter, rest: str) -> list[Said]:
    """``.! command``: a shell command, what it prints said in order."""
    done = subprocess.run(rest, shell=True, capture_output=True, text=True, check=False)
    out = [Said(line) for line in done.stdout.splitlines()]
    return out + [Said(line, True) for line in done.stderr.splitlines()]


def _redirect(interp: Interpreter, rest: str) -> list[Said]:
    """``.> file``: standard output into the file, until ``.>`` alone sends it back."""
    interp.restore()
    if rest:
        interp.redirected = open(rest, "w", encoding="utf-8")
        interp.saved, sys.stdout = sys.stdout, interp.redirected
    return []


def _libraries(interp: Interpreter, rest: str) -> list[Said]:
    """``.libraries``: every library loaded, ROOT's own that it starts with first."""
    from ..library.loading import loaded_names

    return [Said(name) for name in loaded_names()]


def _help(interp: Interpreter, rest: str) -> list[Said]:
    return [Said(line) for line in HELP.splitlines()]


def _quit(interp: Interpreter, rest: str) -> list[Said]:
    interp.finished = True
    return []


def _cancel(interp: Interpreter, rest: str) -> list[Said]:
    return []


def _raw(interp: Interpreter, rest: str) -> list[Said]:
    interp.raw = rest == "1" if rest else not interp.raw
    return [Said("Using raw input" if interp.raw else "Not using raw input")]


def _optimize(interp: Interpreter, rest: str) -> list[Said]:
    if not rest:
        return [Said(f"Current cling optimization level: {interp.level}")]
    if not rest.isdigit():
        return [Said(f"error: .O takes a level, a number such as 2, not {rest!r}", True)]
    interp.level = int(rest)
    return []


def _globals(interp: Interpreter, rest: str) -> list[Said]:
    """``.g``: each variable declared at the prompt, with its type as cling spells it."""
    types = interp.unit.types
    names = [name for name in types if not rest or name == rest]
    if rest and not names:
        return [Said(f"error: no variable {rest} was declared at the prompt", True)]
    return [Said(f"{types[name] or 'auto'} {name}") for name in names]


def _class_named(interp: Interpreter, name: str) -> type | None:
    """The class ``name`` means: one declared at the prompt, else ROOT's - or ``None``."""
    found = interp.unit.namespace.get(name)
    if not isinstance(found, type):
        found = getattr(_root(), name, None)
    return found if isinstance(found, type) else None


def _class(interp: Interpreter, rest: str) -> list[Said]:
    """``.class Name``: the class's bases and public methods - the prompt's or ROOT's."""
    found = _class_named(interp, rest) if rest else None
    if found is None:
        return [Said(f"error: there is no class {rest or '(none given)'} here: .class shows "
                     "the classes declared at the prompt and those xrdroot.pyroot provides, "
                     "by their ROOT names", True)]  # fmt: skip
    return _described(found, rest)


def _described(found: type, name: str) -> list[Said]:
    """A class as ``.class`` shows it: its name and public bases, then its methods."""
    bases = [base.__name__ for base in found.__mro__[1:-1] if not base.__name__.startswith("_")]
    methods = [each for each in dir(found) if not each.startswith("_")]
    methods = [each for each in methods if callable(getattr(found, each, None))]
    heading = f"class {name}" + (f" : {', '.join(bases)}" if bases else "")
    listed = textwrap.wrap(", ".join(methods), 96, initial_indent="    ", subsequent_indent="    ")
    return [Said(heading), Said(f"  {len(methods)} public methods:"), *map(Said, listed)]


#: Every command, by name: each takes the interpreter and the text after the name.
COMMANDS: dict[str, Callable[[Interpreter, str], list[Said]]] = {
    "x": _run,
    "X": _run,
    "L": _load,
    "U": _unload,
    "undo": _undo,
    "I": _include,
    "ls": _ls,
    "cd": _cd,
    "pwd": _pwd,
    "files": _files,
    "!": _shell,
    ">": _redirect,
    "help": _help,
    "?": _help,
    "q": _quit,
    "qqq": _quit,
    "qqqqq": _quit,
    "qqqqqqq": _quit,
    "exit": _quit,
    "quit": _quit,
    "@": _cancel,
    "rawInput": _raw,
    "libraries": _libraries,
    "O": _optimize,
    "g": _globals,
    "class": _class,
}


#: The commands after which ROOT's ``ProcessLine`` hands back ``1`` (ROOT's own, which
#: TRint runs); ``.x`` hands back the macro's value, and every other command ``0``.
ONE = frozenset({"ls", "pwd", "cd", "help", "?"})


def dispatch(interp: Interpreter, line: str) -> list[Said] | None:
    """Run ``line`` if it is a command - ``None`` if it is C++ - and say what it says."""
    found = COMMAND.match(line.strip())
    if found is None:
        return None
    name = found["name"]
    command = COMMANDS.get(name)
    if command is None:
        return [Said(f"error: .{name} is not a command this prompt knows; "
                     ".help lists the ones it does", True)]  # fmt: skip
    if name not in ("x", "X"):
        interp.last_status = int(name in ONE)
    return command(interp, found["rest"])
