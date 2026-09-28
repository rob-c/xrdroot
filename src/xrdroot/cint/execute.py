"""Running a macro as ROOT does: ``root file.C``, ``.x file.C(args)``, ``gROOT->ProcessLine``.

A macro is translated (or its translation found in the cache), the Python
run, and then the function named like the file called with the arguments
``.x`` gave it - ``hsimple.C`` runs ``hsimple()``; an unnamed ``{ ... }``
macro's block is that function. ``file.C+`` (ACLiC, compile first) runs the
same way: there is nothing to compile. What the macro raises as it runs is
raised again as a :class:`MacroError` at the C++ line it happened on.
"""

from __future__ import annotations

import re
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from . import cache, returned
from .errors import MacroError, Refusal, Where
from .runtime import ROOT, cerr, clog, cout
from .runtime.streams import FORMATTERS, ostream
from .translation import Translation, translation

__all__ = [
    "run",
    "run_and_quit",
    "run_source",
    "load",
    "process_line",
    "split_call",
    "arguments",
    "EXTENSIONS",
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
    python = translation(source, "<arguments>").python
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


def _reset_streams() -> None:
    """Every run starts with ``cout`` as C++ starts it: six digits, no ``fixed``, no writers."""
    FORMATTERS.clear()
    fresh = ostream()
    for stream in (cout, cerr, clog):
        for name in ("floatfield", "adjust", "alpha", "base", "plus", "width", "digits", "fill"):
            setattr(stream, name, getattr(fresh, name))


def _translated(source: str, file: str, use_cache: bool) -> Translation:
    found = cache.load(source, file) if use_cache else None
    if found is not None:
        return found
    made = translation(source, file, [Path(file).parent])
    if use_cache:
        cache.save(source, file, made)
    return made


@contextmanager
def _placed(made: Translation, label: str) -> Iterator[None]:
    """Errors raised inside the translated code, re-raised at the C++ line they came from."""
    try:
        yield
    except (MacroError, Refusal, SystemExit, KeyboardInterrupt):
        raise
    except Exception as why:
        where = _where(why, made, label)
        raise MacroError(f"{type(why).__name__}: {why}", where) from why


def _where(why: BaseException, made: Translation, label: str) -> Where | None:
    line = None
    trace: types.TracebackType | None = why.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code.co_filename == label:
            line = trace.tb_lineno
        trace = trace.tb_next
    place = made.source_map.get(line) if line is not None else None
    return Where(*place) if place is not None else None


def run_source(
    source: str,
    file: str = "<macro>",
    args: tuple[Any, ...] = (),
    *,
    root: Any = None,
    use_cache: bool = False,
    call: bool = True,
) -> Any:
    """Translate and run a macro's text; call its function (or unnamed block) with ``args``."""
    return _executed(source, file, args, root=root, use_cache=use_cache, call=call)[0]


def _executed(
    source: str, file: str, args: tuple[Any, ...], *, root: Any, use_cache: bool, call: bool
) -> tuple[Any, Translation]:
    """What running the macro gave, and the translation that was run."""
    made = _translated(source, file, use_cache)
    label = f"<translation of {file}>"
    namespace: dict[str, Any] = {"__name__": "__cint__", "__file__": file}
    binding = ROOT.bind(root) if root is not None else nullcontext()
    _reset_streams()
    with binding:
        with _placed(made, label):
            exec(compile(made.python, label, "exec"), namespace)
        if not call:
            return namespace, made
        entry = _entry(made, namespace, file, args)
        if entry is None:
            return None, made
        with _placed(made, label):
            result = entry(*args)
    sys.stdout.flush()
    return result, made


def _entry(made: Translation, namespace: dict[str, Any], file: str, args: tuple[Any, ...]) -> Any:
    """The function running the macro means - refusing arguments nothing would take."""
    entry = namespace.get(made.entry) if made.entry else None
    if args and made.unnamed:
        raise TypeError(f"{Path(file).name} is an unnamed macro, which takes no arguments")
    if args and entry is None:
        raise TypeError(
            f"{Path(file).name} defines no function {Path(file).stem}() "
            f"to hand {', '.join(map(repr, args))} to"
        )
    return entry


def run(
    path: str | Path, args: tuple[Any, ...] = (), *, root: Any = None, use_cache: bool = True
) -> Any:
    """``.x path(args)``: translate the macro at ``path`` and run it, as ROOT would."""
    where = Path(str(path).rstrip("+"))
    source = where.read_text(encoding="utf-8", errors="replace")
    return run_source(source, str(where), tuple(args), root=root, use_cache=use_cache)


def run_and_quit(path: str | Path, args: tuple[Any, ...] = (), *, use_cache: bool = True) -> int:
    """``root -b -q path(args)``: run the macro, print what it returned, give ROOT's exit status.

    Cling prints the returned value after the macro's own output, and ROOT
    quitting exits with it - see :mod:`xrdroot.cint.returned`.
    """
    where = Path(str(path).rstrip("+"))
    source = where.read_text(encoding="utf-8", errors="replace")
    value, made = _executed(
        source, str(where), tuple(args), root=None, use_cache=use_cache, call=True
    )
    line = returned.shown(value, made.returns)
    if line is not None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    return returned.status(value, made.returns)


def load(path: str | Path, *, root: Any = None) -> dict[str, Any]:
    """``.L path``: translate the macro and define what it declares, calling nothing."""
    where = Path(str(path).rstrip("+"))
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
    body = text if text.endswith((";", "}")) else text + ";"
    return run_source("{\n" + body + "\n}\n", "<ProcessLine>", root=root)
