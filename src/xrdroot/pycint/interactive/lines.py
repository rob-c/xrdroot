"""``gROOT->ProcessLine`` of C++, run in the session's translation unit as cling runs it.

Cling has one translation unit for a session, and a line ``ProcessLine``
hands it is one more input to it: ``TFile *f = ...;`` declares ``f`` for every
line after it, and those lines - typed at the prompt, given to ``ProcessLine``
by a macro, or run by ``.x`` - see it. So a line here is run as the prompt
runs one (:mod:`.unit`), in the prompt's unit once a prompt is open
(:func:`adopt`) and in one of the process's own before that.

Two things differ from the prompt. A line run while an input of the unit is
itself running - ``gROOT->ProcessLine(...)`` typed at the prompt - cannot add
to the unit that input is not yet part of, so it runs by itself, its
declarations ending with it, as before. And what goes wrong is raised, not
said: the macro that called ``ProcessLine`` stops at it, with the line's own
words - a name xrdroot lacks is named as missing, not as undeclared.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from typing import Any

from ...pycxx import execute as _cxx
from ...pycxx.errors import MacroError, Refusal, Where
from ..dialect import ROOT as DIALECT
from ..execute import session_dialect
from .lexer import pieces
from .unit import Diagnostic, Unit
from .values import shown

__all__ = ["adopt", "session_unit", "run_line", "command", "FILE"]

#: What a line's errors are placed in, as ``ProcessLine``'s were before it had a unit.
FILE = "<ProcessLine>"

#: The unit ``ProcessLine`` runs lines in: the prompt's, or the process's own.
_UNITS: list[Unit] = []


def adopt(unit: Unit) -> None:
    """Run the lines ``ProcessLine`` is given in ``unit``: a prompt's, opened now."""
    _UNITS[:] = [unit]


def session_unit() -> Unit:
    """The unit lines run in: the prompt's if one is open, else the process's own."""
    if not _UNITS:
        _UNITS.append(Unit())
    return _UNITS[0]


def run_line(text: str, root: Any = None) -> Any:
    """``ProcessLine(text)`` of C++: run in the session's unit, its value handed back - and
    printed, as cling prints it, when the last expression was given without its ``;``."""
    unit = session_unit()
    if unit.busy:
        return _cxx.run_line(text, file=FILE, dialect=session_dialect(), namespace=root)
    parts = pieces(text)
    if not parts:
        return None
    binding = DIALECT.host_object.bind(root) if root is not None else nullcontext()
    try:
        with binding:
            outcome = unit.outcome(0, parts, text, check=False)
    except Diagnostic as why:
        raised = Refusal if why.translating else MacroError
        raise raised(why.why, Where(FILE, _line(why.where))) from None
    line = shown(outcome.value, outcome.spelled, outcome.lvalue) if outcome.printed else None
    if line:
        sys.stdout.write(line + "\n")
    return outcome.value


def _line(where: str) -> int:
    """The line of the input a diagnostic's ``ROOT_prompt_N:line:column`` is at."""
    parts = where.rsplit(":", 2)
    return int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else 1


#: The prompt's commands ``ProcessLine`` does not run: leaving, and those of the terminal.
NOT_RUN = frozenset({"q", "qqq", "qqqqq", "qqqqqqq", "exit", "quit", "@", ">"})


def command(text: str) -> int | None:
    """``ProcessLine(".undo 1")``, ``(".O 2")``, ``(".ls")``: the prompt's command, on the
    session's unit, what it says printed and what ``ProcessLine`` hands back for it
    handed back - or ``None`` if it is no command ``ProcessLine`` runs."""
    from .commands import COMMAND, COMMANDS, ONE

    found = COMMAND.match(text.strip())
    name = found["name"] if found is not None else ""
    if name not in COMMANDS or name in NOT_RUN or found is None:
        return None
    for said in COMMANDS[name](_holder(), found["rest"]):
        (sys.stderr if said.error else sys.stdout).write(said.text + "\n")
    return int(name in ONE)


#: The settings ``ProcessLine``'s commands keep, with the unit they are of.
_HOLDERS: list[Any] = []


def _holder() -> Any:
    """What the commands ``ProcessLine`` runs reach: kept for the session's unit."""
    unit = session_unit()
    if not _HOLDERS or _HOLDERS[0].unit is not unit:
        _HOLDERS[:] = [_Holder(unit)]
    return _HOLDERS[0]


class _Holder:
    """What a prompt command reaches of its interpreter: the unit, here the session's, and
    the settings a prompt keeps, here new."""

    def __init__(self, unit: Unit) -> None:
        self.unit = unit
        self.level = 2
        self.raw = False
        self.loaded: dict[str, list[str]] = {}
        self.last_status = 0
        self.failed = False
