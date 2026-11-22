"""The prompt itself, without a terminal: lines in, what cling would say out.

:class:`Interpreter` is fed the lines a user types, one at a time, and hands
back what the prompt says to each - a value printed as ``(int) 4``, an error
as ``ROOT_prompt_3:1:1: error: ...`` - while what the C++ prints goes to
standard output as it runs. It keeps the session: the numbered prompt
``root [N]``, the lines of an input that is not finished yet, and the
translation unit everything typed so far makes (:mod:`.unit`). The terminal
loop around it is :mod:`.terminal`; tests feed it directly.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import TextIO

from ...pycxx.errors import CxxError as CintError
from ...pycxx.errors import MacroError
from ..execute import arguments, called, define, exists, run_program, split_call
from ..returned import status as exit_status
from ..runtime.application import TRINT
from .commands import Said, dispatch, failure
from .lexer import Piece, pending, pieces
from .lines import adopt
from .unit import Diagnostic, Outcome, Unit
from .values import shown

__all__ = ["Interpreter", "Said", "MACROS", "CONTINUED"]

#: ``return`` and ``return x;`` typed by themselves, as ``root -e return`` gives them.
RETURN = re.compile(r"^\s*return\b\s*(?P<value>.*?)\s*;?\s*$", re.S)

#: The extensions of the files given on the command line that are ROOT files, not macros.
ROOT_FILES = (".root",)

#: What cling's prompt says while an input is not finished.
CONTINUED = "root (cont'ed, cancel with .@) [{}] "

#: The extensions a macro may have.
MACROS = (".C", ".c", ".cxx", ".cpp", ".cc", ".h", ".hxx")


class Interpreter:
    """cling's prompt, as an object: :meth:`feed` it lines, print what it hands back."""

    def __init__(self, *, exit_on_error: bool = False, directory: Path | None = None) -> None:
        self.unit = Unit(directory)
        adopt(self.unit)  # gROOT->ProcessLine runs in it from now on
        #: The number the next line typed is given: ``root [N]``, and ``ROOT_prompt_N``.
        self.number = 0
        self.buffer: list[str] = []
        self.finished = False
        self.exit_on_error = exit_on_error
        #: The exit status: the value of the last input or macro run, as ``root -q`` exits
        #: with it, or 1 once an error stopped an ``exit_on_error`` session.
        self.last_status = 0
        #: Whether a macro given on the command line failed, which ends their processing.
        self.failed = False
        #: Whether the session was ended with a status - ``exit(n)``, ``Terminate(n)``, an
        #: error under ``-x`` - rather than at ``.q`` or the end of the input.
        self.exited = False
        self.loaded: dict[str, list[str]] = {}
        self.raw = False
        self.level = 2
        self.attached = 0
        self.redirected: TextIO | None = None
        self.saved: TextIO = sys.stdout

    @property
    def prompt(self) -> str:
        """``root [N] ``, or cling's continuation prompt while an input is unfinished."""
        return CONTINUED.format(self.number) if self.buffer else f"root [{self.number}] "

    @property
    def status(self) -> int:
        return self.last_status

    def feed(self, line: str) -> list[Said]:
        """One line typed: a command, a line of C++, or ``.@`` cancelling the input so far."""
        number = self.number
        self.number += 1
        if self.buffer and line.strip() == ".@":
            self.buffer = []
            return []
        if not self.buffer:
            said = dispatch(self, line)
            if said is not None:
                return said
        self.buffer.append(line)
        text = "\n".join(self.buffer)
        if pending(text):
            return []
        self.buffer = []
        return self.run(text, number)

    def cancel(self) -> None:
        """Drop the lines of an unfinished input, as Ctrl-C at cling's prompt does."""
        self.buffer = []

    def run(self, text: str, number: int) -> list[Said]:
        """A whole input of C++, run as prompt ``number``: its value said, or what went wrong.

        ``return x`` - as ``root -e 'return 3'`` - hands back ``x``'s value, unprinted;
        ``return`` alone hands back nothing.
        """
        text = self._returned(text)
        define(text)
        parts = pieces(text)
        if not parts:
            return []
        try:
            outcome = self._outcome(number, parts, text)
        except Diagnostic as why:
            return [self._diagnosed(why)]
        except SystemExit as leaving:
            self.leave(leaving)
            return []
        self.last_status = exit_status(outcome.value, outcome.spelled) if outcome.valued else 0
        line = shown(outcome.value, outcome.spelled, outcome.lvalue) if outcome.printed else None
        return [Said(line)] if line else []

    def _returned(self, text: str) -> str:
        """``return x``'s ``x;``, the status ``0`` until it has run; any other input as it is."""
        returned = RETURN.match(text)
        if returned is None:
            return text
        self.last_status = 0
        return returned["value"] + ";" if returned["value"] else ""

    def _diagnosed(self, why: Diagnostic) -> Said:
        """An input that did not run, said - with the status ROOT's ``ProcessLine`` gives: 1
        for code it could not compile, but 0 when it stopped at a name nothing declares."""
        undeclared = "undeclared identifier" in why.why
        self.last_status = 0 if undeclared else 1
        if not (why.translating or undeclared):
            self._failed_running()
        return Said(str(why), True)

    def _outcome(self, number: int, parts: list[Piece], text: str) -> Outcome:
        """Run the input, its value caught even unprinted - or, if the translator will not
        take it so, as it was typed."""
        return self.unit.outcome(number, parts, text)

    def leave(self, leaving: SystemExit) -> None:
        """The session over, as ``exit(n)`` or ``gApplication->Terminate(n)`` ends it."""
        self.finished = self.exited = True
        self.last_status = leaving.code if isinstance(leaving.code, int) else 0

    def _failed_running(self) -> None:
        """C++ that failed as it ran - cling's exception - ends the session under ``-x``."""
        if self.exit_on_error:
            self.finished = self.exited = True
            self.last_status = 1

    # -- what the command line hands the prompt ------------------------------------------

    def attach(self, path: str) -> list[Said]:
        """A ROOT file given on the command line, opened as ``_fileN`` - as ``root f.root``."""
        name = f"_file{self.attached}"
        self.attached += 1
        quoted = path.replace("\\", "\\\\").replace('"', '\\"')
        sys.stdout.write(f"Attaching file {path} as {name}...\n")
        return self.run(f'TFile *{name} = TFile::Open("{quoted}")', self._next())

    def macro(self, given: str) -> list[Said]:
        """A macro given on the command line, ``x.C(args)`` or ``x.C+``, run as ``root x.C``
        runs it; one that fails ends the processing of the command line's files, as in ROOT."""
        sys.stdout.write(f"Processing {given}...\n")
        sys.stdout.flush()
        spec = called(given)
        self.last_status, self.failed = 1, True  # until it has run
        if not exists(spec):
            path = split_call(given)[0]
            return [Said(f"Error in <TApplication::ExecuteFile>: macro {path} not found in "
                         "path .", True)]  # fmt: skip
        try:
            self.last_status = run_program([spec], arguments(split_call(given)[1]),
                                           session=TRINT)  # fmt: skip
        except SystemExit as leaving:
            self.leave(leaving)
            return []
        except Exception as why:  # a refusal, a failure, a bad argument: said, and on
            if isinstance(why, MacroError):
                self._failed_running()
            said = failure(why) if isinstance(why, CintError) else Said(f"error: {why}", True)
            return [said]
        self.failed = False
        return []

    def given(self, argument: str) -> list[Said]:
        """A file from the command line: a ROOT file attached, a directory gone into, as
        ``root dir`` does, anything else run as a macro."""
        if os.path.isdir(argument):
            os.chdir(argument)
            return []
        if split_call(argument)[0].endswith(ROOT_FILES):
            return self.attach(argument)
        return self.macro(argument)

    def _next(self) -> int:
        number = self.number
        self.number += 1
        return number

    def restore(self) -> None:
        """Standard output back where it was before ``.>`` sent it to a file."""
        if self.redirected is not None:
            sys.stdout = self.saved
            self.redirected.close()
            self.redirected = None

    def close(self) -> None:
        """The session over: standard output restored, if a ``.>`` had it."""
        self.restore()
