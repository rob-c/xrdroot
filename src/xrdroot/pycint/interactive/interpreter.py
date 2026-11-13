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

import sys
from pathlib import Path
from typing import TextIO

from ...pycxx.errors import CxxError as CintError
from ..execute import arguments, run_program, split_call
from .commands import Said, dispatch, failure
from .lexer import pending, pieces
from .unit import Diagnostic, Unit
from .values import shown

__all__ = ["Interpreter", "Said", "MACROS", "CONTINUED"]

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
        #: The number the next line typed is given: ``root [N]``, and ``ROOT_prompt_N``.
        self.number = 0
        self.buffer: list[str] = []
        self.finished = False
        self.exit_on_error = exit_on_error
        #: The exit status: the last macro run's value, as ``root -q`` exits with it, or 1
        #: once an error stopped an ``exit_on_error`` session.
        self.last_status = 0
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
                return self._checked(said)
        self.buffer.append(line)
        text = "\n".join(self.buffer)
        if pending(text):
            return []
        self.buffer = []
        return self._checked(self.run(text, number))

    def cancel(self) -> None:
        """Drop the lines of an unfinished input, as Ctrl-C at cling's prompt does."""
        self.buffer = []

    def run(self, text: str, number: int) -> list[Said]:
        """A whole input of C++, run as prompt ``number``: its value said, or what went wrong."""
        parts = pieces(text)
        if not parts:
            return []
        try:
            outcome = self.unit.run(self.unit.chunk(number, parts, text))
        except Diagnostic as why:
            return [Said(str(why), True)]
        except SystemExit as leaving:
            self.finished = True
            self.last_status = leaving.code if isinstance(leaving.code, int) else 0
            return []
        line = shown(outcome.value, outcome.spelled, outcome.lvalue) if outcome.printed else None
        return [Said(line)] if line else []

    def _checked(self, said: list[Said]) -> list[Said]:
        """What was said, ending the session if it was an error and errors end it (``-x``)."""
        if self.exit_on_error and any(each.error for each in said):
            self.finished = True
            self.last_status = 1
        return said

    # -- what the command line hands the prompt ------------------------------------------

    def attach(self, path: str) -> list[Said]:
        """A ROOT file given on the command line, opened as ``_fileN`` - as ``root f.root``."""
        name = f"_file{self.attached}"
        self.attached += 1
        quoted = path.replace("\\", "\\\\").replace('"', '\\"')
        said = [Said(f"\nAttaching file {path} as {name}...")]
        return said + self._checked(self.run(f'TFile *{name} = TFile::Open("{quoted}")',
                                             self._next()))  # fmt: skip

    def macro(self, given: str) -> list[Said]:
        """A macro given on the command line, ``x.C(args)``, run as ``root x.C`` runs it."""
        path = split_call(given)[0]
        sys.stdout.write(f"\nProcessing {path}...\n")
        sys.stdout.flush()
        if not Path(path.rstrip("+")).is_file():
            self.last_status = 1
            return self._checked([Said(f"Error in <TApplication::ExecuteFile>: macro {path} "
                                       "not found in path .", True)])  # fmt: skip
        try:
            self.last_status = run_program([Path(path)], arguments(split_call(given)[1]))
        except Exception as why:  # a refusal, a failure, a bad argument: said, and on
            self.last_status = 1
            said = failure(why) if isinstance(why, CintError) else Said(f"error: {why}", True)
            return self._checked([said])
        return []

    def given(self, argument: str) -> list[Said]:
        """A file from the command line: a ROOT file attached, anything else run as a macro."""
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
