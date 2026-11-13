"""The terminal around the :class:`~.interpreter.Interpreter`: prompts, history, completion.

At a terminal, lines are read with :mod:`readline` when Python has it - its
history kept in ``~/.xrdroot_history`` between sessions, and Tab completing
ROOT's names and the prompt's own - and the prompt is ``root [N]``. Piped,
``echo 'h.GetEntries()' | xrdroot root -l -b``, lines are read as they come
and no prompt is written, so that what comes out is only what the C++ and
cling print. The session ends at ``.q``, at the end of the input, or when
the C++ calls ``exit()``.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TextIO

from .commands import Said
from .interpreter import Interpreter

__all__ = ["say", "interact", "HISTORY", "Completer"]

#: Where the prompt's history is kept between sessions.
HISTORY = Path("~/.xrdroot_history")

#: How many lines of history are kept.
HISTORY_LENGTH = 1000


def say(said: Iterable[Said]) -> None:
    """Write what the prompt said: errors on standard error, the rest on standard output."""
    for line in said:
        stream = sys.stderr if line.error else sys.stdout
        stream.write(line.text + "\n")
        stream.flush()


class Completer:
    """Tab completion of the names the prompt knows: its own variables, then ROOT's."""

    def __init__(self, interpreter: Interpreter) -> None:
        self.interpreter = interpreter
        self.matches: list[str] = []

    def names(self) -> list[str]:
        from ... import pyroot

        own = list(self.interpreter.unit.types)
        return own + [name for name in dir(pyroot) if not name.startswith("_")]

    def __call__(self, text: str, state: int) -> str | None:
        if state == 0:
            self.matches = sorted({name for name in self.names() if name.startswith(text)})
        return self.matches[state] if state < len(self.matches) else None


def _readline() -> Any:
    """Python's :mod:`readline`, or ``None`` where it has none (Windows)."""
    try:
        return importlib.import_module("readline")
    except ImportError:
        return None


def _terminal_reader(interpreter: Interpreter) -> tuple[Callable[[str], str], Callable[[], None]]:
    """``input()`` with readline's history and completion, and what saves the history."""
    readline = _readline()
    if readline is None:
        return input, lambda: None
    history = HISTORY.expanduser()
    if history.is_file():
        readline.read_history_file(str(history))
    readline.set_history_length(HISTORY_LENGTH)
    readline.set_completer(Completer(interpreter))
    readline.parse_and_bind("tab: complete")
    return input, lambda: readline.write_history_file(str(history))


def _piped_reader(stdin: TextIO) -> Callable[[str], str]:
    """Lines of piped input, without writing a prompt: ``EOFError`` at the end, as ``input``."""

    def read(prompt: str) -> str:
        line = stdin.readline()
        if not line:
            raise EOFError
        return line.rstrip("\n")

    return read


def interact(interpreter: Interpreter, stdin: TextIO | None = None) -> int:
    """Read and run lines until the session ends; the exit status ``root`` would give."""
    stdin = stdin or sys.stdin
    if stdin.isatty():
        read, save = _terminal_reader(interpreter)
    else:
        read, save = _piped_reader(stdin), lambda: None
    try:
        while not interpreter.finished:
            try:
                line = read(interpreter.prompt)
            except EOFError:
                break
            except KeyboardInterrupt:
                interpreter.cancel()
                sys.stdout.write("\n")
                continue
            say(interpreter.feed(line))
    finally:
        save()
        interpreter.close()
    return interpreter.status
