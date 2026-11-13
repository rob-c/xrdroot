"""What the test being run has come to, and the failure being reported as it is written.

GoogleTest prints a failure when the statement that made it ends, with
whatever ``<< message`` was streamed after the assertion. Python has no end
of a statement to wait for, so a failure is a :class:`Note` the message is
streamed onto, printed when GoogleTest next does anything - the next
assertion, or the end of the test. What the test printed in between comes
out before the failure rather than after it; nothing else differs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from ..streams import ostringstream

__all__ = ["Result", "Note", "STATE", "flush", "report", "skip", "pending", "trace", "cxx"]


@dataclass
class Result:
    """One test's outcome as it runs: its failures, whether one was fatal, whether it skipped."""

    suite: str
    name: str
    failures: int = 0
    fatal: bool = False
    skipped: bool = False


class Note:
    """A failure (or a skip) being reported, the ``<<`` after its assertion streaming onto it."""

    def __init__(self, file: str, line: int | None, kind: str, text: str) -> None:
        self.file = file
        self.line = line
        self.kind = kind
        self.text = text
        self.stream = ostringstream()
        self.traces: list[str] = []

    def __lshift__(self, value: Any) -> Note:
        self.stream << value
        return self

    def render(self) -> str:
        """The note as GoogleTest prints it: where, what, the message, and the traces in force."""
        body = "\n".join(part for part in (self.text, self.stream.str()) if part)
        if self.traces:
            body += "\nGoogle Test trace:" + "".join("\n" + trace for trace in self.traces)
        where = self.file if self.line is None else f"{self.file}:{self.line}"
        return f"{where}: {self.kind}\n{body}\n"


@dataclass
class State:
    """Everything a run of tests keeps between one assertion and the next."""

    result: Result | None = None
    pending: Note | None = None
    #: ``SCOPED_TRACE``'s messages in force, by the place that set each: one per place.
    traces: dict[tuple[str, int], str] = field(default_factory=dict)
    #: Failures outside any test - in ``main``, before ``RUN_ALL_TESTS`` - which fail the run.
    stray: int = 0


#: The run's state: one program's tests run at a time.
STATE = State()


def flush() -> None:
    """Print the note being reported, its message now complete."""
    note, STATE.pending = STATE.pending, None
    if note is not None:
        sys.stdout.write(note.render())
        sys.stdout.flush()


def report(file: Any, line: Any, text: str, fatal: Any) -> bool:
    """Record a failure of the test running and begin its note: ``False``, the check's verdict."""
    flush()
    note = Note(str(file), None if line is None else int(line), "Failure", text)
    note.traces = [f"{where}:{at}: {said}" for (where, at), said in reversed(STATE.traces.items())]
    STATE.pending = note
    result = STATE.result
    if result is None:
        STATE.stray += 1
        return False
    result.failures += 1
    result.fatal = result.fatal or bool(fatal)
    return False


def skip(file: Any, line: Any, text: str = "") -> bool:
    """Mark the test running skipped and begin the note saying so: ``False``, so it returns."""
    flush()
    STATE.pending = Note(str(file), int(line), "Skipped", text)
    if STATE.result is not None:
        STATE.result.skipped = True
    return False


def pending() -> Note:
    """What the ``<< message`` after a failed assertion streams onto: the note just begun."""
    if STATE.pending is None:  # nothing failed: the message goes nowhere, as GoogleTest's would
        return Note("", None, "", "")
    return STATE.pending


def cxx(value: Any) -> str:
    """``value`` as a C++ stream writes it: what a message or a trace says."""
    stream = ostringstream()
    stream << value
    return stream.str()


def trace(file: Any, line: Any, message: Any) -> None:
    """``SCOPED_TRACE(message)``: said with every failure until the test ends.

    The trace is kept per place, so a ``SCOPED_TRACE`` in a loop says the
    present turn's message, not every turn's; it lasts to the end of the
    test, not of its block, since the translation has no end of a block.
    """
    flush()
    STATE.traces[(str(file), int(line))] = cxx(message)
