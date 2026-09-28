"""Standard output as a PyROOT script under a pipe has it: two buffers, not one.

A PyROOT script writes to standard output through two libraries at once.
Its own ``print`` goes into Python's buffer, which - when the output is a
pipe or a file rather than a terminal - is written out only when it holds
8 KiB or when the interpreter exits. What ROOT prints goes through the C
library's ``stdout`` instead, which ROOT's C++ mostly flushes as it goes
(``std::endl``, and cling after every line it runs), and which is flushed
last of all at exit, after Python's. So a script that prints a line and
then fits shows the fit first and its own line at the very end; ROOT's
tutorials' expected output is in that order, and xrdroot keeps it.

:func:`split` stands in for ``sys.stdout`` while a script runs: what code
in xrdroot (or a translated C++ line) writes goes out at once, as ROOT's
flushed C++ output would, and what anything else writes goes to Python's
own buffered stream as it always did. :func:`printf` and :func:`unflushed`
are for what ROOT prints with ``printf`` and never flushes: it waits in a
C-sized buffer, and goes out with the next flushed line or at the very end.
Outside a split - the terminal, tests, macros - both are plain writes.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TextIO

__all__ = ["split", "active", "cwrite", "printf", "unflushed", "CStdout"]

#: The size of the C library's buffer on a pipe: it is written out whenever it fills.
C_BUFFER = 4096
#: The module names whose writes are ROOT's own rather than the script's.
ROOT_SIDE = ("xrdroot", "__cint")


class CStdout:
    """The C library's ``stdout`` on file descriptor ``fd``: full buffering, 4 KiB."""

    def __init__(self, fd: int) -> None:
        self.fd = fd
        self.pending = b""

    def write(self, text: str, *, flush: bool) -> None:
        """Buffer ``text``, writing out each full buffer; all of it if ``flush``."""
        self.pending += text.encode("utf-8", errors="replace")
        while len(self.pending) >= C_BUFFER:
            self._out(self.pending[:C_BUFFER])
            self.pending = self.pending[C_BUFFER:]
        if flush:
            self.flush()

    def flush(self) -> None:
        self._out(self.pending)
        self.pending = b""

    def _out(self, data: bytes) -> None:
        while data:
            data = data[os.write(self.fd, data) :]


def _root_side(depth: int) -> bool:
    """Whether the code writing, ``depth`` frames up, is ROOT's side rather than the script's."""
    name = sys._getframe(depth + 1).f_globals.get("__name__", "")
    return isinstance(name, str) and name.startswith(ROOT_SIDE)


class _Split:
    """``sys.stdout`` while a script runs: ROOT's writes to C's stdout, the rest to Python's."""

    def __init__(self, python: TextIO, c: CStdout) -> None:
        self.python = python
        self.c = c
        #: How many :func:`unflushed` blocks are open: ROOT's writes are ``printf``'s in one.
        self.held = 0

    def write(self, text: str) -> int:
        if _root_side(1):
            self.c.write(text, flush=not self.held)
        else:
            self.python.write(text)
        return len(text)

    def flush(self) -> None:
        (self.c if _root_side(1) else self.python).flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.python, name)


def active() -> bool:
    """Whether standard output is split, as a PyROOT script's into a pipe is."""
    return isinstance(sys.stdout, _Split)


def cwrite(text: str, *, flush: bool = True) -> None:
    """Write ``text`` to C's ``stdout`` under a split - flushed, as ``std::endl`` does, or not.

    Unsplit, it is an ordinary write to ``sys.stdout``.
    """
    stream = sys.stdout
    if isinstance(stream, _Split):
        stream.c.write(text, flush=flush)
    else:
        stream.write(text)


def printf(text: str) -> None:
    """Write ``text`` as ROOT's ``printf`` would: into C's buffer, unflushed, under a split."""
    cwrite(text, flush=False)


@contextmanager
def unflushed() -> Iterator[None]:
    """What xrdroot prints in here is ROOT's ``printf``: held in C's buffer, under a split."""
    stream = sys.stdout
    if not isinstance(stream, _Split):
        yield
        return
    stream.held += 1
    try:
        yield
    finally:
        stream.held -= 1


def _pipe(stream: TextIO) -> int | None:
    """The file descriptor under ``stream`` if it is a pipe or a file; None for a terminal."""
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    return None if os.isatty(fd) else fd


@contextmanager
def split() -> Iterator[None]:
    """Give a script's standard output ROOT's two buffers, if it is not a terminal.

    On the way out Python's buffer is written before C's, as the
    interpreter's exit flushes it before the C library's does.
    """
    python = sys.stdout
    fd = _pipe(python)
    if fd is None:
        yield
        return
    c = CStdout(fd)
    sys.stdout = _Split(python, c)
    try:
        yield
    finally:
        sys.stdout = python
        python.flush()
        c.flush()
