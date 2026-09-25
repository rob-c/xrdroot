"""What translating or running a macro can go wrong with, always with the C++ line.

A :class:`Refusal` is the translator saying, by name, that it will not turn a
construct into Python - because the Python it could write would do something
other than what the C++ does, and a macro that runs and gives a different
answer is worse than one that stops. A :class:`MacroError` is a translated
macro failing as it runs, reported against the C++ line it came from rather
than against Python nobody wrote.
"""

from __future__ import annotations

from ..errors import ROOTError

__all__ = ["CintError", "Refusal", "MacroError", "Where"]


class Where:
    """A place in a macro: the file it was read from and the line in it."""

    __slots__ = ("file", "line")

    def __init__(self, file: str, line: int) -> None:
        self.file = file
        self.line = line

    def __repr__(self) -> str:
        return f"{self.file}:{self.line}"


class CintError(ROOTError):
    """Base of what :mod:`xrdroot.cint` raises: a sentence, and the C++ line it is about."""

    def __init__(self, why: str, where: Where | None = None) -> None:
        self.why = why
        self.where = where
        super().__init__(f"{where}: {why}" if where is not None else why)


class Refusal(CintError):
    """A C++ construct this translator will not turn into Python, named, at its line."""


class MacroError(CintError):
    """A translated macro failing as it runs, placed at the C++ line it came from."""
