"""What translating or running C++ can go wrong with, always with the C++ line.

A :class:`Refusal` is the translator saying, by name, that it will not turn a
construct into Python - because the Python it could write would do something
other than what the C++ does, and a program that runs and gives a different
answer is worse than one that stops. A :class:`MacroError` is a translated
program failing as it runs, reported against the C++ line it came from rather
than against Python nobody wrote. :class:`Unsupported` is the runtime naming a
part of a C++ library it does not provide, where a program uses it.

They derive from :class:`CxxError`, a plain :class:`Exception`: this package
stands alone, and a host that has an error hierarchy of its own catches
``CxxError`` beside it.
"""

from __future__ import annotations

__all__ = ["CxxError", "Refusal", "MacroError", "Unsupported", "Where"]


class Where:
    """A place in a program: the file it was read from and the line in it."""

    __slots__ = ("file", "line")

    def __init__(self, file: str, line: int) -> None:
        self.file = file
        self.line = line

    def __repr__(self) -> str:
        return f"{self.file}:{self.line}"


class CxxError(Exception):
    """Base of what :mod:`xrdroot.pycxx` raises: a sentence, and the C++ line it is about."""

    def __init__(self, why: str, where: Where | None = None) -> None:
        self.why = why
        self.where = where
        super().__init__(f"{where}: {why}" if where is not None else why)


class Refusal(CxxError):
    """A C++ construct this translator will not turn into Python, named, at its line."""


class MacroError(CxxError):
    """A translated program failing as it runs, placed at the C++ line it came from."""


class Unsupported(CxxError):
    """A part of a C++ library the runtime does not provide, named where a program uses it."""
