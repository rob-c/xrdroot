"""``std::filesystem``: a path as the parts it is made of, and the few queries over one.

A good part of what a C++ program asks the filesystem is a path built by pieces - ``base /
name``, ``p.filename()``, ``p.extension()`` - and a few questions about it. That is what this
holds: :class:`path` keeps the text and answers those questions from it, and the free functions
-make a temporary directory, remove, exists, create directories - go to :mod:`os` and
:mod:`tempfile`. A path that is a directory has no trailing separator, as C++'s paths have none.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
from typing import Any

__all__ = [
    "create_directories",
    "current_path",
    "exists",
    "file_size",
    "is_directory",
    "is_regular_file",
    "path",
    "remove",
    "temp_directory_path",
]


class path:
    """``std::filesystem::path``: text, and the parts of it a program asks for.

    ``p / "name"`` is C++'s ``operator/``: a new path with ``name`` appended, ``name`` absolute
    or the path empty giving ``name`` itself, as the operator does. ``p.string()`` and ``str(p)``
    are the text; ``filename``, ``stem``, ``extension`` and ``parent_path`` read the parts off
    it with :class:`pathlib.PurePath`'s rules, which are C++'s for a POSIX path.
    """

    #: The path as written: what the parts are read off and what ``string()`` gives back.
    text: str

    def __init__(self, text: Any = "") -> None:
        self.text = text.text if isinstance(text, path) else str(text or "")

    def string(self) -> str:
        """``p.string()``: the path as the text it was made of."""
        return self.text

    def c_str(self) -> str:
        """``p.c_str()``: the same text, which this runtime's ``str`` uses."""
        return self.text

    def empty(self) -> bool:
        return not self.text

    def _pure(self) -> pathlib.PurePath:
        return pathlib.PurePath(self.text)

    def filename(self) -> path:
        return path(self._pure().name)

    def stem(self) -> path:
        return path(self._pure().stem)

    def extension(self) -> path:
        return path(self._pure().suffix)

    def parent_path(self) -> path:
        return path(str(self._pure().parent) if self._pure().name else self.text)

    def is_absolute(self) -> bool:
        return self._pure().is_absolute()

    def replace_filename(self, other: Any) -> path:
        """``p.replace_filename(name)``: the same directory, the new name in it."""
        joined: path = self.parent_path() / other
        return joined

    def __truediv__(self, other: Any) -> path:
        other = path(other)
        if not self.text or other.is_absolute():
            return path(other.text)
        return path(self.text.rstrip("/") + "/" + other.text)

    def __eq__(self, other: Any) -> bool:
        return self.text == (other.text if isinstance(other, path) else str(other))

    def __hash__(self) -> int:
        return hash(self.text)

    def __fspath__(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f'<std::filesystem::path "{self.text}">'


def temp_directory_path() -> path:
    """``temp_directory_path()``: the directory temporary files go in, as this machine has it."""
    return path(tempfile.gettempdir())


def current_path() -> path:
    return path(os.getcwd())


def remove(what: Any) -> bool:
    """``remove(p)``: the file or empty directory removed; ``False`` when nothing was.

    C++ throws when the path names nothing, or names a directory with something in it; here the
    answer is the ``False`` the program would have got from asking first, since a program that
    does ask is the common case and the one this exists for.
    """
    try:
        os.remove(str(what))
    except OSError:
        return False
    return True


def exists(what: Any) -> bool:
    return os.path.exists(str(what))


def is_regular_file(what: Any) -> bool:
    return os.path.isfile(str(what))


def is_directory(what: Any) -> bool:
    return os.path.isdir(str(what))


def file_size(what: Any) -> int:
    return os.path.getsize(str(what))


def create_directories(what: Any) -> bool:
    """``create_directories(p)``: every missing directory on the way, ``True`` if any was made."""
    target = pathlib.Path(str(what))
    if target.is_dir():
        return False
    target.mkdir(parents=True, exist_ok=True)
    return True
