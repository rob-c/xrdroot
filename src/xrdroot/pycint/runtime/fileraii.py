"""ROOT's ``TestSupport`` file guard: ``ROOT::TestSupport::FileRaii``, a scratch file that goes.

ROOT's unit tests make their scratch files with ``FileRaii``: the guard records
the path, the test writes the file, and the file is removed as the guard's scope
ends, however the test ends. **Why** the tests do it that way is that a checking
run leaves nothing behind and a failing test cannot make the next one read a
stale file.

The translation has a scope for the guard, which the C++ dialect asks for by
name (:attr:`xrdroot.pycint.dialect.ROOT.guards`): a local declared as a
``FileRaii`` is destroyed as its block ends - :meth:`FileRaii._destruct` is what
the emitted ``finally`` calls. A guard the test leaves alive - one declared
where no scope is written for it, or kept through a test that crashed - is
emptied with the test, by :func:`remove_all`.

ROOT removes with ``std::remove``, which says nothing about a file that was
never made or is already gone: the tests rely on that, since many guard a file
they expect ROOT to refuse to create.
"""

from __future__ import annotations

import os
from typing import Any

from ...pycxx.runtime.gtest.runner import AFTER_TEST

__all__ = ["FileRaii", "remove_all"]

#: The guards whose file is still theirs to remove: their scope has not ended yet.
LIVE: list[FileRaii] = []


class FileRaii:
    """``ROOT::TestSupport::FileRaii``: a file removed when the guard goes out of scope."""

    def __init__(self, path: Any = "") -> None:
        self.path = str(path)
        self.preserved = False
        LIVE.append(self)

    def GetPath(self) -> str:
        """The guarded file's path, which the test opens, writes and reads."""
        return self.path

    def PreserveFile(self) -> None:
        """Keep the file when the guard ends - ROOT offers this for debugging a test by hand."""
        self.preserved = True

    def remove(self) -> None:
        """Remove the guarded file unless it was preserved: no error if it is not there."""
        if self.preserved:
            return
        try:
            os.remove(self.path)
        except OSError:  # never made, or already gone - as std::remove lets be
            pass

    def _destruct(self) -> None:
        """The guard's scope ends: its file goes with it."""
        if self in LIVE:
            LIVE.remove(self)
        self.remove()


def remove_all() -> None:
    """End every guard still alive, innermost first: the test they were made in has ended."""
    for guard in reversed(LIVE.copy()):
        guard._destruct()


AFTER_TEST.append(remove_all)
