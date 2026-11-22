"""``gApplication``: the ``root`` program a macro runs in - ROOT's ``TRint``.

A macro run by ``root`` (``xrdroot root``, ``xrdroot run``) finds ROOT's
application in ``gApplication``: its name, ``TRint``, the command line it was
started with, and ``Terminate(status)``, which ends the program there and then
with that exit status - what roottest's macros call to fail. ``gROOT`` is
called ``Rint`` there, after the application, so ``gDirectory->pwd()`` says
``Rint:/`` as ``root`` does, where PyROOT's says ``PyROOT:/``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

__all__ = ["TApplication", "gApplication", "started", "TRINT", "PROGRAM", "named"]

#: The name ``gROOT`` has while ``root``'s ``TRint`` session runs - a macro typed at the
#: prompt or given on the command line, which TRint renames the session's directory after.
TRINT = "Rint"

#: The name ``gROOT`` has in a program that is not a session at all: ROOT's ``TROOT`` is
#: ``TROOT("root", "ROOT of everything")`` there, and a program reads that name back -
#: ``TEntryList::Enter`` prefixes a sub-list's tree with it, and ``entrylist_enter`` asks
#: for ``"root/t1"``.
PROGRAM = "root"


class TApplication:
    """``TApplication``: the running program, its arguments, and how it ends."""

    def __init__(self, name: str = "TRint", *given: Any) -> None:
        """``TApplication(name, argv)``, or C++'s ``TApplication(name, &argc, argv[, options,
        numOptions])``, whose count goes with the words."""
        argv: Sequence[str] = (given[1] or ()) if len(given) > 1 else (given[0] if given else ())
        self._name = str(name)
        self._argv = [str(word) for word in argv] or ["root.exe"]

    def restart(self, name: str, argv: Sequence[str]) -> None:
        """The application started anew, as ``name`` with the command line ``argv``."""
        self._name = name
        self._argv = list(argv) or ["root.exe"]

    def GetName(self) -> str:
        return self._name

    def ClassName(self) -> str:
        return self._name

    def Argc(self) -> int:
        return len(self._argv)

    def Argv(self, index: int | None = None) -> Any:
        """``Argv(i)``: the ``i``-th word of the command line; all of them without ``i``."""
        return list(self._argv) if index is None else self._argv[index]

    def Terminate(self, status: int = 0) -> None:
        """``Terminate(status)``: end the program now, with ``status``, as ROOT exits."""
        sys.stdout.flush()
        raise SystemExit(int(status))

    def IsRunning(self) -> bool:
        return True

    def SetReturnFromRun(self, ret: bool) -> None:
        """``SetReturnFromRun``: whether ``Run`` returns; there is no event loop to leave."""

    def Run(self, retrn: bool = False) -> None:
        """``Run``: the event loop, which a batch program has nothing to wait in."""

    def ProcessLine(self, line: Any, sync: bool = False, error: Any = None) -> Any:
        from .root import ROOT

        return ROOT.gROOT.ProcessLine(line)

    def InitializeGraphics(self, force: bool = False) -> None:
        """``InitializeGraphics``: there are never windows to open."""


#: ``gApplication``: ``root``'s ``TRint``.
gApplication = TApplication()


@contextmanager
def started(argv: Sequence[str] = ()) -> Iterator[TApplication]:
    """A ``root`` session: ``gApplication`` the ``TRint`` started by ``argv``, and ``gROOT``
    named ``Rint`` after it - each as it was again once the session is over."""
    from ...pyroot import gROOT

    before = gApplication.GetName(), gApplication.Argv(), gROOT.GetName()
    gApplication.restart("TRint", argv)  # the one object every macro has been handed
    gROOT.SetName(TRINT)
    try:
        yield gApplication
    finally:
        gApplication.restart(before[0], before[1])
        gROOT.SetName(before[2])


@contextmanager
def named(name: str) -> Iterator[None]:
    """``gROOT`` called ``name`` while a translated program runs, and as it was afterwards.

    Which name is right depends on what is running, not on this library: a program started
    on its own is the compiled program's kin, so its ``gROOT`` is ROOT's default ``root``,
    while a macro the prompt runs belongs to ``root``'s ``TRint`` session, which renames
    ``gROOT`` to ``Rint`` (:func:`started`). What a macro reads back - ``gDirectory->pwd()``,
    a ``TEntryList``'s sub-list names - follows from the one it ran under.
    """
    from ...pyroot import gROOT

    before = gROOT.GetName()
    gROOT.SetName(name)
    try:
        yield
    finally:
        gROOT.SetName(before)
