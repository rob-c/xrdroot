"""What every test of :mod:`xrdroot.pyroot` starts from: a clean ROOT session.

ROOT's session is global - the current directory, the objects in memory,
the files open, the functions made - so each test gets it emptied before and
after, and runs in a directory of its own for the files it writes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.core import directories, files, hooks


def expect(*checks: tuple[Any, Any]) -> None:
    """Each check a ``(found, wanted)`` pair; every one compared, and every failure reported.

    A test of ROOT's API asks many small questions of one object; one call
    with a pair per question keeps them together as a table, and says all
    that is wrong at once rather than the first.
    """
    failed = [f"check {at}: {found!r} != {wanted!r}" for at, (found, wanted) in enumerate(checks)
              if not found == wanted]  # fmt: skip
    if failed:
        raise AssertionError("; ".join(failed))


def fresh(tmp_path: Any) -> Iterator[None]:
    """Run a test in ``tmp_path`` with nothing open, nothing in memory, nothing drawn."""
    here = os.getcwd()
    installed = hooks._installed[0]
    os.chdir(tmp_path)
    _reset()
    try:
        yield
    finally:
        _reset()
        hooks.set_draw_hook(installed)
        os.chdir(here)


def _reset() -> None:
    for opened in list(files._OPEN):
        opened.Close()
    ROOT.gROOT.Reset()
    ROOT.gROOT.GetListOfFunctions().Clear()
    ROOT.gROOT.GetListOfFiles().Clear()
    ROOT.gROOT.SetBatch(True)
    directories._CURRENT.clear()
    hooks.set_draw_hook(None)
    hooks.DRAWN.clear()
    ROOT.TH1.AddDirectory(True)
    ROOT.TH1.SetDefaultSumw2(False)
    ROOT.gRandom.SetSeed(4357)
    ROOT.__dict__.pop("gErrorIgnoreLevel", None)
    ROOT.gErrorIgnoreLevel = ROOT.kUnset
