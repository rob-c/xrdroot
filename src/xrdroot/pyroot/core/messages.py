"""``Info``, ``Warning``, ``Error``: ROOT's messages, to standard error, as ROOT words them.

``Warning in <TROOT::Append>: Replacing existing TH1: h (Potential memory leak).``
is one line, of a level, a place and ``printf``-formatted text, and a level
below ``gErrorIgnoreLevel`` is not printed at all. ``ROOT.gErrorIgnoreLevel``
is read where the message is made, so a script setting it is obeyed; a
``Fatal`` message ends the program, as ROOT's aborts it.

Run as PyROOT runs a script (:data:`PYROOT_SCRIPT`), a ``Warning`` is
instead Python's ``RuntimeWarning``, its place the warning's file name and 0
its line - as PyROOT's error handler raises it - so ``assertWarns`` and
``-W error`` see it; and :func:`from_script` tells a class whether Python
called it there, to answer as PyROOT's pythonizations do (``TFile`` raising
``OSError``) rather than as C++ sees it (a zombie).
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

from .cformat import c_format

__all__ = [
    "Info",
    "Warning",
    "Error",
    "SysError",
    "Break",
    "Fatal",
    "kUnset",
    "kPrint",
    "kInfo",
    "kWarning",
    "kError",
    "kBreak",
    "kSysError",
    "kFatal",
    "gErrorIgnoreLevel",
]

kUnset = -1
kPrint = 0
kInfo = 1000
kWarning = 2000
kError = 3000
kBreak = 4000
kSysError = 5000
kFatal = 6000

#: ``gErrorIgnoreLevel``: messages below this level are not printed; a script
#: sets it on the namespace, where :func:`ignore_level` reads it.
gErrorIgnoreLevel = kUnset

#: Whether code runs as PyROOT runs a script - ``xrdroot run script.py``, the suite's
#: shim - where a ``Warning`` is Python's ``RuntimeWarning``.
PYROOT_SCRIPT = False

#: What the code translated from C++ is filed under, which no Python script is.
TRANSLATED = "<translation of "

#: Each message's level, by the word it starts with.
LEVELS = {
    "Info": kInfo,
    "Warning": kWarning,
    "Error": kError,
    "Break": kBreak,
    "SysError": kSysError,
    "Fatal": kFatal,
}


def ignore_level() -> int:
    """``gErrorIgnoreLevel`` as the script left it on the namespace, or ROOT's default."""
    namespace = sys.modules.get("xrdroot.pyroot")
    level = getattr(namespace, "__dict__", {}).get("gErrorIgnoreLevel", kUnset)
    return int(level)


def message(kind: str, location: Any, fmt: Any, *args: Any) -> None:
    """One of ROOT's messages, unless its level is below the one ignored."""
    level = LEVELS[kind]
    if level < ignore_level():
        return
    text = c_format(str(fmt), *args) if args else str(fmt)
    if level == kWarning and PYROOT_SCRIPT:
        warnings.warn_explicit(text, RuntimeWarning, str(location or ""), 0)
        return
    where = f" in <{location}>" if location else ""
    sys.stdout.flush()  # what was printed before comes before, as ROOT's own does
    print(f"{kind}{where}: {text}", file=sys.stderr)
    if level >= kFatal:
        raise SystemExit(1)


def from_script(depth: int = 2) -> bool:
    """Was the caller ``depth`` frames up a PyROOT script's Python, not C++ translated?

    Frames of this package between are passed over, so a class's own helpers
    do not count as the caller.
    """
    if not PYROOT_SCRIPT:
        return False
    frame = sys._getframe(depth)
    package = __name__.split(".")[0] + "."  # xrdroot.
    while _ours(frame, package) and frame.f_back is not None:
        frame = frame.f_back
    return not frame.f_code.co_filename.startswith(TRANSLATED)


def _ours(frame: Any, package: str) -> bool:
    """Is ``frame`` this package's own code, rather than a script's or C++ translated?"""
    translated = frame.f_code.co_filename.startswith(TRANSLATED)
    return not translated and str(frame.f_globals.get("__name__", "")).startswith(package)


def Info(location: Any, fmt: Any, *args: Any) -> None:
    """``Info(location, fmt, ...)``."""
    message("Info", location, fmt, *args)


def Warning(location: Any, fmt: Any, *args: Any) -> None:
    """``Warning(location, fmt, ...)``."""
    message("Warning", location, fmt, *args)


def Error(location: Any, fmt: Any, *args: Any) -> None:
    """``Error(location, fmt, ...)``."""
    message("Error", location, fmt, *args)


def SysError(location: Any, fmt: Any, *args: Any) -> None:
    """``SysError(location, fmt, ...)``."""
    message("SysError", location, fmt, *args)


def Break(location: Any, fmt: Any, *args: Any) -> None:
    """``Break(location, fmt, ...)``."""
    message("Break", location, fmt, *args)


def Fatal(location: Any, fmt: Any, *args: Any) -> None:
    """``Fatal(location, fmt, ...)``: the message, and the end of the program."""
    message("Fatal", location, fmt, *args)
