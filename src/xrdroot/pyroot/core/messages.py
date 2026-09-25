"""``Info``, ``Warning``, ``Error``: ROOT's messages, to standard error, as ROOT words them.

``Warning in <TROOT::Append>: Replacing existing TH1: h (Potential memory leak).``
is one line, of a level, a place and ``printf``-formatted text, and a level
below ``gErrorIgnoreLevel`` is not printed at all. ``ROOT.gErrorIgnoreLevel``
is read where the message is made, so a script setting it is obeyed; a
``Fatal`` message ends the program, as ROOT's aborts it.
"""

from __future__ import annotations

import sys
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
    where = f" in <{location}>" if location else ""
    print(f"{kind}{where}: {text}", file=sys.stderr)
    if level >= kFatal:
        raise SystemExit(1)


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
