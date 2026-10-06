"""``TEnv`` and ``gEnv``: ROOT's resources - names and values, from ``.rootrc`` files and code.

A resource file is lines of ``Name: value``; ``gEnv`` holds the defaults
xrdroot knows of ROOT's ``system.rootrc``, then the user's ``~/.rootrc``,
then the working directory's ``.rootrc``, each over the one before, as
ROOT reads them. ``GetValue(name, default)`` gives the value as the
default's type: an ``int`` asked for of ``yes`` or ``no`` - or ``on``,
``off``, ``true``, ``false`` - is ``1`` or ``0``, as ROOT's ``TEnv`` makes it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .objects import TObject

__all__ = ["TEnv", "TEnvRec", "gEnv", "kEnvGlobal", "kEnvUser", "kEnvLocal", "kEnvChange",
           "kEnvAll"]  # fmt: skip

#: Where a resource was set: ROOT's ``EEnvLevel``.
kEnvGlobal, kEnvUser, kEnvLocal, kEnvChange, kEnvAll = 0, 1, 2, 3, 4

#: The resources of ROOT's ``system.rootrc`` that xrdroot's drawing takes its defaults from.
SYSTEM = {
    "Gui.BackgroundColor": "#e8e8e8",
    "Gui.ForegroundColor": "black",
    "Gui.SelectBackgroundColor": "#d0d0d0",
    "Gui.DocumentBackgroundColor": "white",
    "Canvas.ShowEventStatus": "false",
    "Hist.Binning.1D.x": "100",
    "Root.Stacktrace": "yes",
}

#: The words an ``int`` resource can be written as.
TRUTHS = {"yes": 1, "on": 1, "true": 1, "no": 0, "off": 0, "false": 0}


class TEnvRec(TObject):
    """``TEnvRec``: one resource - its name, its value as written, and where it was set."""

    def __init__(self, name: str, value: str, level: int) -> None:
        super().__init__()
        self.fName, self.fValue, self.fLevel = name, value, level

    def GetName(self) -> str:
        return self.fName

    def GetValue(self) -> str:
        return self.fValue

    def GetLevel(self) -> int:
        return self.fLevel


def _typed(text: str, default: Any) -> Any:
    """A resource's text as the type of the default asked with."""
    if isinstance(default, int):  # a bool too
        word = text.strip().lower()
        return TRUTHS[word] if word in TRUTHS else int(float(word))
    if isinstance(default, float):
        return float(text)
    return text


class TEnv(TObject):
    """``TEnv``: resources by name, read from a resource file or set in code."""

    def __init__(self, name: Any = "") -> None:
        super().__init__()
        self._records: dict[str, TEnvRec] = {}
        self._name = str(name)
        if self._name:
            self.ReadFile(self._name, kEnvLocal)

    def GetRcName(self) -> str:
        return self._name

    def ReadFile(self, fname: Any, level: Any = kEnvLocal) -> int:
        """``ReadFile``: each ``Name: value`` line of the file set; ``-1`` if it cannot be read."""
        path = Path(str(fname)).expanduser()
        if not path.is_file():
            return -1
        for line in path.read_text(errors="replace").splitlines():
            name, colon, value = line.partition(":")
            if colon and name.strip() and not name.lstrip().startswith("#"):
                self.SetValue(name.strip(), value.strip(), level)
        return 0

    def SetValue(self, name: Any, value: Any, level: Any = kEnvChange, type: Any = None) -> None:
        """``SetValue(name, value)``: the resource set, as text; a ``name: value`` one alike."""
        text = str(name)
        if value is None and ":" in text:
            text, _, value = text.partition(":")
        written = ("yes" if value else "no") if isinstance(value, bool) else str(value).strip()
        self._records[text.strip()] = TEnvRec(text.strip(), written, int(level))

    def GetValue(self, name: Any, default: Any = "") -> Any:
        """``GetValue(name, default)``: the resource as the default's type, or the default."""
        found = self._records.get(str(name))
        if found is None:
            return default
        return _typed(found.GetValue(), default)

    def Defined(self, name: Any) -> bool:
        return str(name) in self._records

    def Lookup(self, name: Any) -> TEnvRec | None:
        return self._records.get(str(name))

    def GetTable(self) -> list[TEnvRec]:
        return list(self._records.values())

    def Print(self, option: Any = "") -> None:
        """``Print``: each resource, its value, and where it was set, as ROOT lists them."""
        levels = ("Global", "User", "Local", "Changed", "All")
        for record in self._records.values():
            print(f"{record.fName + ':':<30} {record.fValue:<20} [{levels[record.fLevel]}]")

    def WriteFile(self, fname: Any, level: Any = kEnvAll) -> None:
        """``WriteFile``: the resources of that level - every one for ``kEnvAll`` - to a file."""
        kept = [r for r in self._records.values() if int(level) in (kEnvAll, r.fLevel)]
        Path(str(fname)).write_text("".join(f"{r.fName}: {r.fValue}\n" for r in kept))


def _global() -> TEnv:
    """``gEnv``: ROOT's defaults, then the user's ``.rootrc``, then the working directory's."""
    made = TEnv()
    for name, value in SYSTEM.items():
        made.SetValue(name, value, kEnvGlobal)
    made.ReadFile(Path(os.path.expanduser("~")) / ".rootrc", kEnvUser)
    made.ReadFile(".rootrc", kEnvLocal)
    return made


#: ``gEnv``.
gEnv = _global()
