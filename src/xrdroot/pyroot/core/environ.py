"""``TEnv`` and ``gEnv``: ROOT's resources - names and values, from ``.rootrc`` files and code.

A resource file is lines of ``Name: value``; ``gEnv`` holds the defaults
xrdroot knows of ROOT's ``system.rootrc``, then the user's ``~/.rootrc``,
then the working directory's ``.rootrc``, each over the one before, as
ROOT reads them. ``GetValue(name, default)`` gives the value as the
default's type: an ``int`` asked for of ``yes`` or ``no`` - or ``on``,
``off``, ``true``, ``false`` - is ``1`` or ``0``, as ROOT's ``TEnv`` makes it.

``TEnv(name)`` reads the user's file of that name - in ``$ROOTENV_USER_PATH``
if it is set, else the home directory - then the working directory's, and
``SaveLevel`` writes one level's resources back to its file, keeping the
old one as ``.bak``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .objects import TObject

__all__ = ["TEnv", "TEnvRec", "gEnv", "PyConfig", "kEnvGlobal", "kEnvUser", "kEnvLocal",
           "kEnvChange", "kEnvAll"]  # fmt: skip

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
        #: Set in code since it was last saved, rather than read from a file.
        self.fModified = True

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
            self.ReadFile(_user_file(self._name), kEnvUser)
            if Path.cwd() != Path(os.path.expanduser("~")):  # else it is the user's file
                self.ReadFile(self._name, kEnvLocal)

    def GetRcName(self) -> str:
        return self._name

    def SaveLevel(self, level: Any) -> None:
        """``SaveLevel``: the resources set in code since, at ``level``, written to its file.

        Resources set in code with no level take the one they are saved at.
        The file's other lines are kept, and the file as it was is renamed
        ``.bak``.
        """
        if not self._name or int(level) not in (kEnvUser, kEnvLocal):
            return
        target = _user_file(self._name) if int(level) == kEnvUser else Path(self._name)
        saved = self._changed_at(int(level))
        _rewrite(target, saved)
        for record in saved:
            record.fModified = False

    def _changed_at(self, level: int) -> list[TEnvRec]:
        """The resources set in code at ``level``, those set with none given it now."""
        changed = [record for record in self._records.values() if record.fModified]
        for record in changed:
            if record.fLevel == kEnvChange:
                record.fLevel = level
        return [record for record in changed if record.fLevel == level]

    def ReadFile(self, fname: Any, level: Any = kEnvLocal) -> int:
        """``ReadFile``: each ``Name: value`` line of the file set; ``-1`` if it cannot be read."""
        path = Path(str(fname)).expanduser()
        if not path.is_file():
            return -1
        for line in path.read_text(errors="replace").splitlines():
            name, colon, value = line.partition(":")
            if colon and name.strip() and not name.lstrip().startswith("#"):
                self.SetValue(name.strip(), value.strip(), level)
                self._records[name.strip()].fModified = False
        return 0

    def SetValue(self, name: Any, value: Any = None, level: Any = kEnvChange,
                 type: Any = None) -> None:
        """``SetValue(name, value)``: the resource set, as text.

        Given a name alone, ``name=value`` or ``name: value`` is split, and a
        bare name is set to ``1``, as ROOT's one-argument ``SetValue`` does.
        """
        text = str(name)
        if value is None:
            text, _, value = text.partition("=" if "=" in text else ":")
            value = value or "1"
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


def _user_file(name: str) -> Path:
    """Where a user's resource file of that name is: ``$ROOTENV_USER_PATH``, else home."""
    return Path(os.environ.get("ROOTENV_USER_PATH") or os.path.expanduser("~")) / name


def _rewrite(target: Path, saved: list[TEnvRec]) -> None:
    """``target`` with ``saved`` written over its lines of the same names, the old one ``.bak``."""
    names = {record.fName for record in saved}
    kept = [line for line in _lines(target) if line.partition(":")[0].strip() not in names]
    target.touch()
    written = [*kept, *(f"{r.fName + ':':<40} {r.fValue}" for r in saved)]
    Path(f"{target}.new").write_text("".join(line + "\n" for line in written))
    os.replace(target, f"{target}.bak")
    os.replace(f"{target}.new", target)


def _lines(path: Path) -> list[str]:
    """A file's lines, or none if there is no such file."""
    return path.read_text(errors="replace").splitlines() if path.is_file() else []


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


class _PyConfig:
    """``ROOT.PyConfig``: the switches a PyROOT script sets before ROOT looks at them.

    They are plain flags ROOT reads at start-up - whether to take the command
    line for itself, whether to run the logon macros, whether to install its
    own ``sys.excepthook``. A library that never starts a ROOT application has
    nothing to do with them, so they are kept and reported back, which is what
    a script that sets them needs.
    """

    #: Take ``sys.argv``'s ROOT options at start-up - on, which is ROOT's own default:
    #: PyROOT takes ``-b``, ``-l``, ``-q`` and the rest off a script's command line.
    IgnoreCommandLineOptions = True
    #: Run ``~/.rootlogon.C`` - off for the same reason.
    DisableRootLogon = False
    #: Install ROOT's exception hook - off; this library raises Python exceptions.
    DisablePythonExceptionHandler = False
    #: Whether ROOT's own ``TPython`` is in charge - it is not here.
    EnableTPython = False
    #: Root's banner and start-up message - there is none to print.
    StartGuiThread = False

    def __repr__(self) -> str:
        return "<ROOT.PyConfig>"


#: ``ROOT.PyConfig``: the switches, as ROOT exposes them.
PyConfig = _PyConfig()
