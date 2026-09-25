"""``gROOT``: ROOT's session - memory, the lists of what is open, and batch mode.

``gROOT`` is the top directory: what a script makes before opening a file is
kept there, and it keeps the lists ROOT keeps - the files open, the canvases,
the functions made, the styles - so ``FindObject`` finds by name what a
script made. Beneath it is :data:`xrdroot.session.gROOT`, the session the
rest of this library shares, reached as ``gROOT._xrd``.

Running C++ - ``ProcessLine``, ``Macro``, ``LoadMacro`` - is the translator's
business, :mod:`xrdroot.cint`, and it is handed there when it is installed;
without it each is refused by name. Everything runs as ROOT's ``-b`` does:
batch mode, no windows, whatever ``SetBatch`` was told.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Any, Callable

from ...errors import UnsupportedFeatureError
from .collections import TList
from .directories import TDirectory, current_directory, set_current
from .messages import message
from .strings import TString

__all__ = ["TROOT", "gROOT", "gInterpreter", "TInterpreter", "set_line_processor"]

#: ROOT's release the namespace follows - the oracle it is checked against - two ways more.
RELEASE = "6.40.04"
VERSION_INT = 64004
VERSION_CODE = (6 << 16) + (40 << 8) + 4

#: Whatever runs a line of C++ once the translator installs it with
#: :func:`set_line_processor`.
_PROCESSOR: list[Callable[..., Any]] = []


def set_line_processor(fn: Callable[..., Any] | None) -> None:
    """Install what ``ProcessLine`` runs a line of C++ with; ``None`` takes it out."""
    _PROCESSOR[:] = [] if fn is None else [fn]


def _translator(name: str) -> Any:
    """The translator's function ``name`` - at its top or in ``execute`` - or ``None``."""
    if importlib.util.find_spec("xrdroot.cint") is None:
        return None
    found = getattr(importlib.import_module("xrdroot.cint"), name, None)
    if found is None and importlib.util.find_spec("xrdroot.cint.execute") is not None:
        found = getattr(importlib.import_module("xrdroot.cint.execute"), name, None)
    return found


def _processor(what: str) -> Callable[..., Any]:
    """The translator's ``process_line``, found or loaded, or a refusal naming it."""
    if not _PROCESSOR:
        found = _translator("process_line")
        if found is not None:
            set_line_processor(found)
    if not _PROCESSOR:
        raise UnsupportedFeatureError(
            f"{what} runs C++, and xrdroot.cint - the C++ translator that would run it - is "
            f"not installed"
        )
    return _PROCESSOR[0]


def _graphics(*path: str) -> Any:
    """What the graphics part of the kit has at ``path`` - or ``None`` without it.

    Each step is an attribute, or a submodule of that name imported when the
    package has not imported it itself.
    """
    name = "xrdroot.pyroot.graphics"
    if importlib.util.find_spec(name) is None:
        return None
    found: Any = importlib.import_module(name)
    for part in path:
        found = getattr(found, part, None) or _submodule(f"{name}.{part}")
        name = f"{name}.{part}"
        if found is None:
            return None
    return found


def _submodule(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


class TROOT(TDirectory):
    """``TROOT``: the session, and the directory at the top of every other."""

    def __init__(self, name: str = "PyROOT", title: str = "The ROOT of EVERYTHING") -> None:
        super().__init__(name, title)
        from ... import session

        #: The xrdroot session beneath.
        self._xrd = session.gROOT
        self._batch = True
        self._lists = {
            key: TList()
            for key in (
                "Files", "Canvases", "Functions", "Styles", "Specials", "Globals",
                "Browsers", "Geometries", "Colors", "Types", "Cleanups", "Tasks",
                "ClosedObjects", "DataSets", "MappedFiles", "Sockets",
            )
        }  # fmt: skip
        self._macro_path = "."

    # -- the lists -------------------------------------------------------------------

    def _listed(self, key: str) -> TList:
        return self._lists[key]

    def GetListOfFiles(self) -> TList:
        return self._listed("Files")

    def GetListOfCanvases(self) -> TList:
        """``GetListOfCanvases``: the graphics' canvases, when the graphics are installed."""
        canvases = _graphics("pads", "CANVASES")
        if canvases is None:
            return self._listed("Canvases")
        made = TList()
        for canvas in canvases:
            made.Add(canvas)
        return made

    def MakeDefCanvas(self) -> Any:
        """``MakeDefCanvas``: the graphics' default canvas, ``c1``."""
        maker = _graphics("canvas", "default_canvas")
        if maker is None:
            raise UnsupportedFeatureError(
                "gROOT.MakeDefCanvas makes a canvas, and xrdroot.pyroot.graphics - which draws "
                "them - is not installed"
            )
        return maker()

    def GetListOfFunctions(self) -> TList:
        return self._listed("Functions")

    def GetListOfStyles(self) -> TList:
        return self._listed("Styles")

    def GetListOfSpecials(self) -> TList:
        return self._listed("Specials")

    def GetListOfGlobals(self, load: bool = False) -> TList:
        return self._listed("Globals")

    def GetListOfBrowsers(self) -> TList:
        return self._listed("Browsers")

    def GetListOfGeometries(self) -> TList:
        return self._listed("Geometries")

    def GetListOfColors(self) -> TList:
        return self._listed("Colors")

    def GetListOfTypes(self, load: bool = False) -> TList:
        return self._listed("Types")

    def GetListOfCleanups(self) -> TList:
        return self._listed("Cleanups")

    def GetListOfClosedObjects(self) -> TList:
        return self._listed("ClosedObjects")

    def GetListOfDataSets(self) -> TList:
        return self._listed("DataSets")

    def GetListOfMappedFiles(self) -> TList:
        return self._listed("MappedFiles")

    def GetListOfSockets(self) -> TList:
        return self._listed("Sockets")

    def GetListOfTasks(self) -> TList:
        return self._listed("Tasks")

    def FindObject(self, name: Any) -> Any:
        """``FindObject``: the open files, the functions, canvases, styles, then ``gDirectory``."""
        if not isinstance(name, str):
            return self._list.FindObject(name)
        for key in ("Files", "MappedFiles", "Functions", "Geometries", "Canvases", "Styles",
                    "Specials"):  # fmt: skip
            found = self._lists[key].FindObject(name)
            if found is not None:
                return found
        here = current_directory()
        found = here.Get(name) if here is not self else self._list.FindObject(name)
        anywhere = _graphics("pads", "find_anywhere")
        return found if found is not None or anywhere is None else anywhere(name)

    def FindObjectAny(self, name: Any) -> Any:
        """``FindObjectAny``: as ``FindObject``, then in every directory in memory below."""
        found = self.FindObject(name)
        return found if found is not None else TDirectory.FindObjectAny(self, name)

    def GetFunction(self, name: Any) -> Any:
        """``GetFunction``: a function made by name - ``gaus`` and the rest made when asked."""
        found = self.GetListOfFunctions().FindObject(str(name))
        if found is None:
            from .funcs import standard_function

            found = standard_function(str(name))
        return found

    def GetStyle(self, name: Any) -> Any:
        """``GetStyle``: the graphics' style of that name - or one listed here."""
        getter = _graphics("get_style")
        return self.GetListOfStyles().FindObject(str(name)) if getter is None else getter(str(name))

    def SetStyle(self, name: Any = "Default") -> None:
        """``SetStyle``: make the style called ``name`` current - by the graphics, when there."""
        setter = _graphics("set_style")
        if setter is not None:
            setter(str(name))
            return
        found = self.GetStyle(name)
        if found is None:
            message("Error", "TROOT::SetStyle", "Unknown style:%s", str(name))
            return
        found.cd()

    def ForceStyle(self, force: bool = True) -> None:
        """``ForceStyle``: whether objects read are drawn in the current style; noted."""
        self._forced = bool(force)

    def GetForceStyle(self) -> bool:
        return bool(getattr(self, "_forced", False))

    # -- the session ----------------------------------------------------------------------

    def SetBatch(self, batch: bool = True) -> None:
        self._batch = bool(batch)

    def IsBatch(self) -> bool:
        return self._batch

    def SetWebDisplay(self, where: Any = "") -> None:
        """``SetWebDisplay``: there is no browser to display in."""

    def IsWebDisplay(self) -> bool:
        return False

    def GetVersion(self) -> str:
        """``GetVersion``: ROOT's release this namespace follows, ``6.40.04``."""
        return RELEASE

    def GetVersionInt(self) -> int:
        return VERSION_INT

    def GetVersionCode(self) -> int:
        return VERSION_CODE

    def GetGitCommit(self) -> str:
        return ""

    def GetConfigFeatures(self) -> str:
        """``GetConfigFeatures``: the features built in - this kit's own Python ones."""
        return "pyroot"

    def GetTutorialDir(self) -> TString:
        """``GetTutorialDir``: ``$ROOT_TUTORIAL_DIR``, else ``$ROOTSYS/tutorials``."""
        found = os.environ.get("ROOT_TUTORIAL_DIR")
        if found is None:
            found = os.path.join(os.environ.get("ROOTSYS", "."), "tutorials")
        return TString(found)

    def GetTutorialsDir(self) -> TString:
        return self.GetTutorialDir()

    def GetMacroPath(self) -> str:
        return self._macro_path

    def SetMacroPath(self, path: Any) -> None:
        self._macro_path = str(path)

    def GetDirLevel(self) -> int:
        from .objects import Indent

        return Indent.level

    def Reset(self, option: str = "") -> None:
        """``Reset``: forget what is in memory, and go back to it."""
        self.Clear()
        set_current(self)

    def Time(self, casetime: int = 1) -> None:
        """``Time``: timing each prompt command, which there is no prompt for."""

    def GetFile(self) -> Any:
        return None

    def GetPath(self) -> str:
        return f"{self.GetName()}:/"

    def GetApplication(self) -> Any:
        return None

    def IsInterrupted(self) -> bool:
        return False

    def GetSelectedPad(self) -> Any:
        return None

    def RefreshBrowsers(self) -> None:
        """``RefreshBrowsers``: there are none."""

    def CloseFiles(self) -> None:
        """``CloseFiles``: close every file still open."""
        for opened in list(self.GetListOfFiles()):
            opened.Close()

    def EndOfProcessCleanups(self) -> None:
        self.CloseFiles()

    # -- C++ -------------------------------------------------------------------------------

    def ProcessLine(self, line: Any, error: Any = None) -> Any:
        """``ProcessLine``: a line of C++, run by :mod:`xrdroot.cint`."""
        return _processor(f"gROOT.ProcessLine({str(line)!r})")(str(line))

    def ProcessLineSync(self, line: Any, error: Any = None) -> Any:
        return self.ProcessLine(line, error)

    def ProcessLineFast(self, line: Any, error: Any = None) -> Any:
        return self.ProcessLine(line, error)

    def Macro(self, filename: Any, error: Any = None, padUpdate: bool = True) -> Any:
        """``Macro``: ``.x filename``, run by :mod:`xrdroot.cint`."""
        return self.ProcessLine(f".x {filename}")

    def LoadMacro(self, filename: Any, error: Any = None, check: bool = False) -> int:
        """``LoadMacro``: ``.L filename``, loaded by :mod:`xrdroot.cint`."""
        self.ProcessLine(f".L {filename}")
        return 0


#: ``gROOT``.
gROOT = TROOT()


class TInterpreter:
    """``gInterpreter``: C++ declared and run - by :mod:`xrdroot.cint`, when it is there."""

    def Declare(self, code: Any) -> bool:
        """``Declare``: C++ declarations, translated, and what they declare put in the namespace.

        As PyROOT makes ``ROOT.f`` of a function declared, the functions and
        classes the translation defines become names of ``xrdroot.pyroot``.
        """
        running = _translator("run_source")
        if running is None:
            _processor("gInterpreter.Declare")(str(code))
            return True
        declared = running(str(code), "<Declare>", call=False)
        namespace = sys.modules["xrdroot.pyroot"].__dict__
        for name, value in declared.items():
            if getattr(value, "__module__", None) == "__cint__" and not name.startswith("_"):
                namespace[name] = value
        return True

    def ProcessLine(self, line: Any, error: Any = None) -> Any:
        return gROOT.ProcessLine(line, error)

    def Calc(self, line: Any, error: Any = None) -> Any:
        return gROOT.ProcessLine(line, error)

    def Load(self, filename: Any, system: bool = False) -> int:
        return 0

    def AddIncludePath(self, path: Any) -> None:
        """``AddIncludePath``: nothing is compiled, so there is nothing to include."""

    def GenerateDictionary(self, classes: Any, headers: Any = "", *rest: Any) -> int:
        return 0


#: ``gInterpreter``.
gInterpreter = TInterpreter()
