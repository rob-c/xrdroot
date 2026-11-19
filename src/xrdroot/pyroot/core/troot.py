"""``gROOT``: ROOT's session - memory, the lists of what is open, and batch mode.

``gROOT`` is the top directory: what a script makes before opening a file is
kept there, and it keeps the lists ROOT keeps - the files open, the canvases,
the functions made, the styles - so ``FindObject`` finds by name what a
script made. Beneath it is :data:`xrdroot.session.gROOT`, the session the
rest of this library shares, reached as ``gROOT._xrd``.

Running C++ - ``ProcessLine``, ``Macro``, ``LoadMacro`` - is the translator's
business, :mod:`xrdroot.pycint`, reached through the ``interpreter.*``
providers of :mod:`xrdroot.plugins` when it is installed - it is built on
this namespace, not under it; without it each is refused by name. Everything runs as
ROOT's ``-b`` does:
batch mode, no windows, whatever ``SetBatch`` was told.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import warnings
from collections.abc import Callable
from typing import Any

from ... import plugins
from ...errors import UnsupportedFeatureError
from .collections import THashList, TList
from .directories import TDirectory, current_directory, set_current
from .messages import message
from .objects import TNamed, TObject
from .strings import TString

__all__ = ["TROOT", "gROOT", "gDebug", "gInterpreter", "gObjectTable", "TInterpreter",
           "set_line_processor", "SetHeuristicMemoryPolicy", "TEnum", "TEnumConstant",
           "TListOfEnums", "class_enums", "TListOfEnumsWithLock", "find_enum", "note_enum"]

#: ``gObjectTable``: ROOT's count of objects by class - which is not kept here, so it is null,
#: and a macro testing it skips it.
gObjectTable = None

#: ``gDebug``: ROOT's debug level, which a macro sets to have ROOT print more of what it does.
#: Nothing here prints more for it, but the name is set and read back, which is all a macro
#: that raises it around its own work asks for.
gDebug = 0

#: ROOT's release the namespace follows - the oracle it is checked against - two ways more.
RELEASE = "6.40.04"
VERSION_INT = 64004
VERSION_CODE = (6 << 16) + (40 << 8) + 4
#: ``TROOT::GetVersionDate``: the day 6.40.04 was released, as ROOT writes it - YYYYMMDD,
#: which is why a macro compares it with a number (``< 20030406`` means "older than 2003").
VERSION_DATE = 20250417
#: ``TROOT::GetVersionTime``: the release's time of day, as ROOT writes it - HHMMSS. ROOT's
#: own is the compiler's ``__TIME__``, which for the 6.40.04 build this follows is 0.
VERSION_TIME = 0

#: Whatever runs a line of C++ once the translator installs it with
#: :func:`set_line_processor`.
_PROCESSOR: list[Callable[..., Any]] = []


def set_line_processor(fn: Callable[..., Any] | None) -> None:
    """Install what ``ProcessLine`` runs a line of C++ with; ``None`` takes it out."""
    _PROCESSOR[:] = [] if fn is None else [fn]


#: The package the translator is, whose absence refuses C++ by name.
_TRANSLATOR = "xrdroot.pycint"

#: The translator's functions this namespace calls, each found through its provider.
_PROVIDED: dict[str, Callable[[], Any]] = {
    "process_line": lambda: plugins.lookup("interpreter.process_line"),
    "run_source": lambda: plugins.lookup("interpreter.run_source"),
    "load_library": lambda: plugins.lookup("interpreter.load_library"),
    "loaded_libraries": lambda: plugins.lookup("interpreter.loaded_libraries"),
    "unload_library": lambda: plugins.lookup("interpreter.unload_library"),
    "compile_macro": lambda: plugins.lookup("interpreter.compile_macro"),
    "autoload": lambda: plugins.lookup("interpreter.autoload"),
}


def _translator(name: str) -> Any:
    """The translator's function ``name``, through its provider - or ``None`` without it."""
    try:
        return _PROVIDED[name]()
    except AttributeError:  # a translator that has no such function
        return None
    except ModuleNotFoundError as why:  # no translator - but not one failing to load
        if not str(why.name).startswith(_TRANSLATOR):
            raise
        return None


def _processor(what: str) -> Callable[..., Any]:
    """The translator's ``process_line``, found or loaded, or a refusal naming it."""
    if not _PROCESSOR:
        found = _translator("process_line")
        if found is not None:
            set_line_processor(found)
    if not _PROCESSOR:
        raise UnsupportedFeatureError(
            f"{what} runs C++, and xrdroot.pycint - the C++ translator that would run it - is "
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


#: The names every translation starts with - its runtime's - which a declaration does not add.
_PRELUDE: dict[str, Any] = {}


# -- the interpreter's enumerations ---------------------------------------------------

#: ``kInt_t`` in ROOT's ``EDataType``: what ``TEnum::GetUnderlyingType`` answers with for an
#: enumeration declared with no integer type of its own, which C++ gives ``int`` when its
#: values fit.
K_INT = 3

#: The integer types an enumeration is declared with - ``enum E : short`` - in ROOT's
#: ``EDataType`` codes, as the 6.40.04 oracle answers them: ``char`` is 1, ``short`` 2,
#: ``int`` 3, ``long`` 4. A type not here is answered as ``int``, which is C++'s default.
UNDERLYING_TYPES = {"char": 1, "short": 2, "int": 3, "long": 4}

#: ``gROOT``'s ``fEnums``: every enumeration the interpreter has declared so far, by
#: qualified name, in the order declared. ROOT keeps this table in ``TROOT`` because an
#: enumeration is the interpreter's, not a dictionary's: what a class asks for with
#: ``TClass::GetListOfEnums`` is the part of it that class owns.
_ENUMS: dict[str, TEnum] = {}


class TEnumConstant(TObject):
    """``TEnumConstant``: one enumerator - its name and its value - as a ``TEnum`` holds it."""

    def __init__(self, name: str, value: int, enum: TEnum | None = None) -> None:
        super().__init__()
        self._constant = str(name)
        self._value = int(value)
        self._enum = enum

    def GetName(self) -> str:
        return self._constant

    def GetTitle(self) -> str:
        """``GetTitle``: the enum's title, which ROOT gives each of its constants too."""
        return "" if self._enum is None else self._enum.GetTitle()

    def GetValue(self) -> int:
        return self._value

    def GetEnum(self) -> TEnum | None:
        return self._enum


class TEnum(TNamed):
    """``TEnum``: one enumeration of the interpreter's - its name, the comment ROOT keeps as
    its title, and its constants in the order written.

    A ``TEnum`` is made by :func:`note_enum` as the C++ declaring it is run, and kept in
    :data:`_ENUMS`; ``TEnum::GetEnum`` and ``gROOT.GetListOfEnums`` are how a program finds
    it again.
    """

    def __init__(self, name: str, items: Any = (), title: str = "", underlying: int = K_INT,
                 scoped: bool = False, owner: str = "") -> None:  # fmt: skip
        super().__init__(str(name), str(title))
        self._items = [(str(item), int(value)) for item, value in items]
        self._underlying = int(underlying)
        self._scoped = bool(scoped)
        self._owner = str(owner)

    def GetConstants(self, load: bool = True) -> THashList:
        """``GetConstants``: the constants, in the ``THashList`` ROOT keeps them in.

        The list is made once and then handed back, so a constant a program is holding is
        still the one the enum answers with after the enum has been looked up again - which
        is what roottest's ``execEnumsTest`` compares addresses for.
        """
        held: THashList | None = self.__dict__.get("_constants")
        if held is None:
            held = THashList()
            for item, value in self._items:
                held.Add(TEnumConstant(item, value, self))
            self.__dict__["_constants"] = held
        return held

    def GetConstant(self, name: Any) -> TEnumConstant | None:
        """``GetConstant``: the constant of that name, ``nullptr`` if the enum has none."""
        found = self.GetConstants().FindObject(str(name))
        return found if isinstance(found, TEnumConstant) else None

    def GetQualifiedName(self) -> str:
        """``GetQualifiedName``: the name as C++ writes it - ``N::namespaceEnum``, else the
        name alone for an enumeration of the global namespace."""
        return f"{self._owner}::{self._name}" if self._owner else self._name

    def GetUnderlyingType(self) -> int:
        """``GetUnderlyingType``: the ``EDataType`` code of the integer type the enum was
        declared with, ``kInt_t`` when it was declared with none."""
        return self._underlying

    def IsScoped(self) -> bool:
        """``IsScoped``: whether it is an ``enum class``, whose constants need its name."""
        return self._scoped

    def GetClass(self) -> Any:
        """``GetClass``: the ``TClass`` of the class or namespace owning the enum, or
        ``nullptr`` for one of the global namespace - which no class describes."""
        if not self._owner:
            return None
        from .reflection import TClass

        return TClass.GetClass(self._owner, False)

    #: ``TEnum::EAutoloading``: whether a lookup may load the library providing the enum.
    #: Nothing here loads a library to answer a lookup, so the two read the same either way.
    kAutoload = 0
    kNoAutoload = 1

    @staticmethod
    def GetEnum(name: Any, type: Any = None, autoload: Any = True) -> TEnum | None:
        """``TEnum::GetEnum``: the enumeration C++ calls ``name``, looked up as written.

        ``type`` - the ``std::type_info`` of the enum - is not consulted: a name is what the
        table is keyed by, and a program asking by type asks for an enum it declared, whose
        name the caller has. ``autoload`` likewise changes nothing, since nothing here reads
        a header to find an enum.
        """
        return find_enum(str(name))


class TListOfEnums(TList):
    """``TListOfEnums``: a list of ``TEnum`` - the session's, and a class's own part of it."""


class TListOfEnumsWithLock(TListOfEnums):
    """``TListOfEnumsWithLock``: the list ROOT 6.40 gives the session's enums, written and
    read under the interpreter's lock, which one program running alone has no need of."""


def note_enum(qualified: Any, items: Any = (), title: str = "", underlying: Any = None,
              scoped: bool = False) -> TEnum:  # fmt: skip
    """Note an enumeration C++ has just declared, which is what grows ROOT's table.

    ``qualified`` is the name as C++ writes it (``N::E``, ``A::B::C::enumC1``, ``E``);
    ``items`` its enumerators and their values, in the order written; ``title`` the comment
    written after it; ``underlying`` the integer type after ``:``, if it has one. An enum
    declared again - a header included twice, a macro rerun - is the one already noted, as
    ROOT's table keeps one ``TEnum`` per enumeration.
    """
    full = str(qualified).strip()
    known = _ENUMS.get(full)
    if known is not None:
        return known
    owner, _, own = full.rpartition("::")
    made = TEnum(own, items, title, UNDERLYING_TYPES.get(str(underlying), K_INT), scoped, owner)
    _ENUMS[full] = made
    return made


def find_enum(name: Any) -> TEnum | None:
    """The enumeration declared as ``name``, or ``None`` - ROOT's table is keyed by the name
    C++ writes, so ``N::E`` is found and a bare ``E`` is not (the oracle finds neither)."""
    return _ENUMS.get(str(name).strip())


def class_enums(owner: Any = None) -> TListOfEnums:
    """The enums ``owner`` - a class or namespace name, ``""`` for the global namespace, or
    ``None`` for every one the interpreter knows - in the list a caller asks them in.

    The session's ``GetListOfEnums`` is all of them, as ROOT's ``fEnums`` is, and is the
    locked list ROOT 6.40 gives it; a class's is its own, the same objects filtered by the
    class that owns them.
    """
    made = TListOfEnumsWithLock() if owner is None else TListOfEnums()
    names = None if owner is None else str(owner).strip()
    for enum in _ENUMS.values():
        if names is None or enum._owner == names:
            made.Add(enum)
    return made


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

    def GetColor(self, index: Any) -> Any:
        """``GetColor(n)``: the graphics' ``TColor`` of index ``n``, or ``None``."""
        found = _graphics("colors", "color_object")
        return None if found is None else found(index)

    def GetListOfColors(self) -> TList:
        return self._listed("Colors")

    def GetListOfTypes(self, load: bool = False) -> TList:
        return self._listed("Types")

    def GetListOfEnums(self, load: bool = False) -> TList:
        """``GetListOfEnums``: every enumeration the interpreter has declared.

        ROOT's ``load`` asks the interpreter to read the declarations still waiting to be
        parsed before answering; a declaration here is noted as it is run, so nothing waits
        and it changes nothing. The list is made afresh from the table each time, so an enum
        declared since the last call is in it - which is what a caller looks for.
        """
        return class_enums()

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
        """``FindObject``: the open files, the functions, canvases, styles, what each geometry
        holds, then ``gDirectory``."""
        if not isinstance(name, str):
            return self._list.FindObject(name)
        found = self._found_listed(name)
        if found is not None:
            return found
        here = current_directory()
        found = here.Get(name) if here is not self else self._list.FindObject(name)
        anywhere = _graphics("pads", "find_anywhere")
        return found if found is not None or anywhere is None else anywhere(name)

    def _found_listed(self, name: str) -> Any:
        """The first object of that name in ROOT's own lists, or inside one of its geometries."""
        for key in ("Files", "MappedFiles", "Functions", "Geometries", "Canvases", "Styles",
                    "Specials"):  # fmt: skip
            found = self._lists[key].FindObject(name)
            if found is not None:
                return found
        inside = (geometry.FindObject(name) for geometry in self._lists["Geometries"])
        return next((one for one in inside if one is not None), None)

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

    def GetClass(self, name: Any, load: bool = True, silent: bool = False) -> Any:
        """``GetClass(name)``: the ``TClass`` of that name, ``nullptr`` for one not known."""
        from .reflection import TClass

        return TClass.GetClass(name, load, silent)

    def GetVersion(self) -> str:
        """``GetVersion``: ROOT's release this namespace follows, ``6.40.04``."""
        return RELEASE

    def GetVersionInt(self) -> int:
        return VERSION_INT

    def GetVersionDate(self) -> int:
        """``GetVersionDate``: the release's day as ``YYYYMMDD``, which macros compare."""
        return VERSION_DATE

    def GetVersionTime(self) -> int:
        """``GetVersionTime``: the release's time of day as ``HHMMSS``."""
        return VERSION_TIME

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
        """``ProcessLine``: a line of C++, run by :mod:`xrdroot.pycint`."""
        return _processor(f"gROOT.ProcessLine({str(line)!r})")(str(line))

    def ProcessLineSync(self, line: Any, error: Any = None) -> Any:
        return self.ProcessLine(line, error)

    def ProcessLineFast(self, line: Any, error: Any = None) -> Any:
        return self.ProcessLine(line, error)

    def Macro(self, filename: Any, error: Any = None, padUpdate: bool = True) -> Any:
        """``Macro``: ``.x filename``, run by :mod:`xrdroot.pycint`."""
        return self.ProcessLine(f".x {filename}")

    def LoadMacro(self, filename: Any, error: Any = None, check: bool = False) -> int:
        """``LoadMacro``: ``.L filename``, loaded by :mod:`xrdroot.pycint`."""
        self.ProcessLine(f".L {filename}")
        return 0


#: ``ROOT.SetHeuristicMemoryPolicy``'s deprecation text, as ROOT 6.40 gives it.
_HEURISTIC_DEPRECATED = (
    "ROOT.SetHeuristicMemoryPolicy() is deprecated and will be removed in ROOT 6.44.\n"
    "        Since ROOT 6.40, the heuristic memory policy is disabled by default, and with\n"
    "        ROOT 6.44 it won't be possible to re-enable it with "
    "ROOT.SetHeuristicMemoryPolicy(),\n"
    "        which was only meant to be used during a transition period and will be removed.\n"
)

#: Where a translator that owns the policy keeps its switch - the binding is what runs a
#: translated destructor, so the answer belongs there.
_HEURISTIC_SWITCH = ("xrdroot.pycxx.runtime.objects", "set_heuristic")

#: The policy as this namespace remembers it, for a translator with no switch of its own:
#: ROOT 6.40 turned the heuristic memory policy off, so ``False``.
_heuristic_policy = False


def SetHeuristicMemoryPolicy(enabled: Any) -> bool:
    """``ROOT.SetHeuristicMemoryPolicy``: whether a call that passes an object to a non-const
    pointer parameter is taken as leaving it to C++, and what the answer was before.

    Off since 6.40, and deprecated: a translated class's destructor runs when its Python
    object is collected, where the policy used to say the callee had taken the object over.
    The switch is the translator's, since that is what destroys; one without a switch gets the
    value remembered here, so a program that sets and restores the policy still reads the
    sequence it wrote.
    """
    global _heuristic_policy
    warnings.warn(_HEURISTIC_DEPRECATED, FutureWarning, stacklevel=0)
    module, name = _HEURISTIC_SWITCH
    switch = getattr(_submodule(module), name, None)
    if switch is not None:
        return bool(switch(bool(enabled)))
    previous, _heuristic_policy = _heuristic_policy, bool(enabled)
    return previous


#: ``gROOT``.
gROOT = TROOT()


class TInterpreter:
    """``gInterpreter``: C++ declared and run - by :mod:`xrdroot.pycint`, when it is there."""

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
        if not _PRELUDE:
            _PRELUDE.update(running("", "<Declare>", call=False))
        namespace = sys.modules["xrdroot.pyroot"].__dict__
        for name, value in declared.items():
            ours = getattr(value, "__module__", None) == "__cint__" or name not in _PRELUDE
            if ours and not name.startswith("_"):
                namespace[name] = value  # a function, a class - or a global it declared
        return True

    def ProcessLine(self, line: Any, error: Any = None) -> Any:
        return gROOT.ProcessLine(line, error)

    def Calc(self, line: Any, error: Any = None) -> Any:
        return gROOT.ProcessLine(line, error)

    def Load(self, filename: Any, system: bool = False) -> int:
        """``Load``: ``gSystem->Load`` of the library."""
        from .system import gSystem

        return gSystem.Load(filename)

    def IsLoaded(self, filename: Any) -> bool:
        """``IsLoaded``: whether ``LoadMacro`` has loaded this macro already."""
        return str(filename) in self.__dict__.get("_loaded", set())

    def LoadMacro(self, filename: Any, error: Any = None, check: bool = False) -> None:
        """``LoadMacro``: ``.L filename``, by the translator."""
        gROOT.LoadMacro(filename)
        self.__dict__.setdefault("_loaded", set()).add(str(filename))

    def ExecuteMacro(self, filename: Any, error: Any = None) -> Any:
        return gROOT.Macro(filename)

    def AddIncludePath(self, path: Any) -> None:
        """``AddIncludePath``: nothing is compiled, so there is nothing to include."""

    def GenerateDictionary(self, classes: Any, headers: Any = "", *rest: Any) -> int:
        """``GenerateDictionary``: a dictionary for the classes named - a semicolon separated
        list - which ``TClass::GetState`` reports as ``kHasTClassInit`` from then on.

        Nothing is compiled here: the classes such a dictionary would describe are the ones
        this kit already builds, so what it adds is that ROOT's own answer about them - that a
        dictionary is there - is the one a program gets.
        """
        from .reflection import note_generated_dictionary

        note_generated_dictionary(classes)
        return 0

    #: Whether a name nothing declared is looked for in the libraries a rootmap lists.
    autoloading = True

    def SetClassAutoloading(self, autoload: Any = 1) -> int:
        """``SetClassAutoloading``: turn autoloading on or off; whether it was on."""
        before, self.autoloading = self.autoloading, bool(autoload)
        return int(before)

    SetClassAutoLoading = SetClassAutoloading

    def SetClassAutoparsing(self, autoparse: Any = 1) -> int:
        """``SetClassAutoparsing``: headers are read when a library is built, not parsed late."""
        return 1

    def AutoParse(self, name: Any) -> int:
        return 0

    def AutoLoad(self, classname: Any, knowDictNotLoaded: bool = False) -> int:
        """``AutoLoad``: ``1`` if the library providing ``classname`` is loaded now."""
        autoload = _translator("autoload")
        return int(bool(autoload is not None and autoload(str(classname).split("::")[0])))

    def LoadLibraryMap(self, rootmapfile: Any = None) -> int:
        """``LoadLibraryMap``: a translated library carries its own map, read when needed."""
        return 0

    def GetMapfile(self, library: Any = None) -> Any:
        """``GetMapfile``: the ``.rootmap`` of a library as a ``TEnv``, which nothing here has.

        ROOT reads a library's map at load time and keeps it in the interpreter's; this
        has none, so the answer is ``nullptr`` - which a program that walks the map
        already checks for and skips, as roottest's ``libraryLister.h`` does.
        """
        return None

    def GetInterpreterTypeName(self, name: Any, output: Any = None, quiet: bool = False) -> str:
        """``GetInterpreterTypeName(name, output, quiet)``: the type ``name`` as the interpreter
        knows it, into ``output`` and handed back.

        A name here is already the one the translator resolved - a typedef is its own
        name, ``std::`` or not, as C++ wrote it - so it is echoed.
        """
        spelled = str(name).strip()
        if hasattr(output, "assign"):
            output.assign(spelled)
        elif hasattr(output, "value"):
            output.value = spelled
        return spelled

    def UnloadLibraryMap(self, library: Any) -> int:
        """``UnloadLibraryMap``: nothing is autoloaded that a macro has not asked for by name."""
        return 0


#: ``gInterpreter``.
gInterpreter = TInterpreter()
