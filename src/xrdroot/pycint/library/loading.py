"""Loading a translated library, as ``gSystem->Load`` loads a shared one, and finding it first.

A library is found by name as ROOT's ``TSystem::FindDynamicLibrary`` finds
one: ``libFoo``, ``libFoo.so`` or ``./libFoo.dylib`` all mean
``libFoo.xrdlib``, looked for in the working directory, then each directory
of ``gSystem->GetDynamicPath()``, ``ROOT_LIBRARY_PATH``, ``LD_LIBRARY_PATH``
and ``DYLD_LIBRARY_PATH`` - or only where a name with a directory says.
Loading runs its Python once, in a module of its own, and keeps what it
defines where every later line of C++ (and ``ROOT.<name>`` in Python) looks:
each function and class by its name, and each namespace as an object whose
attributes are its members, so ``mylib::MyClass`` resolves.

``Load`` answers as ROOT's does: ``0`` loaded now, ``1`` loaded already,
``-1`` not found. ROOT's own libraries - ``libTree``, ``libRooFit`` - are
xrdroot's packages, there from the start, so loading one is ``0`` the first
time and ``1`` after, and those ROOT itself starts with are ``1`` at once.

A macro may ``#include`` a header a loaded library was built from: it is
translated again with the macro, so the translator knows its types, but the
names it defines are bound back to the library's own objects once the macro's
Python has run (:func:`rebind`), so there is one class ``Foo``, with one set
of static members, whichever side made the object.

What a library declares is also found by name without loading it first,
as ROOT's rootmap files let it autoload: :func:`autoload` loads the library
on the search path that provides a name nothing else does.
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...pycxx.execute import MODULE, label, placed
from ...pycxx.runtime.namespace import DECLARED
from ...pycxx.translation import Translation
from ..dialect import ROOT as DIALECT
from .manifest import SUFFIX, Library, read, stem_of

__all__ = [
    "Loaded",
    "LOADED",
    "CxxNamespace",
    "search_directories",
    "find",
    "load",
    "load_library",
    "loaded_names",
    "unload",
    "rebind",
    "declare_namespaces",
    "autoload",
    "forget_all",
    "ROOT_LIBRARIES",
]

#: ROOT's own libraries, by name: each is one of xrdroot's packages, there from the start.
ROOT_LIBRARIES = frozenset(
    """libCore libRIO libThread libCling libRint libNet libHist libMatrix libMathCore
    libMathMore libPhysics libGenVector libSmatrix libTree libTreePlayer libTreeViewer
    libGpad libGraf libGraf3d libPostscript libASImage libGui libGed libHistPainter libGeom
    libGeomPainter libGdml libRooFit libRooFitCore libRooFitMore libRooFitHS3 libRooStats
    libHistFactory libRooFitCodegen libRooBatchCompute libTMVA libTMVAUtils libPyMVA libMinuit
    libMinuit2 libFumili libFoam libQuadp libSpectrum libSpectrumPainter libSPlot libMLP
    libXMLIO libXMLParser libSQLIO libROOTDataFrame libROOTVecOps libROOTNTuple
    libROOTNTupleUtil libROOTHist libROOTHistUtil libImt libMultiProc libEG libGenetic
    libUnfold libFFTW libNew libRecorder libROOTTPython libROOTGpadv7 libROOTGraphicsPrimitives
    libROOTCanvasPainter libROOTWebDisplay libWebGui6 libRHTTP libNetxNG libRDAVIX libGviz
    libRGL libEve libRCsg libVc libvectorDict libmapDict liblistDict libdequeDict libsetDict
    libmultimapDict libmultisetDict libunordered_mapDict libunordered_setDict libcomplexDict
    libvalarrayDict libforward_listDict libmap2Dict libmultimap2Dict libTMVAGui libFitPanel
    libROOTMLDataLoader libROOTTMVASofie libROOTTreeMap libGQuartz libGCocoa libGX11 libGX11TTF
    """.split()
)

#: Those ROOT itself loads as it starts, in its order: loading one is ``1``, loaded already.
STARTED = ("libRint", "libCore", "libRIO", "libThread", "libCling")

#: The environment variables whose directories are searched after the dynamic path.
ENVIRONMENT = ("ROOT_LIBRARY_PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")


class CxxNamespace:
    """A C++ namespace a library declares: its members as attributes, nested ones too.

    A class looked up in it is handed to PyROOT's pythonizations first, as
    cppyy hands a class it binds, by its qualified name.
    """

    def __init__(self, name: str, qualified: str = "") -> None:
        self.__name__ = name
        self.__cpp_name__ = qualified or name

    def __repr__(self) -> str:
        return f"<namespace {self.__name__}>"

    def __getattribute__(self, name: str) -> Any:
        found = object.__getattribute__(self, name)
        if not isinstance(found, type) or name.startswith("_"):
            return found
        from ...pyroot.pythonizing import looked_up

        return looked_up(found, name, object.__getattribute__(self, "__cpp_name__"))


@dataclass
class Loaded:
    """A library that has been loaded: what it is, and the module its Python ran in."""

    library: Library
    module: dict[str, Any]
    #: The files it was built from and every one they read, absolute.
    headers: frozenset[str] = field(default_factory=frozenset)


#: Every library loaded, by the absolute path of its file.
LOADED: dict[str, Loaded] = {}
#: ROOT's own libraries loaded so far, by name.
ROOT_LOADED: set[str] = set()
#: The library each header was built into, by the header's absolute path.
HEADERS: dict[str, Loaded] = {}


def _dynamic_path() -> list[str]:
    """``gSystem->GetDynamicPath()``'s directories, if pyroot's gSystem is there to ask."""
    system = sys.modules.get("xrdroot.pyroot.core.system")
    path = str(system.gSystem.GetDynamicPath()) if system is not None else ""
    return [entry for entry in path.split(":") if entry]


def search_directories() -> list[Path]:
    """Where a library named without a directory is looked for, in ROOT's order."""
    entries = [".", *_dynamic_path()]
    for variable in ENVIRONMENT:
        entries += [entry for entry in os.environ.get(variable, "").split(":") if entry]
    return [Path(entry) for entry in dict.fromkeys(entries)]


def find(name: str) -> Path | None:
    """The library file ``name`` means - ``libFoo``, ``libFoo.so``, ``dir/libFoo`` - if any."""
    stem = stem_of(name)
    directory = os.path.dirname(name)
    places = [Path(directory)] if directory else search_directories()
    # ``Foo`` is also ``libFoo``: CMake names a dictionary's library either way.
    stems = [stem] if stem.startswith("lib") else [stem, f"lib{stem}"]
    for each in stems:
        for place in places:
            candidate = place / f"{each}{SUFFIX}"
            if candidate.is_file():
                return candidate
    return None


def _system() -> str:
    """The name ROOT's system class has on this machine, as its messages give it."""
    return "TMacOSXSystem" if sys.platform == "darwin" else "TUnixSystem"


def _not_found(name: str) -> int:
    from ...pyroot.core.messages import message

    path = ":".join(str(each) for each in search_directories())
    message("Error", f"{_system()}::FindDynamicLibrary",
            f"{name}[.so | .dll | .dylib | .sl | .dl | .a] does not exist in {path}")  # fmt: skip
    return -1


def _root_library(stem: str) -> int | None:
    """ROOT's own library ``stem``: ``1`` if it is there already, ``0`` the first time."""
    if stem not in ROOT_LIBRARIES:
        return None
    if stem in STARTED or stem in ROOT_LOADED:
        return 1
    ROOT_LOADED.add(stem)
    return 0


def load(name: str) -> int:
    """``gSystem->Load(name)``: ``0`` loaded now, ``1`` already, ``-1`` not found."""
    text = str(name).strip()
    found = find(text)
    if found is None:
        compiled = _compiled(text)
        if compiled is not None:
            return compiled
        status = _root_library(stem_of(text))
        return _not_found(text) if status is None else status
    if str(found.resolve()) in LOADED:
        return 1
    library = read(found)
    if library is None:
        from ...pyroot.core.messages import message

        message("Error", f"{_system()}::Load",
                f"{found} is not a library xrdroot built, or one an older xrdroot did")  # fmt: skip
        return -1
    load_library(library)
    return 0


def _compiled(name: str) -> int | None:
    """``gSystem->Load("foo_C")`` with no library built yet: ACLiC compiles ``foo.C`` for it.

    ROOT names the library it compiles ``foo.C`` ``foo_C``, so a load of that name is a
    request to build the macro it came from; ``None`` is a name no such macro answers to.
    """
    from . import aclic

    stem = stem_of(name)
    for source in (stem, stem[:-2] if stem.endswith("_C") else ""):
        if source and source + ".C" in {path.name for path in Path().iterdir()}:
            return 0 if aclic.compile_macro(source + ".C", "kc") else -1
    return None


def load_library(library: Library) -> Loaded:
    """Run ``library``'s Python and keep what it defines where later code looks for it."""
    key = str(library.path.resolve()) if library.path is not None else library.name
    if key in LOADED:
        return LOADED[key]
    for dependency in library.libraries:
        load(dependency)
    made = library.translation()
    module: dict[str, Any] = {"__name__": MODULE, "__file__": library.file}
    with DIALECT.session(), placed([made]):
        exec(compile(made.python, label(made), "exec"), module)
    # One class is one class: a class this library's sources see in the header of a library
    # already loaded - a macro compiled against a test library, `macro.C+` with
    # `libTestIoEvent` loaded - is that library's, whose constructors and methods the link
    # would have made the calls in this one reach.
    _rebound(module, made, HEADERS)
    headers = frozenset([*library.resolved(library.sources), *made.included])
    loaded = LOADED[key] = Loaded(library, module, headers)
    HEADERS.update(dict.fromkeys(headers, loaded))
    _declare(loaded)
    own = dict.fromkeys(headers, loaded)
    for watched, translated in WATCHED:
        _rebound(watched, translated, own)
        _undefined(watched, loaded)
    return loaded


def _declare(loaded: Loaded) -> None:
    """What a library defines, kept where every later program looks: its functions and
    classes, its variables, and its namespaces as objects."""
    module = loaded.module
    for name, value in module.items():
        if getattr(value, "__module__", None) == MODULE or name in loaded.library.names:
            DECLARED[name] = value
    for name, members in loaded.library.namespaces.items():
        DECLARED[name] = _namespace(DECLARED.get(name), name, members, module)
    _dictionaried(loaded)


def _dictionaried(loaded: Loaded) -> None:
    """Tell ROOT's class table which classes this library's dictionary describes.

    A program asks ``TClassTable::GetDict("Event")`` to decide whether it must load the
    library first, which under ROOT is answered by the dictionary the library carries.
    """
    classes = [
        name
        for name, value in loaded.module.items()
        if isinstance(value, type) and getattr(value, "__module__", None) == MODULE
    ]
    if not classes:
        return
    from ...pyroot.core.reflection import note_generated_dictionary

    note_generated_dictionary(";".join(classes))


def _namespace(
    existing: Any, name: str, members: dict[str, Any], module: dict[str, Any], outer: str = ""
) -> Any:
    """The namespace ``name`` with ``members`` - added to one already declared, if there is."""
    qualified = f"{outer}::{name}" if outer else name
    space = existing if isinstance(existing, CxxNamespace) else CxxNamespace(name, qualified)
    for member, inner in members.items():
        if isinstance(inner, dict):
            within = vars(space).get(member)
            setattr(space, member, _namespace(within, member, inner, module, qualified))
        elif member in module:
            setattr(space, member, module[member])
    return space


#: Names a program's ``namespace`` block may add to but never makes an object of its own.
OPEN_NAMESPACES = frozenset({"std", "ROOT"})


def declare_namespaces(
    source: str, before: dict[str, Any]
) -> tuple[dict[str, CxxNamespace], set[str]]:
    """The namespaces ``source`` - a ``Declare``, a line - declares, as objects of what they
    hold, kept where later code finds them as a library's are: ``ROOT.NS.A``, ``NS::A``; and
    the names only those namespaces hold.

    Its members are what the run of it just declared, so ``source`` has run first. A
    translation's names are one flat module, so a member named as something declared before
    at the top - ``NS::Foo`` after ``Foo`` - is put back to that, from ``before``."""
    from .translate import declared_in

    found = declared_in(source)
    made: dict[str, CxxNamespace] = {}
    for name, members in found.namespaces.items():
        existing = DECLARED.get(name)
        if name in OPEN_NAMESPACES or not isinstance(existing, (CxxNamespace, type(None))):
            continue
        DECLARED[name] = made[name] = _namespace(existing, name, members, DECLARED)
    hidden = _members(found.namespaces) - set(found.top)
    DECLARED.update({name: before[name] for name in hidden if name in before})
    return made, hidden


def _members(namespaces: dict[str, Any]) -> set[str]:
    """Every name the namespaces hold, at any depth, but the inner namespaces' own."""
    names: set[str] = set()
    for members in namespaces.values():
        for member, inner in members.items():
            names |= _members({member: inner}) if isinstance(inner, dict) else {member}
    return names


def unload(name: str) -> None:
    """``gSystem->Unload(name)``: the library forgotten, so loading it again is ``0``. What it
    declared stays declared, as cling keeps what it has seen."""
    stem = stem_of(str(name))
    ROOT_LOADED.discard(stem)
    for key, loaded in list(LOADED.items()):
        if loaded.library.name == stem:
            del LOADED[key]


def loaded_names() -> list[str]:
    """The libraries loaded, in the order they were - ROOT's own it starts with first, a
    translated one by its file: ``gSystem->GetLibraries()``, ``.libraries``."""
    return [*STARTED, *LOADED, *sorted(ROOT_LOADED)]


def forget_all() -> None:
    """Unload everything: a fresh session's state, for a test that starts over."""
    from ...pyroot.core.reflection import forget_dictionaries

    forget_dictionaries()
    LOADED.clear()
    HEADERS.clear()
    WATCHED.clear()
    ROOT_LOADED.clear()
    INDEX.clear()


# -- one class Foo: a macro's copy of a library's header bound back to the library ----------


#: The programs run so far, each with its translation: a library loaded later binds its
#: names in them too, as cling resolves a declared function only when it is called.
WATCHED: list[tuple[dict[str, Any], Translation]] = []


def rebind(module: dict[str, Any], made: Translation) -> None:
    """The dialect's ``loaded`` hook: names a macro defined from a loaded library's header
    are the library's own objects, as there is one definition of each in C++; and what it
    only declares - ``int fromA();`` - a loaded library defines."""
    WATCHED.append((module, made))
    _rebound(module, made, HEADERS)
    for loaded in LOADED.values():
        _undefined(module, loaded)


def _rebound(module: dict[str, Any], made: Translation, headers: dict[str, Loaded]) -> None:
    """Bind ``module``'s names that came from one of ``headers`` to its library's objects."""
    included = {str(Path(path).resolve()) for path in made.included}
    if not included & headers.keys():
        return
    for name, file in _defined_in(made):
        loaded = headers.get(file)
        if loaded is not None and name in loaded.module:
            module[name] = loaded.module[name]


def _undefined(module: dict[str, Any], loaded: Loaded) -> None:
    """What a program only declares - ``int CountShapes();`` - the library defines: bound
    in ``module``, which has no definition of its own."""
    for name, value in loaded.module.items():
        if getattr(value, "__module__", None) == MODULE and name != "main":  # not its start
            module.setdefault(name, value)


def _defined_in(made: Translation) -> list[tuple[str, str]]:
    """Each name the translation defines at its top, and the C++ file it came from."""
    found = []
    for statement in ast.parse(made.python).body:
        where = _first_mapped(made.source_map, statement)
        if where is None:
            continue
        file = str(Path(where[0]).resolve())
        found += [(name, file) for name in _bound(statement)]
    return found


def _bound(statement: ast.stmt) -> list[str]:
    if isinstance(statement, (ast.ClassDef, ast.FunctionDef)):
        return [statement.name]
    if isinstance(statement, ast.Assign):
        return [target.id for target in statement.targets if isinstance(target, ast.Name)]
    return []


def _first_mapped(source_map: dict[int, tuple[str, int]], statement: ast.stmt) -> Any:
    for line in range(statement.lineno, (statement.end_lineno or statement.lineno) + 1):
        if line in source_map:
            return source_map[line]
    return None


# -- autoloading: the library that provides a name, found as a rootmap finds it -------------

#: Each library file seen on the search path: when it was read, and the library if it has a
#: rootmap - which is what autoloading finds a library by.
INDEX: dict[str, tuple[int, Library | None]] = {}


def _indexed(path: Path) -> Library | None:
    """The library at ``path`` if autoloading may find it, read once while it is unchanged."""
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    kept = INDEX.get(str(path))
    if kept is None or kept[0] != stamp:
        library = read(path)
        kept = INDEX[str(path)] = stamp, library if library and library.rootmap else None
    return kept[1]


def provides(library: Library, name: str) -> bool:
    """Whether ``library`` declares ``name`` at its top: a class, function or namespace."""
    return name in library.names or any(each.split("::")[0] == name for each in library.classes)


def _candidates() -> list[Path]:
    return [path for place in search_directories() for path in sorted(place.glob(f"*{SUFFIX}"))]


#: The libraries being autoloaded now: a name their own Python looks up finds no other.
AUTOLOADING: set[str] = set()


def autoload(name: str) -> bool:
    """Load the library on the search path that provides ``name``; whether one did."""
    if name.startswith("_"):
        return False
    for path in _candidates():
        key = str(path.resolve())
        library = None if key in LOADED or key in AUTOLOADING else _indexed(path)
        if library is None or not provides(library, name):
            continue
        AUTOLOADING.add(key)
        try:
            load_library(library)
        finally:
            AUTOLOADING.discard(key)
        return name in DECLARED
    return False
