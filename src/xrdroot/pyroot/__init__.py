"""``import xrdroot.pyroot as ROOT``: ROOT's own names, over this library's objects.

    >>> import xrdroot.pyroot as ROOT
    >>> h = ROOT.TH1D("h", "a histogram", 10, 0, 1)
    >>> h.Fill(0.25)
    3
    >>> h.GetBinContent(3), h.GetEntries()
    (1.0, 1.0)

A PyROOT script is written against a namespace of ROOT's classes, with ROOT's
names, argument orders and defaults, and prints what ROOT prints. This is that
namespace: each class a thin wrapper keeping the xrdroot object it stands for
in ``._xrd``, so what a script builds is a :class:`~xrdroot.Histogram`,
:class:`~xrdroot.Graph` or :class:`~xrdroot.Function` underneath, written,
fitted and drawn by the rest of the library.

The namespace is put together from :data:`SUBMODULES`, each a module under
this package re-exporting its ``__all__``; a module listed but not there -
another part of the kit not yet installed - is passed over, so the list can
name every part and the namespace holds what there is. A name ROOT has and
this does not is refused by name, so a script that reaches for one fails
saying which.
"""

from __future__ import annotations

import importlib
import importlib.util
import typing as _typing  # not ``Any``: ``ROOT.Any`` is ``ROOT::VecOps::Any``

#: The modules gathered when the namespace is imported: what a macro that reads a tree,
#: fills a histogram or draws a canvas reaches - nearly every macro ROOT's own tests run.
EAGER = (
    "core",
    "stl",
    "classes",
    "stringutils",
    "trees",
    "graphics",
    "mathfit",
)

#: The modules gathered when a name one of them exports is first asked for: RooFit, RooStats,
#: TMVA, the spectra, the geometry and the rest, which a macro names or does not name. They
#: are a third of the namespace's modules and a fifth of a second of every command that never
#: says one of their names.
LATER = (
    "rdf",
    "graphics.ratioplot",
    "graphics.widgets",
    "graphics.views3d",
    "graphics.polar",
    "graphics.polargraph",
    "graphics.texec",
    "graphics.cutg",
    "roofit",
    "roostats",
    "tcut",
    "th2poly",
    "xmlengine",
    "xmldom",
    "vectors",
    "linalg",
    "minuit2",
    "fitshdu",
    "graphics.images",
    "tmva",
    "histv7",
    "ntuple",
    "rcanvas",
    "spectra.tspectrum",
    "spectra.tspectrum2",
    "spectra.transforms",
    "spectra.fits",
    "geom",
    "geom.legacy",
    "geom.legacyshapes",
    "numbadeclare",
    "pythonizing",
    "_pythonization",
    "uhi",
    "cmdlineutils",
)

#: The modules the namespace is made from, in order: a later one's name wins.
SUBMODULES = [*EAGER, *LATER]

#: The names gathered so far, in the order they were gathered: ``__all__``, which is asked
#: for by ``from ROOT import *`` and by ``dir``, and only then has to hold every name.
_EXPORTED: list[str] = []


def _gather(
    namespace: dict[str, _typing.Any] | None = None,
    modules: _typing.Sequence[str] | None = None,
) -> list[str]:
    """Import each listed module that exists and put its ``__all__`` in ``namespace``.

    Without a namespace given it is this module's own, and the names gathered
    join the ones :data:`_EXPORTED` holds.
    """
    into = globals() if namespace is None else namespace
    exported: list[str] = []
    for module in SUBMODULES if modules is None else modules:
        if importlib.util.find_spec(f"{__name__}.{module}") is None:
            continue
        found = importlib.import_module(f"{__name__}.{module}")
        for name in getattr(found, "__all__", ()):
            into[name] = getattr(found, name)
            exported.append(name)
    if namespace is None:
        fresh = [name for name in dict.fromkeys(exported) if name not in _EXPORTED]
        _EXPORTED.extend(fresh)
    return exported


_gather(modules=EAGER)

#: Have the modules the namespace holds only some of been gathered yet?
_gathered_all = False


def _gather_later() -> None:
    """Gather the modules the namespace has not, once: the rest of ROOT's own namespaces."""
    global _gathered_all
    if not _gathered_all:
        _gathered_all = True
        _gather(modules=LATER)

#: The names a script's first touch of the interpreter brings in: ROOT initialises its C++ side
#: then, and cppyy with it, which is why ``import ROOT`` alone does not.
_LAZY = {"gInterpreter"}
globals().pop("gInterpreter", None)


def _interpreter() -> _typing.Any:
    """``ROOT.gInterpreter``: the interpreter, and - as ROOT does on first use - cppyy loaded
    with it, under the name a script's ``sys.modules`` check looks for."""
    import sys

    from .. import plugins

    offered = plugins.lookup("root.cppyy")
    if offered is not None:  # a kit without the binding still has the interpreter
        sys.modules.setdefault("cppyy", offered())
    from .core.troot import gInterpreter

    globals()["gInterpreter"] = gInterpreter  # bound now, as ROOT initialises it once
    return gInterpreter


#: ``ROOT.__version__``: the release this namespace follows, as ``gROOT.GetVersion()`` has it.
__version__: str = globals()["gROOT"].GetVersion()


def __getattr__(name: str) -> _typing.Any:
    """A name one of the modules the namespace has not gathered yet exports; an object
    ``gROOT`` finds by that name, as PyROOT falls back to - ``ROOT.h1``, a geometry's
    ``ROOT.YK01`` - or else a name ROOT has and this namespace does not yet."""
    if name in _LAZY:
        return _interpreter()
    if name == "__all__":
        _gather_later()
        globals()["__all__"] = _EXPORTED  # asked for once, and an attribute from now on
        return _EXPORTED
    if not name.startswith("_"):
        if not _gathered_all:
            _gather_later()
            found = globals().get(name)
            if found is not None:
                return found
        found = globals()["gROOT"].FindObject(name)
        if found is not None:
            return found
        declared = _declared(name)
        if declared is not None:
            from .pythonizing import looked_up

            return looked_up(declared, name)
    raise AttributeError(f"ROOT has {name}; xrdroot.pyroot does not yet")


def __dir__() -> list[str]:
    """Every name the namespace offers: asked for them, it gathers the ones it has not."""
    _gather_later()
    return sorted(globals())


def memset(memory: _typing.Any, value: int = 0, count: int = 1) -> _typing.Any:
    """``memset(memory, value, count)``: C's own, which ROOT's namespace has by that name.

    A translated constructor zeroes a ``char`` array with it - ``memset(fName, 0,
    sizeof(fName))`` - and the name is ROOT's because the macro declares nothing of it, so
    it is looked up in the namespace rather than in a library. Memory ``new char[n]`` hands
    out is filled where it lies, as C fills it, and given back. A ``char name[N]`` is a
    Python ``str``, which no write reaches in place: the text the fill leaves is returned
    for the variable to keep, as ``strcpy``'s is, and a fill that would run past the
    characters the ``str`` holds - whose capacity a ``str`` does not carry - is refused
    rather than guessed at.
    """
    import numpy as np  # only this function needs the array a char buffer is

    from ..cppstd.buffers import put_char  # below this component, asked for when needed
    from ..errors import UnsupportedFeatureError

    filled, length = int(value) & 0xFF, int(count)
    if isinstance(memory, np.ndarray):
        for at in range(min(length, memory.size)):
            put_char(memory, at, filled)
        return memory
    text = str(memory)
    if filled and length > len(text):
        raise UnsupportedFeatureError(
            f"memset cannot fill {length} bytes over the {len(text)} characters this char "
            "array holds: a char[N] is a Python str here, which keeps no capacity past the "
            "characters it holds, so a fill running past them has no memory to land in. "
            "Fill the characters the array holds, or use memory from new char[N] instead"
        )
    written = (chr(filled) if filled else "\\0") * min(length, len(text)) + text[length:]
    return written.split("\\0")[0]


def _declared(name: str) -> _typing.Any:
    """What C++ run so far declared by that name - a macro, ``Declare``, a library loaded -
    or else what the library on the search path providing it declares, autoloaded."""
    from ..pycxx.runtime.namespace import DECLARED
    from .core.troot import _translator, gInterpreter

    for each in (name, name + "_"):  # ``compile_``: a C++ name a Python builtin also has
        if each in DECLARED:
            return DECLARED[each]
    autoload = _translator("autoload") if gInterpreter.autoloading else None
    return DECLARED.get(name) if autoload is not None and autoload(name) else None
