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
    "trees",
    "rdf",
    "graphics",
    "graphics.ratioplot",
    "graphics.widgets",
    "graphics.views3d",
)

#: The modules gathered when a name one of them exports is first asked for: RooFit, RooStats,
#: TMVA, the spectra, the geometry and the rest, which a macro names or does not name. They
#: are a third of the namespace's modules and a fifth of a second of every command that never
#: says one of their names.
LATER = (
    "graphics.polar",
    "graphics.polargraph",
    "graphics.texec",
    "graphics.cutg",
    "roofit",
    "roostats",
    "tcut",
    "th2poly",
    "xmlengine",
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

#: ``ROOT.__version__``: the release this namespace follows, as ``gROOT.GetVersion()`` has it.
__version__: str = globals()["gROOT"].GetVersion()


def __getattr__(name: str) -> _typing.Any:
    """A name one of the modules the namespace has not gathered yet exports; an object
    ``gROOT`` finds by that name, as PyROOT falls back to - ``ROOT.h1``, a geometry's
    ``ROOT.YK01`` - or else a name ROOT has and this namespace does not yet."""
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
