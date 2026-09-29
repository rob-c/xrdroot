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
from typing import Any

#: The modules the namespace is made from, in order: a later one's name wins.
SUBMODULES = [
    "core",
    "stl",
    "trees",
    "rdf",
    "graphics",
    "graphics.ratioplot",
    "roofit",
    "roostats",
    "tcut",
    "tmva",
    "spectra.tspectrum",
    "spectra.tspectrum2",
    "spectra.transforms",
    "spectra.fits",
    "geom",
]

__all__: list[str] = []


def _gather(namespace: dict[str, Any] | None = None) -> list[str]:
    """Import each listed module that exists and put its ``__all__`` in ``namespace``.

    Without a namespace given it is this module's own, and ``__all__`` grows
    by the names gathered.
    """
    into = globals() if namespace is None else namespace
    exported: list[str] = []
    for module in SUBMODULES:
        if importlib.util.find_spec(f"{__name__}.{module}") is None:
            continue
        found = importlib.import_module(f"{__name__}.{module}")
        for name in getattr(found, "__all__", ()):
            into[name] = getattr(found, name)
            exported.append(name)
    if namespace is None:
        fresh = [name for name in dict.fromkeys(exported) if name not in __all__]
        __all__.extend(fresh)
    return exported


_gather()


def __getattr__(name: str) -> Any:
    """A name ROOT has and this namespace does not yet, refused by that name."""
    raise AttributeError(f"ROOT has {name}; xrdroot.pyroot does not yet")
