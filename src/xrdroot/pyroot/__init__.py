"""``import ROOT``, as xrdroot: PyROOT's names over this library's objects.

A placeholder for the namespace the ``core`` agent owns, following the shared
contract exactly so that the two can be put together by uniting the lists: the
namespace is built from :data:`SUBMODULES`, each a module under this package
whose ``__all__`` is re-exported here, and a module that is not there yet is
passed over rather than failing the import. A name ROOT has and nothing here
provides is refused by name, so that a tutorial run over this can count what
is missing.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

#: The modules this namespace is made of, in the order their names are taken.
SUBMODULES = ["core", "stl", "trees", "rdf", "graphics", "roofit"]

__all__: list[str] = []


def _gather() -> None:
    """Import every submodule that is there and take in the names it exports."""
    for name in SUBMODULES:
        full = f"{__name__}.{name}"
        if importlib.util.find_spec(full) is None:  # pragma: no cover - the others' modules
            continue
        module = importlib.import_module(full)
        for exported in module.__all__:
            globals()[exported] = getattr(module, exported)
            __all__.append(exported)


_gather()


def __getattr__(name: str) -> Any:
    raise AttributeError(f"ROOT has {name}; xrdroot.pyroot does not yet")
