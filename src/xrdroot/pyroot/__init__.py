"""ROOT's own Python namespace, over this library: ``import xrdroot.pyroot as ROOT``.

The namespace is put together from the modules listed in :data:`SUBMODULES`,
each re-exporting its ``__all__`` here, so a script written for PyROOT finds
``ROOT.TCanvas`` and ``ROOT.gStyle`` where it looks for them. A name ROOT has
and this does not yet is refused by name, so that what is missing can be
counted rather than guessed.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

#: The modules under this package whose ``__all__`` make up the namespace.
SUBMODULES = ["core", "graphics"]

__all__: list[str] = []


def _gather() -> None:
    """Import every module of :data:`SUBMODULES` there is, and take its names."""
    for name in SUBMODULES:
        if importlib.util.find_spec(f"{__name__}.{name}") is None:
            continue  # another branch's module, not merged in yet
        module = importlib.import_module(f"{__name__}.{name}")
        for exported in getattr(module, "__all__", ()):
            globals()[exported] = getattr(module, exported)
            __all__.append(exported)


_gather()


def __getattr__(name: str) -> Any:
    raise AttributeError(f"ROOT has {name}; xrdroot.pyroot does not yet")
