"""What a translated macro runs on: ``from xrdroot.pycint.runtime import *``.

Every translation in the ROOT dialect starts with that one import. It brings
everything plain C++'s runtime has (:mod:`xrdroot.pycxx.runtime`) - cells,
``printf``, ``cout``, ``<cmath>``, C's arithmetic - with ROOT's meanings of
the helpers where ROOT's objects mean more (:mod:`.adapters`), ``ROOT`` (ROOT's
names, looked up in :mod:`xrdroot.pyroot`), ROOT's ``Printf`` family, the
``std::atomic`` whose reads share the interpreter with the thread being waited
for (:mod:`.spinning`), and the batch fill of an ntuple's column a translated
loop tries first. Loading it also readies GoogleTest for ROOT's unit tests
(:mod:`.diags`).
"""

from __future__ import annotations

from typing import Any

from ...pycxx.runtime import *  # noqa: F403
from ...pycxx.runtime import LIBRARY as _LIBRARY
from . import diags as _diags  # noqa: F401  - ROOT's unit tests' GoogleTest, readied
from .adapters import *  # noqa: F403
from .adapters import __all__ as _adapters
from .root import ROOT, RootProxy
from .spinning import atomic  # over the runtime's: a spin on one must share the GIL


def __getattr__(name: str) -> Any:
    """``fill_ntuple_histogram``, imported when it is asked for and not before.

    Only a translated program that fills an ntuple histogram needs it, and that program has
    already imported the runtime's ROOT; ``xrdroot build dictionary`` emits the name as text and
    never calls it, so it used to import pyroot, the trees and roofit's fitting for nothing.
    """
    if name == "fill_ntuple_histogram":
        from .ntuplefill import fill_ntuple_histogram

        return fill_ntuple_histogram
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ROOT",
    "atomic",
    "fill_ntuple_histogram",  # noqa: F405 - defined by __getattr__ above, not imported here
    "RootProxy",
    *dict.fromkeys([*_LIBRARY, *_adapters]),
]
