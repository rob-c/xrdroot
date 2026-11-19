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

from ...pycxx.runtime import *  # noqa: F403
from ...pycxx.runtime import LIBRARY as _LIBRARY
from . import diags as _diags  # noqa: F401  - ROOT's unit tests' GoogleTest, readied
from .adapters import *  # noqa: F403
from .adapters import __all__ as _adapters
from .ntuplefill import fill_ntuple_histogram
from .root import ROOT, RootProxy
from .spinning import atomic  # over the runtime's: a spin on one must share the GIL

__all__ = [
    "ROOT",
    "atomic",
    "fill_ntuple_histogram",
    "RootProxy",
    *dict.fromkeys([*_LIBRARY, *_adapters]),
]
