"""What a translated macro runs on: ``from xrdroot.cint.runtime import *``.

Every translation starts with that one import. It brings ``ROOT`` (ROOT's
names, looked up in :mod:`xrdroot.pyroot`), :class:`Cell` for variables
whose address is taken, C's ``printf`` family, C++'s ``cout`` with its
manipulators, ``<cmath>`` with C's edge cases, the string functions, and
the helpers for the places C's arithmetic is not Python's: ``idiv`` and
``imod`` for integer division, ``f32`` for a ``float`` store, ``u32`` and
kin for wrapping. Each is documented in its module; this one only gathers
them.
"""

from __future__ import annotations

from .arith import *  # noqa: F403
from .arith import __all__ as _arith
from .cells import *  # noqa: F403
from .cells import __all__ as _cells
from .cmath import *  # noqa: F403
from .cmath import __all__ as _cmath
from .istreams import *  # noqa: F403
from .istreams import __all__ as _istreams
from .objects import *  # noqa: F403
from .objects import __all__ as _objects
from .printf import *  # noqa: F403
from .printf import __all__ as _printf
from .references import *  # noqa: F403
from .references import __all__ as _references
from .root import ROOT, RootProxy
from .streams import *  # noqa: F403
from .streams import __all__ as _streams
from .strings import *  # noqa: F403
from .strings import __all__ as _strings
from .units import *  # noqa: F403
from .units import __all__ as _units

__all__ = [
    "ROOT",
    "RootProxy",
    *_arith,
    *_cells,
    *_cmath,
    *_objects,
    *_printf,
    *_streams,
    *_strings,
    *_istreams,
    *_units,
    *_references,
]
