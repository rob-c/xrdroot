"""What a translated program runs on: ``from xrdroot.pycxx.runtime import *``.

Every translation in plain C++ starts with that one import. It brings
``CXX`` (the host: the standard library's names, and what programs
declared), :class:`Cell` for variables whose address is taken, C's
``printf`` family and ``<cstdio>`` files, C++'s ``cout`` with its manipulators, ``<cmath>`` with
C's edge cases, the string functions, and the helpers for the places C's
arithmetic is not Python's: ``idiv`` and ``imod`` for integer division,
``f32`` for a ``float`` store, ``u32`` and kin for wrapping. Each is
documented in its module; this one only gathers them. A dialect's runtime
gathers these and its library's own.
"""

from __future__ import annotations

from .arith import *  # noqa: F403
from .arith import __all__ as _arith
from .callrefs import *  # noqa: F403
from .callrefs import __all__ as _callrefs
from .cells import *  # noqa: F403
from .cells import __all__ as _cells
from .chrono import *  # noqa: F403
from .chrono import __all__ as _chrono
from .cmath import *  # noqa: F403
from .cmath import __all__ as _cmath
from .cstdio import *  # noqa: F403
from .cstdio import __all__ as _cstdio
from .istreams import *  # noqa: F403
from .istreams import __all__ as _istreams
from .namespace import CXX
from .objects import *  # noqa: F403
from .objects import __all__ as _objects
from .printf import *  # noqa: F403
from .printf import __all__ as _printf
from .references import *  # noqa: F403
from .references import __all__ as _references
from .streams import *  # noqa: F403
from .streams import __all__ as _streams
from .strings import *  # noqa: F403
from .strings import __all__ as _strings
from .threads import *  # noqa: F403
from .threads import __all__ as _threads
from .typeinfo import *  # noqa: F403
from .typeinfo import __all__ as _typeinfo
from .units import *  # noqa: F403
from .units import __all__ as _units

#: Everything but the host: what a dialect's runtime gathers alongside its own host.
LIBRARY = [
    *_arith,
    *_cells,
    *_cmath,
    *_objects,
    *_printf,
    *_cstdio,
    *_streams,
    *_strings,
    *_istreams,
    *_units,
    *_references,
    *_chrono,
    *_threads,
    *_typeinfo,
    *_callrefs,
]

__all__ = ["CXX", *LIBRARY]
