"""``xrdroot.cint.runtime.objects``, kept: pickles made before the split name its classes here.

Its names are the very objects :mod:`xrdroot.pycxx.runtime.objects` exports,
with ROOT's ``delete``, placement ``new`` and ``*p`` from
:mod:`xrdroot.pycint.runtime.adapters`, as a translation in ROOT's dialect
sees them.
"""

from __future__ import annotations

from typing import Any

from ...pycint.runtime.adapters import construct_at as construct_at
from ...pycint.runtime.adapters import delete as delete
from ...pycint.runtime.adapters import deref as deref
from ...pycxx.runtime import objects as _objects

__all__ = _objects.__all__


def __getattr__(name: str) -> Any:
    """Every other name is plain C++'s runtime's own object."""
    return getattr(_objects, name)
