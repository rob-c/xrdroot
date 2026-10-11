"""``ROOT::GetClass``: the class of an object, as ROOT's dictionary names it.

A macro asks what something is - ``ROOT::GetClass(v.GetPtr())->GetName()``
- of objects it made or was handed back: one of ROOT's classes, which says
its own name, a ``std::vector`` of the kit, or the column a frame's
``Take`` gave, which ROOT holds as a ``vector`` of the column's type. The
:class:`~.core.objects.TClass` handed back is the one ``IsA`` gives.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.objects import TClass
from .stl import CPP_NAMES, _Container, cpp_name

__all__ = ["GetClass"]


def _name_of(obj: Any) -> str:
    if hasattr(obj, "ClassName"):
        return str(obj.ClassName())
    if isinstance(obj, _Container):
        return cpp_name(type(obj))
    if isinstance(obj, np.ndarray) and obj.dtype.name in CPP_NAMES:
        return f"vector<{CPP_NAMES[obj.dtype.name]}>"
    raise TypeError(f"ROOT has no dictionary of a {type(obj).__name__} to give the class of")


def GetClass(obj: Any) -> TClass:
    """``ROOT::GetClass(&object)``: the object's class."""
    return TClass(_name_of(obj))
