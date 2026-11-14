"""What a translation calls for objects, pointers and arrays: the standard's, over :mod:`cppstd`.

The helpers here are what plain C++ means by ``delete``, placement ``new``
and ``*p``; a dialect's runtime may export its own of the same name for its
library's objects, which a translation in that dialect then calls instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...cppstd.objects import DTYPES as DTYPES
from ...cppstd.objects import INTEGRAL as INTEGRAL
from ...cppstd.objects import REAL as REAL
from ...cppstd.objects import UNGIVEN as UNGIVEN
from ...cppstd.objects import Candidate as Candidate
from ...cppstd.objects import CppException as CppException
from ...cppstd.objects import Overloaded as Overloaded
from ...cppstd.objects import Pair as Pair
from ...cppstd.objects import _comparison as _comparison
from ...cppstd.objects import _fill as _fill
from ...cppstd.objects import _filled as _filled
from ...cppstd.objects import _fits as _fits
from ...cppstd.objects import _flat as _flat
from ...cppstd.objects import _object_array as _object_array
from ...cppstd.objects import _walked as _walked
from ...cppstd.objects import array as array
from ...cppstd.objects import convertible as convertible
from ...cppstd.objects import dynamic_cast as dynamic_cast
from ...cppstd.objects import function_template as function_template
from ...cppstd.objects import iterate as iterate
from ...cppstd.objects import pointee as pointee
from ...cppstd.objects import postinc as postinc
from ...cppstd.objects import preinc as preinc
from ...cppstd.objects import reverse_range as reverse_range
from ...cppstd.objects import set_attr as set_attr
from ...cppstd.objects import set_item as set_item
from ...cppstd.objects import sort_range as sort_range
from ...cppstd.objects import throw as throw
from ...cppstd.objects import value_copy as value_copy
from ..errors import Refusal

__all__ = [
    "array",
    "convertible",
    "function_template",
    "UNGIVEN",
    "Overloaded",
    "delete",
    "construct_at",
    "value_copy",
    "dynamic_cast",
    "set_item",
    "set_attr",
    "Pair",
    "iterate",
    "sort_range",
    "reverse_range",
    "CppException",
    "throw",
    "deref",
    "pointee",
    "preinc",
    "postinc",
    "INTEGRAL",
    "REAL",
    "uninstantiated",
]


def delete(obj: Any) -> bool:
    """``delete p``: a class's destructor runs, if it has one - and whether it had."""
    destructor = getattr(obj, "_destruct", None)
    if callable(destructor):
        destructor()
        return True
    return False


def construct_at(container: Any, index: Any, obj: Any) -> Any:
    """``new (&a[i]) T(args)``: ``obj`` put in slot ``index`` of ``container``.

    Placement new is handed the memory of slot ``i``; here the object is built
    first and the container given it, which is the same container of the
    same objects afterwards.
    """
    container[index] = obj
    return obj


def deref(pointer: Any) -> Any:
    """``*p`` when the type of ``p`` was not known: a cell's value, an array's first element.

    A library object that holds a value as a cell does says so with a true
    ``_cint_cell``.
    """
    if isinstance(pointer, (np.ndarray, list)):
        return pointer[0]  # a C array
    return held(pointer)


def held(pointer: Any) -> Any:
    """What a cell, a reference or a library's value holder holds - or ``pointer`` itself."""
    named = type(pointer).__name__ in ("Cell", "ItemRef", "AttrRef")
    if hasattr(pointer, "value") and (named or getattr(pointer, "_cint_cell", False)):
        return pointer.value
    return pointer


def uninstantiated(template: str, why: str) -> None:
    """A template used where its arguments were not worked out, whose body needs them.

    Each instance of a template is translated with its arguments in; the
    template itself, for a use whose arguments the translator could not
    tell, only where its body does not depend on them. Where it does, using
    it is refused here, when it is used, with the reason the body could not
    be written.
    """
    raise Refusal(f"{template}, used where its template arguments are not known: {why}")
