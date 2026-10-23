"""Interpreter object helpers retain ROOT lifetime and pointer adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...cppstd.objects import DTYPES as DTYPES
from ...cppstd.objects import INTEGRAL as INTEGRAL
from ...cppstd.objects import REAL as REAL
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
from ...cppstd.objects import dynamic_cast as dynamic_cast
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

__all__ = [
    "array",
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
]


def delete(obj: Any) -> None:
    """``delete p``: a user class's destructor runs; a ROOT file is closed, as its would be."""
    destructor = getattr(obj, "_destruct", None)
    if callable(destructor):
        destructor()
        return
    close = getattr(obj, "Close", None)
    if callable(close) and hasattr(obj, "IsOpen"):
        close()


def construct_at(container: Any, index: Any, obj: Any) -> Any:
    """``new (clones[i]) T(args)``: ``obj`` put in slot ``index`` - ROOT's ``AddAt``, or ``[i] =``.

    A ``TClonesArray`` hands placement new the memory of its slot ``i``;
    here the object is built first and the array given it, which is the
    same array of the same objects afterwards.
    """
    add = getattr(container, "AddAt", None)
    if callable(add):
        add(obj, int(index))
    else:
        container[index] = obj
    return obj


def deref(pointer: Any) -> Any:
    """``*p`` when the type of ``p`` was not known: a cell's value, an array's first element.

    A library object that holds a value as a cell does - an RNTuple field's
    ``shared_ptr<int>`` - says so with a true ``_cint_cell``.
    """
    if isinstance(pointer, (np.ndarray, list)) and not hasattr(pointer, "ClassName"):
        return pointer[0]  # a C array; a ROOT collection that is a list, *list, is the list
    named = type(pointer).__name__ in ("Cell", "ItemRef", "AttrRef")
    if hasattr(pointer, "value") and (named or getattr(pointer, "_cint_cell", False)):
        return pointer.value
    return pointer
