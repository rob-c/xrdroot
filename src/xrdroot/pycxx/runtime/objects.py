"""What a translation calls for objects, pointers and arrays: the standard's, over :mod:`cppstd`.

The helpers here are what plain C++ means by ``delete``, placement ``new``
and ``*p``; a dialect's runtime may export its own of the same name for its
library's objects, which a translation in that dialect then calls instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...cppstd.buffers import CharBuffer
from ...cppstd.buffers import text_into as text_into
from ...cppstd.objects import AN_OBJECT as AN_OBJECT
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
from ...cppstd.objects import sorted_range as sorted_range
from ...cppstd.objects import throw as throw
from ...cppstd.objects import value_copy as value_copy
from ..errors import Refusal

__all__ = [
    "array",
    "char_array",
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
    "sorted_range",
    "reverse_range",
    "CppException",
    "throw",
    "deref",
    "pointee",
    "preinc",
    "postinc",
    "INTEGRAL",
    "REAL",
    "AN_OBJECT",
    "uninstantiated",
    "inherited",
    "base_member",
    "release",
    "set_heuristic",
]


def char_array(shape: Any, init: Any = None) -> Any:
    """``char name[2][8]``: the characters as one buffer of the shape declared.

    A ``char[N][M]`` is N rows of M bytes, not an array of C strings: ``name[i]``
    is a view of row ``i`` - the C string that row holds, which ``strcpy`` and
    ``&name[i][0]`` write in place - and ``name[i][j]`` its j-th byte. Each
    string of ``init`` is copied into a row of its own, NUL and all, as C
    initialises it; a braced list of characters fills the buffer in its own
    order, since that is the order C lays them down in.
    """
    out = np.zeros(tuple(int(size) for size in shape), dtype=np.int8).view(CharBuffer)
    if init is None:
        return out
    flat = out.reshape(-1)
    rows = flat.reshape(-1, out.shape[-1])
    for index, item in enumerate(_flat(init)):
        if isinstance(item, str):
            text_into(rows[index], item)
        else:
            flat[index] = int(item)
    return out


def inherited(obj: Any, name: str, host: Any) -> Any:
    """``f(...)`` in a method of a class derived from a library's class: the member ``f`` the
    object has - its class's or a base's, as C++ looks in the class first - or else the
    library's function of that name."""
    if hasattr(type(obj), name) or name in getattr(obj, "__dict__", ()):  # a data member too
        return getattr(obj, name)
    return getattr(host, name)


def base_member(base: Any, name: str, obj: Any) -> Any:
    """``Base::f(...)`` in a method: ``Base``'s own ``f`` called on this object - or, for a
    static member, the function itself."""
    import inspect

    raw = inspect.getattr_static(base, name)
    if isinstance(raw, (staticmethod, classmethod)) or not hasattr(raw, "__get__"):
        return getattr(base, name)
    return raw.__get__(obj, base)


#: cppyy's heuristic memory policy: off since ROOT 6.40 unless a program turns it on again.
HEURISTIC = False


def set_heuristic(enabled: bool) -> bool:
    """``ROOT.SetHeuristicMemoryPolicy``: turn the heuristic on or off; whether it was."""
    global HEURISTIC
    before, HEURISTIC = HEURISTIC, bool(enabled)
    return before


def release(obj: Any) -> None:
    """A call handing an instance to a non-const pointer parameter: with the heuristic on,
    the C++ side is assumed to keep it, so ``del`` leaves it alone - though an explicit
    ``TClass::Destructor`` still runs the destructor, as cppyy's proxy allows."""
    if HEURISTIC and hasattr(obj, "__dict__"):
        obj.__dict__["_cxx_released"] = True


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
