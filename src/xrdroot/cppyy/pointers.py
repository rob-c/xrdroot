"""Pointers as cppyy hands them to Python: addresses, objects bound to them, and ``nullptr``.

There is no memory to point into here, only objects. An address is a number
standing for one object while it lives - the number ``addressof`` gives,
which ``bind_object`` turns back into the object, seen as the class asked
for. ``nullptr`` is C++'s null pointer, and binding it to a class gives that
class's null object: false, refusing to be compared with ``None`` - the
mistake ROOT 6.40 stopped allowing - and equal to another null.
"""

from __future__ import annotations

import weakref
from typing import Any

__all__ = ["nullptr", "addressof", "bind_object", "is_null"]


class _NullPtr:
    """``cppyy.nullptr``: C++'s null pointer, false and printed as ``nullptr``."""

    _made: _NullPtr | None = None

    def __new__(cls) -> _NullPtr:
        if cls._made is None:
            cls._made = super().__new__(cls)
        return cls._made

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "nullptr"


#: ``cppyy.nullptr``.
nullptr = _NullPtr()

#: The objects ``addressof`` was asked for, by the number it gave: weakly where they can be.
_WEAK: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
_STRONG: dict[int, Any] = {}

#: The null object of each class, made once.
_NULLS: dict[type, type] = {}


def addressof(obj: Any, field: Any = None, byref: bool = False) -> Any:
    """``cppyy.addressof``: the number standing for ``obj`` - or, with a member's name, that
    member's place - as ``ROOT.addressof`` gives it; ``bind_object`` finds the object again."""
    from ..pyroot.core.objects import addressof as roots

    if field is not None:
        return roots(obj, field)
    if is_null(obj):
        return 0
    address = int(roots(obj))
    try:
        _WEAK[address] = obj
    except TypeError:  # an object without a weak reference: kept for good
        _STRONG[address] = obj
    return address


def is_null(obj: Any) -> bool:
    """Is ``obj`` a null pointer: ``nullptr``, ``None``, ``0`` or a class's null object?"""
    if obj is None or obj is nullptr or type(obj) in _NULLS.values():
        return True
    return isinstance(obj, int) and not isinstance(obj, bool) and obj == 0


def bind_object(address: Any, kind: Any) -> Any:
    """``cppyy.bind_object``: the object at ``address`` seen as of class ``kind`` (a class, or
    its name); for a null address, the class's null object."""
    cls = _class(kind)
    if is_null(address):
        return object.__new__(_null_class(cls))
    found = address if not isinstance(address, int) else _WEAK.get(address, _STRONG.get(address))
    if found is None:
        raise TypeError(f"no object is known at address {address:#x}: an address is one "
                        "cppyy.addressof gave, of an object still alive")  # fmt: skip
    return found


def _class(kind: Any) -> type:
    if isinstance(kind, type):
        return kind
    from . import gbl

    found: Any = gbl
    for part in str(kind).split("::"):
        found = getattr(found, part)
    return found  # type: ignore[no-any-return]


def _null_class(cls: type) -> type:
    """The class of ``cls``'s null objects: a ``cls`` that is false and refuses ``== None``."""
    made = _NULLS.get(cls)
    if made is None:
        members = {"__bool__": _false, "__eq__": _equal, "__ne__": _unequal,
                   "__hash__": object.__hash__, "__repr__": _null_repr}  # fmt: skip
        made = _NULLS[cls] = type(cls.__name__, (cls,), members)
    return made


def _false(self: Any) -> bool:
    return False


def _equal(self: Any, other: Any) -> bool:
    if other is None:
        raise TypeError(
            "a C++ null pointer is not compared with None: test it with bool(), or compare it "
            "with cppyy.nullptr"
        )
    return is_null(other)


def _unequal(self: Any, other: Any) -> bool:
    return not _equal(self, other)


def _null_repr(self: Any) -> str:
    return f"<cppyy.gbl.{type(self).__name__} object at 0x0>"
