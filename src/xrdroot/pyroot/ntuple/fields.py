"""What an RNTuple field holds, and where an entry keeps its value while it is filled or read.

A field is declared with a C++ type - ``int``, ``std::string``,
``std::vector<float>`` - which ROOT normalises to the name its files keep,
``std::int32_t`` for ``int``. Its value lives in a holder the model hands
out: for a number or a string a :class:`RFieldPtr`, the ``shared_ptr<T>``
a macro writes through with ``*p = v``; for a vector the ``std::vector``
itself, which a macro fills with ``push_back``. A writer reads the holders
when it fills; a reader writes into them when it loads an entry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ...rntuple.writer import CXX
from ..stl import _spelled, element_type, std, string

__all__ = ["FieldType", "RFieldPtr", "field_type", "typed"]


class FieldType:
    """A field's type: the C++ name ROOT's files keep, and the number type of its values."""

    def __init__(self, cxx: str, dtype: np.dtype[Any] | None, vector: bool) -> None:
        self.cxx, self.dtype, self.vector = cxx, dtype, vector

    @property
    def spec(self) -> Any:
        """What :meth:`xrdroot.WritableFile.rntuple` declares a field of this type with."""
        return self.cxx if self.dtype is not None else str

    def holder(self) -> Any:
        """A fresh value for an entry of this field: zero, empty, or an empty vector."""
        if self.vector:
            return std.vector[CPP_ITEMS[str(self.dtype)]]()
        return RFieldPtr(self)

    def zero(self) -> Any:
        return "" if self.dtype is None else self.dtype.type(0).item()


#: The C++ name a vector of each number type is made with, by NumPy name.
CPP_ITEMS = {"float32": "float", "float64": "double", "bool": "bool", "int8": "signed char",
             "uint8": "unsigned char", "int16": "short", "uint16": "unsigned short",
             "int32": "int", "uint32": "unsigned int", "int64": "long",
             "uint64": "unsigned long"}  # fmt: skip


def _refusal(whole: str) -> UnsupportedFeatureError:
    return UnsupportedFeatureError(
        f"an RNTuple field of type {whole} is not supported: xrdroot's RNTuple writer and "
        f"entries hold numbers, std::string and std::vector of a number, whose layout is "
        f"beyond doubt; records, nested collections and variants are not written."
    )


def _number(text: str, whole: str) -> np.dtype[Any] | None:
    """The NumPy type of a number spelled ``text``, ``None`` for a string; else a refusal."""
    try:
        kind = element_type(text)
    except TypeError:
        raise _refusal(whole) from None
    if kind is string:
        return None
    if isinstance(kind, np.dtype) and kind.name in CXX:
        return kind
    raise _refusal(whole)


#: How a vector's C++ name starts, however it is spelled.
VECTORS = ("vector<", "ROOT::VecOps::RVec<", "ROOT::RVec<", "RVec<")


def field_type(cxx: Any) -> FieldType:
    """The field type a C++ type name declares: ``int`` is ROOT's ``std::int32_t``."""
    text = _spelled(str(cxx))
    if text.startswith(VECTORS) and text.endswith(">"):
        item = _number(text[text.index("<") + 1 : -1], str(cxx))
        if item is None:  # a vector of strings: a nested collection
            raise _refusal(str(cxx))
        return FieldType(f"std::vector<{CXX[item.name]}>", item, True)
    found = _number(text, str(cxx))
    return FieldType("std::string" if found is None else CXX[found.name], found, False)


class _Typed:
    """A template method with its type given: ``model.MakeField["int"]("n")``."""

    def __init__(self, fn: Any, obj: Any, kind: Any) -> None:
        self._fn, self._obj, self._kind = fn, obj, kind

    def __call__(self, *args: Any) -> Any:
        return self._fn(self._obj, self._kind, *args)


class _Untyped:
    """A template method not yet given its type, which it is by ``[...]``."""

    def __init__(self, fn: Any, obj: Any) -> None:
        self._fn, self._obj = fn, obj

    def __getitem__(self, kind: Any) -> _Typed:
        return _Typed(self._fn, self._obj, kind)

    def __call__(self, *args: Any) -> Any:
        """Called without a type: the method finds it from what it is given, if it can."""
        return self._fn(self._obj, None, *args)


class typed:
    """A method C++ declares as a template on a type it needs: called as ``f[T](...)``."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self.__doc__ = fn.__doc__

    def __get__(self, obj: Any, owner: Any = None) -> Any:
        return self if obj is None else _Untyped(self._fn, obj)


class RFieldPtr:
    """``std::shared_ptr<T>`` to a number or a string an entry holds: ``*p`` is ``p.value``."""

    #: What tells the macro runtime's ``deref`` that ``*p`` is this holder's value.
    _cint_cell = True

    def __init__(self, kind: FieldType) -> None:
        self.kind = kind
        self.value: Any = kind.zero()

    def __repr__(self) -> str:
        return f"<shared_ptr<{self.kind.cxx}> holding {self.value!r}>"

    def __getitem__(self, index: int) -> Any:
        return self.value

    def __setitem__(self, index: int, value: Any) -> None:
        self.value = value
