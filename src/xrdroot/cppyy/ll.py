"""``cppyy.ll``: the low-level parts - casts, addresses, and C++ signals as exceptions.

A cast of a number to a fundamental type is the conversion C++ makes
(``ll.cast['int'](3.7)`` is ``3``, ``ll.cast['unsigned char'](300)`` is
``44``); a cast of an object to a class is the object seen as that class,
which here is the object itself - Python's objects know their own type.
Signals are not raised by translated C++, so turning them into exceptions
changes nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from .pointers import addressof, bind_object, nullptr

__all__ = [
    "cast",
    "static_cast",
    "reinterpret_cast",
    "dynamic_cast",
    "addressof",
    "nullptr",
    "array_new",
    "array_delete",
    "FatalError",
    "BusError",
    "SegmentationViolation",
    "IllegalInstruction",
    "AbortSignal",
    "set_signals_as_exception",
    "signals_as_exception",
]


class FatalError(Exception):
    """A C++ signal, raised as a Python exception."""


class BusError(FatalError):
    pass


class SegmentationViolation(FatalError):
    pass


class IllegalInstruction(FatalError):
    pass


class AbortSignal(FatalError):
    pass


def _numeric(kind: str) -> np.dtype[Any] | None:
    from ..cppstd.containers import element_type

    try:
        found = element_type(kind)
    except TypeError:
        return None
    return found if isinstance(found, np.dtype) else None


class _Cast:
    """``cast[T](value)``: ``value`` as the type ``T``, by C++'s conversion for a number."""

    def __getitem__(self, kind: Any) -> Callable[[Any], Any]:
        dtype = _numeric(kind) if isinstance(kind, str) else None

        def cast(value: Any) -> Any:
            if dtype is None:
                return value if not isinstance(value, int) else bind_object(value, kind)
            return np.array(value).astype(dtype).item()

        return cast


#: ``cppyy.ll.cast`` and its C++ spellings, all one here.
cast = static_cast = reinterpret_cast = dynamic_cast = _Cast()


def array_new(kind: Any, size: int) -> Any:
    """``array_new[T](n)``-like: an array of ``n`` zeros of a fundamental type."""
    dtype = _numeric(str(kind))
    return np.zeros(int(size), dtype=dtype if dtype is not None else object)


def array_delete(array: Any) -> None:
    """``array_delete``: what Python's collector does once nothing refers to the array."""


def set_signals_as_exception(on: bool, *args: Any) -> bool:
    """Turn C++ signals into exceptions, or not; translated C++ raises none either way."""
    return True


@contextmanager
def signals_as_exception(*args: Any) -> Iterator[None]:
    """``with cppyy.ll.signals_as_exception():``, which has no signal to catch here."""
    yield
