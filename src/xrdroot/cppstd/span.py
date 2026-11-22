"""``std::span``: a window onto elements someone else owns.

C++'s ``std::span<T>`` is a pointer and a length, and it copies nothing: a
function handed a span writes the caller's own elements. Python cannot hand
out the address of a local ``double``, so what a translated ``&x`` gives is a
:class:`~xrdroot.cppstd.cells.Cell`, and what ``&v[i]`` gives is an
``ItemRef``; both answer ``x[0]`` for reading and for writing. A span
therefore holds *the thing that can be indexed* - a cell, or the sequence
itself (a ``std::vector``, a NumPy array, a list) - and indexes that, which is
how ``result[i] = ...`` inside a function reaches the caller's variable.

The element type is not kept. ``std::span<const double>`` and
``std::span<double>`` are one class here, because the whole of what a span
does - index what it was given - does not depend on it; a program that wrote
through a ``const`` span would not have compiled in C++ either.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

__all__ = ["span"]


class span:
    """``count`` elements starting at what ``data`` indexes; all of it if ``count`` is absent.

    ``std::span<double> s;`` is a legal C++ declaration of an empty span, so a
    span made from nothing is empty rather than an error.
    """

    __slots__ = ("_data", "_size")

    def __init__(self, data: Any = None, count: Any = None) -> None:
        self._data = [] if data is None else data
        self._size = len(self._data) if count is None else int(count)

    @classmethod
    def __class_getitem__(cls, element: Any) -> type[span]:
        """``std::span<float>``: this class, whatever it is a span of."""
        return cls

    def size(self) -> int:
        return self._size

    def empty(self) -> bool:
        return self._size == 0

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> Any:
        return self._data[index]

    def __setitem__(self, index: int, value: Any) -> None:
        self._data[index] = value

    def __iter__(self) -> Iterator[Any]:
        for index in range(self._size):
            yield self._data[index]

    def __repr__(self) -> str:
        return f"span({self._size} elements)"
