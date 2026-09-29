"""A macro's C++ function given to a frame: called once per entry, as ROOT calls it.

xrdroot's frame gives a callable whole batches of entries, which NumPy code
wants; a C++ lambda a macro passes - ``Filter([](double m) { return m < 3; },
{"m"})`` - is written for one entry, so it is called for each: with the
entry's numbers, strings and - for a collection - an ``RVec``, and what it
gives back for each makes the batch's column. A function is a macro's when
the translation defined it, in the module ``__cint__``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np

from ...tree import Jagged
from ..stl import _Vector
from .rvec import _made

__all__ = ["ENTRYWISE", "entrywise", "per_entry"]

#: The frame's methods ROOT calls a function of once per entry, by name.
ENTRYWISE = frozenset({"Filter", "Define", "Redefine", "Foreach", "DefineSlot", "ForeachSlot"})


def _entries(column: Any) -> list[Any]:
    """A batch's values, one for each entry, as a C++ function takes them."""
    if isinstance(column, Jagged):
        return [_made(np.asarray(row)) for row in column]
    if isinstance(column, np.ndarray):
        return list(column.tolist()) if column.dtype != object else list(column)
    return list(column)


def is_macros(function: Any) -> bool:
    """Is ``function`` one a macro's translation defined?"""
    return callable(function) and getattr(function, "__module__", None) == "__cint__"


def entrywise(function: Callable[..., Any]) -> Callable[..., Any]:
    """``function`` called once per entry of a batch, its answers the batch's column."""

    @functools.wraps(function)
    def batch(*columns: Any) -> Any:
        return _column([function(*values) for values in zip(*(_entries(c) for c in columns))])

    return batch


def _column(results: list[Any]) -> Any:
    """What a batch's answers make: numbers an array, collections a ``Jagged``, else a list."""
    if results and all(isinstance(each, (_Vector, np.ndarray)) for each in results):
        rows = [np.asarray(each.data() if isinstance(each, _Vector) else each) for each in results]
        offsets = np.concatenate([[0], np.cumsum([len(row) for row in rows])])
        return Jagged(np.concatenate(rows), offsets)
    found = np.asarray(results)
    return found if found.ndim == 1 and found.dtype.kind in "biuf" else results


def per_entry(name: str, arguments: list[Any]) -> list[Any]:
    """The arguments of frame method ``name``, a macro's function among them made entrywise."""
    if name not in ENTRYWISE:
        return arguments
    return [entrywise(each) if is_macros(each) else each for each in arguments]
