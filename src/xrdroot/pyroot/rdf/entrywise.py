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
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np

from ...tree import Jagged
from ..stl import _Vector
from .rvec import _made

__all__ = ["ENTRYWISE", "entrywise", "per_entry"]

#: The frame's methods ROOT calls a function of once per entry: where the function is among
#: their arguments, and whether it is given the slot first.
ENTRYWISE = {"Filter": (0, False), "Define": (1, False), "Redefine": (1, False),
             "Foreach": (0, False), "DefineSlot": (1, True), "ForeachSlot": (0, True)}  # fmt: skip

#: The column a function of no columns is given, so that it is called once for each entry.
ENTRY = "rdfentry_"


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


def entrywise(function: Callable[..., Any], slot: bool = False,
              counted: bool = False) -> Callable[..., Any]:  # fmt: skip
    """``function`` called once per entry of a batch, its answers the batch's column: after the
    slot, when it takes one; with nothing, when it reads no column but is ``counted`` by one."""

    @functools.wraps(function)
    def batch(*given: Any) -> Any:
        first, columns = (list(given[:1]), given[1:]) if slot else ([], given)
        rows = zip(*(_entries(c) for c in columns), strict=False)
        return _column([function(*first, *([] if counted else values)) for values in rows])

    return batch


def _column(results: list[Any]) -> Any:
    """What a batch's answers make: numbers an array, collections a ``Jagged``, else a list."""
    if results and all(isinstance(each, (_Vector, np.ndarray)) for each in results):
        rows = [np.asarray(each.data() if isinstance(each, _Vector) else each) for each in results]
        offsets = np.concatenate([[0], np.cumsum([len(row) for row in rows])])
        return Jagged(np.concatenate(rows), offsets)
    found = _numbers(results)
    return found if found is not None else results


def _numbers(results: list[Any]) -> Any:
    """The answers as an array of numbers, or ``None`` - collections of collections, ragged,
    or strings, which stay a column of objects."""
    try:
        found = np.asarray(results)
    except ValueError:
        return None
    return found if found.ndim == 1 and found.dtype.kind in "biuf" else None


def _arity(function: Callable[..., Any]) -> int:
    """How many arguments a macro's function takes: its parameters without a default, which
    the translation gives what a lambda captured."""
    found = inspect.signature(function).parameters.values()
    return sum(each.default is inspect.Parameter.empty for each in found)


def per_entry(name: str, arguments: list[Any]) -> list[Any]:
    """The arguments of frame method ``name``, a macro's function among them made entrywise -
    given the entry number to be counted by, when it reads no column."""
    at, slot = ENTRYWISE.get(name, (-1, False))
    if at < 0 or at >= len(arguments) or not is_macros(arguments[at]):
        return arguments
    given = list(arguments)
    columns = list(given[at + 1]) if len(given) > at + 1 else []
    counted = not columns and _arity(given[at]) == int(slot)
    given[at] = entrywise(given[at], slot, counted)
    if counted:
        given[at + 1:at + 2] = [[ENTRY]]
    return given
