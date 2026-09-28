"""Assigning to what a call returns by reference: ``m(i, j) = v``, ``p.X() = v``.

C++ lets a function hand back a reference and the caller assign through
it - ``TMatrixD::operator()(i, j)`` is an element of the matrix, and
``m(i, j) = 1`` writes it. Python has no references to return, so the
translation says what is assigned to instead, and these write it:

* :func:`assign_call` is ``obj(args) = value``: the object's
  ``__setcall__(*args, value)`` if it has one (the counterpart to
  ``__call__`` pyroot's matrices, vectors and tensors keep), else its
  ``__setitem__`` with the arguments as the key;
* :func:`assign_method` is ``obj.Name(args) = value``: ROOT's
  ``SetName(value)`` where there is one and no arguments, else what the
  method returned is assigned into with :func:`assign_into`;
* :func:`assign_into` is copy-assignment into an object someone else
  holds: its own ``operator=`` (``_assign``), a container's contents
  replaced, or an object's state taken from a copy of the value.

Each gives back the value, as C++'s assignment does, for ``a(i) = b(j) = v``.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

__all__ = ["assign_call", "assign_method", "assign_into", "store_through"]


def assign_call(obj: Any, args: tuple[Any, ...], value: Any) -> Any:
    """``obj(*args) = value``, through ``__setcall__`` or ``__setitem__``."""
    setcall = getattr(obj, "__setcall__", None)
    if callable(setcall):
        setcall(*args, value)
        return value
    if hasattr(obj, "__setitem__"):
        obj[args[0] if len(args) == 1 else args] = value
        return value
    raise TypeError(
        f"{type(obj).__name__} has neither __setcall__ nor __setitem__, so what its "
        "operator() returns cannot be assigned to"
    )


def assign_method(obj: Any, name: str, args: tuple[Any, ...], value: Any) -> Any:
    """``obj.name(*args) = value``: ROOT's ``Set<name>``, or an assignment into the result."""
    setter = getattr(obj, f"Set{name}", None)
    if not args and callable(setter):
        setter(value)
        return value
    return assign_into(getattr(obj, name)(*args), value)


def store_through(pointer: Any, value: Any) -> Any:
    """``*p = value``: into the cell (or array) ``p`` is, or the object it points at."""
    # A cell's value is data; an object whose ``value`` is a method is no cell.
    held = getattr(pointer, "value", assign_into)
    if not callable(held) and not hasattr(pointer, "_assign"):
        pointer.value = value
        return value
    if isinstance(pointer, (np.ndarray, list)):
        pointer[0] = value
        return value
    return assign_into(pointer, value)


def assign_into(target: Any, value: Any) -> Any:
    """``target = value`` where ``target`` is an object held elsewhere: copied into, in place."""
    own = getattr(target, "_assign", None)
    if callable(own):
        own(value)
    elif isinstance(target, np.ndarray):
        target[...] = value
    elif isinstance(target, list):
        target[:] = list(value)
    elif isinstance(target, (set, dict)):
        target.clear()
        target.update(value)
    elif hasattr(target, "__dict__") and not isinstance(value, (int, float, str, bool)):
        target.__dict__.update(copy.copy(value).__dict__)
    else:
        raise TypeError(
            f"a {type(target).__name__} returned by value cannot be assigned to in place"
        )
    return value
