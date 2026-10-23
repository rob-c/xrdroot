"""C++ reference stores and copy-assignment preserve their destination lvalues.

Containers register their native layouts explicitly. Generic assignment can
then preserve vector storage without importing or discovering an adapter.
User assignment hooks keep their original priority and return contracts.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .valuecopy import value_copy
from .vectorassign import assigned as _assigned_vector

__all__ = ["assign_call", "assign_into", "store_through", "assign_vector"]


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
    elif _vector_assigned(target, value):
        return target
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


def _vector_assigned(target: Any, value: Any) -> bool:
    """Registered native factories supply layouts; foreign objects never load their adapters."""
    return _assigned_vector(target, value)


def assign_vector(
    target: Any, value: Any, through_pointer: bool = False, copy_on_decline: bool = True,
) -> Any:
    """Keep raw self-assignment native; declined pointer stores retain RHS-copy/hook order."""
    if _vector_assigned(target, value):
        return target
    copied = value_copy(value) if copy_on_decline else value
    return store_through(target, copied) if through_pointer else assign_into(target, copied)
