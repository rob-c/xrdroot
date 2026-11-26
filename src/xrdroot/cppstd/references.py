"""C++ reference stores and copy-assignment preserve their destination lvalues.

Containers register their native layouts explicitly. Generic assignment can
then preserve vector storage without importing or discovering an adapter.
User assignment hooks keep their original priority and return contracts.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from .valuecopy import value_copy
from .vectorassign import assigned as _assigned_vector

__all__ = ["assign_call", "assign_into", "store_through", "assign_vector", "Tie", "tie",
           "IGNORE",
           "store_reference", "increment_reference"]


def store_reference(
    value: Any, reference: Any, conversion: Callable[[Any, Any], Any], compound: bool = False,
) -> Any:
    # RHS is evaluated before the destination; compound stores read that destination once.
    reference.value = conversion(reference.value if compound else None, value)
    return reference


def increment_reference(reference: Any, delta: int) -> Any:
    # Prefix increment is an lvalue; the typed branch address narrows the stored result.
    reference.value = reference.value + delta
    return reference


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


class Tie:
    """``std::tie(a, b)``: the names on one side of an assignment, as tuple unpacking is.

    C++ lets a tuple of references be assigned: ``std::tie(a, b) = pair`` copies into ``a`` and
    ``b``. The translation keeps the call - ``assign_call(std.tie, (a, b), value)`` - so the
    distribution happens here, through :func:`assign_into`, which copies a container *into* the
    object it is given rather than rebinding a name Python cannot rebind.
    """

    __slots__ = ()

    def __setcall__(self, *given: Any) -> Any:
        """``std::tie(...) = value``: each target takes its part, in order."""
        *targets, value = given
        parts = list(value)
        if len(parts) != len(targets):
            raise ValueError(
                f"std::tie names {len(targets)} targets and the right-hand side gave "
                f"{len(parts)} values; C++ needs one for each"
            )
        for target, part in zip(targets, parts, strict=True):
            if target is not IGNORE:  # std::ignore takes its part and keeps nothing
                assign_into(target, part)
        return value

    def __repr__(self) -> str:
        return "<std::tie>"


#: ``std::tie``: what an assignment target of several names is translated to.
tie = Tie()


class _Ignore:
    """``std::ignore``: the tie target that takes its value and keeps nothing."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<std::ignore>"


#: ``std::ignore``: skipped by :meth:`Tie.__setcall__`, as C++ skips it.
IGNORE = _Ignore()


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
