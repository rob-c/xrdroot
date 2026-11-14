"""``cppyy.py``: pythonizations - Python run on a C++ class the first time it is looked up.

``add_pythonization(callback, namespace)`` registers ``callback(klass, name)``,
called once for each class ``cppyy.gbl`` (or a namespace under it) hands out,
so a program can give C++ classes Python's protocols - ``__len__``,
``__iter__`` - where it knows what they mean.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["add_pythonization", "remove_pythonization", "pythonized", "pin_type", "compose_method"]

#: The callbacks registered, each with the namespace whose classes it is called for.
CALLBACKS: list[tuple[Callable[..., Any], str]] = []
#: The classes each callback was run on already, so it runs once for each.
_DONE: set[tuple[int, int]] = set()


def add_pythonization(callback: Callable[..., Any], namespace: str = "") -> None:
    """Call ``callback(klass, name)`` on each class of ``namespace`` looked up from now on."""
    CALLBACKS.append((callback, str(namespace).strip(":")))


def remove_pythonization(callback: Callable[..., Any], namespace: str = "") -> bool:
    """Forget ``callback``; was it registered?"""
    entry = (callback, str(namespace).strip(":"))
    if entry in CALLBACKS:
        CALLBACKS.remove(entry)
        return True
    return False


def pythonized(found: Any, name: str, namespace: str = "") -> Any:
    """``found``, a class, after each callback for its namespace has run on it once."""
    if not isinstance(found, type):
        return found
    for callback, wanted in CALLBACKS:
        key = (id(callback), id(found))
        if wanted == namespace and key not in _DONE:
            _DONE.add(key)
            callback(found, name)
    return found


def pin_type(klass: Any) -> None:
    """``pin_type``: objects of ``klass`` handed back as it, not a class derived from it -
    which they are here already, each knowing its own class."""


def compose_method(klass: Any, name: str, combined: Callable[..., Any]) -> None:
    """``compose_method``: ``klass.name`` replaced by ``combined(self, original, *args)``."""
    original = getattr(klass, name)

    def composed(self: Any, *args: Any, **kwargs: Any) -> Any:
        return combined(self, original, *args, **kwargs)

    setattr(klass, name, composed)
