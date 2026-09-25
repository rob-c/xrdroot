"""What ROOT takes by address: somewhere to put a number, however a script made it.

``gRandom->Rannor(px, py)`` fills two ``Double_t`` by reference. A Python
script hands over a NumPy array of one or more elements, an
``array.array``, a ``ctypes.c_double``, or any object with a ``.value`` to
set - which is what the C++ translator's cells are - and each is filled as
the C++ variable would have been.
"""

from __future__ import annotations

from typing import Any

__all__ = ["store", "load", "store_many"]


def _indexed(target: Any) -> bool:
    """Is ``target`` filled by index - an array, rather than something with ``.value``?"""
    return not hasattr(target, "value") and hasattr(target, "__setitem__")


def store(target: Any, value: Any) -> None:
    """Put ``value`` where ``target`` says, as ``*target = value`` would."""
    if target is None:
        return
    if _indexed(target):
        target[0] = value
        return
    if not hasattr(target, "value"):
        raise TypeError(
            f"a {type(target).__name__} is not somewhere to put a number: hand over a NumPy "
            f"array, an array.array, a ctypes number or anything with a .value"
        )
    target.value = value


def load(target: Any) -> Any:
    """What ``target`` holds now: ``*target``."""
    return target[0] if _indexed(target) else target.value


def store_many(target: Any, values: Any) -> None:
    """Fill an array ``target`` with ``values`` from its start, as a C array is filled."""
    for index, value in enumerate(values):
        target[index] = value
