"""Whole numeric vector copy-assignment keeps the destination object and reserved storage."""
from __future__ import annotations

from typing import Any

import numpy as np

from .vectorvalue import _original_readers, native

#: Factories supply original access descriptors before user specializations can replace them.
_LAYOUTS: dict[type, tuple[str, dict[str, Any]]] = {}
#: ROOT's SmallVectorBase stores its size and capacity in signed 32-bit integers.
_RVEC_MAX = (1 << 31) - 1


def register(kind: type, family: str, readers: dict[str, Any]) -> type:
    """Assignment provenance stays external so numeric row boxing keeps its class fields."""
    _LAYOUTS[kind] = family, readers
    return kind


def _layout(kind: type, source: Any) -> str | None:
    if type(kind) is not type or type(source) is not kind or not native(kind):
        return None
    found = _LAYOUTS.get(kind)
    if found is None or not _original_readers(kind, found[1]):
        return None
    if any("__getattr__" in vars(base) for base in kind.__mro__):
        return None
    return found[0]


def _array(data: np.ndarray[Any, Any], kind: type) -> bool:
    return (data.ndim == 1 and data.dtype == vars(kind)["dtype"]
            and data.dtype.kind in "iuf")


def _writable(data: np.ndarray[Any, Any]) -> bool:
    return bool(data.flags.c_contiguous and data.flags.aligned and data.flags.writeable)


def _storage(value: Any, kind: type) -> tuple[np.ndarray[Any, Any], int] | None:
    try:
        data = object.__getattribute__(value, "_data")
        size = object.__getattribute__(value, "_size")
    except AttributeError:
        return None
    if type(data) is not np.ndarray or type(size) is not int:
        return None
    if not _array(data, kind) or not _writable(data):
        return None
    return (data, size) if 0 <= size <= data.size else None


def _capacity(family: str, current: int, required: int) -> int:
    if family == "std":
        return required
    return min(max(required, 2 * current + 1), _RVEC_MAX)


def assigned(target: Any, source: Any) -> bool:
    """Decline foreign layouts before any public data, iterator or assignment callback."""
    kind = type(target)
    family = _layout(kind, source)
    if family is None:
        return False
    destination, origin = _storage(target, kind), _storage(source, kind)
    if destination is None or origin is None:
        return False
    if target is source:
        return True
    data, _ = destination
    values, size = origin
    if np.shares_memory(data, values):
        return False
    if size > data.size:
        data = np.zeros(_capacity(family, data.size, size), dtype=data.dtype)
    data[:size] = values[:size]
    if data is not destination[0]:
        object.__setattr__(target, "_data", data)
    object.__setattr__(target, "_size", size)
    return True
