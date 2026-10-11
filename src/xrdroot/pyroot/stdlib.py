"""``std::numeric_limits``, the smart pointers and ``std::remove``, for ``ROOT.std``.

``std::numeric_limits<int>::min()`` is a number a macro uses as a sentinel
(df036); ``std::unique_ptr<RooArgSet>{p}`` holds an object the macro goes on
to use, which here is the object itself, Python keeping it while it is
used; ``std::remove(name)`` deletes a file, as C's ``remove`` does, zero
when it did.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

__all__ = ["numeric_limits", "remove", "smart_pointer"]

#: The NumPy type of each C++ arithmetic type ``numeric_limits`` is asked of.
_KINDS = {
    "bool": np.bool_,
    "char": np.int8,
    "signed char": np.int8,
    "unsigned char": np.uint8,
    "short": np.int16,
    "unsigned short": np.uint16,
    "int": np.int32,
    "unsigned int": np.uint32,
    "unsigned": np.uint32,
    "long": np.int64,
    "unsigned long": np.uint64,
    "long long": np.int64,
    "unsigned long long": np.uint64,
    "Long64_t": np.int64,
    "ULong64_t": np.uint64,
    "size_t": np.uint64,
    "float": np.float32,
    "double": np.float64,
    "Float_t": np.float32,
    "Double_t": np.float64,
    "Int_t": np.int32,
    "UInt_t": np.uint32,
}


class _Limits:
    """``std::numeric_limits<T>``: the range and precision of one arithmetic type."""

    def __init__(self, kind: Any) -> None:
        self._kind = np.dtype(kind)

    def _info(self) -> Any:
        if self._kind.kind == "b":
            return None
        return np.finfo(self._kind) if self._kind.kind == "f" else np.iinfo(self._kind)

    def min(self) -> Any:
        """The least value - for a floating type, its least positive normal one, as C++'s."""
        info = self._info()
        if info is None:
            return False
        return float(info.tiny) if self._kind.kind == "f" else int(info.min)

    def max(self) -> Any:
        info = self._info()
        if info is None:
            return True
        return float(info.max) if self._kind.kind == "f" else int(info.max)

    def lowest(self) -> Any:
        info = self._info()
        if info is None:
            return False
        return -float(info.max) if self._kind.kind == "f" else int(info.min)

    def epsilon(self) -> Any:
        info = self._info()
        return float(info.eps) if self._kind.kind == "f" else 0

    def infinity(self) -> float:
        return float("inf")

    def quiet_NaN(self) -> float:
        return float("nan")


class _NumericLimits:
    """``std.numeric_limits``: ``numeric_limits['int']`` is ``numeric_limits<int>``."""

    def __getitem__(self, kind: Any) -> _Limits:
        name = " ".join(str(getattr(kind, "__name__", kind)).replace("std::", "").split())
        if isinstance(kind, type) and kind in (int, float, bool):
            name = {int: "int", float: "double", bool: "bool"}[kind]
        if name not in _KINDS:
            raise TypeError(f"std::numeric_limits is of an arithmetic type, not {name!r}")
        return _Limits(_KINDS[name])


#: ``std.numeric_limits``.
numeric_limits = _NumericLimits()


class _SmartPointer:
    """``std::unique_ptr<T>`` and ``std::shared_ptr<T>``: what they hold is the object."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getitem__(self, kind: Any) -> _SmartPointer:
        return self

    def __call__(self, held: Any = None) -> Any:
        return held

    def __repr__(self) -> str:
        return f"<std.{self._name} template>"


def smart_pointer(name: str) -> _SmartPointer:
    return _SmartPointer(name)


def remove(path: Any) -> int:
    """C's ``remove``: the file deleted, and 0 - or -1, if there was none to delete."""
    try:
        os.remove(str(path))
    except OSError:
        return -1
    return 0
