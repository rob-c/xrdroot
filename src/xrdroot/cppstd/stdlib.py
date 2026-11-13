"""``std::numeric_limits``, the smart pointers and ``std::remove``, for ``ROOT.std``.

``std::numeric_limits<int>::min()`` is a number a macro uses as a sentinel
(df036); ``std::unique_ptr<RooArgSet>{p}`` holds an object the macro goes on
to use, which here is the object itself, Python keeping it while it is
used; ``std::remove(name)`` deletes a file, as C's ``remove`` does, zero
when it did.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

__all__ = ["numeric_limits", "remove", "smart_pointer"]

#: The NumPy type of each C++ arithmetic type ``numeric_limits`` is asked of, as LP64 macOS
#: on ARM64 has them: ``char`` signed, ``wchar_t`` a signed 32-bit integer, ``long double``
#: the same as ``double``.
_KINDS = {
    "bool": np.bool_,
    "char": np.int8,
    "signed char": np.int8,
    "unsigned char": np.uint8,
    "wchar_t": np.int32,
    "char8_t": np.uint8,
    "char16_t": np.uint16,
    "char32_t": np.uint32,
    "short": np.int16,
    "unsigned short": np.uint16,
    "int": np.int32,
    "unsigned int": np.uint32,
    "unsigned": np.uint32,
    "long": np.int64,
    "unsigned long": np.uint64,
    "long long": np.int64,
    "unsigned long long": np.uint64,
    "size_t": np.uint64,
    "float": np.float32,
    "double": np.float64,
    "long double": np.float64,
}

#: The members of ``numeric_limits<T>`` that are functions, ``epsilon()``; the rest are
#: constants, ``digits``.
FUNCTIONS = frozenset(
    "min max lowest epsilon round_error infinity quiet_NaN signaling_NaN denorm_min".split()
)


def _floating(info: Any) -> dict[str, Any]:
    """The members of ``numeric_limits`` of an IEEE floating type, from NumPy's description."""
    digits = int(info.nmant) + 1
    return {
        "min": float(info.tiny),
        "max": float(info.max),
        "lowest": -float(info.max),
        "epsilon": float(info.eps),
        "round_error": 0.5,
        "infinity": math.inf,
        "quiet_NaN": math.nan,
        "signaling_NaN": math.nan,
        "denorm_min": float(info.smallest_subnormal),
        "digits": digits,
        "digits10": int(info.precision),
        "max_digits10": math.ceil(digits * math.log10(2) + 1),
        "is_signed": True,
        "is_integer": False,
        "is_exact": False,
        "has_infinity": True,
        "has_quiet_NaN": True,
        "has_signaling_NaN": True,
        "is_iec559": True,
        "is_modulo": False,
        "min_exponent": int(info.minexp) + 1,
        "max_exponent": int(info.maxexp),
        "min_exponent10": math.ceil(math.log10(float(info.tiny))),
        "max_exponent10": math.floor(math.log10(float(info.max))),
    }


def _integral(kind: Any) -> dict[str, Any]:
    """The members of ``numeric_limits`` of an integer type, ``bool`` among them."""
    low: Any
    high: Any
    nothing: Any
    if kind == np.bool_:
        low, high, digits, signed, nothing = False, True, 1, False, False
    else:
        info = np.iinfo(kind)
        low, high, nothing = int(info.min), int(info.max), 0
        signed = low < 0
        digits = info.bits - signed
    found: dict[str, Any] = dict.fromkeys(
        "epsilon round_error infinity quiet_NaN signaling_NaN denorm_min".split(), nothing
    )
    found.update(dict.fromkeys(
        "max_digits10 min_exponent max_exponent min_exponent10 max_exponent10".split(), 0
    ))  # fmt: skip
    found.update(min=low, max=high, lowest=low, digits=digits, is_signed=signed)
    found.update(digits10=int(digits * math.log10(2)), is_integer=True, is_exact=True)
    found.update(dict.fromkeys("has_infinity has_quiet_NaN has_signaling_NaN is_iec559".split(),
                               False))  # fmt: skip
    found["is_modulo"] = not signed and kind != np.bool_
    return found


def limits_of(kind: Any) -> dict[str, Any]:
    """Every member of ``numeric_limits`` of the arithmetic type whose NumPy type is ``kind``."""
    dtype = np.dtype(kind)
    members = _floating(np.finfo(dtype)) if dtype.kind == "f" else _integral(dtype.type)
    members.update(radix=2, is_specialized=True, is_bounded=True)
    return members


def limit(name: str, member: str) -> Any:
    """``std::numeric_limits<name>::member``, for a C++ arithmetic type spelled canonically -
    or ``None`` when ``name`` is not one or ``member`` not one of its members."""
    kind = _KINDS.get(name)
    return None if kind is None else limits_of(kind).get(member)


class _Limits:
    """``std::numeric_limits<T>``: the range and precision of one arithmetic type."""

    def __init__(self, kind: Any) -> None:
        for member, value in limits_of(kind).items():
            setattr(self, member, (lambda value=value: value) if member in FUNCTIONS else value)


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
