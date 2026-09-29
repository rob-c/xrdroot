"""The functions an expression can call: ROOT's ``TMath``, and C's ``<cmath>``.

Every one of them works on whole arrays at once. ROOT evaluates a formula in
``double``, so the arguments arrive as ``float64`` - one value per entry, or
per element of the collections the expression loops over - and each function
here is the NumPy ufunc that does the same arithmetic, or a few of them put
together. ROOT's ``exp``, ``log``, ``sin`` and the rest are the C library's, and
so are NumPy's - except where AVX-512 gives NumPy vectorised ones of its own,
which differ in the last place now and then - so those go through
:mod:`xrdroot.random.libm`, which uses NumPy's only where it is the C
library's. The rare ones NumPy has no ufunc for, such as ``Erf`` and
``Gamma``, go through :mod:`math` an element at a time: correct, and slow
only for the few expressions that ask for them.

Both spellings ROOT users write are here: ``TMath::Abs`` and ``TMath::Sqrt``
from ``TMath``, and ``abs``, ``fabs`` and ``sqrt`` from C, with ``std::`` in
front of those allowed too.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from ..random import libm

__all__ = ["Function", "FUNCTIONS", "STRING_FUNCTIONS", "CASTS"]

Array = Any


class Function:
    """A function's implementation and how many arguments it takes."""

    __slots__ = ("least", "most", "apply")

    def __init__(self, least: int, most: int, apply: Callable[..., Array]) -> None:
        self.least = least
        self.most = most
        self.apply = apply


def _ufunc(ufunc: Callable[..., Array], arity: int = 1) -> Function:
    return Function(arity, arity, ufunc)


def _elementwise(one: Callable[[float], float]) -> Function:
    """A function of one number from :mod:`math`, run over every element."""
    vectorised = np.frompyfunc(one, 1, 1)

    def apply(values: Array) -> Array:
        return np.asarray(vectorised(values), dtype=np.float64)

    return Function(1, 1, apply)


def _c(fast: Callable[[Array], Array], exact: Callable[[float], float], low: float,
       high: float) -> Function:  # fmt: skip
    """NumPy's ``fast`` where it is the C library's ``exact`` - tried from ``low`` to ``high`` -
    else ``exact`` element by element: ROOT's formulas call the C library."""
    return _ufunc(libm.choose(fast, exact, low, high))


def _constant(value: float) -> Function:
    return Function(0, 0, lambda: np.float64(value))


def _extreme(ufunc: Callable[..., Array]) -> Function:
    """``min``/``max`` of two or more values, as ``TMath::Min`` and C's ``fmin``."""

    def apply(*values: Array) -> Array:
        return ufunc.reduce(np.broadcast_arrays(*values))  # type: ignore[attr-defined]

    return Function(2, 64, apply)


def _sign(a: Array, b: Array) -> Array:
    """``TMath::Sign(a, b)``: the size of ``a`` with the sign of ``b``, zero counting as plus."""
    return np.where(b >= 0, np.abs(a), -np.abs(a))


def _round(values: Array) -> Array:
    """C's ``round``: halves away from zero, where NumPy's ``rint`` rounds them to even."""
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _gaus(x: Array, mean: Array = 0.0, sigma: Array = 1.0, norm: Array = 0.0) -> Array:
    """``TMath::Gaus``: the height of a Gaussian, divided by its area when ``norm`` is true.

    A width of zero is ROOT's ``1e30``, which is what ``TMath::Gaus`` answers
    rather than divide by it.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    safe = np.where(sigma == 0, 1.0, sigma)
    height = libm.exp(-0.5 * ((x - mean) / safe) ** 2)
    height = np.where(norm != 0, height / (math.sqrt(2 * math.pi) * safe), height)
    return np.where(sigma == 0, 1e30, height)


def _breit_wigner(x: Array, mean: Array = 0.0, gamma: Array = 1.0) -> Array:
    """``TMath::BreitWigner``: the Cauchy density of the resonance at ``mean``."""
    return 0.5 * gamma / math.pi / ((x - mean) ** 2 + 0.25 * gamma**2)


def _even(values: Array) -> Array:
    return (values.astype(np.int64) % 2) == 0


def _odd(values: Array) -> Array:
    return (values.astype(np.int64) % 2) != 0


_LOG10 = _c(np.log10, math.log10, 1e-12, 1e6)
_LOG2 = _c(np.log2, math.log2, 1e-12, 1e6)
_ASIN = _c(np.arcsin, math.asin, -1.0, 1.0)
_ACOS = _c(np.arccos, math.acos, -1.0, 1.0)
_TANH = _c(np.tanh, math.tanh, -20.0, 20.0)
_ASINH = _c(np.arcsinh, math.asinh, -50.0, 50.0)
_ACOSH = _c(np.arccosh, math.acosh, 1.0, 50.0)
_ATANH = _c(np.arctanh, math.atanh, -1.0, 1.0)

#: The functions of ``TMath``, as ``TMath::`` is followed by them.
TMATH: dict[str, Function] = {
    "Abs": _ufunc(np.abs),
    "Sqrt": _ufunc(np.sqrt),
    "Sq": Function(1, 1, lambda x: x * x),
    "Power": _ufunc(libm.power, 2),
    "Exp": _ufunc(libm.exp),
    "Log": _ufunc(libm.log),
    "Log10": _LOG10,
    "Log2": _LOG2,
    "Sin": _ufunc(libm.sin),
    "Cos": _ufunc(libm.cos),
    "Tan": _ufunc(libm.tan),
    "ASin": _ASIN,
    "ACos": _ACOS,
    "ATan": _ufunc(libm.atan),
    "ATan2": _ufunc(np.arctan2, 2),
    "SinH": _ufunc(libm.sinh),
    "CosH": _ufunc(libm.cosh),
    "TanH": _TANH,
    "ASinH": _ASINH,
    "ACosH": _ACOSH,
    "ATanH": _ATANH,
    "Min": _extreme(np.minimum),
    "Max": _extreme(np.maximum),
    "Floor": _ufunc(np.floor),
    "Ceil": _ufunc(np.ceil),
    "Nint": _ufunc(np.rint),
    "Sign": Function(2, 2, _sign),
    "Hypot": _ufunc(np.hypot, 2),
    "Erf": _elementwise(math.erf),
    "Erfc": _elementwise(math.erfc),
    "Gamma": _elementwise(math.gamma),
    "LnGamma": _elementwise(math.lgamma),
    "Gaus": Function(1, 4, _gaus),
    "BreitWigner": Function(1, 3, _breit_wigner),
    "IsNaN": _ufunc(np.isnan),
    "Finite": _ufunc(np.isfinite),
    "Even": Function(1, 1, _even),
    "Odd": Function(1, 1, _odd),
    "Pi": _constant(math.pi),
    "TwoPi": _constant(2 * math.pi),
    "PiOver2": _constant(math.pi / 2),
    "PiOver4": _constant(math.pi / 4),
    "InvPi": _constant(1 / math.pi),
    "E": _constant(math.e),
    "Ln10": _constant(math.log(10)),
    "LogE": _constant(math.log10(math.e)),
    "Sqrt2": _constant(math.sqrt(2)),
    "DegToRad": _constant(math.pi / 180),
    "RadToDeg": _constant(180 / math.pi),
    "C": _constant(299792458.0),
}

#: The functions of C's ``<cmath>``, and ROOT's ``sq``, which is spelled like one.
CMATH: dict[str, Function] = {
    "abs": _ufunc(np.abs),
    "fabs": _ufunc(np.abs),
    "sqrt": _ufunc(np.sqrt),
    "cbrt": _ufunc(np.cbrt),
    "pow": _ufunc(libm.power, 2),
    "exp": _ufunc(libm.exp),
    "exp2": _ufunc(np.exp2),
    "log": _ufunc(libm.log),
    "log10": _LOG10,
    "log2": _LOG2,
    "sin": _ufunc(libm.sin),
    "cos": _ufunc(libm.cos),
    "tan": _ufunc(libm.tan),
    "asin": _ASIN,
    "acos": _ACOS,
    "atan": _ufunc(libm.atan),
    "atan2": _ufunc(np.arctan2, 2),
    "sinh": _ufunc(libm.sinh),
    "cosh": _ufunc(libm.cosh),
    "tanh": _TANH,
    "asinh": _ASINH,
    "acosh": _ACOSH,
    "atanh": _ATANH,
    "floor": _ufunc(np.floor),
    "ceil": _ufunc(np.ceil),
    "trunc": _ufunc(np.trunc),
    "round": Function(1, 1, _round),
    "rint": _ufunc(np.rint),
    "fmod": _ufunc(np.fmod, 2),
    "hypot": _ufunc(np.hypot, 2),
    "min": _extreme(np.minimum),
    "max": _extreme(np.maximum),
    "fmin": _extreme(np.minimum),
    "fmax": _extreme(np.maximum),
    "erf": _elementwise(math.erf),
    "erfc": _elementwise(math.erfc),
    "tgamma": _elementwise(math.gamma),
    "lgamma": _elementwise(math.lgamma),
    "isnan": _ufunc(np.isnan),
    "isinf": _ufunc(np.isinf),
    "isfinite": _ufunc(np.isfinite),
    "sq": Function(1, 1, lambda x: x * x),
}

#: Every spelling a function can be called by, against its implementation.
FUNCTIONS: dict[str, Function] = {
    **{f"TMath::{name}": function for name, function in TMATH.items()},
    **CMATH,
    **{f"std::{name}": function for name, function in CMATH.items()},
}


def _strstr(haystack: Array, needle: Array) -> Array:
    """C's ``strstr`` as a condition: does the first string hold the second?"""
    return np.char.find(haystack, needle) >= 0


#: The functions whose arguments are strings rather than numbers.
STRING_FUNCTIONS: dict[str, Function] = {"strstr": Function(2, 2, _strstr)}


#: The C++ types a value can be cast to, ``(int)x`` or ``int(x)``, and what NumPy calls each.
CASTS: dict[str, Any] = {
    "bool": np.bool_,
    "Bool_t": np.bool_,
    "char": np.int8,
    "Char_t": np.int8,
    "unsigned char": np.uint8,
    "UChar_t": np.uint8,
    "short": np.int16,
    "Short_t": np.int16,
    "unsigned short": np.uint16,
    "UShort_t": np.uint16,
    "int": np.int32,
    "Int_t": np.int32,
    "unsigned": np.uint32,
    "unsigned int": np.uint32,
    "UInt_t": np.uint32,
    "long": np.int64,
    "long long": np.int64,
    "Long_t": np.int64,
    "Long64_t": np.int64,
    "unsigned long": np.uint64,
    "unsigned long long": np.uint64,
    "ULong_t": np.uint64,
    "ULong64_t": np.uint64,
    "float": np.float32,
    "Float_t": np.float32,
    "double": np.float64,
    "Double_t": np.float64,
}
