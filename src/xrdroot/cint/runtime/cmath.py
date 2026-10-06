"""``<cmath>`` with C's answers at the edges: ``nan`` and ``inf`` rather than exceptions.

``sqrt(-1)`` is ``nan`` in C and a ``ValueError`` in Python's :mod:`math`;
``log(0)`` is ``-inf``; ``exp(1000)`` is ``inf``; ``pow(0, -1)`` is ``inf``.
A macro that prints one of those prints what C prints. ``round`` rounds
halves away from zero as C's does (Python's rounds them to even), and
``abs`` of an integer stays an integer as ``std::abs`` keeps it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "sqrt",
    "cbrt",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log10",
    "log2",
    "log1p",
    "pow",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "fabs",
    "cabs",
    "floor",
    "ceil",
    "trunc",
    "cround",
    "fmod",
    "hypot",
    "erf",
    "erfc",
    "tgamma",
    "lgamma",
    "isnan",
    "isinf",
    "isfinite",
    "copysign",
    "fmin",
    "fmax",
    "M_PI",
    "M_E",
    "M_SQRT2",
    "M_PI_2",
    "M_PI_4",
    "M_1_PI",
    "M_2_PI",
    "M_LN2",
    "M_LN10",
    "M_2_SQRTPI",
    "M_SQRT1_2",
    "M_LOG2E",
    "M_LOG10E",
]

M_PI = math.pi
M_E = math.e
M_SQRT2 = math.sqrt(2)
M_SQRT1_2 = math.sqrt(0.5)
M_PI_2 = math.pi / 2
M_PI_4 = math.pi / 4
M_1_PI = 1 / math.pi
M_2_PI = 2 / math.pi
M_2_SQRTPI = 2 / math.sqrt(math.pi)
M_LN2 = math.log(2)
M_LN10 = math.log(10)
M_LOG2E = 1 / math.log(2)
M_LOG10E = 1 / math.log(10)

NAN = float("nan")
INF = float("inf")


def _c(function: Callable[..., float], name: str) -> Callable[..., float]:
    def c_function(*args: Any) -> float:
        try:
            return function(*map(float, args))
        except ValueError:
            return NAN
        except OverflowError:
            return INF

    c_function.__name__ = name
    c_function.__doc__ = f"C's {name}: nan and inf where Python's would raise."
    return c_function


def _logarithm(function: Callable[[float], float], name: str) -> Callable[[Any], float]:
    def c_log(value: Any) -> float:
        number = float(value)
        if number == 0:
            return -INF
        if number < 0 or math.isnan(number):
            return NAN
        return function(number)

    c_log.__name__ = name
    return c_log


sqrt = _c(math.sqrt, "sqrt")
exp = _c(math.exp, "exp")
exp2 = _c(lambda x: 2.0**x, "exp2")
expm1 = _c(math.expm1, "expm1")
log = _logarithm(math.log, "log")
log10 = _logarithm(math.log10, "log10")
log2 = _logarithm(math.log2, "log2")
log1p = _c(math.log1p, "log1p")
sin = _c(math.sin, "sin")
cos = _c(math.cos, "cos")
tan = _c(math.tan, "tan")
asin = _c(math.asin, "asin")
acos = _c(math.acos, "acos")
atan = _c(math.atan, "atan")
atan2 = _c(math.atan2, "atan2")
sinh = _c(math.sinh, "sinh")
cosh = _c(math.cosh, "cosh")
tanh = _c(math.tanh, "tanh")
asinh = _c(math.asinh, "asinh")
acosh = _c(math.acosh, "acosh")
atanh = _c(math.atanh, "atanh")
fabs = _c(math.fabs, "fabs")
floor = _c(lambda x: float(math.floor(x)) if math.isfinite(x) else x, "floor")
ceil = _c(lambda x: float(math.ceil(x)) if math.isfinite(x) else x, "ceil")
trunc = _c(lambda x: float(math.trunc(x)) if math.isfinite(x) else x, "trunc")
fmod = _c(math.fmod, "fmod")
hypot = _c(math.hypot, "hypot")
erf = _c(math.erf, "erf")
erfc = _c(math.erfc, "erfc")
tgamma = _c(math.gamma, "tgamma")
lgamma = _c(math.lgamma, "lgamma")
copysign = _c(math.copysign, "copysign")
fmin = _c(min, "fmin")
fmax = _c(max, "fmax")


def cbrt(value: Any) -> float:
    number = float(value)
    return math.copysign(abs(number) ** (1 / 3), number)


def pow(base: Any, exponent: Any) -> float:
    """C's ``pow``: always a double, ``inf`` for a zero to a negative power."""
    if float(base) == 0 and float(exponent) < 0:
        return INF
    try:
        return float(math.pow(float(base), float(exponent)))
    except ValueError:
        return NAN
    except OverflowError:
        return INF


def cabs(value: Any) -> Any:
    """``std::abs``: an integer's stays an integer, a number's is a double, and an ``RVec``'s -
    ``ROOT::VecOps::abs`` - is of each element."""
    if hasattr(value, "__array_ufunc__") and not isinstance(value, np.generic):
        return abs(value)
    if isinstance(value, int) or hasattr(value, "__index__"):
        return abs(int(value))
    return abs(float(value))


def cround(value: Any) -> float:
    """C's ``round``: halves go away from zero."""
    number = float(value)
    if not math.isfinite(number):
        return number
    return math.copysign(math.floor(abs(number) + 0.5), number)


def isnan(value: Any) -> bool:
    return math.isnan(float(value))


def isinf(value: Any) -> bool:
    return math.isinf(float(value))


def isfinite(value: Any) -> bool:
    return math.isfinite(float(value))
