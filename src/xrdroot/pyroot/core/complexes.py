"""``TComplex``: ROOT's complex number, over Python's own.

ROOT's class predates ``std::complex`` in its code and keeps its own
names - ``Re``, ``Im``, ``Rho``, ``Theta`` and static ``Sqrt``, ``Exp``,
``Power`` - and its own print, ``(4,0i)``. Arithmetic takes another
``TComplex`` or a real number on either side, as PyROOT's pythonization of
it does, and refuses anything else with ``TypeError``.
"""

from __future__ import annotations

import cmath
import math
from numbers import Real
from typing import Any

from .objects import TObject

__all__ = ["TComplex"]


def _value(other: Any) -> complex | None:
    """``other`` as a Python complex number, or ``None`` when it is no number."""
    if isinstance(other, TComplex):
        return other._z
    if isinstance(other, Real):
        return complex(float(other))
    return None


class TComplex(TObject):
    """``TComplex(re, im)``, or ``TComplex(rho, theta, True)`` from polar co-ordinates."""

    def __init__(self, re: Any = 0.0, im: Any = 0.0, polar: Any = False) -> None:
        super().__init__()
        if isinstance(re, TComplex):
            self._z: complex = re._z
        elif polar:
            self._z = cmath.rect(float(re), float(im))
        else:
            self._z = complex(float(re), float(im))

    @classmethod
    def _of(cls, z: complex) -> TComplex:
        return cls(z.real, z.imag)

    def Re(self) -> float:
        return self._z.real

    def Im(self) -> float:
        return self._z.imag

    def Rho(self) -> float:
        return abs(self._z)

    def Rho2(self) -> float:
        return abs(self._z) ** 2

    def Theta(self) -> float:
        return math.atan2(self._z.imag, self._z.real) if self._z else 0.0

    def _arith(self, other: Any, operation: Any) -> Any:
        value = _value(other)
        if value is None:
            return NotImplemented
        return self._of(operation(self._z, value))

    def __add__(self, other: Any) -> Any:
        return self._arith(other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> Any:
        return self._arith(other, lambda a, b: b + a)

    def __sub__(self, other: Any) -> Any:
        return self._arith(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> Any:
        return self._arith(other, lambda a, b: b - a)

    def __mul__(self, other: Any) -> Any:
        return self._arith(other, lambda a, b: a * b)

    def __rmul__(self, other: Any) -> Any:
        return self._arith(other, lambda a, b: b * a)

    def __truediv__(self, other: Any) -> Any:
        return self._arith(other, lambda a, b: a / b)

    def __rtruediv__(self, other: Any) -> Any:
        return self._arith(other, lambda a, b: b / a)

    def __neg__(self) -> TComplex:
        return self._of(-self._z)

    def __complex__(self) -> complex:
        return self._z

    def __eq__(self, other: object) -> bool:
        value = _value(other)
        return value is not None and value == self._z

    def __hash__(self) -> int:
        return hash(self._z)

    def __str__(self) -> str:
        return f"({self._z.real:g},{self._z.imag:g}i)"

    def Print(self, option: str = "") -> None:
        print(self)

    @staticmethod
    def I() -> TComplex:  # noqa: E743 - ROOT's name for the imaginary unit
        return TComplex(0.0, 1.0)

    @staticmethod
    def One() -> TComplex:
        return TComplex(1.0, 0.0)

    @staticmethod
    def Abs(c: Any) -> float:
        return abs(TComplex(c)._z)

    @staticmethod
    def Conjugate(c: Any) -> TComplex:
        return TComplex._of(TComplex(c)._z.conjugate())

    @staticmethod
    def Sqrt(c: Any) -> TComplex:
        return TComplex._of(cmath.sqrt(TComplex(c)._z))

    @staticmethod
    def Exp(c: Any) -> TComplex:
        return TComplex._of(cmath.exp(TComplex(c)._z))

    @staticmethod
    def Log(c: Any) -> TComplex:
        return TComplex._of(cmath.log(TComplex(c)._z))

    @staticmethod
    def Power(c: Any, power: Any) -> TComplex:
        return TComplex._of(TComplex(c)._z ** TComplex(power)._z)
