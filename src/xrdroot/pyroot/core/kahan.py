"""``ROOT::Math::KahanSum``: a sum that keeps the rounding it lost, and gives it back.

Adding a small number to a large one loses the low bits; Kahan's compensation keeps them in a
carry beside the sum, so ``Result()`` is the sum plus what the additions dropped. ROOT's
``Math/Util.h`` is the reference for the arithmetic, including how two compensated sums combine:
their sums and their carries add separately, because the carry of one is no more precise than
the other's.

``KahanSum<T>`` here is one class: the element type and the unroll width of
``KahanSum<double, 4>`` say which template a macro meant, not how the sum is kept.
"""

from __future__ import annotations

from typing import Any

__all__ = ["KahanSum"]


def _deref(value: Any) -> Any:
    """The element an iterator is at, or the value itself when it is no iterator."""
    deref = getattr(value, "__deref__", None)
    return deref() if callable(deref) else value


class KahanSum:
    """``ROOT::Math::KahanSum<T>``: a compensated sum with its carry.

    ``KahanSum<double, 4>`` is the same class - the width says how many accumulators that
    template unrolls in C++, which changes the rounding but not the interface - so a macro's
    subscript reaches this one class and ``Accumulate`` is its own.
    """

    def __init__(self, initialValue: Any = 0.0, carry: Any = 0.0) -> None:
        self.fSum = initialValue
        self.fCarry = carry

    @classmethod
    def __class_getitem__(cls, arguments: Any) -> type[KahanSum]:
        """``KahanSum<double>``, ``KahanSum<double, 4>``: the one class of a compensated sum."""
        return cls

    def Add(self, x: Any, end: Any = None) -> KahanSum:
        """``Add(value)``, or ``Add(begin, end)`` of an iterator range, as ROOT has both."""
        if end is not None:
            at = x
            while at != end:
                self.Add(_deref(at))
                at += 1
            return self
        corrected = x - self.fCarry
        total = self.fSum + corrected
        self.fCarry = (total - self.fSum) - corrected
        self.fSum = total
        return self

    def Sum(self) -> Any:
        return self.fSum

    def Carry(self) -> Any:
        return self.fCarry

    def Result(self) -> Any:
        return self.fSum

    def _combined(self, other: Any, add: bool) -> KahanSum:
        """Two compensated sums combined: sums with sums, carries with carries, as ROOT does."""
        if not isinstance(other, KahanSum):
            other = KahanSum(other)
        pair = (self.fSum + other.fSum, self.fCarry + other.fCarry) if add else (
            self.fSum - other.fSum, self.fCarry - other.fCarry)
        return KahanSum(*pair)

    def __add__(self, other: Any) -> KahanSum:
        return self._combined(other, True)

    __radd__ = __add__

    def __sub__(self, other: Any) -> KahanSum:
        return self._combined(other, False)

    def __rsub__(self, other: Any) -> KahanSum:
        return KahanSum(other)._combined(self, False)

    def __mul__(self, other: Any) -> KahanSum:
        other = other if isinstance(other, KahanSum) else KahanSum(other)
        return KahanSum(self.fSum * other.fSum, self.fCarry * other.fCarry)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> KahanSum:
        other = other if isinstance(other, KahanSum) else KahanSum(other)
        return KahanSum(self.fSum / other.fSum, self.fCarry / other.fCarry)

    def __iadd__(self, other: Any) -> KahanSum:
        made = self.__add__(other)
        self.fSum, self.fCarry = made.fSum, made.fCarry
        return self

    def __isub__(self, other: Any) -> KahanSum:
        made = self.__sub__(other)
        self.fSum, self.fCarry = made.fSum, made.fCarry
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KahanSum):
            return NotImplemented
        return (self.fSum, self.fCarry) == (other.fSum, other.fCarry)

    __hash__ = None  # type: ignore[assignment]

    @staticmethod
    def Accumulate(begin: Any, end: Any, initialValue: Any = 0.0) -> KahanSum:
        """``KahanSum<T>::Accumulate(begin, end, initial)``: the range's compensated sum."""
        made = KahanSum(initialValue)
        made.Add(begin, end)
        return made

    def __repr__(self) -> str:
        return f"<ROOT::Math::KahanSum sum={self.fSum!r} carry={self.fCarry!r}>"
