"""``ROOT::Math::KahanSum``: the compensated sum RooFit adds a likelihood's terms with.

Minuit differentiates the likelihood numerically, so the last bits of its
value move where Minuit goes - and a correctly rounded sum is not the same
number as RooFit's compensated one. This is RooFit's, step for step: one
running sum, one carry, each term added as ``KahanSum::Add`` adds it, the
value the running sum alone (``Sum()``).
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["Kahan"]


class Kahan:
    """A running compensated sum."""

    __slots__ = ("total", "carry")

    def __init__(self, total: float = 0.0, carry: float = 0.0) -> None:
        self.total = total
        self.carry = carry

    def add(self, x: float) -> None:
        y = x - self.carry
        t = self.total + y
        self.carry = (t - self.total) - y
        self.total = t

    def extend(self, values: Iterable[float]) -> Kahan:
        total, carry = self.total, self.carry
        for x in values:
            y = x - carry
            t = total + y
            carry = (t - total) - y
            total = t
        self.total, self.carry = total, carry
        return self
