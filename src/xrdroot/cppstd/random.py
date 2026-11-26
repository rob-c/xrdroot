"""``std::mt19937`` and the distributions a macro draws from it, number for number as libc++'s.

ROOT's interpreter on macOS is built against libc++, so a macro's random
numbers are libc++'s: the Mersenne Twister of the standard (seeded 5489
unless told), ``generate_canonical`` of two of its words for a double,
``uniform_real_distribution`` scaling that, and ``normal_distribution`` by
Marsaglia's polar method, keeping the second of each pair for the next call.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["default_random_engine", "mt19937", "normal_distribution",
           "uniform_real_distribution", "uniform_int_distribution"]

#: The twister's word count, its shift, and the constants of its recurrence and tempering.
N, M = 624, 397
MATRIX_A, UPPER, LOWER = 0x9908B0DF, 0x80000000, 0x7FFFFFFF
DEFAULT_SEED = 5489


class mt19937:
    """``std::mt19937``: 32-bit words from the standard Mersenne Twister."""

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed(seed)

    def seed(self, value: int = DEFAULT_SEED) -> None:
        state = [int(value) & 0xFFFFFFFF]
        for i in range(1, N):
            previous = state[-1]
            state.append((1812433253 * (previous ^ (previous >> 30)) + i) & 0xFFFFFFFF)
        self._state = np.array(state, dtype=np.uint64)
        self._words: list[int] = []

    def _twist(self) -> None:
        """The next 624 words, the recurrence worked in the runs whose inputs are known."""
        mt = self._state
        for start, stop in ((0, N - M), (N - M, 2 * (N - M)), (2 * (N - M), N)):
            i = np.arange(start, stop)
            y = (mt[i] & UPPER) | (mt[(i + 1) % N] & LOWER)
            mt[i] = mt[(i + M) % N] ^ (y >> 1) ^ ((y & 1) * MATRIX_A)
        self._words = _tempered(mt)


    def __call__(self) -> int:
        if not self._words:
            self._twist()
        return self._words.pop()

    def discard(self, count: int) -> None:
        for _ in range(int(count)):
            self()

    @staticmethod
    def min() -> int:
        return 0

    @staticmethod
    def max() -> int:
        return 0xFFFFFFFF


class default_random_engine:
    """``std::default_random_engine``: libc++'s ``minstd_rand0``, the 31-bit LCG of
    ``16807 * x mod (2**31 - 1)``, seeded 1. `std::shuffle` and the distributions
    take one where a macro names no engine of its own."""

    MODULUS = 2147483647
    MULTIPLIER = 16807

    def __init__(self, seed: int = 1) -> None:
        self.seed(seed)

    def seed(self, value: int = 1) -> None:
        self._state = int(value) % self.MODULUS

    def __call__(self) -> int:
        self._state = (self.MULTIPLIER * self._state) % self.MODULUS
        return self._state

    def discard(self, count: int) -> None:
        for _ in range(int(count)):
            self()

    @staticmethod
    def min() -> int:
        return 1

    @staticmethod
    def max() -> int:
        return default_random_engine.MODULUS - 1


def _tempered(state: np.ndarray[Any, Any]) -> list[int]:
    """The words a twisted state gives, in the order they are drawn (last first, for ``pop``)."""
    y = state.copy()
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    return list((y & 0xFFFFFFFF).tolist()[::-1])


def canonical(engine: Any) -> float:
    """libc++'s ``generate_canonical<double, 53>``: two words, the second the high one."""
    low = float(engine())
    return (low + float(engine()) * 4294967296.0) / 18446744073709551616.0


class uniform_real_distribution:
    """``std::uniform_real_distribution(a, b)``."""

    def __init__(self, a: float = 0.0, b: float = 1.0) -> None:
        self._a, self._b = float(a), float(b)

    def __class_getitem__(cls, kind: Any) -> type:
        return cls

    def __call__(self, engine: Any) -> float:
        return (self._b - self._a) * canonical(engine) + self._a

    def a(self) -> float:
        return self._a

    def b(self) -> float:
        return self._b

    def reset(self) -> None:
        """``reset``: nothing is kept between calls."""


class uniform_int_distribution:
    """``std::uniform_int_distribution<T>(a, b)``: each integer of ``[a, b]`` alike likely.

    The engine's word is scaled into the interval the way libstdc++/libc++ do it - one word
    per draw, the whole range of the engine mapped onto ``b - a + 1`` values - so a stream of
    draws is the same as C++'s for the same engine and seed.
    """

    def __init__(self, a: int = 0, b: int | None = None) -> None:
        low = int(a)
        high = low if b is None else int(b)
        self._a, self._b = min(low, high), max(low, high)

    def __class_getitem__(cls, kind: Any) -> type:
        return cls

    def __call__(self, engine: Any) -> int:
        span = int(engine.max()) - int(engine.min()) + 1
        width = self._b - self._a + 1
        if width >= span:  # C++ refuses rather than looping for ever
            raise ValueError(f"uniform_int_distribution({self._a}, {self._b}) is wider than "
                             "the engine it draws from")
        drawn = int(engine()) - int(engine.min())
        return self._a + drawn * width // span

    def a(self) -> int:
        return self._a

    def b(self) -> int:
        return self._b

    def min(self) -> int:
        return self._a

    def max(self) -> int:
        return self._b

    def reset(self) -> None:
        """``reset``: nothing is kept between calls."""


class normal_distribution:
    """``std::normal_distribution(mean, stddev)``: Marsaglia's polar method, a pair at a time."""

    def __init__(self, mean: float = 0.0, stddev: float = 1.0) -> None:
        self._mean, self._stddev = float(mean), float(stddev)
        self._kept: float | None = None

    def __class_getitem__(cls, kind: Any) -> type:
        return cls

    def __call__(self, engine: Any) -> float:
        if self._kept is not None:
            up, self._kept = self._kept, None
        else:
            up = self._pair(engine)
        return up * self._stddev + self._mean

    def _pair(self, engine: Any) -> float:
        while True:
            u = 2.0 * canonical(engine) - 1.0
            v = 2.0 * canonical(engine) - 1.0
            s = u * u + v * v
            if 0.0 < s <= 1.0:
                break
        factor = math.sqrt(-2.0 * math.log(s) / s)
        self._kept = v * factor
        return u * factor

    def mean(self) -> float:
        return self._mean

    def stddev(self) -> float:
        return self._stddev

    def reset(self) -> None:
        self._kept = None
