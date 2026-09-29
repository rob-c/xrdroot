"""``std::shuffle`` with ``TMVA::RandomGenerator<TRandom3>``: TMVA's random splits, draw for draw.

Which events TMVA trains on is decided by shuffling them with a
``RandomGenerator<TRandom3>`` seeded with ``SplitSeed`` (100 unless said)
- a generator whose every draw is ``TRandom3::Integer(4294967295)`` - handed
to the C++ library's ``std::shuffle``. The shuffle is libc++'s, the library
ROOT is built with on macOS: a Fisher-Yates pass whose each index is drawn by
``uniform_int_distribution`` from the low bits of one draw, drawn again while
it is out of range. Doing exactly that here gives the very events ROOT picks,
and so the same training sample, the same correlation matrices and - for the
methods that are closed-form - the same classifier.
"""

from __future__ import annotations

from typing import Any

from ..random.mersenne import TRandom3

__all__ = ["RandomGenerator", "shuffle"]

#: ``kMaxUInt``, the top of the generator's range.
MAX_UINT = 4294967295


class RandomGenerator:
    """``TMVA::RandomGenerator<TRandom3>``: whole numbers in ``[0, 2**32 - 1[``."""

    def __init__(self, seed: int = 0) -> None:
        self._random = TRandom3(seed)

    def __call__(self) -> int:
        return int(self._random.integer(MAX_UINT))

    def rndm(self) -> float:
        """The generator's ``Rndm()``, for the few places TMVA asks it directly."""
        return float(self._random.rndm())


def _bits(span: int) -> int:
    """How many low bits of a draw libc++ keeps to pick one of ``span`` values."""
    width = span.bit_length() - 1
    if span & ((1 << width) - 1):
        width += 1
    return width


def draw_below(generator: Any, span: int) -> int:
    """``uniform_int_distribution<ptrdiff_t>(0, span - 1)(generator)`` as libc++ draws it."""
    if span == 1:
        return 0
    mask = (1 << _bits(span)) - 1
    while True:
        value = generator() & mask
        if value < span:
            return int(value)


def shuffle(items: list[Any], generator: Any) -> None:
    """``std::shuffle(items.begin(), items.end(), generator)``, in place, as libc++ does it."""
    remaining = len(items) - 1
    for first in range(len(items) - 1):
        offset = draw_below(generator, remaining + 1)
        if offset:
            items[first], items[first + offset] = items[first + offset], items[first]
        remaining -= 1
