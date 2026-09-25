"""``TRandom2``: L'Ecuyer's maximally equidistributed combined Tausworthe generator.

Three 32-bit shift registers, each stepped by ROOT's ``TAUSWORTHE`` macro,
XORed together into one word; ``Rndm()`` is the word over 2**32, drawn again
if it is zero. The state is three words and the period about 10**26.

Each register's step is a linear map on 32 bits, so a thousand steps is a
32-by-32 matrix of bits too, found by squaring. That is what vectorises it:
a run of draws is laid out as a few thousand streams side by side, each
started that many steps further on by the matrix, and all of them are then
stepped together, a pass of NumPy per step. Ten million draws are three
thousand passes over three thousand streams.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np

from .trandom import UINT, TRandom, _member, nonzero, uuid_bytes

__all__ = ["TRandom2", "step"]

#: ``TRandom2::Rndm``'s scale: 1/2**32.
SCALE = 2.3283064365386963e-10
#: Each register's ``TAUSWORTHE(s, a, b, c, d)`` arguments, as ``TRandom2`` passes them.
REGISTERS = ((13, 19, 4294967294, 12), (2, 25, 4294967288, 4), (3, 11, 4294967280, 17))
#: The smallest value each register may be seeded with, below which ROOT adds it on.
FLOORS = (2, 8, 16)


def step(s: Any, a: int, b: int, c: int, d: int) -> Any:
    """ROOT's ``TAUSWORTHE`` macro, for a number or an array of 64-bit integers."""
    return (((s & c) << d) & 0xFFFFFFFF) ^ ((((s << a) & 0xFFFFFFFF) ^ s) >> b)


def _apply(columns: tuple[int, ...], word: int) -> int:
    """A linear map on 32 bits, given by where each bit goes, applied to one word."""
    out = 0
    for column in columns:
        if word & 1:
            out ^= column
        word >>= 1
    return out


@functools.cache
def jump(register: int, steps: int) -> tuple[int, ...]:
    """Where each bit goes after ``steps`` steps of one register: its step's matrix, powered."""
    if steps == 1:
        return tuple(step(1 << bit, *REGISTERS[register]) for bit in range(32))
    half = jump(register, steps // 2)
    twice = tuple(_apply(half, column) for column in half)
    return tuple(_apply(jump(register, 1), column) for column in twice) if steps % 2 else twice


def _floored(word: int, floor: int) -> int:
    """A register's seed as ROOT leaves it: raised by its floor if it is below it."""
    return word + floor if word < floor else word


class TRandom2(TRandom):
    """ROOT's ``TRandom2``: fast, three words of state, a period of about 10**26."""

    #: The seed ROOT's constructor defaults to.
    DEFAULT_SEED = 1

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        super().__init__(seed)

    def _engine_init(self) -> None:
        self._registers = [0, 0, 0]

    def _engine_seed(self, seed: int) -> None:
        if seed:
            words: list[int] = []
            value = seed
            for _ in FLOORS:
                value = _floored((69069 * value) % UINT, FLOORS[len(words)])
                words.append(value)
        else:
            raw = uuid_bytes()
            first, second, third, fourth = (
                int.from_bytes(raw[i : i + 4], "little") for i in (0, 4, 8, 12)
            )
            words = [first, second, (third + fourth) % UINT]
        self._registers = [_floored(word, floor) for word, floor in zip(words, FLOORS)]
        self._engine_fill(6)

    def _engine_fill(self, count: int) -> Any:
        return nonzero(self._words, count).astype(np.float64) * SCALE

    def _words(self, count: int) -> Any:
        """The next ``count`` words, as streams a jump apart stepped side by side."""
        if not count:
            return np.empty(0, dtype=np.int64)
        width = max(1, math.isqrt(count))
        rows = -(-count // width)
        streams = np.array([self._starts(k, width, rows) for k in range(3)], dtype=np.int64)
        out = np.empty((width, rows), dtype=np.int64)
        last, finish = (count - 1) % width, (count - 1) // width
        for m in range(width):
            streams = np.array([step(streams[k], *REGISTERS[k]) for k in range(3)])
            out[m] = streams[0] ^ streams[1] ^ streams[2]
            if m == last:
                self._registers = [int(word) for word in streams[:, finish]]
        return out.T.reshape(-1)[:count]

    def _starts(self, register: int, width: int, rows: int) -> list[int]:
        """Where each stream of one register starts: here, and ``width`` steps on, and on."""
        columns = jump(register, width)
        starts = [self._registers[register]]
        for _ in range(rows - 1):
            starts.append(_apply(columns, starts[-1]))
        return starts

    def _engine_export(self) -> dict[str, Any]:
        first, second, third = self._registers
        return {"fSeed": first, "fSeed1": second, "fSeed2": third}

    def _engine_import(self, state: dict[str, Any]) -> None:
        self._registers = [_member(state, name, 0, UINT) for name in ("fSeed", "fSeed1", "fSeed2")]

    def _engine_get_seed(self) -> int:
        return self._registers[0]
