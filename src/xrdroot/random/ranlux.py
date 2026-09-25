"""``TRandom1``: RANLUX, Lüscher's generator, as ROOT runs it - and as an LCG, to run it fast.

RANLUX is Marsaglia and Zaman's subtract-with-borrow generator on 24-bit
digits - each new digit the one ten back less the one twenty-four back less
a borrow - of which only 24 digits are kept in every ``24 + p``, the rest
thrown away to break its correlations; ``p`` is the luxury level's 0, 24,
73, 199 or 365. ROOT's ``TRandom1`` returns each digit over 2**24 as a
``float``, and where a digit is below 2**12 it tops the number up with the
digit nine further back, so that small numbers keep 24 bits of precision.

Run as ROOT runs it, a draw costs a dozen steps of a loop, each depending on
the last; ROOT itself takes 80 ns a draw. But the digits are the base-2**24
expansion of a fraction with denominator ``m = b**24 - b**10 + 1``, and each
step divides that fraction's numerator by ``b`` modulo ``m`` (Tezuka and
L'Ecuyer; Sibidanov's RANLUX++ is built on the same fact). So the ``24 + p``
steps between one kept block and the next are one multiplication of a
576-bit number, and the block's digits are the low digits of the numerator
times ``m``'s inverse - a few big-integer operations per 24 draws, in
Python's own arithmetic, and exact. The digits are then made into ROOT's
``float`` values an array at a time.

The numerator of the generator's state is ``L10 - L24 - carry``: the last
ten digits and the last twenty-four, read as numbers, less the borrow. For
any digits and either carry it lies in ``[-m, 0]``, where a numerator is
known from its residue modulo ``m`` - but for the two fixed points, all
zeros and all ``b - 1``, which share one and never move. Everything in
ROOT's state - the ring of 24 floats, its two indices, the carry and the
count into the block - is made from the numerator again after every run of
draws, so ``state`` is ROOT's, member for member.
"""

from __future__ import annotations

import functools
import itertools
import operator
from typing import Any

import numpy as np

from .mersenne import TRandom3
from .trandom import UINT, TRandom, _member

__all__ = ["TRandom1", "advance", "expand"]

#: The base of RANLUX's digits.
B = 1 << 24
#: The modulus of the LCG RANLUX is: ``b**24 - b**10 + 1``.
M = B**24 - B**10 + 1
#: One step of RANLUX, as a multiplier modulo ``m``: the inverse of ``b``.
A = pow(B, -1, M)
#: The digits thrown away after every 24 kept, for each of ROOT's luxury levels.
LEVELS = (0, 24, 73, 199, 365)
#: L'Ecuyer's constants, with which ROOT spreads one seed over 24 digits.
ECUYER_A, ECUYER_B, ECUYER_C, ECUYER_D = 53668, 40014, 12211, 2147483563
#: ``2**-24``, ``fMantissaBit24``.
BIT24 = 2.0**-24
# fmt: off
#: The first seed of each of the 215 pairs in ROOT's ``fgSeedTable``, the only
#: one of the two the default constructor reads.
SEED_TABLE = (
    9876, 1299961164, 669708517, 190904760, 1289741558, 1803730167, 489854550,
    1348037628, 350557787, 591502945, 1901084678, 1988640932, 1873836227, 1146416592,
    1837193353, 38219936, 349152748, 744459040, 1983990104, 309164507, 362993787,
    556776976, 1584900822, 1249892722, 1686600998, 1127381380, 1999420861, 1972906041,
    84636481, 1186362995, 2141621785, 1969581251, 1150606439, 95187861, 940517655,
    215350428, 786161212, 1450830056, 1696578057, 1803414346, 1017898585, 1184497978,
    633338765, 430889421, 492544653, 389386975, 1720322786, 1868768216, 443210610,
    1191938868, 616890172, 935835339, 1058009367, 1463148129, 1795336935, 274019517,
    483689317, 2070804364, 930226308, 989324440, 410606853, 1583588576, 2102034391,
    2005037790, 1244218766, 49312790, 721012176, 1719909107, 1156177430, 307561322,
    906041433, 1591375755, 461522398, 2145930725, 1938419274, 174405412, 494343653,
    1025534808, 2048959776, 950636517, 1828843197, 211109723, 825474095, 374915657,
    1241296328, 1260624655, 900676210, 697951025, 1007920268, 264596520, 1977924811,
    1440257718, 1928778847, 1307807366, 1498732610, 1617712402, 1261800762, 949929273,
    1766170474, 1849939248, 887262858, 483086133, 1330541052, 1850591475, 1431775678,
    922493739, 1058517206, 709067283, 1044787723, 999707003, 2140038663, 1803100150,
    867445693, 408583729, 1166715497, 430738528, 1388022681, 1664028066, 1767741172,
    1625723550, 464486085, 754082421, 1315342834, 960416608, 1262630671, 59809238,
    1205644919, 1615183160, 1024474681, 1703877649, 1821417852, 1738806466, 620780646,
    1070174101, 658537995, 2055317555, 1647371686, 29067379, 1763495989, 1602690753,
    885787576, 1853512561, 1798585498, 1819261032, 1133245275, 689459799, 1730609912,
    1556832175, 251375753, 2083946182, 2142981854, 763711891, 1581256466, 2121337132,
    1004003636, 569816524, 626626425, 632086003, 1008211580, 1134217766, 1423829292,
    942037869, 1959429535, 778311037, 1531372185, 241935492, 272453504, 390441332,
    1230238834, 1242956379, 515648357, 364477932, 2096008713, 1409752526, 1288158292,
    407562666, 1071056491, 1014143949, 203080461, 125647866, 2015685843, 1425476020,
    1673735652, 1714199325, 1389137652, 288547803, 200159281, 1580828223, 1832286107,
    182557704, 1688025575, 1508287706, 36721212, 1968679856, 279109231, 1358617667,
    740626186, 1882655908, 648016670, 780255321, 857296483, 1631676846, 1906971307,
    1541899600, 1267051693, 1998673940, 1914117058, 426068513, 139365577, 2129910384,
    661788054, 1104640222, 356133630, 242242374, 957935844,
)
# fmt: on


def number(digits: Any) -> int:
    """Digits of base 2**24, least significant first, read as one number."""
    return sum(int(digit) << (24 * k) for k, digit in enumerate(digits))


@functools.cache
def _inverse(count: int) -> int:
    """``m``'s inverse modulo ``b**count``, which turns a numerator into its next digits."""
    inverse: int = pow(M, -1, B**count)
    return inverse


def _digits(raw: bytes) -> Any:
    """Little-endian 24-bit digits, three bytes each, as an array."""
    parts = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
    return parts[:, 0] | (parts[:, 1] << 8) | (parts[:, 2] << 16)


def expand(numerator: int, count: int) -> Any:
    """The next ``count`` digits RANLUX makes from a state with this numerator."""
    value = numerator * _inverse(count) % B**count
    return _digits(value.to_bytes(3 * count, "little"))


def advance(numerator: int, steps: int) -> int:
    """The numerator ``steps`` steps on: a multiplication by ``b``'s inverse modulo ``m`` per step.

    The numerator is kept in ``[-m, 0]``; the residue 0 belongs to the fixed
    points 0 and ``-m``, and so to whichever of them it started as.
    """
    residue = numerator * pow(A, steps, M) % M
    return residue - M if residue else numerator


def skip_for(luxury: int) -> int:
    """How many digits ROOT throws away after every 24 at a luxury level."""
    if 0 <= luxury < len(LEVELS):
        return LEVELS[luxury]
    return luxury - 24 if luxury >= 24 else LEVELS[3]


def _ecuyer(seed: int) -> int:
    """One step of L'Ecuyer's generator, as ``TRandom1::SetSeeds`` takes it."""
    k = seed // ECUYER_A
    seed = ECUYER_B * (seed - k * ECUYER_A) - k * ECUYER_C
    return seed + ECUYER_D if seed < 0 else seed


def seed_digits(seeds: list[int]) -> list[int]:
    """``SetSeeds``: the 24 digits a list of seeds, up to its first zero, starts RANLUX from."""
    table = [seed % B for seed in itertools.takewhile(bool, seeds)][:24]
    value = table[-1]
    while len(table) < 24:
        value = _ecuyer(value)
        table.append(value % B)
    return table


def _outputs(x: Any, fine: Any) -> Any:
    """ROOT's ``float`` for each kept digit, topped up from nine digits back where it is small."""
    coarse = x * BIT24
    finer = ((x * 2.0**24 + fine) * 2.0**-48).astype(np.float32).astype(np.float64)
    finer[finer == 0] = 2.0**-48
    return np.where(x < 4096, finer, coarse)


def _blocks(numerator: int, count24: int, skip: int, blocks: int) -> Any:
    """Thirty-three digits for each block of 24 kept from the third on: nine before it, and it."""
    cycle = 24 + skip
    anchor = advance(numerator, 24 - count24 + skip + cycle - 9)
    jump, inverse, scale = pow(A, cycle, M), _inverse(33), B**33
    raw = []
    for _ in range(blocks):
        raw.append((anchor * inverse % scale).to_bytes(99, "little"))
        residue = anchor * jump % M
        anchor = residue - M if residue else anchor
    return _digits(b"".join(raw)).reshape(blocks, 33)


class TRandom1(TRandom):
    """ROOT's ``TRandom1``: RANLUX at a luxury level, 3 unless said otherwise.

    With no seed it takes the next of ROOT's table of 215 seeds, as ROOT's
    default constructor does, so that generators made one after another in
    a session start apart - the first here from the same seed as the first
    in ROOT. ``lux`` is ROOT's: 0 to 4 for its levels, 24 or more for that
    many digits in every block (24 kept, the rest thrown away).
    """

    #: How many generators the default constructor has seeded, ROOT's ``fgNumEngines``.
    engines = 0

    def __init__(self, seed: int | None = None, lux: int = 3) -> None:
        self._luxury = operator.index(lux)
        if seed is None:
            seed = _table_seed(TRandom1.engines)
            TRandom1.engines += 1
        super().__init__(seed)

    def _engine_init(self) -> None:
        self._digits = [0] * 24
        self._ilag, self._carry, self._count24, self._skip = 23, 0, 0, LEVELS[3]

    def _engine_seed(self, seed: int) -> None:
        word = seed % UINT
        seeds = [word] if word else (TRandom3(0).rndm(24) * 4294967296.0).astype(np.int64).tolist()
        self._skip = skip_for(self._luxury)
        self._digits = seed_digits(seeds)
        self._ilag, self._count24 = 23, 0
        self._carry = 0 if self._digits[23] else 1

    def _history(self) -> list[int]:
        """The last 24 digits made, oldest first, out of ROOT's ring."""
        return [self._digits[(self._ilag - k) % 24] for k in range(24)]

    def _engine_fill(self, count: int) -> Any:
        if not count:
            return np.empty(0)
        history = self._history()
        numerator = number(history[14:]) - number(history) - self._carry
        where = np.arange(count) + self._count24
        block = where // 24
        raw = np.arange(count) + block * self._skip
        head = np.concatenate([history, expand(numerator, 48 - self._count24 + self._skip)])
        early = np.minimum(raw, len(head) - 25)
        x, fine = head[early + 24], head[early + 15]
        late = block >= 2
        if late.any():
            digits = _blocks(numerator, self._count24, self._skip, int(block[-1]) - 1)
            x[late] = digits[block[late] - 2, 9 + where[late] % 24]
            fine[late] = digits[block[late] - 2, where[late] % 24]
        self._count24 = int(where[-1] + 1) % 24
        self._finish(numerator, head, int(raw[-1]) + 1 + (0 if self._count24 else self._skip))
        return _outputs(x, fine)

    def _finish(self, numerator: int, head: Any, end: int) -> None:
        """Leave ROOT's ring, indices, carry and count as ``end`` steps on from ``numerator``."""
        after = advance(numerator, end)
        if end + 24 <= len(head):
            history = head[end : end + 24]
        else:
            history = expand(advance(numerator, end - 24), 24)
        self._ilag = (self._ilag - end) % 24
        for k, digit in enumerate(history.tolist()):
            self._digits[(self._ilag - k) % 24] = digit
        self._carry = number(history[14:]) - number(history) - after

    def _engine_export(self) -> dict[str, Any]:
        return {
            "fFloatSeedTable": (np.array(self._digits) * BIT24).astype(np.float32),
            "fIlag": self._ilag,
            "fJlag": (self._ilag + 10) % 24,
            "fCarry": self._carry * BIT24,
            "fCount24": self._count24,
            "fNskip": self._skip,
            "fLuxury": self._luxury,
        }

    def _engine_import(self, state: dict[str, Any]) -> None:
        self._digits = _table_digits(state.get("fFloatSeedTable", ()))
        self._ilag = _member(state, "fIlag", 0, 24)
        if _member(state, "fJlag", 0, 24) != (self._ilag + 10) % 24:
            raise ValueError("A TRandom1 state's fJlag is always ten places after its fIlag.")
        carry = state.get("fCarry")
        if carry not in (0, BIT24):
            raise ValueError(f"A TRandom1 state's fCarry is 0 or 2**-24, not {carry!r}.")
        self._carry = 1 if carry else 0
        self._count24 = _member(state, "fCount24", 0, 24)
        self._skip = _member(state, "fNskip", 0, 1 << 31)
        self._luxury = operator.index(state.get("fLuxury", 3))

    def _engine_get_seed(self) -> int:
        return self._digits[0]


def _table_seed(engines: int) -> int:
    """The seed ROOT's default constructor takes for the generator after ``engines`` others."""
    cycle, index = divmod(engines, len(SEED_TABLE))
    return SEED_TABLE[index] ^ ((cycle & 0x007FFFFF) << 8)


def _table_digits(table: Any) -> list[int]:
    """The 24 digits of a saved ``fFloatSeedTable``, which must be whole multiples of 2**-24."""
    digits = np.asarray(table, dtype=np.float64) * 2.0**24
    whole = (digits == np.floor(digits)) & (digits >= 0) & (digits < B)
    if digits.shape != (24,) or not np.all(whole):
        raise ValueError("A TRandom1 state's fFloatSeedTable is 24 multiples of 2**-24 in [0, 1).")
    return [int(digit) for digit in digits]
