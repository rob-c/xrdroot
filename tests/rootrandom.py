"""ROOT's generators transcribed line for line, one draw at a time, to check the fast ones against.

Nothing here is quick or clever: each class is the C++ of ``TRandom.cxx``,
``TRandom1.cxx``, ``TRandom2.cxx`` or ``TRandom3.cxx`` written out in Python
integers and single-precision floats, statement by statement, so that it can
be read against ROOT's source. The library makes the same numbers an array
at a time by quite different means - NumPy's twister, jump matrices, RANLUX
as an LCG on 576-bit numbers - and the tests hold the two to each other.
"""

from __future__ import annotations

import numpy as np

MASK = 0xFFFFFFFF


class MT:
    """``TRandom3``: ``SetSeed`` for a non-zero seed and ``operator()``, as written in ROOT."""

    def __init__(self, seed: int) -> None:
        self.mt = [seed & MASK]
        for i in range(1, 624):
            last = self.mt[-1]
            self.mt.append((1812433253 * (last ^ (last >> 30)) + i) & MASK)
        self.count = 624

    def twist(self) -> None:
        mt = self.mt
        for i in range(623):
            y = (mt[i] & 0x80000000) | (mt[i + 1] & 0x7FFFFFFF)
            mt[i] = mt[(i + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
        y = (mt[623] & 0x80000000) | (mt[0] & 0x7FFFFFFF)
        mt[623] = mt[396] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
        self.count = 0

    def word(self) -> int:
        if self.count >= 624:
            self.twist()
        y = self.mt[self.count]
        self.count += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        return y ^ (y >> 18)

    def rndm(self) -> float:
        y = self.word()
        return y * 2.3283064365386963e-10 if y else self.rndm()


class LCG:
    """``TRandom``: BSD ``rand`` in ``UInt_t``, over ROOT's ``kCONS``."""

    def __init__(self, seed: int) -> None:
        self.seed = seed & MASK

    def rndm(self) -> float:
        self.seed = ((1103515245 * self.seed + 12345) & MASK) & 0x7FFFFFFF
        return 4.6566128730774e-10 * self.seed if self.seed else self.rndm()


def taus(s: int, a: int, b: int, c: int, d: int) -> int:
    """ROOT's ``TAUSWORTHE`` macro."""
    return (((s & c) << d) & MASK) ^ ((((s << a) & MASK) ^ s) >> b)


class Taus:
    """``TRandom2``: ``SetSeed`` for a non-zero seed, and ``Rndm``."""

    def __init__(self, seed: int) -> None:
        self.s = (69069 * seed) & MASK
        if self.s < 2:
            self.s += 2
        self.s1 = (69069 * self.s) & MASK
        if self.s1 < 8:
            self.s1 += 8
        self.s2 = (69069 * self.s1) & MASK
        if self.s2 < 16:
            self.s2 += 16
        for _ in range(6):
            self.rndm()

    def rndm(self) -> float:
        self.s = taus(self.s, 13, 19, 4294967294, 12)
        self.s1 = taus(self.s1, 2, 25, 4294967288, 4)
        self.s2 = taus(self.s2, 3, 11, 4294967280, 17)
        iy = self.s ^ self.s1 ^ self.s2
        return 2.3283064365386963e-10 * iy if iy else self.rndm()


class Ranlux:
    """``TRandom1``: ``SetSeeds`` from one seed, and ``Rndm``, in ``float`` as ROOT has them."""

    LEVELS = (0, 24, 73, 199, 365)

    def __init__(self, seed: int, lux: int = 3) -> None:
        self.nskip = self.LEVELS[lux] if 0 <= lux <= 4 else (lux - 24 if lux >= 24 else 199)
        table = [seed % 0x1000000]
        next_seed = table[0]
        while len(table) < 24:
            k = next_seed // 53668
            next_seed = 40014 * (next_seed - k * 53668) - k * 12211
            if next_seed < 0:
                next_seed += 2147483563
            table.append(next_seed % 0x1000000)
        self.table = [np.float32(t * 2.0**-24) for t in table]
        self.ilag, self.jlag, self.count24 = 23, 9, 0
        self.carry = np.float32(2.0**-24 if self.table[23] == 0 else 0.0)

    def step(self) -> np.float32:
        uni = self.table[self.jlag] - self.table[self.ilag] - self.carry
        if uni < 0.0:
            uni = np.float32(float(uni) + 1.0)
            self.carry = np.float32(2.0**-24)
        else:
            self.carry = np.float32(0.0)
        self.table[self.ilag] = uni
        self.ilag = (self.ilag - 1) % 24
        self.jlag = (self.jlag - 1) % 24
        return uni

    def rndm(self) -> float:
        uni = self.step()
        if uni < 2.0**-12:
            uni = np.float32(float(uni) + 2.0**-24 * float(self.table[self.jlag]))
            if uni == 0:
                uni = np.float32(2.0**-48)
        self.count24 += 1
        if self.count24 == 24:
            self.count24 = 0
            for _ in range(self.nskip):
                self.step()
        return float(uni)


def draws(oracle: object, count: int) -> np.ndarray:
    """The next ``count`` values of an oracle's ``Rndm()``."""
    return np.array([oracle.rndm() for _ in range(count)])  # type: ignore[attr-defined]
