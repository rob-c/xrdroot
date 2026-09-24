"""XXH3-64, the checksum every RNTuple envelope, page and anchor carries.

ROOT's older LZ4 blocks carry XXH64, which :mod:`..compression` computes;
RNTuple moved to its successor, XXH3, which is a different function with the
same width. The ``xxhash`` package - part of the ``lz4`` extra - computes it
in C, and when it is installed every checksum is checked on the way in, so a
damaged header, footer or page is refused rather than decoded into numbers
that look like data. Without it they are passed over on the way in, as the
LZ4 checksums are, because in Python they would cost more than the reading.

Writing cannot pass over them: a file whose checksums are wrong is a file ROOT
refuses. So XXH3 is here in Python too, seed zero and the default secret,
which is the only variant RNTuple uses. The long-input path - everything over
240 bytes, which is nearly every page - is the one that matters for time, and
it is arranged so that NumPy does the arithmetic a block of sixteen stripes at
a time: within a block the accumulators only ever add, and addition modulo
two to the sixty-four does not mind the order, so a block's stripes can be
summed at once and only the scrambling between blocks is a Python loop.

The test suite holds this against the C implementation, and against the
checksums ROOT itself wrote into the files under ``tests/data/rntuple``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

__all__ = ["xxh3_64", "checksum", "verifying"]

#: XXH3's default secret, the 192 bytes every seedless hash is keyed by.
SECRET = bytes.fromhex(
    "b8fe6c3923a44bbe7c01812cf721ad1cded46de9839097db7240a4a4b7b3671f"
    "cb79e64eccc0e578825ad07dccff7221b8084674f743248ee03590e6813a264c"
    "3c2852bb91c300cb88d0658b1b532ea371644897a20df94e3819ef46a9deacd8"
    "a8fa763fe39c343ff9dcbbc7c70b4f1d8a51e04bcdb45931c89f7ec9d9787364"
    "eac5ac8334d3ebc3c581a0fffa1363eb170ddd51b7f0da49d316552629d4689e"
    "2b16be587d47a1fc8ff8b8d17ad031ce45cb3a8f95160428afd7fbcabb4b407e"
)

#: The primes XXH3 borrows from XXH32 and XXH64, and the two of its own.
P32_1, P32_2, P32_3 = 0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D
P64_1, P64_2, P64_3, P64_4, P64_5 = (
    0x9E3779B185EBCA87,
    0xC2B2AE3D27D4EB4F,
    0x165667B19E3779F9,
    0x85EBCA77C2B2AE63,
    0x27D4EB2F165667C5,
)
MX1, MX2 = 0x165667919E3779F9, 0x9FB21C651E98DF25
M64 = (1 << 64) - 1

#: A stripe is sixty-four bytes, eight lanes of eight; a block is sixteen of
#: them, which is how many the default secret can key before it runs out.
STRIPE = 64
STRIPES_PER_BLOCK = (len(SECRET) - STRIPE) // 8
BLOCK = STRIPE * STRIPES_PER_BLOCK

#: The secret as sixty-four-bit words, which is how the long path reads it.
_WORDS = np.frombuffer(SECRET, dtype="<u8")
#: For stripe ``s`` of a block and lane ``i``, the secret word keying it.
_KEYS = np.stack([_WORDS[s : s + 8] for s in range(STRIPES_PER_BLOCK)])
#: Each lane's neighbour, which its raw input is added to.
_SWAP = np.array([1, 0, 3, 2, 5, 4, 7, 6])
_LOW = np.uint64(0xFFFFFFFF)
_ACC0 = (P32_3, P64_1, P64_2, P64_3, P64_4, P32_2, P64_5, P32_1)


def _u64(data: bytes, at: int) -> int:
    return int.from_bytes(data[at : at + 8], "little")


def _u32(data: bytes, at: int) -> int:
    return int.from_bytes(data[at : at + 4], "little")


def _fold(a: int, b: int) -> int:
    """The 128-bit product of two words, its halves XORed together."""
    product = a * b
    return (product & M64) ^ (product >> 64)


def _avalanche(h: int) -> int:
    h ^= h >> 37
    h = (h * MX1) & M64
    return h ^ (h >> 32)


def _xxh64_avalanche(h: int) -> int:
    h ^= h >> 33
    h = (h * P64_2) & M64
    h ^= h >> 29
    h = (h * P64_3) & M64
    return h ^ (h >> 32)


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & M64


def _rrmxmx(h: int, length: int) -> int:
    h ^= _rotl(h, 49) ^ _rotl(h, 24)
    h = (h * MX2) & M64
    h ^= (h >> 35) + length
    h = (h * MX2) & M64
    return h ^ (h >> 28)


def _mix16(data: bytes, at: int, secret_at: int) -> int:
    return _fold(
        _u64(data, at) ^ _u64(SECRET, secret_at),
        _u64(data, at + 8) ^ _u64(SECRET, secret_at + 8),
    )


def _upto3(data: bytes) -> int:
    length = len(data)
    combined = (data[0] << 16) | (data[length >> 1] << 24) | data[-1] | (length << 8)
    return _xxh64_avalanche(combined ^ (_u32(SECRET, 0) ^ _u32(SECRET, 4)))


def _upto8(data: bytes) -> int:
    length = len(data)
    joined = _u32(data, length - 4) + (_u32(data, 0) << 32)
    return _rrmxmx(joined ^ (_u64(SECRET, 8) ^ _u64(SECRET, 16)), length)


def _upto16(data: bytes) -> int:
    length = len(data)
    low = _u64(data, 0) ^ (_u64(SECRET, 24) ^ _u64(SECRET, 32))
    high = _u64(data, length - 8) ^ (_u64(SECRET, 40) ^ _u64(SECRET, 48))
    swapped = int.from_bytes(low.to_bytes(8, "little"), "big")
    return _avalanche((length + swapped + high + _fold(low, high)) & M64)


def _upto128(data: bytes) -> int:
    """Pairs of sixteen bytes from either end, working inwards."""
    length = len(data)
    acc = (length * P64_1) & M64
    rounds = (length - 1) // 32  # 0 for 17-32 bytes, up to 3 for 97-128
    for index in range(rounds, -1, -1):
        acc += _mix16(data, 16 * index, 32 * index)
        acc += _mix16(data, length - 16 * (index + 1), 32 * index + 16)
    return _avalanche(acc & M64)


def _upto240(data: bytes) -> int:
    """Sixteen bytes at a time, the first eight rounds then the rest."""
    length = len(data)
    acc = (length * P64_1) & M64
    for index in range(8):
        acc += _mix16(data, 16 * index, 16 * index)
    acc = _avalanche(acc & M64)
    for index in range(8, length // 16):
        acc += _mix16(data, 16 * index, 16 * (index - 8) + 3)
    acc += _mix16(data, length - 16, 136 - 17)
    return _avalanche(acc & M64)


def _stripes(lanes: np.ndarray[Any, Any], keys: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """What a run of stripes adds to the accumulators, lane by lane.

    ``lanes`` is ``(..., stripes, 8)`` words of input and ``keys`` the secret
    words keying each; what comes back has the stripe axis summed away.
    """
    keyed = lanes ^ keys
    product = (keyed & _LOW) * (keyed >> np.uint64(32))
    total: np.ndarray[Any, Any] = (product + lanes[..., _SWAP]).sum(axis=-2, dtype=np.uint64)
    return total


def _scramble(acc: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    acc = acc ^ (acc >> np.uint64(47))
    acc = acc ^ _WORDS[16:24]
    return acc * np.uint64(P32_1)


def _long(data: bytes) -> int:
    """Everything over 240 bytes: blocks of stripes, scrambled between blocks."""
    length = len(data)
    blocks = (length - 1) // BLOCK
    whole = np.frombuffer(data, dtype="<u8", count=blocks * BLOCK // 8)
    sums = _stripes(whole.reshape(blocks, STRIPES_PER_BLOCK, 8), _KEYS)
    acc = np.array(_ACC0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for block in sums:
            acc = _scramble(acc + block)
        tail = (length - 1 - blocks * BLOCK) // STRIPE
        start = blocks * BLOCK
        rest = np.frombuffer(data, dtype="<u8", count=tail * 8, offset=start)
        acc = acc + _stripes(rest.reshape(tail, 8), _KEYS[:tail])
        last = np.frombuffer(data, dtype="<u8", count=8, offset=length - STRIPE)
        key = np.frombuffer(SECRET, dtype="<u8", count=8, offset=len(SECRET) - STRIPE - 7)
        acc = acc + _stripes(last.reshape(1, 8), key.reshape(1, 8))
    words = [int(value) for value in acc]
    result = (length * P64_1) & M64
    for index in range(4):
        result += _fold(
            words[2 * index] ^ _u64(SECRET, 11 + 16 * index),
            words[2 * index + 1] ^ _u64(SECRET, 19 + 16 * index),
        )
    return _avalanche(result & M64)


def _short(data: bytes) -> int:
    """Sixteen bytes or fewer, each length range its own recipe."""
    if not data:
        return _xxh64_avalanche(_u64(SECRET, 56) ^ _u64(SECRET, 64))
    if len(data) <= 3:
        return _upto3(data)
    if len(data) <= 8:
        return _upto8(data)
    return _upto16(data)


def xxh3_64(data: bytes) -> int:
    """The XXH3-64 of ``data`` with seed zero, in Python, as RNTuple stores it."""
    data = bytes(data)
    if len(data) <= 16:
        return _short(data)
    if len(data) <= 128:
        return _upto128(data)
    if len(data) <= 240:
        return _upto240(data)
    return _long(data)


#: XXH3-64 in C, when the ``xxhash`` package is there to provide it.
_fast: Callable[[bytes], int] | None
try:  # pragma: no cover - depends on the optional lz4 extra
    from xxhash import xxh3_64_intdigest as _fast
except ImportError:  # pragma: no cover - the Python XXH3 above fills in
    _fast = None


def checksum(data: bytes) -> int:
    """XXH3-64 of ``data``, in C when ``xxhash`` is installed and in Python if not."""
    if _fast is None:
        return xxh3_64(data)
    return _fast(data)


def verifying() -> bool:
    """Whether checksums are checked on the way in: only when they are cheap."""
    return _fast is not None
