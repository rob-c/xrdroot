"""``ranluxpp/mulmod.h``: the arithmetic ROOT's ranlux++ generator works in.

``m = 2 ** 576 - 2 ** 240 + 1`` is the modulus, and a number below it is nine 64-bit
limbs, little-endian. ROOT writes the four functions limb by limb so that they run without
128-bit integers; here they are Python's own integers, masked back into the limbs the
caller passes, which is the same arithmetic without the machine's limit.
"""

from __future__ import annotations

from typing import Any

__all__ = ["multiply9x9", "mod_m", "mulmod", "powermod", "to_lcg", "MODULUS", "LIMBS"]

#: A number this arithmetic works on, as limbs of this many bits.
LIMBS = 9
#: One limb's width in bits, and the mask that keeps a value inside it.
WORD = 64
#: ``m = 2 ** 576 - 2 ** 240 + 1``: the modulus ``to_lcg`` and the engines reduce by.
MODULUS = (1 << (LIMBS * WORD)) - (1 << 240) + 1


def multiply9x9(first: Any, second: Any, out: Any) -> None:
    """``multiply9x9(a, b, out)``: the 18-limb product of two 9-limb numbers."""
    product = _number(first, LIMBS) * _number(second, LIMBS)
    _store(out, product, 2 * LIMBS)


def mod_m(mul: Any, out: Any) -> None:
    """``mod_m(mul, out)``: an 18-limb number reduced modulo ``m``, in 9 limbs."""
    _store(out, _number(mul, 2 * LIMBS) % MODULUS, LIMBS)


def mulmod(first: Any, inout: Any) -> None:
    """``mulmod(a, b)``: ``b`` replaced by ``a * b mod m``, both 9 limbs."""
    product = _number(first, LIMBS) * _number(inout, LIMBS)
    _store(inout, product % MODULUS, LIMBS)


def powermod(base: Any, res: Any, n: Any) -> None:
    """``powermod(base, res, n)``: ``res = base ** n mod m``.

    ``base`` and ``res`` may be the same array, as ROOT's comment allows: the base is read
    once into a number before anything is written.
    """
    _store(res, pow(_number(base, LIMBS), int(n), MODULUS), LIMBS)


def _number(limbs: Any, count: int) -> int:
    """The number a list of ``count`` 64-bit limbs, lowest first, stands for."""
    return sum((int(limbs[index]) & _MASK) << (WORD * index) for index in range(count))


def _store(limbs: Any, value: int, count: int) -> None:
    """``value`` written into ``count`` limbs of ``limbs``, lowest first, as ``uint64_t``."""
    for index in range(count):
        limbs[index] = (value >> (WORD * index)) & _MASK


#: What keeps one limb to 64 bits: ``uint64_t``'s own wrap.
_MASK = (1 << WORD) - 1



#: How far a RANLUX state's top bits sit above the LCG residue's: ``2 ** 240``.
#: The 240 high bits a residue differs from its state by: ``r - (r >> 336)``.
_SHIFT = 336


def to_lcg(ranlux: Any, c: Any, lcg: Any) -> None:
    """``to_lcg(ranlux, c, lcg)``: the LCG state a RANLUX state and its carry stand for.

    The RANLUX numbers are 576 bits and ``c`` one more; the residue is
    ``r - (r >> 336) + c`` modulo ``2 ** 576``, which is what ROOT's limb loop computes.
    """
    state = _number(ranlux, LIMBS)
    _store(lcg, (state - (state >> _SHIFT) + int(c)) % (1 << (LIMBS * WORD)), LIMBS)
