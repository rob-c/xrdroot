"""``ranluxpp/mulmod.h``: the arithmetic ROOT's ranlux++ generator works in.

``m = 2 ** 576 - 2 ** 240 + 1`` is the modulus, and a number below it is nine 64-bit
limbs, little-endian. ROOT writes the four functions limb by limb so that they run without
128-bit integers; here they are Python's own integers, masked back into the limbs the
caller passes, which is the same arithmetic without the machine's limit.
"""

from __future__ import annotations

from typing import Any

__all__ = ["multiply9x9", "mod_m", "mulmod", "powermod", "to_lcg", "to_ranlux",
           "MODULUS", "LIMBS"]

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


#: ``helpers.h``'s four limb primitives: ``add_overflow``/``sub_overflow`` *set* the flag,
#: ``add_carry``/``sub_carry`` *add to* it - which is how ROOT's loops carry a borrow across
#: two subtractions of one limb.
def _added(a: int, b: int) -> tuple[int, int]:
    """``add_overflow(a, b, overflow)``: the sum in 64 bits, and whether it carried out."""
    total = a + b
    return total & _MASK, 1 if total > _MASK else 0


def _add_to_carry(a: int, b: int, carry: int) -> tuple[int, int]:
    """``add_carry(a, b, carry)``: the sum, and ``carry`` grown by its own carry out."""
    total, carried = _added(a, b)
    return total, carry + carried


def _subtracted(a: int, b: int) -> tuple[int, int]:
    """``sub_overflow(a, b, overflow)``: the difference in 64 bits, and whether it borrowed."""
    return (a - b) & _MASK, 1 if b > a else 0


def _sub_from_carry(a: int, b: int, carry: int) -> tuple[int, int]:
    """``sub_carry(a, b, carry)``: the difference, and ``carry`` grown by its borrow."""
    difference, borrowed = _subtracted(a, b)
    return difference, carry + borrowed


def _bits_of(upper: list[int], index: int) -> int:
    """The ``index``-th limb of ``upper >> 336``: the high 240 bits, two limbs per piece."""
    bits = 0
    if index < 4:
        bits += upper[index + 5] >> 16
        if index < 3:
            bits += (upper[index + 6] << 48) & _MASK
    return bits & _MASK


def _compute_r(upper: list[int]) -> int:
    """``compute_r(upper, r)`` of a zeroed ``r``: the correction ROOT reduces by, -1, 0 or 1.

    ``r`` is ``-upper``, then less its own high 240 bits, then more the pair shifted up by
    240 - each loop reporting its borrow or carry - and one more when what is left holds at
    least ``m``. Only the correction is needed here, so ``r`` is not returned.
    """
    rest = [0] * LIMBS
    carry = 0
    for index in range(LIMBS):
        rest[index], carry = _subtracted(rest[index], carry)
        rest[index], carry = _sub_from_carry(rest[index], upper[index], carry)
    correction = -carry
    carry = 0
    for index in range(LIMBS):
        rest[index], carry = _subtracted(rest[index], carry)
        rest[index], carry = _sub_from_carry(rest[index], _bits_of(upper, index), carry)
    correction -= carry
    carry = 0
    limb, carry = _add_to_carry(rest[3], ((upper[5] >> 16) << 48) & _MASK, carry)
    limb, carry = _add_to_carry(limb, (upper[0] << 48) & _MASK, carry)
    rest[3] = limb
    for index in range(3):
        limb, carry = _added(rest[index + 4], carry)
        limb, carry = _add_to_carry(limb, _shifted_pair(upper, index), carry)
        limb, carry = _add_to_carry(limb, _high_pair(upper, index), carry)
        rest[index + 4] = limb
    limb, carry = _added(rest[7], carry)
    limb, carry = _add_to_carry(limb, upper[8] >> 32, carry)
    limb, carry = _add_to_carry(limb, _high_pair(upper, 3), carry)
    rest[7] = limb
    limb, carry = _added(rest[8], carry)
    limb, carry = _add_to_carry(limb, _high_pair(upper, 4), carry)
    rest[8] = limb
    correction += carry
    return correction + (1 if correction == 0 and _at_least_m(rest) else 0)


def _shifted_pair(upper: list[int], index: int) -> int:
    """The ``t2`` limb of ROOT's third loop: ``upper`` read from limb ``index + 5``."""
    return ((upper[5 + index] >> 32) + ((upper[6 + index] << 32) & _MASK)) & _MASK


def _high_pair(upper: list[int], index: int) -> int:
    """The ``t3`` limb of ROOT's third loop: ``upper`` read from limb ``index``."""
    return ((upper[index] >> 16) + ((upper[index + 1] << 48) & _MASK)) & _MASK


def _at_least_m(rest: list[int]) -> bool:
    """``r >= m``: the low 240 bits are not all zero and the 336 above them are all set."""
    low = rest[0] | rest[1] | rest[2] | (rest[3] & 0x0000FFFFFFFFFFFF)
    return low != 0 and (rest[3] >> 48) == 0xFFFF and all(
        rest[index] == _MASK for index in range(4, LIMBS)
    )


def to_ranlux(lcg: Any, ranlux: Any, c_out: Any) -> None:
    """``to_ranlux(lcg, ranlux, c_out)``: the state and carry whose residue this ``lcg`` is.

    ``to_lcg`` loses the carry, so a residue comes back as the state ROOT's own arithmetic
    finds: ``lcg + (lcg >> 336)``, then the correction from ``compute_r`` added - one to
    the first limb, or one taken from every limb when the correction is ``-1``, which ROOT
    does not write back for the limbs after the first. ``c_out`` is the carry the last of
    those additions left.
    """
    upper = [int(word) & _MASK for word in lcg]
    correction = _compute_r(upper)
    state = [0] * LIMBS
    carry = 0
    for index in range(LIMBS):
        state[index], carry = _added(upper[index], carry)
        state[index], carry = _add_to_carry(state[index], _bits_of(upper, index), carry)
    taken = correction >> 1
    state[0], carry = _added(state[0], correction & _MASK)
    for index in range(1, LIMBS):
        limb, carry = _added(state[index], carry)
        limb, carry = _add_to_carry(limb, taken & _MASK, carry)
    _store(ranlux, _number(state, LIMBS), LIMBS)
    c_out.value = carry
