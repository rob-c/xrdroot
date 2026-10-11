"""``ROOT::Math::LFSR``: pseudo-random bit sequences from a linear feedback shift register.

A ``std::bitset`` of ``k`` bits is the register; each step XORs the bits at
the taps - the exponents of the polynomial, exponent ``E`` the bit ``E-1`` -
shifts the register and feeds the new bit in, as ``Math/LFSR.h`` does.
``GenerateSequence`` steps from a seed until a register value comes round
again (or ``2^k - 1`` steps), the bits it fed in its answer.
"""

from __future__ import annotations

from typing import Any

from .messages import message

__all__ = ["LFSR"]


def _sequence(kind: str = "unsigned char") -> Any:
    from ..stl import std

    return std.vector[kind]()


class LFSR:
    """``ROOT::Math::LFSR``: its two functions."""

    @staticmethod
    def NextLFSR(lfsr: Any, taps: Any, left: bool = True) -> bool:
        """The new bit - the taps' bits XORed - and the register shifted, the bit fed in."""
        taps = [int(tap) for tap in taps]
        bit = lfsr[taps[0] - 1]
        for tap in taps[1:]:
            bit ^= lfsr[tap - 1]
        if left:
            lfsr <<= 1
            lfsr[0] = bit
        else:
            lfsr >>= 1
            lfsr[lfsr.size() - 1] = bit
        return bool(bit)

    @staticmethod
    def GenerateSequence(start: Any, taps: Any, left: bool = True, wrapping: bool = False,
                         oppositeBit: bool = False) -> Any:  # fmt: skip
        """The bits fed in from the seed ``start`` until the register repeats a value."""
        result = _sequence()
        if _refused(start, [int(tap) for tap in taps]):
            return result
        k, lfsr = start.size(), type(start)(start.size(), start.to_ulong())
        if oppositeBit:
            result.push_back(_far(lfsr, left))
        seen: set[int] = set()
        for _ in range(2**k - 1):
            bit = LFSR.NextLFSR(lfsr, taps, left)
            result.push_back(_far(lfsr, left) if oppositeBit else int(bit))
            if _repeated(lfsr, seen, wrapping) or lfsr == start:
                break
        if oppositeBit:
            result.pop_back()
        return result


def _refused(start: Any, taps: list[int]) -> bool:
    """A tap outside the register, or a seed of no bits set: ROOT's error, and no sequence."""
    k = start.size()
    bad = [tap for tap in taps if tap > k or tap == 0]
    if bad:
        message("Error", "ROOT::Math::LFSR", f"Tap {bad[0]} is out of range [1,{k}]")
    elif start.none():
        message("Error", "ROOT::Math::LFSR", "A non-zero start value is needed")
    return bool(bad) or start.none()


def _far(lfsr: Any, left: bool) -> int:
    """The bit at the register's far end - its top for a left shift, its bottom for a right."""
    return int(lfsr[lfsr.size() - 1] if left else lfsr[0])


def _repeated(lfsr: Any, seen: set[int], wrapping: bool) -> bool:
    """Whether the register holds a value it held before - which, wrapping, is not looked at."""
    if wrapping:
        return False
    value = lfsr.to_ulong()
    found = value in seen
    seen.add(value)
    return found
