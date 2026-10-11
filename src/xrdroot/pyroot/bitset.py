"""``std::bitset<N>``: ``N`` bits held as an integer, bit 0 the lowest, as C++ holds them.

``std.bitset[3]()`` is three zero bits, ``bitset[8](0b1010)`` those of a
number; the bits index, flip, shift within their width, compare and print
as C++'s do (``to_string`` the highest bit first): what a macro of shift
registers asks of them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["bitset"]


class Bitset:
    """``N`` bits."""

    __slots__ = ("_bits", "_size")

    def __init__(self, size: int, value: Any = 0) -> None:
        self._size = int(size)
        text = str(value) if isinstance(value, str) else None
        number = int(text, 2) if text else int(value)
        self._bits = number & self._mask()

    def _mask(self) -> int:
        return (1 << self._size) - 1

    def size(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> bool:
        return bool(self._bits >> int(index) & 1)

    def __setitem__(self, index: int, value: Any) -> None:
        bit = 1 << int(index)
        self._bits = self._bits | bit if value else self._bits & ~bit

    def flip(self, index: Any = None) -> Bitset:
        """``flip()`` every bit, or ``flip(i)`` one - in place, as C++'s, and the set returned."""
        self._bits ^= self._mask() if index is None else 1 << int(index)
        return self

    def count(self) -> int:
        return bin(self._bits).count("1")

    def any(self) -> bool:
        return self._bits != 0

    def none(self) -> bool:
        return self._bits == 0

    def to_ulong(self) -> int:
        return self._bits

    def to_string(self) -> str:
        return format(self._bits, f"0{self._size}b") if self._size else ""

    def __ilshift__(self, shift: int) -> Bitset:
        self._bits = (self._bits << int(shift)) & self._mask()
        return self

    def __irshift__(self, shift: int) -> Bitset:
        self._bits >>= int(shift)
        return self

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Bitset) and (self._size, self._bits) == (other._size, other._bits)

    def __hash__(self) -> int:
        return hash((self._size, self._bits))

    def __str__(self) -> str:
        return self.to_string()


class _BitsetTemplate:
    """``std.bitset[N]``: the maker of bitsets of ``N`` bits."""

    def __repr__(self) -> str:
        return "<std.bitset template>"

    def __getitem__(self, size: int) -> Any:
        def made(value: Any = 0) -> Bitset:
            return Bitset(int(size), value)

        return made


#: ``ROOT.std.bitset``.
bitset = _BitsetTemplate()
