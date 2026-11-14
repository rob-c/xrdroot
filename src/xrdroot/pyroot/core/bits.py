"""``TBits``: a row of bits that grows as bits past its end are set.

ROOT keeps the bits in bytes, the lowest bit of the first byte first, and
``fNbits`` is one past the highest bit ever set - or the size it was made
with. Setting a bit past the end grows the row - its bytes to twice what that
bit needs, as ROOT grows them, which is what a file then holds; testing one
past it is false.
"""

from __future__ import annotations

from typing import Any

from .objects import TObject

__all__ = ["TBits"]

#: ``kMaxUInt``: where ``LastSetBit`` and ``LastNullBit`` look back from unless told.
MOST = 0xFFFFFFFF


class TBits(TObject):
    """``TBits``: bits by number, as many as were asked for or set."""

    CLASS_TITLE = "Bit container"

    def __init__(self, nbits: Any = 8) -> None:
        super().__init__()
        if isinstance(nbits, TBits):
            self._nbits: int = nbits._nbits
            self._bytes: bytearray = bytearray(nbits._bytes)
            return
        self._nbits = int(nbits) if int(nbits) > 0 else 8
        self._bytes = bytearray((self._nbits - 1) // 8 + 1)

    def _grow(self, bit: int) -> None:
        if bit < self._nbits:
            return
        need = bit // 8 + 1
        if need > len(self._bytes):
            self._bytes.extend(bytes(2 * need - len(self._bytes)))
        self._nbits = bit + 1

    def SetBitNumber(self, bit: int, value: bool = True) -> None:
        bit = int(bit)
        if not value:
            self.ResetBitNumber(bit)
            return
        self._grow(bit)
        self._bytes[bit >> 3] |= 1 << (bit & 7)

    def ResetBitNumber(self, bit: int) -> None:
        bit = int(bit)
        if bit < self._nbits:
            self._bytes[bit >> 3] &= ~(1 << (bit & 7)) & 0xFF

    def TestBitNumber(self, bit: int) -> bool:
        bit = int(bit)
        return 0 <= bit < self._nbits and bool(self._bytes[bit >> 3] & (1 << (bit & 7)))

    def __getitem__(self, bit: int) -> bool:
        return self.TestBitNumber(bit)

    def __setitem__(self, bit: int, value: Any) -> None:
        self.SetBitNumber(bit, bool(value))

    # -- finding bits: the row as one integer, the lowest bit the first ----------------

    def _value(self) -> int:
        return int.from_bytes(self._bytes, "little")

    def _within(self) -> int:
        """The bits below ``fNbits``, as one integer."""
        return self._value() & ((1 << self._nbits) - 1)

    def _set(self) -> list[int]:
        value, found = self._within(), []
        while value:
            low = value & -value
            found.append(low.bit_length() - 1)
            value ^= low
        return found

    def CountBits(self, start: int = 0) -> int:
        """How many bits are set from ``start`` on - in every byte kept, as ROOT counts."""
        return (self._value() >> int(start)).bit_count()

    def _first(self, value: int, start: int) -> int:
        shifted = value >> start
        return start + (shifted & -shifted).bit_length() - 1 if shifted else self._nbits

    def _last(self, value: int, last: int) -> int:
        kept = value & ((1 << (min(int(last), self._nbits - 1) + 1)) - 1)
        return kept.bit_length() - 1 if kept else self._nbits

    def _nulls(self) -> int:
        return ~self._value() & ((1 << self._nbits) - 1)

    def FirstSetBit(self, start: int = 0) -> int:
        """The first bit set from ``start`` on - ``fNbits`` if none is."""
        return self._first(self._within(), max(int(start), 0))

    def LastSetBit(self, last: int = MOST) -> int:
        """The last bit set up to ``last`` - ``fNbits`` if none is."""
        return self._last(self._within(), last)

    def FirstNullBit(self, start: int = 0) -> int:
        return self._first(self._nulls(), max(int(start), 0))

    def LastNullBit(self, last: int = MOST) -> int:
        return self._last(self._nulls(), last)

    def GetNbits(self) -> int:
        return self._nbits

    def GetNbytes(self) -> int:
        return len(self._bytes)

    def ResetAllBits(self, value: bool = False) -> None:
        """``ResetAllBits``: every bit cleared - whatever ``value`` says, as ROOT's does."""
        self._bytes = bytearray(len(self._bytes))

    def Clear(self, option: str = "") -> None:
        self._nbits = 0
        self._bytes = bytearray(1)

    def Compact(self) -> None:
        """``Compact``: down to the last byte with a bit set - whole bytes, as ROOT keeps."""
        needed = max((index + 1 for index, byte in enumerate(self._bytes) if byte), default=1)
        if needed != len(self._bytes):
            self._nbits = 8 * needed
            self._bytes = self._bytes[:needed]

    def Print(self, option: str = "") -> None:
        """``Print``: each bit set, by number, as ROOT lists them."""
        for bit in self._set():
            print(f" bit:{bit:4d} = 1")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TBits) and self._within() == other._within()

    def __hash__(self) -> int:
        return hash(self._within())

    # -- bitwise, as ROOT's operators: the left side's size, its bytes beyond the right's -----

    def _combined(self, other: TBits, how: Any) -> TBits:
        made = TBits(self)
        count = len(made._bytes)
        mine = made._value()
        theirs = other._value() & ((1 << (8 * count)) - 1)
        made._bytes = bytearray(how(mine, theirs).to_bytes(count, "little"))
        return made

    def __and__(self, other: TBits) -> TBits:
        return self._combined(other, lambda a, b: a & b)

    def __or__(self, other: TBits) -> TBits:
        return self._combined(other, lambda a, b: a | b)

    def __xor__(self, other: TBits) -> TBits:
        return self._combined(other, lambda a, b: a ^ b)

    def __invert__(self) -> TBits:
        made = TBits(self)
        made._bytes = bytearray(~byte & 0xFF for byte in made._bytes)
        return made

    def __ilshift__(self, shift: int) -> TBits:
        """``<<=``: every bit ``shift`` places up, the row longer by as many."""
        value, bits = self._value() << int(shift), self._nbits + int(shift)
        self._nbits = bits
        self._bytes = bytearray(value.to_bytes(max((bits + 7) // 8, len(self._bytes)), "little"))
        return self

    def __lshift__(self, shift: int) -> TBits:
        made = TBits(self)
        made <<= shift
        return made

    def __rshift__(self, shift: int) -> TBits:
        made = TBits(self)
        made >>= shift
        return made

    def cxx_ostream(self) -> str:
        """``operator<<``: every bit of every byte, the highest first - ``TBits::Output``."""
        return "".join(f"{byte:08b}" for byte in reversed(self._bytes))

    def __irshift__(self, shift: int) -> TBits:
        """``>>=``: every bit ``shift`` places down, the row shorter by as many."""
        value = self._value() >> int(shift)
        self._nbits = max(self._nbits - int(shift), 0)
        self._bytes = bytearray(value.to_bytes(len(self._bytes), "little"))
        return self

    # -- what is written ----------------------------------------------------------

    def _members(self) -> dict[str, Any]:
        """``fNbits``, ``fNbytes`` and ``fAllBits``, as ``TBits`` streams them."""
        return {"fNbits": self._nbits, "fNbytes": len(self._bytes), "fAllBits": bytes(self._bytes)}

    @classmethod
    def _from_members(cls, members: dict[str, Any]) -> TBits:
        made = cls()
        held = members.get("fAllBits")
        if held is not None:
            made._bytes = bytearray(int(value) & 0xFF for value in held)
        made._nbits = int(members.get("fNbits", 8 * len(made._bytes)))
        return made
