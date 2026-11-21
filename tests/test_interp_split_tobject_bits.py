"""A split ``TObject``'s ``fBits`` read back as the object holds it, not as the file wrote it.

ROOT masks ``kIsOnHeap|kNotDeleted`` out of ``fBits`` when it writes and sets them on the
object its reader fills, so a split collection's ``fBits`` column carries those two bits.
``TTree::Show`` prints that word, and the oracle reading a ``TClonesArray`` of ``TNamed``
prints ``50331648`` for a word the file holds as 0.
"""

from __future__ import annotations

from xrdroot.cxx import Prim
from xrdroot.interp import (
    HEAP_BITS,
    Flat,
    HeapFlat,
    HeapRows,
    Rows,
    Values,
    _in_memory,
)


def test_a_split_objects_bits_carry_the_heap_bits_the_reader_sets() -> None:
    rows = HeapRows(Prim("uint32", "u", 4), 0, False)
    values = rows.decode(b"\x00\x00\x00\x00\x00\x00\x00\x18")
    assert values.tolist() == [HEAP_BITS, HEAP_BITS | 0x18]


def test_an_object_held_whole_keeps_its_own_word_with_the_heap_bits() -> None:
    flat = HeapFlat(Prim("uint32", "u", 4), 1)
    assert flat.decode(b"\x00\x00\x00\x10").tolist() == [HEAP_BITS | 0x10]


def test_only_a_split_objects_bits_column_is_touched() -> None:
    prim = Prim("uint32", "u", 4)
    assert isinstance(_in_memory(Rows(prim, 0, False)), HeapRows)
    assert isinstance(_in_memory(Flat(prim, 1)), HeapFlat)
    assert isinstance(_in_memory(Values("list", lambda buf: None)), Values)
