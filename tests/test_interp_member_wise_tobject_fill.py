"""A ``TObject`` read a member at a time carries the two words its object has, not the file's.

ROOT writes an object's ``fBits`` with ``kIsOnHeap|kNotDeleted`` masked out and its
``fUniqueID`` as the number alone, then sets the two bits and folds the reading process's
byte in on the object it fills - so a container written a member at a time, which ROOT reads
straight into objects it has already made, holds neither of the file's two words.
:data:`~xrdroot.interp.FOLDED` is what turns them round, and the columns a member-wise
container is read with are the ones these tests drive.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xrdroot.buffer import Buffer
from xrdroot.interp import FOLDED, HEAP_BITS, PID_OFFSET, _column, _columns, _steps
from xrdroot.streamers import Member

#: ``TStreamerBasicType``: a member that is one number, as ``TObject``'s two words are.
BASIC = 13


class Described:
    """A file that describes the classes given it, and nothing else.

    ``_columns`` asks a source for the streamer information it holds, so one mapping of class
    name to its members is all a class needs to be read here.
    """

    def __init__(self, classes: dict[str, list[Member]]) -> None:
        self.classes = {
            name: {member.name: member for member in held} for name, held in classes.items()
        }

    def streamers(self) -> dict[str, dict[str, Member]]:
        return self.classes


def a_word(name: str) -> Member:
    """One ``UInt_t`` member, which is what both of ``TObject``'s two words are."""
    return Member(name, "", BASIC, "UInt_t", 1)


def a_tobject(classes: dict[str, list[Member]] | None = None) -> Described:
    """A file describing ``TObject``'s two words, and whatever else a test adds to it."""
    found = {"TObject": [a_word("fUniqueID"), a_word("fBits")]}
    return Described({**found, **(classes or {})})


def words(*values: int) -> Buffer:
    """The bytes a file holds for ``values``: four big-endian bytes each, one after another."""
    return Buffer(b"".join(value.to_bytes(4, "big") for value in values))


def read(columns: list[tuple[str, Any]], *values: int, rows: list[dict[str, Any]]) -> None:
    """Every one of ``columns``, read into ``rows`` as reading a container's entry does.

    Each member's words are its own run in the file - one column reads only its own member -
    so every column is handed the bytes afresh.
    """
    for _label, column in columns:
        column(words(*values), rows)


def test_a_member_wise_tobjects_identifier_gains_the_processes_byte() -> None:
    rows: list[dict[str, Any]] = [{}, {}]
    read(_columns("TObject", a_tobject(), ()), 9, 0, rows=rows)
    assert [int(row["fUniqueID"]) for row in rows] == [(PID_OFFSET << 24) | 9, 0]


def test_a_member_wise_tobjects_bits_gain_the_heaps_two_bits() -> None:
    rows: list[dict[str, Any]] = [{}, {}]
    read(_columns("TObject", a_tobject(), ()), 0x18, 0, rows=rows)
    assert [int(row["fBits"]) for row in rows] == [HEAP_BITS | 0x18, HEAP_BITS]


def test_a_class_that_is_not_tobject_keeps_the_words_the_file_holds() -> None:
    """Only ``TObject``'s two words are not the file's: another class's word is its own."""
    held = {"TThing": [Member("fBits", "", BASIC, "UInt_t", 1)]}
    rows: list[dict[str, Any]] = [{}]
    read(_columns("TThing", a_tobject(held), ()), 0x18, rows=rows)
    assert rows == [{"fBits": 0x18}]


def test_a_member_of_tobject_read_with_no_fold_keeps_the_files_word() -> None:
    """The fold is given to a ``TObject``'s members and to no other column, so a caller that
    asks for none gets the word the file holds."""
    source = a_tobject()
    step = dict(_steps("TObject", source))["fBits"]
    column = _column("fBits", a_word("fBits"), step, source, ())
    rows: list[dict[str, Any]] = [{}]
    column(words(0x18), rows)
    assert rows == [{"fBits": 0x18}]


def test_a_folded_column_of_no_objects_reads_nothing() -> None:
    """An entry of no objects has no words to fold, and takes none from the buffer."""
    buf = words()
    for _label, column in _columns("TObject", a_tobject(), ()):
        column(buf, [])
    assert buf.pos == 0


def test_a_fold_is_the_one_a_split_branchs_column_gets() -> None:
    """One mechanism, two routes: a branch's column and a member-wise column agree."""
    assert sorted(FOLDED) == ["fBits", "fUniqueID"]
    assert FOLDED["fBits"](np.array([0], dtype=np.uint32)).tolist() == [HEAP_BITS]
