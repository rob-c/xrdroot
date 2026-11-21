"""A split ``TObject``'s ``fUniqueID`` read back with the process that numbered it.

ROOT's writer stores an object's number alone and its reader folds the byte naming the process
into the identifier's top byte - ``((index + 1) << 24) | number`` - so ``TTree::Show`` prints
``16777225`` for a word the file holds as ``9``. An identifier no process ever gave an object,
zero, keeps its zero: the oracle prints an unreferenced collection's ``fUniqueID`` as ``0``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xrdroot.cxx import Prim
from xrdroot.interp import (
    PID_OFFSET,
    UID_MASK,
    Flat,
    HeapRows,
    NumberedFlat,
    NumberedRows,
    Rows,
    Values,
    _numbered,
    _numbered_column,
    build,
)
from xrdroot.objects import BranchRecord, LeafRecord


def identifiers(*values: int) -> bytes:
    """The words a file holds for ``values``, as the big-endian run a basket carries."""
    return b"".join(value.to_bytes(4, "big") for value in values)


def a_member(classname: str, member: str, ltype: int = 13, counted: bool = True) -> Any:
    """The column of ``member`` of ``classname``: of each object of a split collection, or one
    per entry when the leaf has no counter."""
    source = type("Described", (), {})()
    source.streamers = lambda: {}
    branch, leaf = BranchRecord(), LeafRecord("TLeafElement")
    branch.classname, leaf.name, leaf.ltype = classname, f"hits.{member}", ltype
    if counted:
        leaf.count = LeafRecord("TLeafElement")
        leaf.count.name = "hits_"
    return build(branch, leaf, source)


def test_a_numbered_identifier_gains_the_reading_processes_byte() -> None:
    rows = NumberedRows(Prim("uint32", "I", 4), 0, False)
    assert rows.decode(identifiers(9, 0, 8)).tolist() == [
        (PID_OFFSET << 24) | 9,
        0,
        (PID_OFFSET << 24) | 8,
    ]


def test_an_identifier_held_whole_gains_the_processes_byte_too() -> None:
    flat = NumberedFlat(Prim("uint32", "I", 4), 1)
    assert flat.decode(identifiers(3)).tolist() == [(PID_OFFSET << 24) | 3]


def test_the_processes_byte_leaves_the_objects_own_number_alone() -> None:
    """The byte goes above the number, which is what a reference looks the object up by."""
    numbered = _numbered(np.array([9, 0], dtype=np.uint32))
    assert [int(value) & UID_MASK for value in numbered] == [9, 0]


def test_only_a_split_objects_identifier_is_folded() -> None:
    prim = Prim("uint32", "I", 4)
    assert isinstance(_numbered_column(Rows(prim, 0, False)), NumberedRows)
    assert isinstance(_numbered_column(Flat(prim, 1)), NumberedFlat)
    assert isinstance(_numbered_column(Values("list", lambda buf: None)), Values)


def test_a_split_tobjects_identifier_branch_is_a_folded_column() -> None:
    column = a_member("TObject", "fUniqueID")
    assert isinstance(column, NumberedRows)
    assert column.decode(identifiers(9)).tolist() == [(PID_OFFSET << 24) | 9]


def test_a_split_tobjects_bits_branch_keeps_its_own_rule() -> None:
    """``fBits`` carries the heap's two bits and no process byte, so it is not folded."""
    column = a_member("TObject", "fBits")
    assert isinstance(column, HeapRows) and not isinstance(column, NumberedRows)
    assert isinstance(a_member("TObject", "fBits"), HeapRows)


def test_an_identifier_held_whole_by_an_entry_is_folded_too() -> None:
    """One object per entry, a fixed-size member: the same fold, a flat column."""
    column = a_member("TObject", "fUniqueID", counted=False)
    assert isinstance(column, NumberedFlat)
    assert column.decode(identifiers(4)).tolist() == [(PID_OFFSET << 24) | 4]


def test_a_member_that_is_not_an_identifier_is_left_as_the_file_wrote_it() -> None:
    column = a_member("TObject", "fSurname", 13)
    assert isinstance(column, Rows) and not isinstance(column, NumberedRows)
    assert column.decode(identifiers(7)).tolist() == [7]


def test_another_class_member_named_funiqueid_is_not_an_identifier() -> None:
    """Only a ``TObject`` base's identifier is one a process numbers."""
    column = a_member("Hit", "fUniqueID")
    assert not isinstance(column, NumberedRows) and isinstance(column, Rows)
