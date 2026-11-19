"""A class whose ``ClassDef`` version is zero, in the record it was written as.

ROOT's ``TBufferFile::WriteVersion`` puts a checksum where the version goes only for a
*foreign* class - one declared without ``ClassDef``, whose layout is told apart by that
checksum - and folds its version 1 to 0 when it does. A class that carries
``ClassDef(Name, 0)`` - ``roottest``'s ``io/abstractclass`` ``PureAbstractInterface``, an
interface with no data of its own - writes a plain zero version and nothing after it, so a
reader that takes any zero version for a checksum swallows four bytes of the class's own
record and reports it ending before it said it would.

ROOT tells the two apart by the record's byte count, as its class-less branch of
``ReadVersion`` does: a count of 6 is a version and a checksum, a count of 2 is the version
alone.
"""

from __future__ import annotations

from typing import Any

from xrdroot.buffer import Buffer
from xrdroot.interp import _streamed


def _positions() -> tuple[Any, list[int]]:
    """A reader for a record's contents, and where it found them."""
    seen: list[int] = []

    def read(buf: Buffer) -> str:
        seen.append(buf.pos)
        return "read"

    return _streamed(read), seen


def test_a_version_zero_class_is_read_as_a_version_and_not_as_a_checksum() -> None:
    read, seen = _positions()
    # the bytes roottest's io/abstractclass file holds for PureAbstractInterface: a count
    # of 2, which is the version alone, and that version, zero
    assert read(Buffer(bytes.fromhex("40000002 0000"))) == "read"
    assert seen == [6]


def test_a_foreign_class_still_carries_its_checksum_where_its_version_goes() -> None:
    read, seen = _positions()
    assert read(Buffer(bytes.fromhex("40000006 0000 1918368b"))) == "read"
    assert seen == [10]
