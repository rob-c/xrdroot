"""How long ROOT 6.40 streams what this writer streams: what ``TTree::Print`` counts.

``TBranch::GetTotalSize`` - what ``Print`` adds to a branch's baskets -
streams the branch on its own into a fresh buffer, and ``Print``'s total
for the tree adds the tree's own key and record, and what that record takes
on file. Four things make what ROOT 6.40 streams other than the records this
writer puts in a file, which are of the ROOT 6.08 vintage :mod:`.winfo`
describes: a class named a second time is a four-byte reference back to the
first; a 6.40 ``TTree`` and ``TBranch`` are a version on and carry their
``fIOFeatures``, and its ``TAttMarker`` is a version on too; ``TObject::Streamer``
no longer writes the bits saying an object is on the heap and not deleted;
and the places of baskets are there only while the tree that wrote them is
open - ``held``.

A :class:`Measuring` buffer is one the writers stream into as they do into a
record, and that makes those differences, byte for byte: its length is
ROOT's, and so - its bytes being ROOT's - is what compressing it takes.
"""

from __future__ import annotations

from typing import Any

from .writer import BITS, WBuffer

__all__ = ["IO_FEATURES", "Measuring", "version"]

#: What a ROOT 6.40 ``TBranch`` or ``TTree`` streams as its ``fIOFeatures``: a record of
#: version 0, so with the class's checksum, around its one byte of bits, none set.
IO_FEATURES = bytes.fromhex("40000007" "0000" "1aa12f10" "00")
#: The versions ROOT 6.40 streams the classes at that this writer streams at older ones.
ROOT_VERSIONS = {"TTree": 20, "TBranch": 13, "TAttMarker": 3}
#: ``TObject``'s ``kIsOnHeap`` and ``kNotDeleted``, which 6.40's streamer leaves out.
TRANSIENT = 0x03000000
#: What ROOT adds to a place in a buffer to make a reference to it, and marks a class with.
MAP_OFFSET, CLASS_MASK = 2, 0x80000000


def version(buf: WBuffer, classname: str, ours: int) -> int:
    """The version to stream ``classname`` at: this writer's, or ROOT 6.40's when measuring."""
    return ROOT_VERSIONS.get(classname, ours) if isinstance(buf, Measuring) else ours


class Measuring(WBuffer):
    """A buffer streamed into as a record is, whose bytes are what ROOT 6.40's would be."""

    def __init__(self, held: bool, origin: int = 0) -> None:
        super().__init__()
        self.held = held
        self.origin = origin
        self._named: dict[str, int] = {}

    def tag(self, classname: str) -> int:
        first = self._named.get(classname)
        if first is None:
            index = super().tag(classname)
            self._named[classname] = index
            return index
        index = len(self.data)
        self.data += b"\x00\x00\x00\x00"  # the byte count, filled in when the object ends
        self.u32(CLASS_MASK | (self.origin + first + 4 + MAP_OFFSET))
        return index

    def tobject(self, bits: int = BITS, unique: int = 0) -> None:
        super().tobject(bits & ~TRANSIENT, unique)  # a TNamed's TObject comes through here too

    def features(self) -> None:
        """Where ROOT 6.40 streams an ``fIOFeatures`` this writer does not."""
        self.raw(IO_FEATURES)


def measured(write: Any, held: bool, origin: int = 0) -> bytes:
    """What ``write(buf)`` streams, as ROOT 6.40 would stream it."""
    buf = Measuring(held, origin)
    write(buf)
    return bytes(buf.data)
