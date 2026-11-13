"""A record moved under a key of another length: its references made to point right again.

Inside a record ROOT points back at what it wrote earlier - a class named in
full the first time, an object met before - by place, and every place is
counted from the start of the record's *key*. A record copied under a name
of another length has a key of another length in front of it, so every one
of those places is off by the difference, and the copy reads back wrong.
ROOT itself never meets this: it writes a renamed object afresh.

This finds every place by reading the record as its file describes the
class - the same walk :func:`~xrdroot.interp.whole_object` takes, through
:meth:`Buffer.any`, which is where every reference is read - noting where
each reference sits, and moves each by the difference. It refuses unless
that walk read every byte and every reference resolved, and unless the
moved record reads back the same way under its new key, so a record whose
class streams itself some way the file does not describe is refused, never
moved by a guess.
"""

from __future__ import annotations

import struct
from typing import Any

from ..buffer import BYTE_COUNT_MASK, CLASS_MASK, MAP_OFFSET, NEW_CLASS_TAG, Buffer
from ..errors import FormatError, UnsupportedFeatureError
from ..file import Key, Source

__all__ = ["relocated"]

#: A reference's word.
WORD = struct.Struct(">I")
#: The reference to the record's own object, which ROOT files at place 1 whatever its key.
SELF = 1


class _Noting(Buffer):
    """A buffer that notes where every reference it reads sits, and anything it steps over."""

    __slots__ = ("places", "whole", "begun")

    def __init__(self, data: bytes, offset: int) -> None:
        super().__init__(data, offset)
        #: Where each reference read is, counted as the buffer counts.
        self.places: list[int] = []
        #: Whether every byte was read and every reference found what it points at.
        self.whole = True
        #: Where each object begins, as a reference names it, since ROOT counts an
        #: object as met from the moment it starts on it.
        self.begun: set[int] = set()

    def _word(self, at: int) -> int:
        value: int = WORD.unpack_from(self.data, at - self.offset)[0]
        return value

    def any(self, classes: dict[str, Any]) -> Any:
        start = self.pos
        count = self._word(start)
        counted = bool(count & BYTE_COUNT_MASK) and count != NEW_CLASS_TAG
        at = start + 4 if counted else start
        tag = self._word(at)
        if tag & CLASS_MASK:
            self._object(start, at, tag, counted, classes)
        elif tag not in (0, SELF):
            self.places.append(at)
            self.whole &= tag in self.begun  # an object pointed at that was never read
        return super().any(classes)

    def _object(self, start: int, at: int, tag: int, counted: bool, classes: Any) -> None:
        """An object, its class named in full or pointed at: noted, and whether it is read."""
        self.begun.add(start + MAP_OFFSET)
        if tag != NEW_CLASS_TAG:
            self.places.append(at)
        name = self.cstring_at(at) if tag == NEW_CLASS_TAG else self.refs.get(tag & ~CLASS_MASK)
        if counted and isinstance(name, str) and classes.get(name) is None:
            self.whole = False  # stepped over, unread

    def cstring_at(self, at: int) -> str:
        """The class name written in full after a new-class tag at ``at``."""
        index = at + 4 - self.offset
        end = self.data.index(b"\0", index)
        return self.data[index:end].decode("latin-1")

    def resume(self, end: int | None) -> None:
        if end is not None and end > self.pos:
            self.whole = False
        super().resume(end)

    def skip_record(self) -> int:
        self.whole = False
        return super().skip_record()


def _walked(payload: bytes, keylen: int, classname: str, source: Source) -> _Noting:
    """The record read as its file describes the class, noting every reference."""
    from ..interp import Refused, whole_object

    column = whole_object(classname, source)
    if isinstance(column, Refused):
        raise UnsupportedFeatureError(f"a {classname} is not read here: {column.reason}")
    buf = _Noting(payload, keylen)
    try:
        column.value(buf, buf.pos)
    except FormatError as why:
        raise UnsupportedFeatureError(f"a {classname} streams itself its own way ({why})") from why
    if buf.remaining or not buf.whole:
        raise UnsupportedFeatureError(
            f"a {classname} streams some of itself in a way its file does not describe"
        )
    return buf


def relocated(key: Key, source: Source, keylen: int) -> bytes:
    """``key``'s record, uncompressed, its references moved for a key ``keylen`` long."""
    payload = key.payload(source)
    walked = _walked(payload, key.keylen, key.classname, source)
    moved = bytearray(payload)
    shift = keylen - key.keylen
    for at in walked.places:
        index = at - key.keylen
        value = WORD.unpack_from(payload, index)[0]
        flag = value & CLASS_MASK
        WORD.pack_into(moved, index, flag | ((value & ~CLASS_MASK) + shift))
    _walked(bytes(moved), keylen, key.classname, source)  # and it reads, every place found
    return bytes(moved)
