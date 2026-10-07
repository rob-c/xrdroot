"""Objects streamed by their class's description, as ``TBufferFile::WriteClassBuffer`` does.

A whole object in a branch is its class's streamer run over it: a byte count
and the class version, then each element of the class's ``TStreamerInfo`` in
order - a base class as an object of its own, a ``TObject`` base as
``TObject::Streamer`` writes one, a number as its streamer type packs it, an
object member as that object. :func:`stream` does that for any class
:data:`~.winfo.INFOS` describes, reading the members through ``get``.

A ``TClonesArray`` asked to bypass its objects' streamers writes them member
by member instead, ``TStreamerInfo::WriteBufferClones``: every object's
``TObject`` part, then the first member of every object, then the second...
:func:`stream_clones` writes the whole array that way. And a ``TBranchObject``
names an object's class in front of it (:func:`named`), as its
``TLeafObject`` writes every entry.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .winfo import INFOS
from .wpacking import pack_member
from .writer import WBuffer

__all__ = ["IGNORED", "named", "stream", "stream_clones"]

#: The classes whose ``TObject`` part is not streamed: ``TClass::IgnoreTObjectStreamer``.
IGNORED: set[str] = set()
#: The bits ``TObject::Streamer`` never writes: ``kIsOnHeap`` and ``kNotDeleted``.
TRANSIENT_BITS = 0x01000000 | 0x02000000
#: ``TClonesArray::kBypassStreamer``: the array's objects are written member by member.
BYPASS_STREAMER = 1 << 12

#: How a member is read off the object being streamed: by its name.
Getter = Callable[[str], Any]


def _tobject(buf: WBuffer, bits: int = 0) -> None:
    """``TObject::Streamer``: version 1, the identifier, the bits it keeps."""
    buf.raw(struct.pack(">hII", 1, 0, bits & ~TRANSIENT_BITS))


def stream(buf: WBuffer, classname: str, get: Getter) -> None:
    """One object of ``classname``, its members read by ``get``, as its streamer writes it."""
    _checksum, version, elements = INFOS[classname]
    at = buf.start(version)
    for kind, name, _title, stype, _s, _a, _d, _m, typename, _x in elements:
        if kind == "TStreamerBase":
            if name != "TObject":
                stream(buf, name, get)
            elif classname not in IGNORED:
                _tobject(buf)
        elif kind in ("TStreamerObject", "TStreamerObjectAny"):
            stream(buf, typename, get(name))
        else:
            buf.raw(pack_member(stype, _title, np.asarray([get(name)])))
    buf.end(at)


def named(classname: str, payload: bytes) -> bytes:
    """An object as a ``TBranchObject`` holds it: its class's name, NUL and all, first."""
    name = classname.encode()
    return bytes([len(name)]) + name + b"\0" + payload


def stream_clones(classname: str, objects: Sequence[Getter], bypass: bool) -> bytes:
    """A ``TClonesArray`` of ``objects`` of ``classname``, as ``TClonesArray::Streamer`` writes it."""
    _checksum, version, _elements = INFOS[classname]
    buf = WBuffer()
    at = buf.start(INFOS["TClonesArray"][1])
    _tobject(buf, BYPASS_STREAMER if bypass else 0)
    buf.string(f"{classname}s")  # fName: the class's name, made plural
    buf.string(f"{classname};{version}")
    buf.raw(struct.pack(">ii", len(objects), 0))  # how many, and the lower bound
    if bypass:
        _members(buf, classname, objects)
    else:
        for get in objects:
            buf.u8(1)  # this slot holds an object
            stream(buf, classname, get)
    buf.end(at)
    return bytes(buf.data)


def _members(buf: WBuffer, classname: str, objects: Sequence[Getter]) -> None:
    """``TStreamerInfo::WriteBufferClones``: each element of the class, for every object."""
    for kind, name, title, stype, _s, _a, _d, _m, _typename, _x in INFOS[classname][2]:
        if kind == "TStreamerBase" and name == "TObject":
            for _get in objects:
                _tobject(buf)
        elif kind == "TStreamerBase":
            _members(buf, name, objects)
        else:
            buf.raw(pack_member(stype, title, np.asarray([get(name) for get in objects])))
