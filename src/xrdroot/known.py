"""Classes a file holds without describing, read by the layout ROOT declares.

A file normally carries the streamer information of every class written into
it, and this reader goes by that. A few files do not: the recordings ROOT's
GUI tutorials replay hold a ``TRecorder`` and no description of it. Its
layout is short, fixed and public - ``TRecorder.h`` declares, for version 2,
a ``TObject`` base and a ``TString fFilename``, its state being transient -
so it is read by that declaration, and only at the version it was made for.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .buffer import Buffer
from .errors import UnsupportedFeatureError

__all__ = [
    "KNOWN",
    "STREAMS_ITSELF",
    "read_recorder",
    "read_tobject",
    "read_tprocessid",
    "read_tref",
    "read_trefarray",
]

#: ``TObject::kHasUUID``: the reference names its process by UUID rather than by number.
HAS_UUID = 1 << 5


def read_recorder(_described: Any) -> Callable[[Buffer], dict[str, Any]]:
    """How a ``TRecorder`` reads: its ``TObject``, then the file its events went to."""

    def read(buf: Buffer) -> dict[str, Any]:
        version, end = buf.header()
        if version != 2:
            raise UnsupportedFeatureError(
                f"this TRecorder is version {version}, and without the file's description "
                f"of it only the version 2 that TRecorder.h declares is known"
            )
        unique, bits = buf.tobject()
        row = {"TObject": {"fUniqueID": unique, "fBits": bits}, "fFilename": buf.string()}
        buf.resume(end)
        return row

    return read


def read_tobject(_described: Any) -> Callable[[Buffer], dict[str, Any]]:
    """How a bare ``TObject`` reads, which ROOT writes no description of: ``TObject::Streamer``,
    no byte count, and for one a ``TRef`` may point at, which ``ProcessID`` record numbered it."""

    def read(buf: Buffer) -> dict[str, Any]:
        unique, bits, process = buf.referenced()
        found: dict[str, Any] = {"fUniqueID": unique, "fBits": bits}
        if process is not None:
            found["pidf"] = process
        return found

    return read


def read_tprocessid(_described: Any) -> Callable[[Buffer], dict[str, Any]]:
    """How a ``TProcessID`` reads when its file does not describe it: a ``TNamed`` - the
    record's name, ``ProcessID0`` and on, and the process's UUID as its title."""

    def read(buf: Buffer) -> dict[str, Any]:
        _version, end = buf.header()
        name, title = buf.named()
        buf.resume(end)
        return {"TNamed": {"fName": name, "fTitle": title}}

    return read


def read_tref(_described: Any) -> Callable[[Buffer], dict[str, Any]]:
    """How a ``TRef`` reads: a ``TObject`` whose identifier is the referenced object's.

    ``TRef::Streamer`` writes its ``TObject`` with no byte count and then the
    process the object belongs to - its number, or for ``kHasUUID`` its UUID
    as a string. The object itself is whichever one in the same file carries
    that identifier; resolving it is left to whoever holds both.
    """

    def read(buf: Buffer) -> dict[str, Any]:
        unique, bits = buf.tobject()
        process = buf.string() if bits & HAS_UUID else buf.u16()
        return {"TObject": {"fUniqueID": unique, "fBits": bits}, "fPID": process}

    return read


def read_trefarray(_described: Any) -> Callable[[Buffer], dict[str, Any]]:
    """How a ``TRefArray`` reads: its ``TObject`` and name, how many slots and the lower bound,
    the process its objects belong to, and each object's identifier - ``TRefArray::Streamer``."""

    def read(buf: Buffer) -> dict[str, Any]:
        _version, end = buf.header()
        unique, bits = buf.tobject()
        name = buf.string()
        count, lower = buf.i32(), buf.i32()
        process = buf.u16()
        uids = [buf.u32() for _ in range(count)]
        buf.resume(end)
        row = {"TObject": {"fUniqueID": unique, "fBits": bits}, "fName": name}
        return {**row, "fLowerBound": lower, "fPID": process, "fUIDs": uids}

    return read


#: Classes read here even when the file describes them: ``TObject::Streamer`` writes, for an
#: object a ``TRef`` may point at, two bytes its description does not mention.
STREAMS_ITSELF = frozenset({"TObject"})

#: The classes this module reads, against how each is read.
KNOWN: dict[str, Callable[[Any], Callable[[Buffer], Any]]] = {
    "TRecorder": read_recorder,
    "TObject": read_tobject,
    "TProcessID": read_tprocessid,
    "TRef": read_tref,
    "TRefArray": read_trefarray,
}
