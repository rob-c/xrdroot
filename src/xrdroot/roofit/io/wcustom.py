"""Hand-written RooFit streamers preserve errors, binning and deferred proxy identifiers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .stream import Streamed

if TYPE_CHECKING:
    from .wstream import Writer


def code_repo(writer: Writer, value: Streamed) -> None:
    """Python workspaces contain no C++ source files, class relations or extra headers."""
    writer.buf.raw(bytes(12))


def real_var(writer: Writer, value: Streamed) -> None:
    writer.object(value.m["RooAbsRealLValue"], "RooAbsRealLValue")
    for name in ("_error", "_asymErrLo", "_asymErrHi"):
        writer.number("d", value.get(name))
    writer.pointer(value.get("_binning"))
    writer.object(value.get("_sharedProp"), "RooRealVarSharedProperties")


def ref_array(writer: Writer, value: Streamed) -> None:
    at = writer.buf.start(1)
    writer.tobject(value.get("TObject"))
    writer.buf.string(value.get("fName"))
    uids = value.get("uids")
    writer.buf.i32(len(uids))
    writer.buf.i32(value.get("fLowerBound"))
    writer.buf.u16(value.get("pid"))
    for uid in uids:
        writer.buf.u32(uid)
    writer.buf.end(at)


def linked(writer: Writer, value: Streamed) -> None:
    """RooLinkedList writes a bare version without a surrounding byte count."""
    writer.buf.u16(3)
    writer.tobject(value.get("TObject"))
    items = value.get("items")
    writer.buf.i32(len(items))
    for item in items:
        writer.pointer(item)
    writer.buf.string(value.get("_name"))


#: Streamers whose on-file representation differs from their declared data members.
CUSTOM: dict[str, Callable[[Writer, Streamed], None]] = {
    "RooWorkspace::CodeRepo": code_repo,
    "RooRealVar": real_var,
    "RooRefArray": ref_array,
}
