"""The one class of a canvas that writes itself by hand: ``TCanvas``.

Nearly everything a canvas holds is written by the file's own description of
its class - the pad under it, the pads inside it, and what they draw. The
canvas itself is not: ``TCanvas::Streamer`` writes its pad, then a run of
sizes and flags in an order of its own, and ROOT puts no description of it
in the file at all. This is that order, as ROOT's source writes it, for the
versions ROOT has written it in.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..buffer import Buffer

__all__ = ["STREAMED", "read_canvas"]

#: How one class the file does describe reads, handed in by the reader that
#: knows the file: the pad under a canvas and its attributes.
Described = Callable[[str], Callable[[Buffer], Any]]

#: The sizes the canvas keeps as ``float``, in the order they are written.
SIZES = ("fXsizeUser", "fYsizeUser", "fXsizeReal", "fYsizeReal")
#: The flags a version 2 canvas adds at the end, and a version 4 one between.
LATER_FLAGS = (("kShowEventStatus", 2), ("kAutoExec", 4), ("kMenuBar", 2))


def _window(buf: Buffer, version: int, row: dict[str, Any]) -> None:
    """Where the window was and how big it and the canvas were, in pixels.

    A canvas of version 2 or older wrote no window size, which was then the
    size of the canvas itself.
    """
    row["fWindowTopX"], row["fWindowTopY"] = buf.i32(), buf.i32()
    if version > 2:
        row["fWindowWidth"], row["fWindowHeight"] = buf.u32(), buf.u32()
    row["fCw"], row["fCh"] = buf.u32(), buf.u32()
    if version <= 2:
        row["fWindowWidth"], row["fWindowHeight"] = row["fCw"], row["fCh"]


def _flags(buf: Buffer, version: int, row: dict[str, Any]) -> None:
    """The canvas's bits of behaviour, the later ones as later versions added them."""
    row["kMoveOpaque"], row["kResizeOpaque"] = buf.bool(), buf.bool()
    row["fHighLightColor"] = buf.i16()
    row["fBatch"] = buf.bool()
    for name, since in LATER_FLAGS:
        if version >= since:
            row[name] = buf.bool()


def read_canvas(described: Described) -> Callable[[Buffer], dict[str, Any]]:
    """How a ``TCanvas`` reads: ``TCanvas::Streamer``, a member at a time."""
    pad, attributes = described("TPad"), described("TAttCanvas")

    def read(buf: Buffer) -> dict[str, Any]:
        version, end = buf.header()
        row: dict[str, Any] = {"TPad": pad(buf), "fDISPLAY": buf.string()}
        row["fDoubleBuffer"], row["fRetained"] = buf.i32(), buf.bool()
        row.update((name, buf.f32()) for name in SIZES)
        _window(buf, version, row)
        row["fCatt"] = attributes(buf)
        _flags(buf, version, row)
        buf.resume(end)
        return row

    return read


#: The classes that stream themselves by hand, against how each is read.
STREAMED: dict[str, Callable[[Described], Callable[[Buffer], Any]]] = {"TCanvas": read_canvas}
