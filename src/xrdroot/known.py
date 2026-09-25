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

__all__ = ["KNOWN", "read_recorder"]


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


#: The classes this module reads, against how each is read.
KNOWN: dict[str, Callable[[Any], Callable[[Buffer], Any]]] = {"TRecorder": read_recorder}
