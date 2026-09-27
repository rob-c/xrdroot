"""``std::cout``: where RooFit's own printing goes, ordered as C++'s is against Python's.

Under PyROOT, what C++ prints goes straight to the process's standard output,
while a script's own ``print`` waits in Python's buffer - so when the output
is a pipe, as the tutorials' is, every line RooFit prints comes before the
script's own lines that were printed before it, until Python's buffer fills.
To print as ROOT does, RooFit's lines here are written the same way: to the
file descriptor, past Python's buffer, when a PyROOT script is running
(``import ROOT`` is :mod:`xrdroot.pyroot`) and standard output is the
process's own and not a terminal. Everywhere else - a translated macro,
whose ``cout`` is Python's stream too, or a test capturing output - they go
to ``sys.stdout`` like anything else.
"""

from __future__ import annotations

import os
import sys

__all__ = ["line", "write"]


def _direct() -> bool:
    root = sys.modules.get("ROOT")
    if root is None or getattr(root, "__name__", "") != "xrdroot.pyroot":
        return False
    stream = sys.stdout
    return stream is sys.__stdout__ and stream is not None and not stream.isatty()


def write(text: str) -> None:
    """Print ``text`` as C++'s ``std::cout`` would."""
    if _direct():
        os.write(1, text.encode())
        return
    sys.stdout.write(text)
    sys.stdout.flush()


def line(text: str = "") -> None:
    write(text + "\n")


class _Stream:
    """``std::cout`` as a stream object, for what takes one: a message service's stream."""

    def write(self, text: str) -> None:
        write(text)

    def flush(self) -> None:
        """Nothing is kept back to flush: each write goes out whole."""


#: ``std::cout``.
STREAM = _Stream()
