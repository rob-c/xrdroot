"""RNTuple, ROOT 7's columnar successor to the ``TTree``: read, and written.

An RNTuple keeps a column of plain values for every leaf of its schema, in
compressed pages grouped into clusters of entries, and describes itself in
little-endian envelopes of its own rather than in ROOT's streamer format. The
only part of it a ROOT file's directory lists is the anchor, a
``ROOT::RNTuple`` key saying where the rest is - which is why
``f["Events"]`` on one gives back an :class:`RNTuple` rather than a
``TTree``, with the same ways in: ``len``, ``keys``, a field by name,
``arrays`` and ``iterate``.

:meth:`~xrdroot.WritableFile.rntuple` writes one, from numbers, booleans,
strings and vectors of numbers.
"""

from __future__ import annotations

from .reader import RField, RNTuple
from .writer import WritableRNTuple

__all__ = ["RNTuple", "RField", "WritableRNTuple"]
