"""RNTuples merged and copied: every input's entries, read and written again.

An RNTuple is merged the slow way only - its fields read a batch of entries
at a time and written through :class:`~xrdroot.WritableRNTuple` - since its
pages, unlike a tree's baskets, are listed in a page list that the writer
builds from what it wrote itself. The fields have to be the same, by name
and C++ type, in every input, and the ones the writer does not write -
records, nested collections - are refused by the writer by name.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..rntuple.writer import spec_of

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from ..rntuple import RNTuple, WritableRNTuple
    from ..writer import WritableDirectory

__all__ = ["NtupleMerge"]

#: How many entries an RNTuple is read at a time.
STEP = 100_000


class NtupleMerge:
    """One RNTuple being written from one or more read, each after the one before."""

    def __init__(self, directory: WritableDirectory, name: str, first: RNTuple) -> None:
        self.directory = directory
        self.name = name
        self.types = first.cxx_types()
        self.description = first.description
        self.written: WritableRNTuple | None = None

    def _declared(self, batch: Mapping[str, Any]) -> WritableRNTuple:
        if self.written is None:
            fields = {name: spec_of(name, values) for name, values in batch.items()}
            self.written = self.directory.rntuple(
                self.name, fields, description=self.description
            )
        return self.written

    def add(self, ntuple: RNTuple, where: str) -> None:
        """Take in one more RNTuple's entries, after those taken in before."""
        theirs = ntuple.cxx_types()
        if theirs != self.types:
            differ = sorted(
                name
                for name in set(theirs) | set(self.types)
                if theirs.get(name) != self.types.get(name)
            )
            raise ValueError(
                f"the RNTuple {where} does not have the fields of the first one merged into "
                f"it ({', '.join(differ)} differ); RNTuples are merged only when their "
                f"fields are the same"
            )
        names = list(self.types)
        for batch in ntuple.iterate(names, step=STEP):
            self._declared(batch).extend(batch)
        if self.written is None:
            self._declared(ntuple.arrays(names, 0, 0))
