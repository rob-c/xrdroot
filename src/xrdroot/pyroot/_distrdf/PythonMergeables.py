"""The partial results distributed tasks send back, and how two of them are added up.

A task's result is a :class:`MergeableValue` - the partial xrdroot's frame
left of it (:class:`xrdroot.rdf.distributed.Partial`) and how to present the
sum to a PyROOT script - except a ``Snapshot``'s, which is the
:class:`SnapshotResult` of the file the task wrote. The reducer adds them up
pairwise with ``Merge``; ``GetValue`` gives the result.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .Backends.Base import BaseBackend

__all__ = ["MergeableValue", "SnapshotResult"]


class MergeableValue:
    """``RMergeableValue``: a result's partial, and how its value is handed to the user."""

    def __init__(self, partial: Any, present: Callable[[Any], Any]) -> None:
        self.partial = partial
        self.present = present

    def __repr__(self) -> str:
        return f"<MergeableValue {self.partial!r}>"

    def Merge(self, other: MergeableValue) -> None:
        self.partial.Merge(other.partial)

    def GetValue(self) -> Any:
        return self.present(self.partial.GetValue())


class SnapshotResult:
    """A distributed ``Snapshot``: the tree's name and the file each task wrote it into."""

    def __init__(self, treename: str, filenames: list[str], resultptr: Any = None) -> None:
        self.treename = treename
        self.filenames = filenames
        #: The task's booked ``Snapshot``; it stays in the task.
        self._resultptr = resultptr

    def __repr__(self) -> str:
        return f"<SnapshotResult {self.treename!r} in {self.filenames}>"

    def Merge(self, other: SnapshotResult) -> None:
        self.filenames.extend(other.filenames)

    def GetValue(self, backend: BaseBackend) -> Any:
        """A distributed frame of the files written, on the same backend, as local
        ``Snapshot`` gives a frame of the file it wrote."""
        from ..trees import TChain

        chain = TChain(self.treename)
        for filename in self.filenames:
            chain.Add(filename)
        return backend.make_dataframe(chain)
