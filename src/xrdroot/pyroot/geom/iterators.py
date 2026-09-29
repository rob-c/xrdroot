"""``TGeoIterator`` and ``TGeoIteratorPlugin``: the tree of placed volumes, node by node.

``TGeoIterator(top)`` visits every node below ``top`` depth first; at each
it knows its depth (``GetLevel``, 1 for a daughter of ``top``), the node
at every depth above it (``GetNode(level)``) and its path of node names
(``GetPath``, ``/TOP_1/A_1/B_2``). ``Skip`` passes over what is below the
node just visited. A plugin - a class deriving from ``TGeoIteratorPlugin``
- is handed the iterator in ``fIterator`` and has its ``ProcessNode``
called at each node the painter paints, as ROOT's is.
"""

from __future__ import annotations

from typing import Any

from ...geom import IDENTITY, Matrix
from ..core.objects import TObject
from .painting import walk

__all__ = ["TGeoIterator", "TGeoIteratorPlugin"]


def _put(path: Any, text: str) -> None:
    """Into a ``TString`` passed by reference, or a cell's ``value``."""
    if hasattr(path, "Clear") and hasattr(path, "Append"):
        path.Clear()
        path.Append(text)
    else:
        path.value = text


class _Position:
    """Where an iterator is: the node at each depth from the top down, and the top."""

    def __init__(self, top: Any) -> None:
        self.top = top
        self.path: tuple[Any, ...] = ()
        self.matrix: Matrix = IDENTITY

    def GetLevel(self) -> int:
        return len(self.path)

    def GetNode(self, level: int) -> Any:
        """The node at depth ``level`` on the way to this one (none at 0, the top)."""
        return self.path[int(level) - 1] if int(level) > 0 else None

    def GetPath(self, path: Any) -> None:
        top = f"/{self.top.GetName()}_1"
        _put(path, top + "".join(f"/{node.GetName()}" for node in self.path))

    def GetCurrentMatrix(self) -> Matrix:
        return self.matrix

    def GetTopVolume(self) -> Any:
        return self.top


class TGeoIterator(_Position, TObject):
    """``TGeoIterator(top)``: call it for the next node, ``None`` when there are no more."""

    def __init__(self, top: Any) -> None:
        _Position.__init__(self, top)
        TObject.__init__(self)
        self._walk = walk(top)
        self._skip: tuple[Any, ...] | None = None

    def __call__(self) -> Any:
        for node, matrix, _, path in self._walk:
            if self._skip is not None and path[: len(self._skip)] == self._skip:
                continue
            self._skip = None
            self.path, self.matrix = path, matrix
            return node
        return None

    Next = __call__

    def __iter__(self) -> Any:
        return iter(self, None)

    def Skip(self) -> None:
        """Pass over what is below the node just visited."""
        self._skip = self.path

    def Reset(self, top: Any = None) -> None:
        self.__init__(top if top is not None else self.top)  # type: ignore[misc]


class TGeoIteratorPlugin(TObject):
    """``TGeoIteratorPlugin``: derive from it, and ``ProcessNode`` sees each painted node."""

    def __init__(self) -> None:
        super().__init__()
        self.fIterator: Any = None

    def _visit(self, path: tuple[Any, ...], level: int) -> None:
        from .manager import current_manager

        position = _Position(current_manager().GetGeomPainter().top or path[0].GetMotherVolume())
        position.path = path
        self.fIterator = position

    def ProcessNode(self) -> None:
        """What to do at each node: nothing, until a derived class says."""

    def SetIterator(self, iterator: Any) -> None:
        self.fIterator = iterator
