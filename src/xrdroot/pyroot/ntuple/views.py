"""A collection's views: its size per entry, the range of its items, and the items by number.

``reader.GetCollectionView("v")`` is a vector field seen as a collection:
``view(i)`` is entry ``i``'s size, ``GetCollectionRange(i)`` the numbers
of its items, and ``GetView<T>("_0")`` the items themselves, read by
those numbers - how a program walks a vector too large to load whole.
The numbers run on over the whole RNTuple, where ROOT's count within a
cluster; a program that only hands them from a range to a view sees no
difference.
"""

from __future__ import annotations

from typing import Any

from .fields import typed

__all__ = [
    "RNTupleReadOptions",
    "RNTupleCollectionView",
    "RNTupleLocalRange",
    "kInvalidDescriptorId",
]

#: ``ROOT::kInvalidDescriptorId``: the number no cluster, field or column has.
kInvalidDescriptorId = 2**64 - 1


class _ClusterCache:
    kOff = 0
    kOn = 1
    kDefault = 1


class RNTupleReadOptions:
    """``RNTupleReadOptions``: how pages are cached - which a reading here does a field at a
    time, the first time it is asked for, whatever is chosen."""

    EClusterCache = _ClusterCache

    def __init__(self) -> None:
        self._cache = _ClusterCache.kDefault

    def SetClusterCache(self, cache: Any) -> None:
        self._cache = int(cache)

    def GetClusterCache(self) -> int:
        return self._cache


class _RangeIterator:
    """``RNTupleLocalRange::RIterator``: an item's number, stepped on by ``++``."""

    def __init__(self, at: Any) -> None:
        self._at = int(at)

    def __iadd__(self, step: int) -> _RangeIterator:
        self._at += int(step)
        return self

    def __index__(self) -> int:
        return self._at

    __int__ = __index__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _RangeIterator) and other._at == self._at

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = None  # type: ignore[assignment]


class RNTupleLocalRange:
    """``RNTupleLocalRange``: the numbers of one entry's items, ``begin()`` to ``end()``."""

    RIterator = _RangeIterator

    def __init__(self, cluster: Any, start: Any, stop: Any) -> None:
        self._start, self._stop = int(start), int(stop)

    def begin(self) -> _RangeIterator:
        return _RangeIterator(self._start)

    def end(self) -> _RangeIterator:
        return _RangeIterator(self._stop)

    def __iter__(self) -> Any:
        return iter(range(self._start, self._stop))

    def size(self) -> int:
        return self._stop - self._start


class RNTupleCollectionView:
    """A vector field as a collection: its sizes by entry, and views of its items."""

    def __init__(self, reader: Any, name: str | None = None) -> None:
        """A collection's view, or a copy of one: ``RNTupleCollectionView(std::move(view))``."""
        if isinstance(reader, RNTupleCollectionView):
            reader, name = reader.reader, reader.name
        self.reader: Any = reader
        self.name = str(name)

    def __call__(self, entry: Any) -> int:
        """The entry's size: how many items its vector holds."""
        offsets = self.reader.column(self.name).offsets
        return int(offsets[int(entry) + 1] - offsets[int(entry)])

    def GetCollectionRange(self, entry: Any) -> RNTupleLocalRange:
        offsets = self.reader.column(self.name).offsets
        start, stop = offsets[int(entry)], offsets[int(entry) + 1]
        return RNTupleLocalRange(kInvalidDescriptorId, start, stop)

    @typed
    def GetView(self, kind: Any, name: Any) -> Any:
        """``GetView<T>("_0")``: the items, read by their numbers."""
        from .reading import RNTupleView

        return RNTupleView(self.reader, f"{self.name}.{name}")
