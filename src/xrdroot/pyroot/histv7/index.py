"""``RBinIndex`` and its ranges: how ROOT 7's histograms name a bin, and walk a run of them.

A bin index is a normal bin's number from 0, or one of three special
values - the underflow, the overflow and no bin at all - kept, as
``ROOT::Experimental::RBinIndex`` keeps them, as the top three values of a
64-bit unsigned integer, so that arithmetic that leaves the normal bins
lands on the invalid index as C++'s does. A range is ``[begin, end)`` of
them, with the number of normal bins after which a walk steps to the
overflow; a multidimensional range walks several at once, the last
fastest, which is the order of the bins' global indices.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from typing import Any

__all__ = ["RBinIndex", "RBinIndexMultiDimRange", "RBinIndexRange", "RLinearizedIndex",
           "CreateBinIndexRange", "as_index", "index_vector"]  # fmt: skip

#: ``std::uint64_t``'s range, which an index wraps around as C++'s does.
WORD = 1 << 64
#: The special values, as ``RBinIndex`` keeps them: ``static_cast<std::uint64_t>(-3)`` and on.
UNDERFLOW, OVERFLOW, INVALID = WORD - 3, WORD - 2, WORD - 1


class RBinIndex:
    """``RBinIndex``: a normal bin's number, the underflow, the overflow, or no bin."""

    __slots__ = ("fIndex",)

    def __init__(self, index: Any = None) -> None:
        if isinstance(index, RBinIndex):
            self.fIndex: int = index.fIndex
        else:  # a negative number, which C++ asserts against, wraps as its cast does
            self.fIndex = INVALID if index is None else int(index) % WORD

    @staticmethod
    def Underflow() -> RBinIndex:
        return _special(UNDERFLOW)

    @staticmethod
    def Overflow() -> RBinIndex:
        return _special(OVERFLOW)

    def GetIndex(self) -> int:
        return self.fIndex

    def IsNormal(self) -> bool:
        return self.fIndex < UNDERFLOW

    def IsUnderflow(self) -> bool:
        return self.fIndex == UNDERFLOW

    def IsOverflow(self) -> bool:
        return self.fIndex == OVERFLOW

    def IsInvalid(self) -> bool:
        return self.fIndex == INVALID

    def __add__(self, step: Any) -> RBinIndex:
        """``index + a``: a special value, or a sum past the normal bins, is the invalid index.

        A negative step is the translator's ``--index``, written as ``index + -1``.
        """
        count = int(step)
        if count < 0:
            return self - (-count)
        moved = self.fIndex + count
        return RBinIndex(moved if self.IsNormal() and moved < UNDERFLOW else None)

    def __sub__(self, step: Any) -> RBinIndex:
        count = int(step)
        if count < 0:
            return self + (-count)
        return RBinIndex(self.fIndex - count if self.IsNormal() and self.fIndex >= count else None)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RBinIndex) and self.fIndex == other.fIndex

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __lt__(self, other: RBinIndex) -> bool:
        return self.IsNormal() and other.IsNormal() and self.fIndex < other.fIndex

    def __gt__(self, other: RBinIndex) -> bool:
        return self.IsNormal() and other.IsNormal() and self.fIndex > other.fIndex

    def __le__(self, other: RBinIndex) -> bool:
        return self == other or self < other

    def __ge__(self, other: RBinIndex) -> bool:
        return self == other or self > other

    def __hash__(self) -> int:
        return hash(self.fIndex)

    def __repr__(self) -> str:
        names = {UNDERFLOW: "Underflow", OVERFLOW: "Overflow", INVALID: "Invalid"}
        return f"RBinIndex({names.get(self.fIndex, self.fIndex)})"


def _special(value: int) -> RBinIndex:
    made = RBinIndex()
    made.fIndex = value
    return made


def as_index(given: Any) -> RBinIndex:
    """A bin given as an ``RBinIndex``, or as a normal bin's number - or, for an element of a
    ``std::vector<RBinIndex>(n)`` the STL stand-in left empty, the invalid index."""
    return given if isinstance(given, RBinIndex) else RBinIndex(given)


class RLinearizedIndex:
    """``RLinearizedIndex``: where a bin sits among an axis's (or the histogram's) bins, and
    whether there is such a bin."""

    __slots__ = ("fIndex", "fValid")

    def __init__(self, fIndex: int = 0, fValid: bool = False) -> None:
        self.fIndex, self.fValid = int(fIndex), bool(fValid)

    def __repr__(self) -> str:
        return f"RLinearizedIndex({self.fIndex}, {self.fValid})"


class RBinIndexRange:
    """``RBinIndexRange``: the bins ``[begin, end)``, stepping to the overflow after the last
    of ``nNormalBins`` normal bins. Made by an axis, or ``Internal::CreateBinIndexRange``."""

    __slots__ = ("_begin", "_end", "_normal")

    def __init__(self) -> None:
        self._begin, self._end, self._normal = RBinIndex(), RBinIndex(), 0

    def GetBegin(self) -> RBinIndex:
        return self._begin

    def GetEnd(self) -> RBinIndex:
        return self._end

    def IsInvalid(self) -> bool:
        """An invalid end is the end of a full range; only an invalid begin is no range."""
        return self._begin.IsInvalid()

    def begin(self) -> RangeIterator:
        return RangeIterator(self._begin, self._normal)

    def end(self) -> RangeIterator:
        return RangeIterator(self._end, self._normal)

    def __iter__(self) -> Iterator[RBinIndex]:
        at, end = self.begin(), self._end
        while at.fIndex != end:
            yield at.fIndex
            at._preinc()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RBinIndexRange) and (self._begin, self._end, self._normal) == (
            other._begin, other._end, other._normal)  # fmt: skip

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"RBinIndexRange({self._begin}, {self._end})"


def CreateBinIndexRange(begin: Any, end: Any, nNormalBins: Any) -> RBinIndexRange:
    """``Internal::CreateBinIndexRange``: the range ``[begin, end)`` of ``nNormalBins`` bins."""
    made = RBinIndexRange()
    made._begin, made._end, made._normal = as_index(begin), as_index(end), int(nNormalBins)
    return made


class _Iterator:
    """What an iterator is to the translated code: ``*it`` reads ``value``, ``++it`` is
    ``_preinc`` (or ``it += 1``), and ``it->member`` is the member of what it points at."""

    #: ``*it`` reads ``value``, as the translator reads through a cell.
    _cint_cell = True

    @property
    def value(self) -> Any:
        return self._deref()

    def _deref(self) -> Any:
        raise NotImplementedError

    def _preinc(self) -> None:
        raise NotImplementedError

    def __copy__(self) -> Any:
        raise NotImplementedError

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        """``auto it = other;``: a copy stepping on its own over the same range, not a copy of
        the range."""
        return self.__copy__()

    def __iadd__(self, steps: Any) -> Any:
        for _ in range(int(steps)):
            self._preinc()
        return self

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = None  # type: ignore[assignment]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._deref(), name)


class RangeIterator(_Iterator):
    """``RBinIndexRange::RIterator``: the underflow, then the normal bins, then the overflow."""

    def __init__(self, index: RBinIndex, normal: int) -> None:
        self.fIndex, self._normal = index, normal

    def _deref(self) -> RBinIndex:
        return self.fIndex

    def __copy__(self) -> RangeIterator:
        return RangeIterator(self.fIndex, self._normal)

    def _preinc(self) -> None:
        index = self.fIndex
        if index.IsUnderflow():
            self.fIndex = RBinIndex(0)
        elif index.IsOverflow():
            self.fIndex = RBinIndex()
        elif not index.IsInvalid():  # an invalid iterator stays where it is, as ROOT's does
            index = index + 1
            self.fIndex = RBinIndex.Overflow() if index.fIndex == self._normal else index

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RangeIterator) and (self.fIndex, self._normal) == (
            other.fIndex, other._normal)  # fmt: skip


def index_vector(indices: Iterable[RBinIndex]) -> Any:
    """A ``std::vector<RBinIndex>`` of ``indices``, as ROOT's ranges hand them out."""
    from ...cppstd.containers import std

    return std.vector["ROOT::Experimental::RBinIndex"](list(indices))


class RBinIndexMultiDimRange:
    """``RBinIndexMultiDimRange``: every combination of the bins of several ranges, the last
    range's bins fastest."""

    def __init__(self, ranges: Iterable[RBinIndexRange] = ()) -> None:
        self._ranges = list(ranges)
        self._empty = any(range_.GetBegin() == range_.GetEnd() for range_ in self._ranges)

    def GetRanges(self) -> Any:
        from ...cppstd.containers import std

        return std.vector["ROOT::Experimental::RBinIndexRange"](list(self._ranges))

    def begin(self) -> MultiDimIterator:
        made = MultiDimIterator(self)
        if not self._empty:
            made._iterators = [range_.begin() for range_ in self._ranges]
            made._indices = [it.fIndex for it in made._iterators]
        return made

    def end(self) -> MultiDimIterator:
        return MultiDimIterator(self)

    def combinations(self) -> Iterator[tuple[RBinIndex, ...]]:
        """Each combination of the bins, as a tuple: the walk the iterators make, quickly."""
        if not self._ranges:
            return iter(())
        return itertools.product(*(list(range_) for range_ in self._ranges))

    def __iter__(self) -> Iterator[Any]:
        return (index_vector(each) for each in self.combinations())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RBinIndexMultiDimRange) and self._ranges == other._ranges

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = None  # type: ignore[assignment]


class MultiDimIterator(_Iterator):
    """``RBinIndexMultiDimRange::RIterator``: ``*it`` is the vector of the current indices."""

    def __init__(self, owner: RBinIndexMultiDimRange) -> None:
        self._owner = owner
        self._iterators: list[RangeIterator] = []
        self._indices: list[RBinIndex] = []

    def _deref(self) -> Any:
        return index_vector(self._indices)

    def __copy__(self) -> MultiDimIterator:
        """A copy that steps on its own: ``auto it = other;``, and ``std::distance``'s walk."""
        made = MultiDimIterator(self._owner)
        made._iterators = [it.__copy__() for it in self._iterators]
        made._indices = list(self._indices)
        return made

    def _preinc(self) -> None:
        """Advance the last iterator, carrying into the ones before it as an odometer does."""
        ranges, count = self._owner._ranges, len(self._iterators)
        carried = 0
        for at in reversed(range(count)):
            self._iterators[at]._preinc()
            self._indices[at] = self._iterators[at].fIndex
            if self._indices[at] != ranges[at].GetEnd():
                break
            carried += 1
        if carried == count:  # every iterator at its end: the end iterator, which has none
            self._iterators, self._indices = [], []
            return
        for at in range(count - carried, count):
            self._iterators[at] = ranges[at].begin()
            self._indices[at] = self._iterators[at].fIndex

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, MultiDimIterator) and self._owner is other._owner
                and self._iterators == other._iterators)  # fmt: skip
