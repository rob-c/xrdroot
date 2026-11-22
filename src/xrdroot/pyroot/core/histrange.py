"""``THistRange``: the global bin numbers a histogram's current range covers, one at a time.

ROOT's ``THistRange`` holds a ``TBinIterator`` pair, and ``for (auto &bin : r)`` walks the
bins a histogram is *drawn* over - the axes' current ranges, which ``SetRange`` narrows.
Nothing is copied out of the histogram: the iterator carries ROOT's own global bin number,
x fastest, ``binx + nx * (biny + ny * binz)`` with each ``n`` an axis size *including* its
two flow bins, because that is the number ``GetBinContent`` answers. The four range kinds
are ``TBinIterator``'s: the axis ranges, the bins within the axis limits, every bin with
the flow bins, and the flow bins themselves - which ROOT's own code visits exactly like
every bin, so this does too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["THistRange", "TBinIterator"]

#: ``TH2Poly``'s flow bins are numbered from the negative end of its one-dimensional range,
#: as ROOT's ``TBinIterator`` takes them for a ``TH2Poly`` or a ``TProfile2Poly``.
POLY_UNDERFLOW = -9
#: The bin number a walk stops at. ROOT compares against exactly this - which is why its
#: ``kUnOfBins`` never visits a poly's last flow bin, whose number *is* -1, and this does not
#: visit it either: a range that stops short as ROOT's does is the range ROOT has.
END = -1


@dataclass(frozen=True)
class _Bounds:
    """One range: the axis sizes ROOT numbers a global bin by, and the bins to visit."""

    dim: int
    nx: int
    ny: int
    nz: int
    xmin: int
    xmax: int
    ymin: int
    ymax: int
    zmin: int
    zmax: int


def _span(axis: Any, kind: int) -> tuple[int, int]:
    """The first and last bin of ``axis`` that ``kind`` visits."""
    if kind == TBinIterator.kHistRange:
        return int(axis.GetFirst()), int(axis.GetLast())
    if kind == TBinIterator.kAxisBins:
        return 1, int(axis.GetNbins())
    return 0, int(axis.GetNbins()) + 1


def _poly_bounds(count: int, kind: int) -> _Bounds:
    """The range of a ``TH2Poly``, whose bins are one dimension however many axes it draws."""
    if kind == TBinIterator.kAllBins:
        first, last = POLY_UNDERFLOW, count
    elif kind == TBinIterator.kUnOfBins:
        first, last = POLY_UNDERFLOW, -1
    else:
        first, last = 1, count
    return _Bounds(1, 0, 0, 0, first, last, 0, 0, 0, 0)


def _bounds(hist: Any, kind: int) -> _Bounds:
    """What ``kind`` covers of ``hist``, worked out once however long the walk is."""
    counted = getattr(hist, "GetNumberOfBins", None)
    if callable(counted):
        return _poly_bounds(int(counted()), kind)
    xmin, xmax = _span(hist.GetXaxis(), kind)
    ymin, ymax = _span(hist.GetYaxis(), kind)
    zmin, zmax = _span(hist.GetZaxis(), kind)
    return _Bounds(
        int(hist.GetDimension()),
        int(hist.GetNbinsX()) + 2,
        int(hist.GetNbinsY()) + 2,
        int(hist.GetNbinsZ()) + 2,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
    )


class TBinIterator:
    """``TBinIterator``: a histogram's bins, one global bin number at a time.

    Its four range kinds are ROOT's own spellings, ``kHistRange`` being the default and the
    reason the class exists: a user range visits the bins inside it and no flow bin.
    """

    #: ``TBinIterator::ERangeType``, in ROOT's declaration order, unscoped as ROOT has it.
    kHistRange = 0
    kAxisBins = 1
    kAllBins = 2
    kUnOfBins = 3

    def __init__(self, hist: Any, range_type: int = kHistRange) -> None:
        self._bounds = _bounds(hist, range_type)
        self._x = self._bounds.xmin
        self._y = self._bounds.ymin
        self._z = self._bounds.zmin
        self._bin = self._global()

    @classmethod
    def End(cls) -> TBinIterator:
        """``TBinIterator::End()``: the bin number a walk stops at, ``-1`` and no bins."""
        made = cls.__new__(cls)
        made._bounds = _Bounds(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        made._x = made._y = made._z = 0
        made._bin = END
        return made

    def __iter__(self) -> TBinIterator:
        return self

    def __next__(self) -> int:
        if self._bin == END:
            raise StopIteration
        visited = self._bin
        self._advance()
        return visited

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TBinIterator) and self._bin == other._bin

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __repr__(self) -> str:
        return f"TBinIterator({self._bin})"

    def _global(self) -> int:
        """ROOT's global bin number of the bin the iterator is on, x running fastest."""
        found = self._bounds
        if found.dim == 1:
            return self._x
        if found.dim == 2:
            return self._x + found.nx * self._y
        return self._x + found.nx * (self._y + found.ny * self._z)

    def _advance(self) -> None:
        """``++it``: the next bin, or the end sentinel once the last one has been visited."""
        found = self._bounds
        if self._x < found.xmax:
            self._x += 1
        elif not self._stepped_out(found):
            self._bin = END
            return
        if self._x == 0:
            self._x = 1  # only a poly's range starts below 0, and bin 0 is not a bin
        self._bin = self._global()

    def _stepped_out(self, found: _Bounds) -> bool:
        """Carry one bin into the next axis; false when there is no next axis to carry into."""
        if found.dim < 2:
            return False
        self._x = found.xmin
        if self._y < found.ymax:
            self._y += 1
            return True
        if found.dim < 3 or self._z >= found.zmax:
            return False
        self._y = found.ymin
        self._z += 1
        return True


class THistRange:
    """``THistRange``: a histogram's bins as a range, which a ``for`` loop walks.

    ROOT's ``TH1`` gives no iterator of its own; this is the one a macro builds by hand, and
    the range is the histogram's, not a copy: ``SetRange`` on an axis before the loop narrows
    what it visits.
    """

    def __init__(self, hist: Any, range_type: int = TBinIterator.kHistRange) -> None:
        self._hist = hist
        self._range_type = range_type

    def begin(self) -> TBinIterator:
        return TBinIterator(self._hist, self._range_type)

    def end(self) -> TBinIterator:
        return TBinIterator.End()

    def __iter__(self) -> TBinIterator:
        return self.begin()
