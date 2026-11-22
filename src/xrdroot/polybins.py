"""``TH2Poly``'s bins: polygons of the plane, filled by which one a point falls in.

A bin is one polygon, or several (a country and its islands); a point is
in it by ``TMath::IsInside``'s crossing count, and a fill goes to the first
bin - in the order they were added - that holds it. Outside the axes'
range a fill goes to one of the eight regions round it, and inside the
range but in no bin to the ninth, ``fOverflow[4]``, as ROOT keeps them. A
histogram with no range given (``TH2Poly()``) widens its axes to hold each
bin as it is added - from a range of ``0..0``, so its axes always hold the
origin, as ROOT's do.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["PolyBins", "inside"]

#: The regions round a histogram's range, and the one within it outside every bin.
REGIONS = 9


def inside(xs: Any, ys: Any, x: float, y: float) -> bool:
    """``TMath::IsInside``: whether ``(x, y)`` is inside the polygon, by crossings."""
    odd = False
    j = len(xs) - 1
    for i in range(len(xs)):
        if (ys[i] < y <= ys[j]) or (ys[j] < y <= ys[i]):
            if xs[i] + (y - ys[i]) / (ys[j] - ys[i]) * (xs[j] - xs[i]) < x:
                odd = not odd
        j = i
    return odd


def _cell(value: float, origin: float, span: float, steps: int) -> int:
    """Which cell of the search grid along one axis ``value`` falls in, clamped to it."""
    return min(max(int((value - origin) / span * steps), 0), steps - 1)


class PolyBins:
    """The bins of a ``TH2Poly``, their contents, and the fills that missed them."""

    def __init__(self, xrange: Sequence[float] = (0.0, 0.0), yrange: Sequence[float] = (0.0, 0.0),
                 floating: bool = True) -> None:  # fmt: skip
        self.xrange, self.yrange = [float(v) for v in xrange], [float(v) for v in yrange]
        self.floating = floating
        self.polygons: list[list[tuple[list[float], list[float]]]] = []
        self.boxes = np.zeros((0, 4))
        self.contents = np.zeros(0)
        self.overflow = np.zeros(REGIONS)
        #: ``fTsumw, fTsumw2, fTsumwx, fTsumwx2, fTsumwy, fTsumwy2``, the entries, then
        #: ``fTsumwxy, fTsumwz, fTsumwz2`` - the moments ``GetStats`` reports, kept apart
        #: from the count of fills because a fill into a region adds to neither. Plain
        #: floats, not an array: a fill adds to them seven to ten times, and NumPy's
        #: scalar arithmetic is five times what a list slot costs.
        self.sums = [0.0] * 10
        #: ``fSumw2``: the sum of squared weights per bin and per region, laid out as ROOT
        #: lays it - the nine regions, then the bins - or ``None`` while no weight other
        #: than one has been filled, when an error is the root of the content, as ROOT's is.
        self.squares: np.ndarray[Any, Any] | None = None
        #: Which bins each cell of the search grid holds, built by :meth:`grid`.
        self._table: dict[tuple[int, int], tuple[int, ...]] | None = None

    def copy(self) -> PolyBins:
        """A copy of these bins, deep enough that neither histogram can move the other's."""
        made = PolyBins(self.xrange, self.yrange, self.floating)
        made.polygons = [[(list(x), list(y)) for x, y in polygons] for polygons in self.polygons]
        made.boxes, made.contents = self.boxes.copy(), self.contents.copy()
        made.overflow, made.sums = self.overflow.copy(), list(self.sums)
        made.squares = None if self.squares is None else self.squares.copy()
        return made

    def add(self, polygons: Sequence[tuple[Any, Any]]) -> int:
        """A bin of these polygons - each ``(xs, ys)`` - and its number, from 1."""
        made = [([float(v) for v in x], [float(v) for v in y]) for x, y in polygons]
        xs = np.concatenate([np.asarray(x) for x, _ in made])
        ys = np.concatenate([np.asarray(y) for _, y in made])
        box = [xs.min(), xs.max(), ys.min(), ys.max()]
        if self.floating:
            self.range_of(box)
        self.polygons.append(made)
        self.boxes = np.vstack([self.boxes, box])
        self.contents = np.append(self.contents, 0.0)
        if self.squares is not None:
            self.squares = np.append(self.squares, 0.0)
        self._table = None
        return len(self.polygons)

    def range_of(self, box: Sequence[float]) -> None:
        """Widen the axes - a histogram given no range of its own - to hold this bin."""
        self.xrange = [min(self.xrange[0], box[0]), max(self.xrange[1], box[1])]
        self.yrange = [min(self.yrange[0], box[2]), max(self.yrange[1], box[3])]

    def slot(self, number: int) -> int:
        """Where a bin - or a region, numbered -1 to -9 - sits in :attr:`squares`."""
        return -number - 1 if number < 0 else REGIONS + number - 1

    def sumw2(self) -> np.ndarray[Any, Any]:
        """``TH1::Sumw2``: keep the squared weights, seeded with the errors so far.

        ROOT seeds them from the bin contents - before this the error of a bin is the
        root of its content - and only once something has been filled, which is why a
        histogram whose contents were all set by hand starts from zero.
        """
        if self.squares is None:
            squares = np.zeros(REGIONS + len(self.contents))
            if self.sums[6] > 0:
                squares[:REGIONS], squares[REGIONS:] = np.abs(self.overflow), np.abs(self.contents)
            self.squares = squares
        return self.squares

    def error(self, number: int) -> float:
        """``GetBinError``: the root of the squared weights kept, or of the content."""
        if self.squares is not None:
            return math.sqrt(float(self.squares[self.slot(number)]))
        return math.sqrt(abs(self.content(number)))

    def content(self, number: int) -> float:
        """A bin's content by number from 1, or a region's, numbered -1 to -9."""
        if number < 0:
            return float(self.overflow[-number - 1]) if -number <= REGIONS else 0.0
        return float(self.contents[number - 1]) if 0 < number <= len(self.contents) else 0.0

    def region(self, x: float, y: float) -> int:
        """Which region round the range ``(x, y)`` is in: -1..-9, -5 within the range."""
        row = -1 if y > self.yrange[1] else -4 if y > self.yrange[0] else -7
        return row + (-2 if x > self.xrange[1] else -1 if x > self.xrange[0] else 0)

    def find(self, x: float, y: float) -> int:
        """The bin holding ``(x, y)``, or the region it fell in, as ``TH2Poly::Fill`` finds."""
        region = self.region(x, y)
        if region != -5:
            return region
        table = self._table if self._table is not None else self.grid()
        for index in table.get(self._key(x, y), ()):
            for px, py in self.polygons[index]:
                if inside(px, py, x, y):
                    return index + 1
        return -5

    def grid(self) -> dict[tuple[int, int], tuple[int, ...]]:
        """Which bins a point in each cell might be in, built once and kept.

        A fill asks which bin holds a point, and a program asks it a million times
        over; testing every bin's box each time is ten times the cost of looking the
        cell up, so the bins are dealt into a grid about one bin to a cell.
        """
        if self._table is None:
            side = max(1, math.isqrt(max(1, len(self.polygons))))
            #: ``x0, y0, 1/stepx, 1/stepy, ncellsx, ncellsy``: what a cell lookup needs.
            self._fast = (self.xrange[0], self.yrange[0],
                          side / ((self.xrange[1] - self.xrange[0]) or 1.0),
                          side / ((self.yrange[1] - self.yrange[0]) or 1.0),
                          side, side)  # fmt: skip
            self.steps = (side, side)
            found: dict[tuple[int, int], list[int]] = {}
            for index, box in enumerate(self.boxes.tolist()):
                for key in self._covered(box):
                    found.setdefault(key, []).append(index)
            self._table = {key: tuple(value) for key, value in found.items()}
        return self._table

    def _covered(self, box: Sequence[float]) -> list[tuple[int, int]]:
        """Every cell of the grid one bin's box overlaps, in add order."""
        x0 = _cell(box[0], self.xrange[0], (self.xrange[1] - self.xrange[0]) or 1.0, self.steps[0])
        x1 = _cell(box[1], self.xrange[0], (self.xrange[1] - self.xrange[0]) or 1.0, self.steps[0])
        y0 = _cell(box[2], self.yrange[0], (self.yrange[1] - self.yrange[0]) or 1.0, self.steps[1])
        y1 = _cell(box[3], self.yrange[0], (self.yrange[1] - self.yrange[0]) or 1.0, self.steps[1])
        return [(ix, iy) for ix in range(x0, x1 + 1) for iy in range(y0, y1 + 1)]

    def _key(self, x: float, y: float) -> tuple[int, int]:
        """The cell of the grid a point falls in, clamped to it."""
        ox, oy, sx, sy, nx, ny = self._fast
        return (min(max(int((x - ox) * sx), 0), nx - 1),
                min(max(int((y - oy) * sy), 0), ny - 1))  # fmt: skip

    def moments(self, w: float, x: float, y: float) -> None:
        """One fill's moments: the weight, its square, and x's and y's, which only a fill
        inside a bin adds."""
        sums = self.sums
        sums[0] += w
        sums[1] += w * w
        sums[2] += w * x
        sums[3] += w * x * x
        sums[4] += w * y
        sums[5] += w * y * y
        sums[6] += 1

    def fill(self, x: float, y: float, w: float = 1.0) -> int:
        """``Fill``: ``w`` into the bin holding ``(x, y)`` - or its region - and its number."""
        if self.squares is None and w != 1.0:
            self.sumw2()  # the first weight that is not one, as ``TH2Poly::Fill`` starts them
        found = self.find(x, y)
        if found < 0:
            self.overflow[-found - 1] += w
        else:
            self.contents[found - 1] += w
            self.moments(w, x, y)
        if self.squares is not None:
            self.squares[self.slot(found)] += w * w
        return found
