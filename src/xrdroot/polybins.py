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


class PolyBins:
    """The bins of a ``TH2Poly``, their contents, and the fills that missed them."""

    def __init__(self, xrange: Sequence[float] = (0.0, 0.0), yrange: Sequence[float] = (0.0, 0.0),
                 floating: bool = True) -> None:  # fmt: skip
        self.xrange, self.yrange = [float(v) for v in xrange], [float(v) for v in yrange]
        self.floating = floating
        self.polygons: list[list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]] = []
        self.boxes = np.zeros((0, 4))
        self.contents = np.zeros(0)
        self.overflow = np.zeros(REGIONS)
        #: ``fTsumw, fTsumw2, fTsumwx, fTsumwx2, fTsumwy, fTsumwy2`` and the entries.
        self.sums = np.zeros(7)

    def add(self, polygons: Sequence[tuple[Any, Any]]) -> int:
        """A bin of these polygons - each ``(xs, ys)`` - and its number, from 1."""
        made = [(np.asarray(x, dtype=float), np.asarray(y, dtype=float)) for x, y in polygons]
        xs = np.concatenate([x for x, _ in made])
        ys = np.concatenate([y for _, y in made])
        box = [xs.min(), xs.max(), ys.min(), ys.max()]
        if self.floating:
            self.xrange = [min(self.xrange[0], box[0]), max(self.xrange[1], box[1])]
            self.yrange = [min(self.yrange[0], box[2]), max(self.yrange[1], box[3])]
        self.polygons.append(made)
        self.boxes = np.vstack([self.boxes, box])
        self.contents = np.append(self.contents, 0.0)
        return len(self.polygons)

    def region(self, x: float, y: float) -> int:
        """Which region round the range ``(x, y)`` is in: -1..-9, -5 within the range."""
        row = -1 if y > self.yrange[1] else -4 if y > self.yrange[0] else -7
        return row + (-2 if x > self.xrange[1] else -1 if x > self.xrange[0] else 0)

    def find(self, x: float, y: float) -> int:
        """The bin holding ``(x, y)``, or the region it fell in, as ``TH2Poly::Fill`` finds."""
        region = self.region(x, y)
        if region != -5:
            return region
        b = self.boxes
        near = np.flatnonzero((b[:, 0] <= x) & (x <= b[:, 1]) & (b[:, 2] <= y) & (y <= b[:, 3]))
        for index in near:
            if any(inside(px, py, x, y) for px, py in self.polygons[index]):
                return int(index) + 1
        return -5

    def fill(self, x: float, y: float, w: float = 1.0) -> int:
        """``Fill``: ``w`` into the bin holding ``(x, y)`` - or its region - and its number."""
        found = self.find(x, y)
        if found < 0:
            self.overflow[-found - 1] += w
            return found
        self.contents[found - 1] += w
        self.sums += (w, w * w, w * x, w * x * x, w * y, w * y * y, 1)
        return found
