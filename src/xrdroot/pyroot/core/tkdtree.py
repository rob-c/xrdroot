"""``TKDTreeBinning``: a k-d tree over a data set, its leaves taken as the bins of a grid.

ROOT's class splits the points it is given down the widest dimension of each group at that
group's median until it has as many leaves as the caller asked for, and each leaf becomes
the hyperrectangle its points fall in. A ``TH2Poly`` is then built from those rectangles and
given each leaf's count as a bin content - which is what makes this useful: a *data-driven*
binning with no empty bins.

The edges this kit computes are a k-d tree of the data and not ROOT's own tree: the leaves
depend on the order the split is chosen in, so a caller who needs ROOT's exact rectangles is
on different ground. What is identical is what the class promises - one bin per leaf, its
count, its box, and the data's own extremes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .objects import TObject

__all__ = ["TKDTreeBinning"]


class TKDTreeBinning(TObject):
    """``TKDTreeBinning``: ``size`` points of ``dim`` dimensions, binned into ``bins``.

    The points come as ROOT keeps them - one run of ``size`` values per dimension, so a 2-D
    set is its x's and then its y's - which is how a macro fills the array it hands over.
    """

    def __init__(self, size: Any = 0, dim: Any = 1, data: Any = None,
                 bins: Any = 100, adjust: Any = False) -> None:  # fmt: skip
        super().__init__()
        self._size, self._dim, self._bins = int(size), int(dim), int(bins)
        if self._bins < 1 or self._size < self._bins:
            raise ValueError(
                f"a k-d binning of {self._size} points cannot hold {self._bins} bins: "
                "ask for no more bins than there are points, or give it more data."
            )
        given = np.asarray(data if data is not None else [], dtype=float)
        self._data = given.reshape(self._dim, self._size).T.copy()
        self._min, self._max, self._counts = _split(self._data, self._bins)
        self._adjust = bool(adjust)

    def GetNBins(self) -> int:
        return self._bins

    def GetDim(self) -> int:
        return self._dim

    def GetBinContent(self, number: Any) -> int:
        """How many of the data's points the ``number``'s leaf holds - ROOT counts from 0."""
        return int(self._counts[int(number)])

    def GetBinsMinEdges(self) -> Any:
        """Every bin's lower corner, ``dim`` numbers per bin, as the ``double *`` ROOT gives."""
        return self._min.reshape(-1)

    def GetBinsMaxEdges(self) -> Any:
        """Every bin's upper corner, ``dim`` numbers per bin, as the ``double *`` ROOT gives."""
        return self._max.reshape(-1)

    def GetBinMinEdges(self, number: Any) -> Any:
        return self._min[int(number)]

    def GetBinMaxEdges(self, number: Any) -> Any:
        return self._max[int(number)]

    def GetDataMin(self, axis: Any) -> float:
        """The smallest value the data has along ``axis`` - ROOT's 0-based dimension."""
        return float(self._data[:, int(axis)].min())

    def GetDataMax(self, axis: Any) -> float:
        """The largest value the data has along ``axis`` - ROOT's 0-based dimension."""
        return float(self._data[:, int(axis)].max())

    def GetBinVolume(self, number: Any) -> float:
        return float(np.prod(self._max[int(number)] - self._min[int(number)]))

    def GetBinDensity(self, number: Any) -> float:
        volume = self.GetBinVolume(number)
        return float(self._counts[int(number)]) / volume if volume else 0.0


def _widest(points: np.ndarray) -> int:
    """The dimension ``points`` spreads over most, the lowest index winning a tie."""
    return int(np.argmax(points.max(axis=0) - points.min(axis=0)))


def _halves(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``points`` cut at the median of its widest dimension, as a k-d tree splits a node."""
    values = points[:, _widest(points)]
    middle = len(points) // 2
    if values.max() == values.min():  # no spread to cut: halve the group by position alone
        return points[:middle], points[middle:]
    order = np.argpartition(values, middle)
    return points[order[:middle]], points[order[middle:]]


def _split(points: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut ``points`` into ``bins`` leaves, and answer each leaf's box and its count.

    The largest group is cut first, which keeps the leaves as even as the data allows - the
    walk ROOT makes, and the reason a leaf's box is the extent of the points it holds.
    """
    groups = [points]
    while len(groups) < bins:
        grown = max(range(len(groups)), key=lambda found: len(groups[found]))
        left, right = _halves(groups.pop(grown))
        groups.extend([left, right])
    low = np.array([group.min(axis=0) for group in groups])
    high = np.array([group.max(axis=0) for group in groups])
    return low, high, np.array([len(group) for group in groups])
