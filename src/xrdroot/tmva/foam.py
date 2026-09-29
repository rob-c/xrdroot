"""``TMVA::PDEFoam``: the variable space cut into cells where the training events' density changes.

The foam lives in the unit hypercube, each variable mapped from its range
(:func:`ranges` - the training range less ``TailCut`` of the events at each
end). A cell is explored by sampling ``nSampl`` points in it from
``TRandom3(4356)`` and weighing each by a density of the training events in
a small box around it (``VolFrac`` of each range): the discriminant
``S/(S+B)``, the event count or the mean target. Each variable's samples are
histogrammed in ``nBin`` bins, and ``Varedu`` finds the cut that most reduces
the spread of the weights; the cell whose spread is largest is then cut
there, until there are ``nActiveCells`` cells or none worth cutting. All of
it is TMVA's code, draw for draw; the boxes are searched with SciPy's
k-d tree, then tested exactly as TMVA's ``BinarySearchTree`` tests them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..random.mersenne import TRandom3
from .foamcells import Cells
from .log import Logger

__all__ = ["Density", "Foam", "ranges"]

#: ``std::numeric_limits<float>::epsilon()`` and ``<double>``'s.
FLT_EPSILON, DBL_EPSILON = 1.1920928955078125e-07, 2.220446049250313e-16
#: ``kHigh``: the least weight a cell's exploration starts from.
HIGH = 1.0e150

f32 = np.float32


def ranges(values: Any, fraction: float) -> tuple[Any, Any]:
    """``CalcXminXmax``: each variable's range, less ``fraction`` of the events at each end."""
    outside = int(len(values) * float(f32(fraction)))
    lows, highs = [], []
    for column in np.asarray(values, dtype=np.float64).T:
        low = float(f32(min(column.min(), np.finfo(np.float32).max)))
        high = float(f32(max(column.max(), np.finfo(np.float32).tiny)))
        width = (high - low) / 10000
        bins = np.clip(np.floor(10000 * (column - low) / (high - low)).astype(int) + 1, 0, 10001)
        counts = np.bincount(bins, minlength=10002)
        left, right = np.cumsum(counts), np.cumsum(counts[::-1])[::-1]
        first = np.flatnonzero(left[1:10001] > outside)
        last = np.flatnonzero(right[1:10001] > outside)
        lows.append(float(f32(low + first[0] * width)) if len(first) else low)
        highs.append(float(f32(low + (last[-1] + 1) * width)) if len(last) else high)
    return np.asarray(lows), np.asarray(highs)


@dataclass
class Density:
    """``PDEFoam*Density``: what the training events in a box around a point say of it."""

    values: Any
    weights: Any
    box: Any
    kind: str = "discriminant"
    marks: Any = None
    tree: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from scipy.spatial import cKDTree

        self.volume = float(np.prod(self.box))
        self.tree = cKDTree(self.values / self.box)

    def __call__(self, points: Any) -> tuple[Any, Any]:
        """The density at each point, and its ``event_density`` (events in the box per volume)."""
        from scipy.spatial import cKDTree

        pairs = cKDTree(points / self.box).sparse_distance_matrix(
            self.tree, 0.5 * (1 + 1e-9), p=np.inf, output_type="ndarray"
        )
        rows, cols = pairs["i"].astype(np.intp), pairs["j"].astype(np.intp)
        low, high = points[rows] - self.box / 2.0, points[rows] + self.box / 2.0
        found = self.values[cols]
        keep = np.all((low < found) & (high >= found), axis=1)
        rows, cols = rows[keep], cols[keep]
        size = len(points)
        count = np.bincount(rows, minlength=size).astype(np.float64)
        total = np.bincount(rows, self.weights[cols], minlength=size)
        inverse = 1.0 / self.volume
        if self.kind == "event":
            return (total + 0.1) * inverse, count * inverse
        marked = np.bincount(rows, (self.weights * self.marks)[cols], minlength=size)
        return marked / (total + 0.1) * inverse, count * inverse


class Foam(Cells):
    """A foam being grown, and the cells it has grown."""

    def __init__(
        self, name: str, xmin: Any, xmax: Any, density: Density, options: dict[str, Any]
    ) -> None:
        super().__init__(len(xmin), 2 * int(options["nActiveCells"]) - 1)
        self.name, self.xmin, self.xmax, self.density = name, xmin, xmax, density
        self.samples, self.nbin = int(options["nSampl"]), int(options["nBin"])
        self.per_bin, self.nmin = 10000, int(options["Nmin"])
        self.max_depth = int(options["MaxDepth"])
        self.random = TRandom3(4356)
        self.log = Logger(name)

    def grow(self) -> None:
        """``Create``: the root explored, then the cell of largest driver cut until done."""
        self.fill(None)
        self.explore(0)
        while self.last + 2 < self.ncells:
            chosen = self.peek()
            if chosen < 0:
                self.ncells = self.last + 1
                break
            self.divide(chosen)
        self.elements[:] = 0.0

    def peek(self) -> int:
        """``PeekMax``: the active cell of largest driver that may yet be cut."""
        best, chosen, enough, shallow = 0.0, -1, True, True
        for cell in range(self.last + 1):
            if not self._candidate(cell):
                continue
            enough, shallow = self._limits(cell)
            if self.driv[cell] > best and enough and shallow:
                best, chosen = float(self.driv[cell]), cell
        if chosen < 0 and enough and shallow:
            self.log.warning(
                "<PDEFoam::PeekMax>: no more candidate cells (drivMax>0) found for "
                "further splitting."
            )
        return chosen

    def _candidate(self, cell: int) -> bool:
        """An active cell with a driver, and a cut inside it rather than at an edge."""
        if self.status[cell] != 1 or self.driv[cell] < FLT_EPSILON:
            return False
        xdiv = abs(self.xdiv[cell])
        return bool(DBL_EPSILON < xdiv < 1.0 - DBL_EPSILON)

    def _limits(self, cell: int) -> tuple[bool, bool]:
        """Whether a cell holds more than ``Nmin`` events, and is above ``MaxDepth``, if asked."""
        enough = self.nmin <= 0 or bool(self.elements[cell, 0] > self.nmin)
        shallow = self.max_depth <= 0 or self.depth(cell) < self.max_depth
        return enough, shallow

    def divide(self, cell: int) -> None:
        """``Divide``: the cell made inactive, and its two daughters made and explored."""
        self.status[cell] = 0
        first, second = self.fill(cell), self.fill(cell)
        self.dau0[cell], self.dau1[cell] = first, second
        self.explore(first)
        self.explore(second)

    def _sample(self, cell: int) -> tuple[Any, Any, Any]:
        posi, size = self.hcub(cell)
        alpha = np.asarray(self.random.rndm(self.samples * self.dim), dtype=np.float64)
        alpha = alpha.reshape(self.samples, self.dim)
        points = (alpha * size + posi).astype(f32).astype(np.float64)
        spans = self.xmax - self.xmin
        points = (points * spans + self.xmin).astype(f32).astype(np.float64)
        value, events = self.density(points)
        return alpha, value, events

    def explore(self, cell: int) -> None:
        """``Explore``: the cell sampled, its best cut found, its and its parents' sums updated."""
        dx = self.volume(cell) * float(np.prod(self.xmax - self.xmin))
        old_intg, old_driv, old_total = self.intg[cell], self.driv[cell], self.elements[cell, 0]
        alpha, value, events = self._sample(cell)
        weights = dx * value
        bins = np.minimum((alpha * self.nbin).astype(int), self.nbin - 1)
        squares = np.stack(
            [
                np.bincount(bins[:, k], weights * weights, minlength=self.nbin)
                for k in range(self.dim)
            ]
        )
        total = _sequential(events) * dx / self.samples
        count = float(len(weights))
        sums = (_sequential(weights), _sequential(weights * weights), count)
        best, xbest = self.varedu(sums, np.sqrt(squares) ** 2)
        intg = sums[0] / (count + 0.000001)
        driv = np.sqrt(sums[1] / count) - intg
        self.best[cell], self.xdiv[cell] = best, xbest
        self.intg[cell], self.driv[cell], self.elements[cell, 0] = intg, driv, total
        parent = self.parent[cell]
        while parent >= 0:
            self.intg[parent] += intg - old_intg
            self.driv[parent] += driv - old_driv
            self.elements[parent, 0] += total - old_total
            parent = self.parent[parent]

    def varedu(self, sums: tuple[float, float, float], squares: Any) -> tuple[int, float]:
        """``Varedu``: the variable and edge whose cut most reduces the weights' spread."""
        entries, all_squares = sums[2], sums[1]
        spread = np.sqrt(all_squares) / np.sqrt(entries)
        best, xbest, ybest, gain_max = -1, 0.5, 1.0, 0.0
        for k in range(self.dim):
            gain, x_min, x_max = _best_window(squares[k], entries, all_squares, spread, self.nbin)
            if gain >= gain_max:
                gain_max, best = gain, k
                xbest, ybest = x_min, x_max
                if int(self.nbin * x_min) == 0:
                    xbest = ybest
        return best, xbest


def _sequential(values: Any) -> float:
    """A sum taken one value after another, as TMVA's loops take it."""
    return float(np.cumsum(values)[-1]) if len(values) else 0.0


def _best_window(
    squares: Any, entries: float, total: float, spread: float, nbin: int
) -> tuple[float, float, float]:
    """The bins ``[jLo, jUp]`` whose inside and outside spreads sum least, and the gain."""
    best, gain, x_min, x_max = HIGH, 0.0, 0.0, 0.0
    for low in range(1, nbin + 1):
        inside = 0.0
        for up in range(low, nbin + 1):
            inside += float(squares[up - 1])
            x_lo, x_up = (low - 1.0) / nbin, (up * 1.0) / nbin
            width, rest = x_up - x_lo, 1.0 - x_up + x_lo
            s_in = (
                0.0 if width < DBL_EPSILON else np.sqrt(inside) / np.sqrt(entries * width) * width
            )
            if rest < DBL_EPSILON or total - inside < DBL_EPSILON:
                s_out = 0.0
            else:
                s_out = np.sqrt(total - inside) / np.sqrt(entries * rest) * rest
            if s_in + s_out < best:
                best, gain, x_min, x_max = s_in + s_out, spread - (s_in + s_out), x_lo, x_up
    return float(gain), x_min, x_max
