"""The rectangular cut optimisation of ``MethodCuts``: the efficiencies of a box, and its fitters.

Each variable's cut is a lower edge and a width - with ``FMax`` only the
lower edge is fitted, the width reaching past the top; with ``FMin`` only
the width, the edge fixed at the bottom - and a box's signal and background
efficiencies are the weighted fractions of the training events inside it
(``lower < x <= upper`` in every variable, as ``BinarySearchTree::InVolume``
has it), in TMVA's single precision. Every box tried is kept if it has the
least background efficiency yet seen in its bin of signal efficiency: that
table of 100 bins is what the method is. The Monte Carlo fitter draws its
boxes from ``TRandom3(Seed)`` exactly as TMVA's ``MCFitter`` does, and here
evaluates them a batch at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..random.mersenne import TRandom3
from .genetic import Interval

__all__ = ["CutTable", "Sample", "draw_mc"]

#: ``fgMaxAbsCutVal``: the cut that is no cut.
MAX_CUT = 1.0e30
#: How many boxes are evaluated at once.
BATCH = 2000


@dataclass
class Sample:
    """The events the efficiencies are counted over: values, and each class's weights."""

    values: Any
    signal: Any
    weights: Any

    def efficiencies(self, lower: Any, upper: Any) -> tuple[Any, Any]:
        """``GetEffsfromSelection`` of every box - rows of ``lower`` and ``upper`` - at once."""
        inside = np.ones((len(lower), len(self.values)), dtype=bool)
        for index in range(self.values.shape[1]):
            column = self.values[:, index][None, :]
            inside &= (lower[:, index][:, None] < column) & (upper[:, index][:, None] >= column)
        found = []
        for mask in (self.signal, ~self.signal):
            selected = (inside[:, mask] * self.weights[mask]).sum(axis=1).astype(np.float32)
            total = np.float32(np.sum(self.weights[mask]))
            found.append((selected / total).astype(np.float64) if total else np.zeros(len(lower)))
        return found[0], found[1]


@dataclass
class CutTable:
    """``fEffBvsSLocal`` and ``fCutMin``/``fCutMax``: the best box in each signal-efficiency bin."""

    nbins: int = 100
    effb: Any = None
    lower: Any = None
    upper: Any = None
    nvar: int = 0

    def __post_init__(self) -> None:
        if self.effb is None:
            self.effb = np.full(self.nbins, -0.1)
            self.lower = np.zeros((self.nbins, self.nvar))
            self.upper = np.zeros((self.nbins, self.nvar))

    def bins(self, effs: Any) -> Any:
        """``FindBin`` of each signal efficiency, from 1."""
        return np.where(effs < 1.0, (np.floor(np.asarray(effs) * self.nbins)).astype(int) + 1, self.nbins + 1)

    def offer(self, effs: Any, effb: Any, lower: Any, upper: Any) -> None:
        """Each box, in order, kept where its background efficiency beats its bin's."""
        where = self.bins(effs)
        for row in range(len(effs)):
            index = where[row] - 1
            if 0 <= index < self.nbins and (self.effb[index] < 0 or self.effb[index] > effb[row]):
                self.effb[index] = effb[row]
                self.lower[index], self.upper[index] = lower[row], upper[row]

    def offer_batch(self, effs: Any, effb: Any, lower: Any, upper: Any) -> None:
        """:meth:`offer` for many boxes: each bin's first box of its least background efficiency."""
        where = self.bins(effs) - 1
        valid = (where >= 0) & (where < self.nbins)
        rows = np.flatnonzero(valid)
        if not len(rows):
            return
        order = np.lexsort((rows, effb[rows], where[rows]))
        ranked = rows[order]
        first = np.concatenate(([True], where[ranked][1:] != where[ranked][:-1]))
        for row in ranked[first]:
            index = where[row]
            if self.effb[index] < 0 or self.effb[index] > effb[row]:
                self.effb[index] = effb[row]
                self.lower[index], self.upper[index] = lower[row], upper[row]


@dataclass
class Ranges:
    """The fit's intervals - an edge and a width per variable - and whether each is drawn."""

    intervals: list[Interval] = field(default_factory=list)

    def boxes(self, parameters: Any) -> tuple[Any, Any]:
        """``MatchParsToCuts``: each row of parameters as its box's lower and upper edges."""
        lower = parameters[:, 0::2]
        return lower, lower + parameters[:, 1::2]


def draw_mc(intervals: list[Interval], samples: int, seed: int) -> Any:
    """Every parameter set ``MCFitter`` tries, drawn from its ``TRandom3(seed)`` in its order."""
    random = TRandom3(seed)
    random.rndm()
    drawn = sum(1 for i in intervals if i.nbins > 0 or i.low != i.high)
    if not drawn:
        return np.tile([i.low for i in intervals], (samples, 1))
    random.rndm(drawn)
    draws = np.asarray(random.rndm(drawn * samples)).reshape(samples, drawn)
    parameters = np.empty((samples, len(intervals)))
    column = 0
    for index, interval in enumerate(intervals):
        if interval.nbins > 0:
            positions = (draws[:, column] * interval.nbins).astype(int)
            parameters[:, index] = [interval.GetElement(p) for p in positions]
            column += 1
        elif interval.low == interval.high:
            parameters[:, index] = interval.low
        else:
            parameters[:, index] = interval.low + (interval.high - interval.low) * draws[:, column]
            column += 1
    return parameters
