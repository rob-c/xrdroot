"""``KNN``: TMVA's k nearest neighbours, as ``ModulekNN`` finds them.

Each variable is scaled by the width of the central ``ScaleFrac`` of its
training values (from the 10% to the 90% point by default), and an event's
output is the weight of signal among its ``nkNN`` nearest training events
over their total weight - or, for a regression, their weighted mean
target. ``UseKernel`` weighs each neighbour by a Gaussian (``Kernel=Gaus``)
or a polynomial (``Poln``) of its distance, as TMVA's formulas do, the
query taken unscaled where TMVA takes it so. The neighbours are found by
brute force in NumPy, which finds the same events a k-d tree finds.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..dataset import Events
from ..log import Logger
from ..method import CLASSIFICATION, REGRESSION, Method
from ..xmlfile import Node, children, floats, number

__all__ = ["MethodKNN", "scale_widths"]

#: How many query events are compared with the training events at once.
CHUNK = 512


def scale_widths(values: Any, fraction: int) -> Any:
    """``ComputeMetric``: each variable's width, its ``(100 - f)/2`` to ``100 - that`` point."""
    size = len(values)
    low, high = (100 - fraction) // 2, 100 - (100 - fraction) // 2
    positions = (100 * np.arange(size)) // size
    ordered = np.sort(values, axis=0)
    first = np.flatnonzero(positions == low)
    last = np.flatnonzero(positions == high)
    if not len(first) or not len(last):
        return ordered[-1] - ordered[0]
    return ordered[last[0]] - ordered[first[0]]


class MethodKNN(Method):
    """``TMVA::MethodKNN``."""

    type_name = "KNN"
    analyses = frozenset({CLASSIFICATION, REGRESSION})
    defaults: ClassVar[dict[str, Any]] = {
        "nkNN": 20,
        "BalanceDepth": 6,
        "ScaleFrac": 0.8,
        "SigmaFact": 1.0,
        "Kernel": "Gaus",
        "Trim": False,
        "UseKernel": False,
        "UseWeight": True,
        "UseLDA": False,
    }

    def process_options(self) -> None:
        self.k = max(int(self.opt("nkNN")), 1)
        self.fraction = min(max(float(self.opt("ScaleFrac")), 0.0), 1.0)

    def train(self, events: Events) -> None:
        self.log.header("<Train> start...")
        if self.opt("IgnoreNegWeightsInTraining"):
            events = events.take(events.weights > 0)
        self.log.info(f"Reading {len(events)} events")
        signal = events.classes == self.dsi.GetSignalClassIndex()
        self.log.info(f"Number of signal events {float(np.sum(events.weights[signal])):g}")
        self.log.info(f"Number of background events {float(np.sum(events.weights[~signal])):g}")
        self.values = np.asarray(events.values, dtype=np.float32)
        self.types = np.where(signal, 1, 2)
        self.weights = np.asarray(events.weights, dtype=np.float64)
        self.targets = np.asarray(events.targets, dtype=np.float32)
        self._make()

    def _make(self) -> None:
        """``MakeKNN``: the training events scaled, and TMVA's account of the tree it builds."""
        self.log.info(f"Creating kd-tree with {len(self.values)} events")
        self.scales = np.ones(self.values.shape[1], dtype=np.float32)
        percent = int(100.0 * self.fraction)
        if percent > 0:
            low, high = (100 - percent) // 2, 100 - (100 - percent) // 2
            self.log.info(
                "Computing scale factor for 1d distributions: "
                f"(ifrac, bottom, top) = ({percent}%, {low}%, {high}%)"
            )
            self.scales = np.float32(scale_widths(self.values.astype(np.float64), percent))
        self.scaled = (self.values / self.scales).astype(np.float32)
        Logger("ModulekNN").header(
            f"Optimizing tree for {self.values.shape[1]} variables with {len(self.values)} values"
        )
        for kind in (1, 2):
            count = int(np.sum(self.types == kind))
            if count:
                self.log.info(f"<Fill> Class {kind} has {count:>8d} events")

    def _neighbours(self, values: Any) -> tuple[Any, Any]:
        """The ``nkNN`` nearest training events of each query, and their squared distances."""
        query = (np.asarray(values, dtype=np.float32) / self.scales).astype(np.float64)
        train = self.scaled.astype(np.float64)
        k = min(self.k, len(train))
        found, distances = [], []
        for start in range(0, len(query), CHUNK):
            block = query[start : start + CHUNK]
            squared = ((block[:, None, :] - train[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argpartition(squared, k - 1, axis=1)[:, :k]
            near = np.take_along_axis(squared, nearest, axis=1)
            order = np.argsort(near, axis=1, kind="stable")
            found.append(np.take_along_axis(nearest, order, axis=1))
            distances.append(np.take_along_axis(near, order, axis=1))
        return np.concatenate(found), np.concatenate(distances)

    def _kernel(self, values: Any, index: Any, distance: Any) -> Any:
        """Each neighbour's kernel weight, as ``GausKernel`` or ``PolnKernel`` gives it."""
        if not self.opt("UseKernel"):
            return np.ones(index.shape)
        if str(self.opt("Kernel")) == "Poln":
            radius = np.where(distance > 0, distance, 0).max(axis=1, keepdims=True)
            scaled = np.sqrt(distance) / np.sqrt(np.where(radius > 0, radius, 1))
            return np.where(scaled < 1, (1 - scaled**3) ** 3, 0.0)
        raw = np.asarray(values, dtype=np.float64)[:, None, :]
        difference = self.scaled.astype(np.float64)[index] - raw
        counted = (distance > 0)[:, :, None]
        spread = np.sqrt((difference**2 * counted).sum(axis=1) / np.maximum(counted.sum(axis=1), 1))
        sigma = abs(float(self.opt("SigmaFact"))) * np.where(spread > 0, spread, 1.0)
        return np.exp(-(difference**2 / (2.0 * sigma[:, None, :] ** 2)).sum(axis=2))

    def evaluate(self, values: Any) -> Any:
        index, distance = self._neighbours(values)
        weight = self._kernel(values, index, distance)
        if self.opt("UseWeight"):
            weight = weight * self.weights[index]
        if self.analysis == REGRESSION:
            found = (weight[:, :, None] * self.targets[index]).sum(axis=1) / weight.sum(axis=1)[
                :, None
            ]
            return self.handler.inverse_targets(found)
        return (weight * (self.types[index] == 1)).sum(axis=1) / weight.sum(axis=1)

    def add_weights(self, node: Node) -> None:
        weights = node.add(
            "Weights",
            NEvents=len(self.values),
            NVar=self.values.shape[1],
            NTgt=self.targets.shape[1],
        )
        for row in range(len(self.values)):
            numbers = [*self.values[row], *self.targets[row]]
            event = weights.add(
                "Event", Type=int(self.types[row]), Weight=number(self.weights[row])
            )
            event.text = " ".join(number(value) for value in numbers)

    def read_weights(self, node: Any) -> None:
        nvar, ntgt = int(node.get("NVar")), int(node.get("NTgt", 0))
        rows = [floats(item) for item in children(node, "Event")]
        table = np.array(rows, dtype=np.float64).reshape(len(rows), nvar + ntgt)
        self.values = table[:, :nvar].astype(np.float32)
        self.targets = table[:, nvar:].astype(np.float32)
        self.types = np.array([int(str(item.get("Type"))) for item in children(node, "Event")])
        self.weights = np.array(
            [float(str(item.get("Weight"))) for item in children(node, "Event")]
        )
        self._make()
