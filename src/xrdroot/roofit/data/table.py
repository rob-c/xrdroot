"""``Roo1DTable``: how many events - summed weights - a dataset has in each state of a category.

``data.table(c)`` counts, ``data.table(c, "x>8")`` counts those passing a
cut, and the table prints as ``Roo1DTable::c = (A=12,B=30)`` or, verbose, as
a box of labels and counts in ``Roo1DTable::printMultiline``'s widths.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..printing import RooPrintable, g, kClassName, kName, kValue

__all__ = ["Roo1DTable", "table_of"]


class Roo1DTable(RooPrintable):
    """Counts per state of a category."""

    def __init__(self, name: str, title: str, labels: list[str], counts: list[float]) -> None:
        self._name, self._title = name, title
        self.labels, self.counts = labels, counts

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        return self._title

    def printName(self) -> str:
        return self._name

    def printClassName(self) -> str:
        return "Roo1DTable"

    def defaultPrintContents(self, option: Any) -> int:
        return kName | kClassName | kValue

    def printValue(self) -> str:
        parts = [
            f"{label}={g(count)}" for label, count in zip(self.labels, self.counts) if count > 0
        ]
        return "(" + ",".join(parts) + ")"

    def printMultiline(self, contents: int, verbose: bool, indent: str) -> str:
        width = max([0, *(len(label) for label in self.labels)])
        digits = int(math.log10(max([1.0, *self.counts]))) + 1
        rule = f"{indent}  +-{'-' * width}-+-{'-' * digits}-+\n"
        text = f"{indent}\n{indent}  Table {self._name} : {self._title}\n" + rule
        for label, count in zip(self.labels, self.counts):
            if count > 0 or verbose:
                text += f"  | {label:>{width}} | {g(count):>{digits}} |\n"
        return text + rule + f"{indent}\n"

    def get(self, label: str, silent: bool = False) -> float:
        return float(self.counts[self.labels.index(str(label))])

    def getFrac(self, label: str, silent: bool = False) -> float:
        total = sum(self.counts)
        return self.get(label) / total if total else 0.0

    def getOverflow(self) -> float:
        return 0.0


def table_of(data: Any, category: Any, cut: Any = None) -> Roo1DTable:
    """``RooAbsData::table``: the weights of the events in each state, after ``cut``."""
    from ..collections import as_list
    from .selection import context_of

    if not hasattr(category, "states"):
        return multi_table(data, as_list(category), cut)
    keep = data.mask(cut or None)
    found = data.variable(category.GetName())
    column = (
        data.column(category.GetName())
        if found is not None
        else np.broadcast_to(category.compute(context_of(data)), keep.shape)
    )[keep]
    weights = data.weights()[keep]
    labels = list(category.states())
    counts = [float(np.sum(weights[column == category.states()[label]])) for label in labels]
    title = data.GetName() + (f"({cut})" if cut else "")
    return Roo1DTable(category.GetName(), title, labels, counts)


def multi_table(data: Any, categories: list[Any], cut: Any = None) -> Roo1DTable:
    """A table over several categories at once: ``RooMultiCategory``'s states, the first fastest.

    A Python set of categories has no order ROOT can see either; they are
    taken sorted by name, as the set's hashes happened to give them in
    ROOT's own run.
    """
    import itertools

    ordered = sorted(categories, key=lambda one: one.GetName())
    keep = data.mask(cut or None)
    weights = data.weights()[keep]
    columns = [data.column(one.GetName())[keep] for one in ordered]
    labels, counts = [], []
    states = [list(one.states().items()) for one in ordered]
    for combination in itertools.product(*reversed(states)):
        chosen = list(reversed(combination))
        mask = np.ones(len(weights), dtype=bool)
        for column, (_, index) in zip(columns, chosen):
            mask &= column == index
        labels.append("{" + ";".join(label for label, _ in chosen) + "}")
        counts.append(float(np.sum(weights[mask])))
    name = "(" + " x ".join(one.GetName() for one in ordered) + ")"
    return Roo1DTable(name, data.GetName() + (f"({cut})" if cut else ""), labels, counts)
