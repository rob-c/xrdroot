"""Reading a data set's events from its trees: ``DataSetFactory::BuildEventVector``.

Each class's trees are read in the order they were added: every variable,
target and spectator expression evaluated over the whole tree at once -
through :meth:`xrdroot.TTree.arrays`, which speaks ``TTreeFormula``'s
language - the class's cut applied, and the weight made of the tree's
weight times the class's weight expression, in single precision as TMVA's
``Float_t weight`` is. What is counted on the way - events before and after
the cut, and their weights - is what the summary TMVA prints is made of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dataset import MAX_TREE_TYPE, DataSetInfo, Events, as_float
from .log import Logger
from .variables import VariableInfo

__all__ = ["ClassCounts", "TreeInput", "read_class"]


@dataclass
class TreeInput:
    """``TMVA::TreeInfo``: a tree, the class it is of, its weight and what it is for."""

    tree: Any
    class_name: str
    weight: float = 1.0
    tree_type: int = MAX_TREE_TYPE


@dataclass
class ClassCounts:
    """``DataSetFactory::EventStats``: one class's events, counted as they are read."""

    initial: int = 0
    before: int = 0
    weight_before: float = 0.0
    after: int = 0
    weight_after: float = 0.0
    train_requested: int = 0
    test_requested: int = 0
    split_requested: float = 0.0
    #: The events read, by the tree type they came in: training, testing, either.
    events: dict[int, list[Events]] = field(default_factory=dict)

    def cut_scaling(self) -> float:
        return self.after / self.before if self.before else 1.0


def backing(tree: Any) -> Any:
    """What a tree reads from: a pyroot tree's view, or an xrdroot tree as it is."""
    view = getattr(tree, "_view", None)
    return view() if view is not None else tree


def _column(value: Any, info: VariableInfo, size: int) -> Any:
    """One variable's values: the whole column, or one element of an array expression."""
    array = np.asarray(value if info.index is None else _element(value, info.index))
    if array.shape == ():
        array = np.full(size, array)
    return as_float(array)


def _element(value: Any, index: int) -> Any:
    """The ``index``-th element of every entry of an array expression."""
    if isinstance(value, np.ndarray) and value.ndim == 2:
        return value[:, index]
    return np.array([np.asarray(entry)[index] for entry in value], dtype=np.float64)


def _evaluate(tree: Any, expressions: list[str]) -> dict[str, Any]:
    """Every distinct expression over every entry of ``tree``."""
    wanted = list(dict.fromkeys(text for text in expressions if text))
    return dict(backing(tree).arrays(wanted)) if wanted else {}


def _block(infos: list[VariableInfo], found: dict[str, Any], size: int) -> Any:
    """The columns of ``infos`` side by side: a row per entry."""
    if not infos:
        return np.zeros((size, 0))
    return np.column_stack([_column(found[info.expression], info, size) for info in infos])


def _weights(found: dict[str, Any], expression: str, tree_weight: float, size: int) -> Any:
    """Each entry's ``Float_t`` weight: the tree's weight times the weight expression's value."""
    base = np.float32(tree_weight)
    if not expression:
        return np.full(size, float(base))
    values = np.asarray(found[expression], dtype=np.float64)
    return as_float(base * values)


def _read_tree(dsi: DataSetInfo, item: TreeInput, number: int, counts: ClassCounts) -> None:
    info = dsi.classes[number]
    expressions = [v.expression for v in dsi.variables + dsi.targets + dsi.spectators]
    found = _evaluate(item.tree, [*expressions, info.cut, info.weight])
    size = len(backing(item.tree))
    values = _block(dsi.variables, found, size)
    weights = _weights(found, info.weight, item.weight, size)
    passed = np.ones(size, dtype=bool)
    if info.cut:
        passed = np.asarray(found[info.cut], dtype=np.float64) >= 0.5
    counts.initial += size
    counts.before += size
    counts.weight_before += float(np.sum(weights[~np.isnan(weights)]))
    events = Events(
        values[passed],
        _block(dsi.targets, found, size)[passed],
        _block(dsi.spectators, found, size)[passed],
        np.full(int(passed.sum()), number, dtype=np.int64),
        weights[passed],
    )
    events = _finite(events, dsi)
    counts.after += len(events)
    counts.weight_after += float(np.sum(events.weights))
    counts.events.setdefault(item.tree_type, []).append(events)


def _finite(events: Events, dsi: DataSetInfo) -> Events:
    """The events without a NaN or an infinity, each one left out said so, as TMVA says it."""
    rows = np.column_stack([events.values, events.targets, events.spectators, events.weights])
    bad = ~np.all(np.isfinite(rows), axis=1)
    if not bad.any():
        return events
    log = Logger("DataSetFactory")
    for index in np.flatnonzero(bad):
        log.warning(f"Dataset[{dsi.name}] : NaN or +-inf in Event {index}")
    return events.take(~bad)


def read_class(dsi: DataSetInfo, inputs: list[TreeInput], number: int, log: Logger) -> ClassCounts:
    """Every tree of class ``number`` read, as ``BuildEventVector``'s loop over them reads it."""
    counts = ClassCounts()
    name = dsi.classes[number].name
    for item in inputs:
        if item.class_name != name:
            continue
        log.info(f"Building event vectors for type {item.tree_type} {name}")
        log.info(f"Dataset[{dsi.name}] :  create input formulas for tree {item.tree.GetName()}")
        _read_tree(dsi, item, number, counts)
    return counts
