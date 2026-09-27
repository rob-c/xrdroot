"""A dataset made of others, one per state of an index category: ``Index(c), Import({...})``.

``RooDataSet("d", "d", {x}, Index=sample, Import={"physics": a, "control": b})``
is ``a``'s events with ``sample`` set to ``physics`` and ``b``'s with it set to
``control``, appended in the order of the states' labels - a ``std::map``'s
order, which is how ``RooDataSet::loadValuesFromSlices`` walks them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..messages import INFO, log

__all__ = ["import_slices"]


def _slices(options: Any) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for one in options.every("Import"):
        first = one.value(0)
        if isinstance(first, dict):
            found.update({str(k): v for k, v in first.items()})
        elif isinstance(first, str):
            found[first] = one.value(1)
    return found


def _define(data: Any, index: Any, category: Any, label: str) -> None:
    """A state the index category lacks, defined in it and in the dataset's copy - and said."""
    if index.hasLabel(label):
        return
    index.defineType(label)
    category.defineType(label)
    log(
        data,
        INFO,
        "InputArguments",
        f"RooDataSet::ctor({data.GetName()}) defining state "
        f'"{label}" in index category {index.GetName()}',
    )


def _slice_columns(data: Any, index: Any, label: str, source: Any) -> dict[str, Any]:
    """One slice's columns: its events, and the index at the slice's state for each."""
    n = source.numEntries()
    return {
        one.GetName(): np.full(n, float(index.lookupIndex(label)))
        if one.GetName() == index.GetName()
        else np.asarray(source.column(one.GetName()), dtype=np.float64)
        for one in data.get()
    }


def import_slices(data: Any, options: Any) -> None:
    """Fill ``data`` from the datasets the ``Import`` options give per state of the ``Index``."""
    index = options.get("Index")
    category = data.variable(index.GetName())
    columns: dict[str, list[Any]] = {one.GetName(): [] for one in data.get()}
    weights: list[Any] = []
    for label, source in sorted(_slices(options).items()):
        _define(data, index, category, label)
        for name, column in _slice_columns(data, index, label, source).items():
            columns[name].append(column)
        weights.append(source.weights())
    _assemble(data, columns, weights)


def _assemble(data: Any, columns: dict[str, list[Any]], weights: list[Any]) -> None:
    """The slices one after another; weights kept if any slice - or the dataset - has them."""
    for name, parts in columns.items():
        data._columns[name] = np.concatenate(parts) if parts else np.zeros(0)
    if data._weights is not None or any(np.any(w != 1) for w in weights):
        data._weights = np.concatenate(weights) if weights else np.zeros(0)
