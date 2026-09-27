"""Which events of a dataset a cut and a range select.

A cut is a formula over the dataset's variables - ``"y>5.17"``,
``"c==c::Plus"`` - or a function object; a range is the name of one or more
ranges, ``"left,right"``, an event being in it when every variable it names
a range of is inside that range, and in ``"a,b"`` when it is in either.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["context_of", "in_range", "selected"]


def context_of(data: Any) -> dict[str, Any]:
    """The dataset's columns as a context: each variable's name to its column."""
    return dict(data.columns())


def in_range(data: Any, rng: Any, variables: Any = None) -> np.ndarray[Any, Any]:
    """Which events lie in the named range(s) ``rng``, of every variable that has one."""
    n = data.numEntries()
    if not rng:
        return np.ones(n, dtype=bool)
    found = np.zeros(n, dtype=bool)
    for part in (one for one in str(rng).split(",") if one):
        found |= _inside(data, part, variables)
    return found


def _inside(data: Any, part: str, variables: Any) -> np.ndarray[Any, Any]:
    keep = np.ones(data.numEntries(), dtype=bool)
    for var in variables if variables is not None else data.get():
        if not var.hasRange(part) or not (var.InheritsFrom("RooAbsRealLValue") or
                                          var.InheritsFrom("RooCategory")):  # fmt: skip
            continue
        column = data.column(var.GetName())
        if var.InheritsFrom("RooCategory"):
            keep &= np.isin(column, var.range_indices(part))
        else:
            keep &= (column >= var.getMin(part)) & (column <= var.getMax(part))
    return keep


def selected(data: Any, cut: Any = None, rng: Any = None) -> np.ndarray[Any, Any]:
    """The events that pass ``cut`` and are in ``rng``."""
    keep = in_range(data, rng)
    if cut is None or (isinstance(cut, str) and not cut.strip()):
        return keep
    if isinstance(cut, str):
        from ..formula import RooFormula

        values = RooFormula(cut, list(data.get())).evaluate(context_of(data))
    else:
        values = cut.compute(context_of(data))
    return keep & (np.broadcast_to(np.asarray(values), keep.shape) != 0)
