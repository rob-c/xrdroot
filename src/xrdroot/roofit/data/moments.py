"""``RooAbsData::moment``, ``mean`` and ``sigma``: a variable's moments over a dataset's events.

The sum is RooFit's compensated one, event by event, divided by the
dataset's weighted count of the events the cut and range select. A range
given skips the events *inside* it from the sum - ``allInRange`` tested the
wrong way round in RooFit - while the count still takes them: that is kept,
since it is the number RooFit prints.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..fitting.kahan import Kahan
from ..messages import ERROR, log

__all__ = ["mean", "moment", "sigma"]


def moment(data: Any, var: Any, order: float, *rest: Any) -> float:
    """``moment(var, order, [offset], [cut], [range])``: about the mean for an order above
    one when no offset is given."""
    if rest and isinstance(rest[0], (int, float)):
        return _about(data, var, float(order), float(rest[0]), *rest[1:])
    offset = moment(data, var, 1.0, 0.0, *rest) if order > 1 else 0.0
    return _about(data, var, float(order), offset, *rest)


def _about(
    data: Any, var: Any, order: float, offset: float, cut: Any = None, rng: Any = None
) -> float:
    where = f"RooDataSet::moment({data.GetName()})"
    if data.get().find(var.GetName()) is None:
        log(data, ERROR, "InputArguments", f"{where} ERROR: unknown variable: {var.GetName()}")
        return 0.0
    total = data.sumEntries(cut, rng)
    if total == 0.0:
        log(data, ERROR, "InputArguments", f"{where} WARNING: empty dataset")
        return 0.0
    keep = data.mask(cut, None)
    if rng is not None:
        keep = keep & ~data.mask(None, rng)
    values = np.asarray(data.column(var.GetName()), dtype=np.float64)[keep]
    weights = np.asarray(data.weights(), dtype=np.float64)[keep]
    terms = weights * np.power(values - offset, order)
    return Kahan().extend(terms.tolist()).total / total


def mean(data: Any, var: Any, cut: Any = None, rng: Any = None) -> float:
    return moment(data, var, 1.0, 0.0, cut, rng)


def sigma(data: Any, var: Any, cut: Any = None, rng: Any = None) -> float:
    return math.sqrt(moment(data, var, 2.0, cut, rng))
