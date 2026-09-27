"""What a curve on a frame projects, slices and averages: ``Slice`` and ``ProjWData``.

A frame knows the variables of the data drawn on it; a density's curve is
the density of the frame's variable with each of the others either
*integrated out* (projected), *held* at one value (a slice - ``Slice(tag,
"B0")`` sets the category to that state), or *averaged over a dataset*
(``ProjWData(data)``: the mean, over the dataset's events, of the density
of the frame's variable given theirs). ``RooAbsReal::plotOn`` works this out
in ``makeProjectionSet`` and says what it decided, and so does :func:`view`:
the projected variables in the frame's order, the slice in the density's.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..cmdargs import Commands
from ..collections import as_list
from ..messages import INFO, log

__all__ = ["View", "sliced", "view"]


class View:
    """The variables a curve integrates over, holds fixed and averages over data - and the data."""

    def __init__(self) -> None:
        self.projected: list[str] = []
        self.sliced: list[str] = []
        self.averaged: list[str] = []
        self.data: Any = None
        self.binned = False

    def average(self, value: Any, ctx: dict[str, Any]) -> Any:
        """``RooDataWeightedAverage``: ``value`` averaged over the projection data's events."""
        source = self.data.binnedClone() if self.binned else self.data
        weights = np.asarray(source.weights(), dtype=np.float64)
        keep = weights != 0
        inner = dict(ctx)
        for name, column in ((one, np.asarray(source.column(one))[keep]) for one in self.averaged):
            inner[name] = column[None, :]
        inner.update(
            {k: v[:, None] for k, v in ctx.items() if isinstance(v, np.ndarray) and v.ndim}
        )
        found = np.broadcast_to(
            value(inner), np.broadcast_shapes(*(np.shape(v) for v in inner.values()))
        )
        return np.sum(found * weights[keep], axis=-1) / np.sum(weights)


def sliced(options: Commands) -> list[tuple[Any, str]]:
    """The ``Slice`` options' categories and states - each category set to its state, as ROOT sets it."""
    found: list[tuple[Any, str]] = []
    for one in options.every("Slice") + options.every("SliceCat"):
        pairs = (
            one.value(0).items()
            if isinstance(one.value(0), dict)
            else [(one.value(0), one.value(1))]
        )
        for category, label in pairs:
            category.setLabel(str(label))
            found.append((category, str(label)))
    return found


def _projection_data(options: Commands) -> tuple[Any, bool]:
    args = options.args("ProjWData")
    if not args:
        return None, False
    if hasattr(args[0], "numEntries"):
        return args[0], bool(args[1]) if len(args) > 1 else False
    return args[1], bool(args[2]) if len(args) > 2 else False


def _projected(pdf: Any, frame: Any, held: set[str]) -> list[str]:
    """``makeProjectionSet``: the frame's variables the density depends on, but its own and ``held``."""
    plot_var, deps = frame.getPlotVar().GetName(), pdf.dependents()
    names = [one.GetName() for one in frame.norm_vars or ()]
    return [one for one in names if one != plot_var and one in deps and one not in held]


def _slice_set(pdf: Any, frame: Any, projected: list[str]) -> list[str]:
    """The frame's variables the density depends on, neither drawn nor projected - in its order."""
    frame_vars = {one.GetName() for one in frame.norm_vars or ()} - {frame.getPlotVar().GetName()}
    return [one.GetName() for one in pdf.leaves() if one.GetName() in frame_vars - set(projected)]


def view(pdf: Any, frame: Any, options: Commands, function: str = "plotOn") -> View:
    """``makeProjectionSet`` and the rest of ``RooAbsReal::plotOn``'s preprocessing, with its messages."""
    made = View()
    made.data, made.binned = _projection_data(options)
    projected = _projected(pdf, frame, {category.GetName() for category, _ in sliced(options)})
    made.sliced = _slice_set(pdf, frame, projected)
    if made.sliced:
        log(pdf, INFO, "Plotting", f"RooAbsReal::{function}({pdf.GetName()}) plot on "
            f"{frame.getPlotVar().GetName()} represents a slice in ({','.join(made.sliced)})")  # fmt: skip
    data = as_list(made.data.get()) if made.data is not None else []
    data_vars = {one.GetName() for one in data}
    made.averaged = [one for one in projected if one in data_vars]
    made.projected = [one for one in projected if one not in data_vars]
    return made


def announce_average(pdf: Any, frame: Any, made: View) -> None:
    if made.averaged:
        log(pdf, INFO, "Plotting", f"RooAbsReal::plotOn({pdf.GetName()}) plot on "
            f"{frame.getPlotVar().GetName()} averages using data variables "
            f"({','.join(made.averaged)})")  # fmt: skip
