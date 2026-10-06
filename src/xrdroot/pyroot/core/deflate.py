"""``LabelsDeflate`` and ``LabelsInflate``: an axis of categories cut to its labels, or doubled.

A histogram filled by label on an axis that may grow doubles that axis each
time it runs out of bins, and leaves bins beyond the last label empty;
``LabelsDeflate`` keeps only the bins up to the last label, the same width
each, with their contents, errors, labels and the histogram's statistics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...booking import Binning
from ...reshaping import _rebuilt

__all__: list[str] = []


def _binning(axis: Any, nbins: int | None = None) -> Binning:
    count = axis.GetNbins() if nbins is None else nbins
    low = axis.GetXmin()
    return Binning(count, low, low + count * axis.GetBinWidth(1), np.zeros(0))


def _kept(cells: Any, shape: tuple[int, ...], along: int, count: int) -> Any:
    """The cells up to bin ``count`` along one axis, and that axis' two flows."""
    grid = np.asarray(cells).reshape(shape)
    last = shape[along] - 1
    indices = [*range(count + 1), last]
    return np.take(grid, indices, axis=along).ravel()


def _placed(cells: Any, shape: tuple[int, ...], along: int, grown: int) -> Any:
    """The cells on an axis doubled: each where it was, the overflow moved to the new end."""
    grid = np.asarray(cells).reshape(shape)
    wider = list(shape)
    wider[along] = grown + 2
    made = np.zeros(wider, dtype=grid.dtype)
    keep: list[Any] = [slice(None)] * len(shape)
    keep[along] = slice(0, shape[along] - 1)
    made[tuple(keep)] = grid[tuple(keep)]
    keep[along] = -1
    made[tuple(keep)] = np.take(grid, -1, axis=along)
    return made.ravel()


def inflated(histogram: Any, at: int) -> Any:
    """``LabelsInflate``: axis ``at`` of ``histogram`` with twice the bins, each as wide."""
    axes = histogram._axes()
    grown = 2 * axes[at].GetNbins()
    old = histogram._xrd
    made = _rebuilt(
        old, [_binning(a, grown if i == at else None) for i, a in enumerate(axes)], None
    )
    shape = tuple(a.GetNbins() + 2 for a in reversed(axes))
    along = len(axes) - 1 - at
    made._home[made._key] = _placed(old._cells(), shape, along, grown).astype(made._cells().dtype)
    if old._sumw2() is not None:
        made._core["fSumw2"] = _placed(old._sumw2(), shape, along, grown)
    _carry_labels(made, axes)
    return made


def _carry_labels(made: Any, axes: list[Any]) -> None:
    for letter, axis in zip("XYZ", axes, strict=False):
        row = made._core[f"f{letter}axis"]
        row["_labels"] = dict(axis._labels())
        row["fBits2"] = axis._row.get("fBits2", 0)


def deflated(histogram: Any, at: int) -> Any:
    """The xrdroot histogram under ``histogram`` with axis ``at`` cut to its last label."""
    axes = histogram._axes()
    labels = axes[at]._labels()
    count = max((int(bin) for bin in labels), default=0)
    if count == 0 or count >= axes[at].GetNbins():
        return None
    old = histogram._xrd
    made = _rebuilt(
        old, [_binning(a, count if i == at else None) for i, a in enumerate(axes)], None
    )
    shape = tuple(a.GetNbins() + 2 for a in reversed(axes))
    along = len(axes) - 1 - at
    made._home[made._key] = _kept(old._cells(), shape, along, count).astype(made._cells().dtype)
    if old._sumw2() is not None:
        made._core["fSumw2"] = _kept(old._sumw2(), shape, along, count)
    _carry_labels(made, axes)
    return made


def sorted_by_label(histogram: Any, order: str) -> None:
    """The labelled bins of a histogram of one axis put in order: by label, or by content."""
    axis = histogram.GetXaxis()
    labels = {int(bin): label for bin, label in axis._labels().items()}
    bins = sorted(labels)
    cells, errors = histogram._xrd._cells(), histogram._xrd._sumw2()
    if order == "a":
        chosen = sorted(bins, key=lambda bin: labels[bin])
    else:
        chosen = sorted(bins, key=lambda bin: float(cells[bin]), reverse=order == ">")
    cells[bins] = cells[chosen]
    if errors is not None:
        errors[bins] = errors[chosen]
    axis._labels().clear()
    for bin, old in zip(bins, chosen, strict=False):
        axis.SetBinLabel(bin, labels[old])
