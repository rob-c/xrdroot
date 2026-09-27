"""``RooAbsData::plotOn``: a dataset binned on a frame, as points with error bars.

The events are counted in the frame's bins - or ``Binning(...)``'s -
after ``Cut`` and ``CutRange``, and become a :class:`~.hist.RooHist`
called ``h_<data>``, its bars Poisson intervals unless the weights are not
whole numbers, when they are the square root of the summed squared
weights (``DataError`` chooses outright).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..binning import RooAbsBinning
from ..cmdargs import Commands, commands
from ..messages import INFO, log
from ..printing import g
from .curves import style
from .hist import AUTO, POISSON, SUMW2, from_counts

__all__ = ["plot_data"]


def _edges(frame: Any, options: Commands, data: Any = None) -> tuple[np.ndarray[Any, Any], bool]:
    """The bin edges, and whether the options chose them rather than the frame."""
    var = frame.getPlotVar()
    given = options.args("Binning")
    own = getattr(data, "default_binning", None)
    if not given and own is not None and own(var) is not None:
        return own(var).array(), True
    if not given:
        if not var.getBinning().isUniform():
            return var.getBinning().array(), False
        return np.linspace(frame.GetXmin(), frame.GetXmax(), frame.GetNbinsX() + 1), False
    first = given[0]
    if isinstance(first, RooAbsBinning):
        return first.array(), True
    if isinstance(first, str):
        return var.getBinning(first).array(), True
    low, high = (given[1], given[2]) if len(given) > 2 and given[1] != given[2] else (var.getMin(), var.getMax())
    return np.linspace(low, high, int(first) + 1), True


def _error_type(data: Any, options: Commands) -> int:
    etype = int(options.get("DataError", 0, AUTO))
    if etype != AUTO:
        return etype
    if data.isNonPoissonWeighted():
        log(data, INFO, "InputArguments", f"RooAbsData::plotOn({data.GetName()}) INFO: dataset has "
            "non-integer weights, auto-selecting SumW2 errors instead of Poisson errors")  # fmt: skip
        return SUMW2
    return POISSON


def _label(var: Any, edges: np.ndarray[Any, Any]) -> str:
    width = (edges[-1] - edges[0]) / (len(edges) - 1)
    unit = f" {var.getUnit()}" if var.getUnit() else ""
    return f"Events / ( {g(width)}{unit} )"


def plot_data(data: Any, frame: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``data.plotOn(frame, options...)``."""
    options = commands(args, kwargs)
    var = frame.getPlotVar()
    edges, chosen = _edges(frame, options, data)
    cut, rng = options.get("Cut"), options.get("CutRange")
    keep = data.mask(cut, rng)
    values = data.column(var.GetName())[keep]
    weights = data.weights()[keep]
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    sumw2, _ = np.histogram(values, bins=edges, weights=data.weights_squared()[keep])
    nominal = 0.0
    if chosen:
        nominal = frame.getFitRangeBinW() if frame.getFitRangeNEvt() else float(np.mean(np.diff(edges)))
    etype = _error_type(data, options)
    name = str(options.get("Name", 0, "")) or _name(data, cut, rng)
    hist = from_counts(name, edges, counts, sumw2, etype, nominal,
                       float(options.get("XErrorSize", 0, 1.0)), float(options.get("Rescale", 0, 1.0)))
    hist.y_label = _label(var, edges)
    if cut or rng:
        total = data.sumEntries()
        log(data, INFO, "Plotting", f"RooTreeData::plotOn: plotting {g(float(np.sum(counts)))} events "
            f"out of {g(total)} total events")  # fmt: skip
        hist.raw_entries = total
    hist.data_values, hist.data_weights = values, weights
    style(hist, options)
    frame.update_norm_vars(list(data.get()))
    frame.add_plotable(hist, str(options.get("DrawOption", 0, "P")),
                       bool(options.get("Invisible", 0, False)),
                       bool(options.get("RefreshNorm", 0, True)))  # fmt: skip
    return frame


def _name(data: Any, cut: Any, rng: Any) -> str:
    name = f"h_{data.GetName()}"
    if rng:
        name += f"_CutRange[{rng}]"
    if cut:
        name += f"_Cut[{cut}]"
    return name
