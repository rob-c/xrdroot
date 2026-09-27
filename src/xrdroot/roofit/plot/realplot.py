"""``RooAbsReal::plotOn``: the curve itself - one per range - or an error band round it.

The last step of plotting, over the shared option list: a ``Range`` of
several names becomes one call per name, all normalised in the lot; then
the options are read (warning of any given twice) and either the curve is
sampled and added (:func:`.curves._add_curve`), or, with
``VisualizeError``, :func:`band` draws the curve and its variations and
adds the band between them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..cmdargs import RooCmdArg
from .cmdlist import CmdList
from .curves import _add_curve, _announce_plot, _range_fraction

__all__ = ["real_plot"]


def real_plot(func: Any, frame: Any, cmds: CmdList, chosen: Any = None, nset: Any = frozenset()) -> Any:
    """``RooAbsReal::plotOn(frame, argList)``."""
    ranges = cmds.find("RangeWithName")
    if ranges is not None and "," in str(ranges.value(0)):
        joint = str(ranges.value(0))
        with cmds.adding(RooCmdArg("NormRange", joint)):
            for part in joint.split(","):
                ranges.args = (part, *ranges.args[1:])
                real_plot(func, frame, cmds, chosen, nset)
        return frame
    options = cmds.process(f"RooAbsReal::plotOn({func.GetName()})")
    fit = options.get("VisualizeError")
    if fit is not None and "P" not in str(options.get("DrawOption", 0, "L")):
        from .band import band

        return band(func, frame, cmds, options)
    return _draw(func, frame, options, chosen, nset)


def _draw(func: Any, frame: Any, options: Any, chosen: Any, nset: frozenset[str]) -> Any:
    """One curve, over its range, scaled as the options say and normalised within ``NormRange``."""
    average = None
    if nset and "ProjWData" in options:
        nset, average = _data_average(func, frame, options, nset)
    if nset:
        _announce_plot(func, frame, nset, average[2] if average else ())
    scale = float(options.get("Normalization", 0, 1.0))
    low, high, post, norm_range = _extent(frame, options)
    ranged = "Range" in options or "RangeWithName" in options
    wings = "VLines" in options or not ranged  # a range drawn has no wings, unless asked
    if nset and (post or "NormRange" in options):
        scale /= _range_fraction(func, frame, nset, norm_range, [(low, high)])
    suffix = str(options.get("CurveNameSuffix", 0, "") or "")
    return _add_curve(func, frame, options, nset, scale, chosen, suffix, (low, high, wings), average)


def _data_average(func: Any, frame: Any, options: Any, nset: frozenset[str]) -> tuple[frozenset[str], Any]:
    """``ProjWData``: the observables the projection data has, averaged over its events, not integrated.

    The curve is then ``sum_i w_i f(x; y_i) / sum_i w_i`` - the density normalised
    over the rest, at each event's ``y`` - as ``RooDataWeightedAverage`` has it.
    """
    data = next(one for one in options.args("ProjWData") if hasattr(one, "numEntries"))
    held = [one.GetName() for one in data.get()]
    needed = [name for name in held if name in nset - {frame.getPlotVar().GetName()}]
    if not needed:
        return nset, None
    columns = {name: np.asarray(data.column(name), dtype=np.float64) for name in needed}
    weights = np.asarray(data.weights(), dtype=np.float64)
    return nset - frozenset(needed), (columns, weights, needed, len(needed) < len(held))


def _extent(frame: Any, options: Any) -> tuple[float, float, Any, Any]:
    """Where the curve is drawn, whether it is normalised within that range, and which range."""
    var = frame.getPlotVar()
    norm_range = options.get("NormRange")
    if "Range" in options:
        low, high = float(options.get("Range", 0)), float(options.get("Range", 1))
        return low, high, bool(options.get("Range", 2, True)), norm_range
    if "RangeWithName" in options:
        name = str(options.get("RangeWithName"))
        adjust = bool(options.get("RangeWithName", 1, True))
        return var.getMin(name), var.getMax(name), adjust, norm_range or name
    return frame.GetXmin(), frame.GetXmax(), False, norm_range


def evaluate_scale(values: Any) -> Any:
    return np.asarray(values, dtype=np.float64)
