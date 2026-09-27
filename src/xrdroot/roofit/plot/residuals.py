"""``chiSquare``, ``residHist`` and ``pullHist``: how data on a frame differ from a curve on it.

Each is RooFit's own arithmetic (``RooCurve::chiSquare``, ``RooHist::makeResidHist``): a
data point is compared with the curve's average over the point's bin - its integral over the
bin divided by the width - and the error that counts is the one on the curve's side.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fitting.kahan import Kahan
from ..messages import WARNING, log

__all__ = ["chi_square", "residuals"]


def _pair(frame: Any, curvename: Any, histname: Any) -> tuple[Any, Any]:
    """The curve and the data compared: those named, or the last drawn of each."""
    from .curve import RooCurve
    from .hist import RooHist

    return frame.findObject(curvename, RooCurve), frame.findObject(histname, RooHist)


def _inside(curve: Any, hist: Any) -> list[int]:
    """The data points within the curve's extent."""
    low, high = float(curve.x[0]), float(curve.x[-1])
    return [i for i, x in enumerate(hist.x) if low <= x <= high]


def chi_square(frame: Any, curvename: Any, histname: Any, nFitParam: int) -> float:
    """``RooPlot::chiSquare``: the pulls' squares summed, per degree of freedom."""
    curve, hist = _pair(frame, curvename, histname)
    ylow, yhigh = hist.errors()
    xlow, xhigh = np.asarray(hist.members["fEXlow"]), np.asarray(hist.members["fEXhigh"])
    total, bins = Kahan(), 0
    for i in _inside(curve, hist):
        x, y = float(hist.x[i]), float(hist.y[i])
        mean = curve.average(x - xlow[i], x + xhigh[i])
        if y != 0:
            pull = (y - mean) / (ylow[i] if y > mean else yhigh[i])
            total.add(pull * pull)
            bins += 1
    return float(total.total) / (bins - nFitParam)


def _same_curves(frame: Any, curvename: Any) -> list[Any]:
    """The curve named - or the last drawn - and every other of its name: a multi-range fit's."""
    from .curve import RooCurve

    found: list[Any] = []
    for obj, _, _ in reversed(frame.items):
        if isinstance(obj, RooCurve) and (not curvename or obj.GetName() == curvename):
            curvename = obj.GetName()
            found.append(obj)
    return found


def residuals(frame: Any, histname: Any, curvename: Any, normalize: bool, useAverage: bool) -> Any:
    """``RooPlot::residHist``: each point less the curve - in its errors, for a pull."""
    from .hist import RooHist

    curves = _same_curves(frame, curvename)
    _, hist = _pair(frame, None, histname)
    rows: list[tuple[float, float, float, float]] = []
    for curve in curves:
        rows += _rows(hist, curve, normalize, useAverage)
    kind = ("pull", "Pull") if normalize else ("resid", "Residual")
    xs, ys, lows, highs = (list(one) for one in zip(*rows)) if rows else ([], [], [], [])
    zeros = [0.0] * len(xs)
    made = RooHist(
        f"{kind[0]}_{hist.GetName()}_{curves[0].GetName()}",
        f"{kind[1]} of {hist.GetTitle()} and {curves[0].GetTitle()}",
        xs, ys, zeros, zeros, lows, highs,
    )  # fmt: skip
    made.nominal_width = hist.nominal_width
    made.entries = float(np.sum(ys))
    return made


def _rows(hist: Any, curve: Any, normalize: bool, useAverage: bool) -> list[Any]:
    """``fillResidHist``: each point inside the curve less the curve's average over its bin."""
    ylow, yhigh = hist.errors()
    xlow, xhigh = np.asarray(hist.members["fEXlow"]), np.asarray(hist.members["fEXhigh"])
    half = 0.5 * hist.nominal_width
    rows = []
    for i in _inside(curve, hist):
        x, point = float(hist.x[i]), float(hist.y[i])
        if useAverage:
            low, high = float(xlow[i]) or half, float(xhigh[i]) or half
            found = point - curve.average(x - low, x + high)
        else:
            found = point - float(curve.interpolate(x))
        dyl, dyh = float(ylow[i]), float(yhigh[i])
        if normalize:
            found, dyl, dyh = _pull(hist, i, found, dyl, dyh)
        rows.append((x, found, dyl, dyh))
    return rows


def _pull(hist: Any, i: int, found: float, low: float, high: float) -> tuple[float, float, float]:
    """A residual in units of the error on its side - or zero, with a warning, for none."""
    norm = low if found > 0 else high
    if norm == 0.0:
        log(hist, WARNING, "Plotting", f"RooHist::makeResisHist({hist.GetName()}) WARNING: point "
            f"{i} has zero error, setting residual to zero")  # fmt: skip
        return 0.0, 0.0, 0.0
    return found / norm, low / norm, high / norm
