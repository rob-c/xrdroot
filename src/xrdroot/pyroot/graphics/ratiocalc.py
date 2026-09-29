"""What a ``TRatioPlot`` draws in its lower pad, worked out as ``TRatioPlot::BuildLowerPlot`` does.

Two histograms are divided - by ``TGraphAsymmErrors::Divide``, its option
passed on (``pois`` unless told otherwise), or with ``divsym`` by
``TH1::Divide`` into symmetric errors - or subtracted, ``diff``, or
subtracted and each difference divided by its bin's error, ``diffsig``.
One fitted histogram gives the fit's residuals, each over its bin's error
(``errasym`` the error on the side the function lies, ``errfunc`` the
square root of the function), with the fit's confidence bands at one and
two sigma - ``TVirtualFitter::GetConfidenceIntervals`` of the latest fit, or
of the ``TFitResult`` given - each over the same error. Each mode has the
reference lines ROOT draws dashed across the lower pad.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.graphs import TGraphAsymmErrors, TGraphErrors

__all__ = ["DIFFERENCE", "DIVIDE", "DIVIDE_HIST", "FIT_RESIDUAL", "SIGNIFICANCE", "build"]

#: ``TRatioPlot::CalculationMode``: how the lower plot is worked out.
DIVIDE, DIVIDE_HIST, DIFFERENCE, FIT_RESIDUAL, SIGNIFICANCE = 1, 2, 3, 4, 5
#: ``TRatioPlot::ErrorMode``: which error a residual or significance is over.
SYMMETRIC, ASYMMETRIC, FUNCTION = 1, 2, 3
#: Where each mode's reference lines are: about one for a ratio, at zero for a
#: difference - ROOT asks for three there and has one, so draws one - and at
#: zero and one either side for a significance or residual.
GRIDLINES = {
    DIVIDE: [0.7, 1.0, 1.3],
    DIVIDE_HIST: [0.7, 1.0, 1.3],
    DIFFERENCE: [0.0],
    SIGNIFICANCE: [1.0, 0.0, -1.0],
    FIT_RESIDUAL: [1.0, 0.0, -1.0],
}


def _clone(h: Any) -> Any:
    """A copy of ``h`` kept in no directory, as ROOT's temporaries are deleted after use."""
    made = h.Clone()
    made.SetDirectory(None)
    return made


def _divided(rp: Any) -> Any:
    """``TGraphAsymmErrors::Divide`` of the two histograms, each scaled first."""
    upper, lower = _clone(rp._h1), _clone(rp._h2)
    upper.Scale(rp._c1)
    lower.Scale(rp._c2)
    graph = TGraphAsymmErrors()
    graph.Divide(upper, lower, rp._option)
    return graph


def _subtracted(rp: Any) -> Any:
    """``h1*c1 - h2*c2``, a point per bin with its errors."""
    made = _clone(rp._h1)
    made.Reset()
    made.Add(rp._h1, rp._h2, rp._c1, -1 * rp._c2)
    return TGraphErrors(made)


def _divided_histograms(rp: Any) -> Any:
    """``TH1::Divide(h1, h2, c1, c2, option)``, a point per bin with its symmetric errors."""
    made = _clone(rp._h1)
    made.Reset()
    made.Divide(rp._h1, rp._h2, rp._c1, rp._c2, rp._option)
    return TGraphErrors(made)


def _error(rp: Any, i: int, above: bool, function: float | None) -> float:
    """The error a bin's difference is over: its own, the side it is on, or the function's."""
    if rp._error_mode == ASYMMETRIC:
        return float(rp._h1.GetBinErrorLow(i) if above else rp._h1.GetBinErrorUp(i))
    if rp._error_mode == SYMMETRIC:
        return float(rp._h1.GetBinError(i))
    if rp._error_mode == FUNCTION and function is not None:
        with np.errstate(invalid="ignore"):  # below zero: a NaN, as C's sqrt gives
            return float(np.sqrt(function))
    rp.Warning("BuildLowerPlot", "error mode is invalid")
    return 0.0


def _point(graph: Any, at: int, h: Any, i: int, value: float) -> None:
    """A point at bin ``i``'s centre, half a bin wide either way and half a unit high."""
    half = h.GetBinWidth(i) / 2.0
    graph.SetPoint(at, h.GetBinCenter(i), value)
    graph.SetPointError(at, half, half, 0.5, 0.5)


def _significances(rp: Any) -> Any:
    """Each bin's ``(h1 - h2) / error``, the underflow's included as ROOT's loop includes it."""
    graph = TGraphAsymmErrors()
    at = 0
    for i in range(rp._h1.GetNbinsX() + 1):
        val, val2 = rp._h1.GetBinContent(i), rp._h2.GetBinContent(i)
        error = _error(rp, i, val - val2 > 0, None)
        if error != 0.0:
            _point(graph, at, rp._h1, i, (val - val2) / error)
            at += 1
    return graph


def fit_function(h: Any) -> Any:
    """The function hung first on ``h``, if it is a ``TF1``: what its residuals are of."""
    functions = h.GetListOfFunctions()
    first = functions.At(0) if functions.GetSize() else None
    return first if first is not None and first.InheritsFrom("TF1") else None


def _bands(rp: Any, xs: np.ndarray[Any, Any]) -> tuple[Any, Any]:
    """The fit's band half-widths at the bin centres, at the two confidence levels."""
    from ..core.fitters import LATEST, band

    held = rp._fit_result
    result = getattr(held.Get() if hasattr(held, "Get") else held, "_xrd", None)
    result = result if result is not None else LATEST["result"]
    if result is None:
        return np.zeros(len(xs)), np.zeros(len(xs))
    return band(result, xs, rp._cl1)[1], band(result, xs, rp._cl2)[1]


def _residuals(rp: Any, function: Any) -> Any:
    """Each bin's ``(h1 - f) / error``, and the bands over the same error in ``rp``'s graphs."""
    h = rp._h1
    nbins = h.GetNbinsX()
    xs = np.array([h.GetBinCenter(i) for i in range(1, nbins + 1)])
    ci1, ci2 = _bands(rp, xs)
    graph = TGraphAsymmErrors()
    at = 0
    for i in range(1, nbins + 1):
        val, x = h.GetBinContent(i), h.GetBinCenter(i)
        fx = float(function.Eval(x))
        error = _error(rp, i, val - fx > 0, fx)
        if error != 0.0:
            _point(graph, at, h, i, (val - fx) / error)
            for band_graph, halves in ((rp._ci1, ci1), (rp._ci2, ci2)):
                band_graph.SetPoint(at, x, 0)
                band_graph.SetPointError(at, x, float(halves[i - 1]) / error)
            at += 1
    return graph


def build(rp: Any) -> bool:
    """``BuildLowerPlot``: the lower pad's graph, its bands and its reference lines, in ``rp``."""
    rp._ci1 = rp._ci1 if rp._ci1 is not None else TGraphErrors()
    rp._ci2 = rp._ci2 if rp._ci2 is not None else TGraphErrors()
    rp._gridline_positions = list(GRIDLINES[rp._mode])
    if rp._mode == FIT_RESIDUAL:
        function = fit_function(rp._h1)
        if function is None:
            rp.Error("BuildLowerPlot", "h1 does not have a fit function")
            return False
        graph = _residuals(rp, function)
    else:
        makers = {DIVIDE: _divided, DIFFERENCE: _subtracted, SIGNIFICANCE: _significances,
                  DIVIDE_HIST: _divided_histograms}  # fmt: skip
        graph = makers[rp._mode](rp)
    for made in (graph, rp._ci1, rp._ci2):
        made.SetTitle("")
    rp._ratio_graph = graph
    return True
