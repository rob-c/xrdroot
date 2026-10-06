"""``TVirtualFitter``: the last fit, as ROOT's old fitting interface reaches it, and its bands.

ROOT keeps the fitter of the most recent ``Fit`` behind
``TVirtualFitter::GetFitter()``; what tutorials still ask of it is the choice
of minimiser, which xrdroot's fits always make as Minuit, and
``GetConfidenceIntervals``, the band a fitted function's parameter errors
give it - :func:`confidence_intervals`, which ``TFitResult`` offers too.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .distributions import normal_quantile, student_quantile
from .messages import message
from .objects import TNamed

__all__ = ["TVirtualFitter", "TBackCompFitter"]

#: The fit ``GetFitter`` hands back: the result of the latest ``Fit``.
LATEST: dict[str, Any] = {"result": None}


def correction(result: Any, cl: float, norm: bool = True) -> float:
    """The band's width per unit of error: Student's quantile scaled by the fit's chi2/ndf.

    Without a chi2 to scale by, as for a likelihood fit, it is the normal quantile.
    """
    if norm and result.chi2 > 0 and result.ndf > 0:
        scale = np.sqrt(result.chi2 / result.ndf)
        return float(student_quantile(0.5 + cl / 2, result.ndf) * scale)
    return float(normal_quantile(0.5 + cl / 2, 1.0))


def _gradients(function: Any, points: np.ndarray[Any, Any], params: np.ndarray[Any, Any]) -> Any:
    """The function's derivative in each parameter at each point, by a four-point difference."""
    columns = []
    for index, value in enumerate(params):
        step = 1e-3 * max(abs(value), 1.0)
        values = []
        for shift in (-2, -1, 1, 2):
            moved = params.copy()
            moved[index] = value + shift * step
            values.append(np.asarray(function.evaluate(points, moved), dtype=np.float64))
        columns.append((values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step))
    return np.stack(columns, axis=-1)


def band(result: Any, points: Any, cl: float = 0.95, norm: bool = False) -> Any:
    """The fitted function's value at each point, and the half-width of its band there.

    ``TVirtualFitter`` asks for the band unscaled by chi2/ndf, as ROOT's does.
    """
    function = result.function
    points = np.asarray(points, dtype=np.float64)
    params = np.asarray(result.parameters, dtype=np.float64)
    grad = _gradients(function, points, params)
    spread = np.einsum("ni,ij,nj->n", grad, result.covariance, grad)
    values = np.asarray(function.evaluate(points, params), dtype=np.float64)
    return values, np.sqrt(np.maximum(spread, 0.0)) * correction(result, cl, norm)


def confidence_intervals(result: Any, obj: Any, cl: float = 0.95) -> None:
    """``GetConfidenceIntervals(obj, cl)``: fill a ``TGraphErrors`` or a histogram with the band.

    A graph's points keep their ``x`` and take the function's value and the
    band's half-width as ``y`` and its error; a histogram's bins, at their centres, do too.
    """
    if hasattr(obj, "SetBinContent"):
        _into_histogram(result, obj, cl)
    elif obj.ClassName() == "TGraphErrors":
        xs = np.asarray(obj.GetX(), dtype=np.float64)[: obj.GetN()]
        values, halves = band(result, xs, cl)
        for i, (x, y, half) in enumerate(zip(xs, values, halves, strict=False)):
            obj.SetPoint(i, x, y)
            obj.SetPointError(i, 0.0, half)
    else:
        message("Error", "GetConfidenceIntervals", "This object type is not supported")


def _into_histogram(result: Any, h: Any, cl: float) -> None:
    """Each bin, at its centre, the function's value and the band's half-width as its error."""
    axes = [h.GetXaxis(), h.GetYaxis(), h.GetZaxis()][: result.function.dimensions]
    bins = [np.arange(1, axis.GetNbins() + 1) for axis in axes]
    cells = [grid.ravel() for grid in np.meshgrid(*bins, indexing="ij")]
    centres = [
        np.array([axis.GetBinCenter(int(b)) for b in column])
        for axis, column in zip(axes, cells, strict=False)
    ]
    values, halves = band(result, centres[0] if len(axes) == 1 else np.column_stack(centres), cl)
    for at, (value, half) in enumerate(zip(values, halves, strict=False)):
        bin = h.GetBin(*(int(column[at]) for column in cells))
        h.SetBinContent(bin, value)
        h.SetBinError(bin, half)


class TVirtualFitter(TNamed):
    """``TVirtualFitter``: the fitter behind the latest ``Fit``, and the minimiser to use."""

    CLASS_TITLE = "Abstract interface for fitting"
    _default = "Minuit"

    @staticmethod
    def GetFitter() -> Any:
        """``GetFitter``: the latest fit's fitter - ``None`` before any fit."""
        return TBackCompFitter(LATEST["result"]) if LATEST["result"] is not None else None

    @staticmethod
    def Fitter(obj: Any = None, maxpar: int = 25) -> Any:
        """``TVirtualFitter::Fitter(obj, maxpar)``: the fitter, as ``GetFitter`` - where ROOT
        would also make room for ``maxpar`` parameters, which a fit here never runs short of."""
        return TVirtualFitter.GetFitter()

    @staticmethod
    def SetDefaultFitter(name: Any = "") -> None:
        """``SetDefaultFitter``: noted; every xrdroot fit minimises with Minuit's Migrad."""
        TVirtualFitter._default = str(name) or "Minuit"

    @staticmethod
    def GetDefaultFitter() -> str:
        return TVirtualFitter._default


class TBackCompFitter(TVirtualFitter):
    """``TBackCompFitter``: what ``GetFitter`` hands back, over the latest fit's result."""

    def __init__(self, result: Any) -> None:
        super().__init__("BCFitter", "BackCompFitter")
        self._result = result

    def GetConfidenceIntervals(self, obj: Any, cl: float = 0.95) -> None:
        confidence_intervals(self._result, obj, cl)

    def GetNumberTotalParameters(self) -> int:
        return len(self._result.parameters)

    def GetParameter(self, i: int) -> float:
        return float(self._result.parameters[int(i)])

    def GetParError(self, i: int) -> float:
        return float(self._result.errors[int(i)])

    def GetCovarianceMatrixElement(self, i: int, j: int) -> float:
        return float(self._result.covariance[int(i), int(j)])
