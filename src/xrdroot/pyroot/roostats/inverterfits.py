"""ROOT graph/function adapters for the inverter's existing fit algorithms.

The statistical engine owns the search and interval state. These adapters keep
the same TF1 and TGraph fit calls, options, starting values and error rules while
leaving their public ROOT object construction in the wrapper layer.
"""

from __future__ import annotations

from typing import Any

from ...roofit.messages import WARNING, log
from ...roostats.inverterlimits import closest_point_index, sorted_order
from ...roostats.inverterscan import _log


def error_graph(result: Any, xmin: float, xmax: float) -> tuple[Any, int]:
    """The points inside ``[xmin, xmax]`` with errors, as a ``TGraphErrors`` - and how many
    points were inside at all."""
    from ..core.graphs import TGraphErrors

    graph, inside = TGraphErrors(), 0
    for i in sorted_order(result._x):
        if xmin < xmax and xmin <= result.GetXValue(i) <= xmax:
            inside += 1
            if result.GetYError(i) > 1e-6:
                at = graph.GetN()
                graph.SetPoint(at, result.GetXValue(i), result.GetYValue(i))
                graph.SetPointError(at, 0.0, result.GetYError(i))
    return graph, inside


def fitted_error(result: Any, graph: Any, target: float, lower: bool, limit: float,
                  given: tuple[float, float]) -> float:  # fmt: skip
    from ..core.funcs import TF1

    xs = sorted(result._x)
    low, high = given if given[0] < given[1] else (xs[0], xs[-1])
    fct = TF1("fct", "exp([0] * (x - [2] ) + [1] * (x-[2])**2)", low, high)
    scale = high - low
    if lower:
        fct.SetParameters(2.0 / scale, 0.1 / scale, graph.GetX()[0])
        fct.SetParLimits(0, 0, 100.0 / scale)
        fct.SetParLimits(1, 0, 10.0 / scale)
    else:
        fct.SetParameters(-2.0 / scale, -0.1 / scale)
        fct.SetParLimits(0, -100.0 / scale, 0)
        fct.SetParLimits(1, -100.0 / scale, 0)
    if graph.GetN() < 3:
        fct.FixParameter(1, 0.0)
    status = int(graph.Fit(fct, "Q EX0"))
    index = closest_point_index(result, target, 1, limit)
    if status != 0:
        log(result, WARNING, "Eval", "HypoTestInverterResult::CalculateEstimatedError - cannot "
            f"estimate  the {'lower' if lower else 'upper'} limit error ")  # fmt: skip
        return 0.0
    if result.GetYError(index) <= 0:
        return 0.0
    slope = fct.Derivative(result.GetXValue(index))
    return float(min(abs(result.GetYError(index) / slope), high - low))


def exponential(search: Any, low: float, high: float) -> Any:
    """``expoFit``: through the target at the limit, of the slope between the ends."""
    from ..core.funcs import TF1

    expo = TF1("expoFit", "[0]*exp([1]*(x-[2]))", low, high)
    expo.FixParameter(0, search.target)
    slope = _log(search.cls_max[0] / search.cls_min[0]) / (search.r_max - search.r_min)
    expo.SetParameter(1, slope)
    expo.SetParameter(2, search.limit)
    return expo
