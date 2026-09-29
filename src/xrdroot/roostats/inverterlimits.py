"""How a ``HypoTestInverterResult`` finds its limits, observed and expected.

The points, sorted in the scanned parameter, are joined by straight lines
- ``TGraph::Eval`` - and MathCore's root finder finds where that curve
crosses the target, one less the confidence level; a limit found from the
whole range is refined within the part of it that crosses. The expected
limits invert, point by point, the quantiles of each point's expected
p-values.
"""

from __future__ import annotations

import bisect
import math
from typing import Any

from ..roofit.messages import ERROR, WARNING, log

__all__ = ["closest_point_index", "find_interpolated_limit", "graph_eval", "graph_x"]


def sorted_order(xs: list[float]) -> list[int]:
    """``TMath::SortItr``, ascending: the indices of ``xs`` in order."""
    return sorted(range(len(xs)), key=lambda i: xs[i])


def _neighbours(xs: list[float], x: float) -> tuple[int, int]:
    """``TGraph::Eval``'s two points for ``x``: those either side - or the two nearest, beyond
    either end."""
    below = sorted((i for i in range(len(xs)) if xs[i] < x), key=lambda i: -xs[i])
    above = sorted((i for i in range(len(xs)) if xs[i] > x), key=lambda i: xs[i])
    if not above:
        return below[1], below[0]
    if not below:
        return above[0], above[1]
    return below[0], above[0]


def graph_eval(xs: list[float], ys: list[float], x: float) -> float:
    """``TGraph::Eval``, linear: the line through the neighbours of ``x`` - or the two nearest at
    either end, extrapolated."""
    if len(xs) < 2:
        return ys[0] if xs else 0.0
    if x in xs:
        return ys[xs.index(x)]
    low, up = _neighbours(xs, x)
    if xs[low] == xs[up]:
        return ys[low]
    return ys[up] + (x - xs[up]) * (ys[low] - ys[up]) / (xs[low] - xs[up])


def _variable_range(result: Any) -> tuple[float, float]:
    var = next(iter(result._parameters), None)
    return (var.getMin(), var.getMax()) if var is not None else (-math.inf, math.inf)


def graph_x(result: Any, xs: list[float], ys: list[float], y0: float, low: bool,
            xmin: float = 1.0, xmax: float = 0.0) -> tuple[float, float, float]:  # fmt: skip
    """``GetGraphX``: where the curve crosses ``y0`` - and the range searched."""
    from ..numerics.rootfinder import brent_root_finder

    n = len(xs)
    trivial = _no_crossing(result, ys, y0, low)
    if trivial is not None:
        return trivial, xmin, xmax
    varmin, varmax = _variable_range(result)
    given = xmin < xmax
    if not given:
        xmin, xmax = _whole_range(xs, ys, y0, low, varmin, varmax)
    if result._interpolation == result.kSpline:
        from ..errors import UnsupportedFeatureError

        raise UnsupportedFeatureError("HypoTestInverterResult interpolates its points linearly "
                                      "here; the spline option is not here yet")  # fmt: skip
    found, limit = brent_root_finder(lambda x: graph_eval(xs, ys, x) - y0, xmin, xmax,
                                     max(2 * n, 100), 100, 1e-16, 1e-6)  # fmt: skip
    if not found:
        return _failed(result, xs, ys, y0, xmin, xmax), xmin, xmax
    if not given:
        limit = _refined(result, xs, ys, y0, low, limit)
    return limit, xmin, xmax


def _no_crossing(result: Any, ys: list[float], y0: float, low: bool) -> Any:
    """The answer without a search: too few points, or a curve all on one side of ``y0``."""
    if len(ys) < 2:
        log(result, ERROR, "Eval", "HypoTestInverterResult::GetGraphX - need at least 2 points for "
            f"interpolation (n={len(ys)})")  # fmt: skip
        return ys[0] if ys else 0.0
    varmin, varmax = _variable_range(result)
    if max(ys) < y0:
        return varmax if low else varmin
    if min(ys) > y0:
        return varmin if low else varmax
    return None


def _whole_range(xs: list[float], ys: list[float], y0: float, low: bool, varmin: float,
                 varmax: float) -> tuple[float, float]:  # fmt: skip
    """No range given: the points', out to the parameter's ends where the curve must be
    extrapolated."""
    xmin, xmax = xs[0], xs[-1]
    if ys[0] > y0 and low:  # RooStats' test of the curve below the target too: never so here
        xmin = varmin
    if ys[-1] > y0 and not low:
        xmax = varmax
    return xmin, xmax


def _failed(result: Any, xs: list[float], ys: list[float], y0: float, xmin: float,
            xmax: float) -> float:  # fmt: skip
    from ..roofit.printing import g

    log(result, ERROR, "Eval", f"HypoTestInverterResult - interpolation failed for interval "
        f"[{g(xmin)},{g(xmax)} ]  g(xmin,xmax) ={g(graph_eval(xs, ys, xmin))},"
        f"{g(graph_eval(xs, ys, xmax))} target={g(y0)} return inf\nOne may try to clean up "
        "invalid points using HypoTestInverterResult::ExclusionCleanup().")  # fmt: skip
    return math.inf


def _refined(result: Any, xs: list[float], ys: list[float], y0: float, low: bool,
             limit: float) -> float:  # fmt: skip
    """The root again, within the points beyond the one found: where the curve crosses more
    than once, the crossing an upper limit is. RooStats also looks below a lower limit, for
    a crossing the grid found no earlier one of - which a grid from the lower end cannot miss."""
    n = len(xs)
    index = bisect.bisect_right(xs, limit) - 1  # TMath::BinarySearch
    if not low and index < n - 2 and (ys[-1] - y0) * (ys[index + 1] - y0) < 0:
        return graph_x(result, xs, ys, y0, low, xs[index + 1], xs[-1])[0]
    return limit


def points(result: Any) -> tuple[list[float], list[float], list[int]]:
    """The scan's points in order of the parameter: its values, their ``y``, their indices."""
    order = sorted_order(result._x)
    return [result.GetXValue(i) for i in order], [result.GetYValue(i) for i in order], order


def find_interpolated_limit(result: Any, target: float, low: bool, xmin: float = 1.0,
                            xmax: float = 0.0) -> float:  # fmt: skip
    """``FindInterpolatedLimit``: the limit - lower or upper - where the curve crosses ``target``,
    kept on the result; the side searched chosen by where the curve peaks, if no range is given."""
    from ..roofit.printing import g

    varmin, varmax = _variable_range(result)
    if result.ArraySize() < 2:
        log(result, WARNING, "Eval", "HypoTestInverterResult::FindInterpolatedLimit - not enough "
            f"points to get the inverted interval - return {g(xmin if low else xmax)}")  # fmt: skip
        result._lower, result._upper = varmin, varmax
        return varmin if low else varmax
    xs, ys, _ = points(result)
    if xmin >= xmax:
        chosen = _search_range(result, xs, ys, target, low, varmin, varmax)
        if chosen is None:
            return float(result._lower if low else result._upper)
        low, xmin, xmax = chosen
    limit, _, _ = graph_x(result, xs, ys, target, low, xmin, xmax)
    if low:
        result._lower = limit
    else:
        result._upper = limit
    estimated_error(result, target, low, xmin, xmax)
    return float(result._lower if low else result._upper)


def _search_range(result: Any, xs: list[float], ys: list[float], target: float, low: bool,
                  varmin: float, varmax: float) -> Any:  # fmt: skip
    """Where to look: below the curve's peak for a lower limit, above it for an upper - or,
    the peak at an end, the parameter's end - and which side, if the peak is below target."""
    n = len(xs)
    peak = max(range(n), key=lambda i: (ys[i], -i))
    if ys[peak] > target:
        if low and peak == 0:
            result._lower = varmin
            return None
        if not low and peak == n - 1:
            result._upper = varmax
            return None
        if low:
            return low, (xs[0] if ys[0] <= target else varmin), xs[peak]
        return low, xs[peak], (xs[-1] if ys[-1] <= target else varmax)
    if peak <= (n - 1) // 2:
        result._lower = varmin
        return False, 1.0, 0.0
    result._upper = varmax
    return True, 1.0, 0.0


def _most_precise(result: Any, target: float) -> int:
    """Mode 0: the point of smallest error within three errors of ``target`` - or the nearest."""
    best, closest, smallest, nearest = -1, -1, 2.0, 2.0
    for i in range(result.ArraySize()):
        dist, error = abs(result.GetYValue(i) - target), result.GetYError(i)
        if dist < 3 * error and error < smallest:
            smallest, best = error, i
        if dist < nearest:
            nearest, closest = dist, i
    return best if best >= 0 else closest


def closest_point_index(result: Any, target: float, mode: int = 0, xtarget: float = 0.0) -> int:
    """``FindClosestPointIndex``: mode 0, the most precise point within three errors of
    ``target`` - or the nearest; else the points about ``xtarget``: 2 the lower, 3 the higher,
    1 the one nearer ``target``."""
    if mode == 0:
        return _most_precise(result, target)
    xs, _, order = points(result)
    first = bisect.bisect_right(xs, xtarget) - 1
    if first < 0 or first >= len(xs) - 1:
        return order[0] if first < 0 else order[-1]
    one, two = order[first], order[first + 1]
    if mode in (2, 3):
        lower = result.GetXValue(one) < result.GetXValue(two)
        return one if lower == (mode == 2) else two
    near = abs(result.GetYValue(one) - target) <= abs(result.GetYValue(two) - target)
    return one if near else two


def _error_graph(result: Any, xmin: float, xmax: float) -> tuple[Any, int]:
    """The points inside ``[xmin, xmax]`` with errors, as a ``TGraphErrors`` - and how many
    points were inside at all."""
    from ..pyroot.core.graphs import TGraphErrors

    graph, inside = TGraphErrors(), 0
    for i in sorted_order(result._x):
        if xmin < xmax and xmin <= result.GetXValue(i) <= xmax:
            inside += 1
            if result.GetYError(i) > 1e-6:
                at = graph.GetN()
                graph.SetPoint(at, result.GetXValue(i), result.GetYValue(i))
                graph.SetPointError(at, 0.0, result.GetYError(i))
    return graph, inside


def estimated_error(result: Any, target: float, lower: bool, xmin: float = 1.0,
                    xmax: float = 0.0) -> float:  # fmt: skip
    """``CalculateEstimatedError``: the error of the point nearest the limit over the slope there
    of an exponential of a parabola fitted to the points' ``y`` - kept on the result."""
    if result.ArraySize() < 2:
        log(result, WARNING, "Eval", "HypoTestInverterResult::CalculateEstimateError"
            + ("Empty result " if not result.ArraySize() else " only  points - return its error"))
        return result.GetYError(0) if result.ArraySize() else 0.0
    if result.GetNullTestStatDist(0) is None:
        return 0.0
    graph = _valid_graph(result, lower, xmin, xmax)
    if graph is None:
        return 0.0
    limit = result._lower if lower else result._upper
    if math.isnan(limit):
        return 0.0
    error = _fitted_error(result, graph, target, lower, limit, (xmin, xmax))
    result._errors[0 if lower else 1] = error
    return error


def _valid_graph(result: Any, lower: bool, xmin: float, xmax: float) -> Any:
    """The points to fit - ``None``, said if there were points but none with errors, if fewer
    than two."""
    graph, inside = _error_graph(result, xmin, xmax)
    if graph.GetN() >= 2:
        return graph
    if inside >= 2:
        log(result, WARNING, "Eval", "HypoTestInverterResult::CalculateEstimatedError - no valid "
            f"points - cannot estimate  the {'lower' if lower else 'upper'} limit error ")
    return None


def _fitted_error(result: Any, graph: Any, target: float, lower: bool, limit: float,
                  given: tuple[float, float]) -> float:  # fmt: skip
    from ..pyroot.core.funcs import TF1

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


#: ``fgAsymptoticMaxSigma``: the asymptotic expected p-values from -5 to 5 sigma, a sigma apart.
MAX_SIGMA = 5.0
NUM_POINTS = 11


def quantile(values: list[float], p: float, kind: int = 7) -> float:
    """``TMath::Quantiles``: of type 7, between the order statistics about ``(n - 1) p``; of
    type 1, the inverse of the empirical distribution function."""
    ordered = sorted(values)
    n = len(ordered)
    eps = 4 * 2.220446049250313e-16
    if kind == 1:
        nppm = n * p
        j = math.floor(nppm + eps)
        gamma = 1.0 if (nppm - j) > j * 2.220446049250313e-16 else 0.0
    else:
        nppm = 1.0 + p * (n + 1 - 2.0)
        j = math.floor(nppm + eps)
        gamma = nppm - j
        gamma = 0.0 if gamma < eps else gamma
    first = j - 1 if 0 < j <= n else (0 if j <= 0 else n - 1)
    second = j if 0 < j < n else (0 if j <= 0 else n - 1)
    return (1 - gamma) * ordered[first] + gamma * ordered[second]


def copied(dist: Any) -> Any:
    """``SamplingDistribution``'s copy: its values, weights and variable's name."""
    from .sampling import SamplingDistribution

    made = SamplingDistribution(dist.GetName(), dist.GetTitle(),
                                list(dist.GetSamplingDistribution()),
                                list(dist.GetSampleWeights()))  # fmt: skip
    made._var_name = dist.GetVarName()
    return made


def expected_p_value_dist(result: Any, index: int) -> Any:
    """``GetExpectedPValueDist``: ``CLs`` - or ``CLs+b`` - for each background toy as data; or,
    with no toys, the asymptotic ones from -5 to 5 sigma."""
    from .hypotest import HypoTestResult
    from .sampling import SamplingDistribution

    if not 0 <= index < result.ArraySize():
        return None
    background = result.GetBackgroundTestStatDist(index)
    signal = result.GetSignalAndBackgroundTestStatDist(index)
    found = result._results[index]
    if background is not None and signal is not None:
        temp = HypoTestResult()
        temp.SetPValueIsRightTail(found.GetPValueIsRightTail())
        temp.SetBackgroundAsAlt(True)
        temp.SetNullDistribution(copied(signal))
        temp.SetAltDistribution(copied(background))
        values = []
        for value in background.GetSamplingDistribution():
            temp.SetTestStatisticData(value)
            values.append(temp.CLs() if result._use_cls else temp.CLsplusb())
        return SamplingDistribution("expected values", "expected values", values)
    return _asymptotic_values(result, found)


def _asymptotic_values(result: Any, found: Any) -> Any:
    from .asympformulae import expected_p_values
    from .sampling import SamplingDistribution

    step = 2 * MAX_SIGMA / (NUM_POINTS - 1)
    values = []
    for i in range(NUM_POINTS):
        pval = expected_p_values(found.NullPValue(), found.AlternatePValue(), -MAX_SIGMA + step * i,
                                 result._use_cls, not result._two_sided)  # fmt: skip
        if pval < 0:
            return None
        values.append(pval)
    return SamplingDistribution("Asymptotic expected values", "Asymptotic expected values", values)


def _size(dist: Any) -> int:
    return 0 if dist is None else int(dist.GetSize())


def _quantiles_of(dist: Any, size: int) -> list[float]:
    """``size`` quantiles of a point's expected p-values - none, if it has none."""
    if dist is None:
        return []
    values = dist.GetSamplingDistribution()
    return [quantile(values, min((b + 1) / size, 1.0), 1) for b in range(size)]


def _limits(result: Any, quantiles: list[list[float]], size: int, lower: bool) -> list[float]:
    """For each quantile, the limit of the curve through the points' quantiles."""
    kept = [k for k in sorted_order(result._x) if quantiles[k]]
    xs = [result.GetXValue(k) for k in kept]
    return [graph_x(result, xs, [quantiles[k][j] for k in kept], 1 - result._cl, lower)[0]
            for j in range(size)]  # fmt: skip


def limit_distribution(result: Any, lower: bool) -> Any:
    """``GetLimitDistribution``: for each quantile of the points' expected p-values, the limit
    the curve through those quantiles gives - at least ten of them."""
    from .sampling import SamplingDistribution

    n = result.ArraySize()
    if n < 2:
        log(result, ERROR, "Eval", "HypoTestInverterResult::GetLimitDistribution not  enough "
            "points -  return 0 ")  # fmt: skip
        return None
    dists = [expected_p_value_dist(result, i) for i in range(n)]
    size = int(sum(_size(d) for d in dists) / n)
    if size < 10:
        log(result, WARNING, "InputArguments", "HypoTestInverterResult - set a minimum size of 10 "
            "for limit distribution")  # fmt: skip
        size = 10
    quantiles = [_quantiles_of(d, size) for d in dists]
    limits = _limits(result, quantiles, size, lower)
    title = "Expected lower limits" if lower else "Expected upper limits"
    return SamplingDistribution("Expected lower Limit" if lower else "Expected upper Limit", title,
                                limits)  # fmt: skip


def expected_limit(result: Any, nsig: float, lower: bool, opt: str = "") -> float:
    """``GetExpectedLimit``: the limit ``nsig`` from the median - asymptotically, the one at that
    many sigma; with toys, the quantile of the limit distribution, or with ``"P"`` the limit of
    the curve through each point's quantile."""
    from ..function.analytic import gaussian_cdf

    if result.ArraySize() <= 0:
        return 1.0 if lower else 0.0
    first = result._results[0]
    if first.GetNullDistribution() is None and first.GetAltDistribution() is None:
        dist = limit_distribution(result, lower)
        values = [] if dist is None else dist.GetSamplingDistribution()
        if len(values) <= 1:
            return 0.0
        step = 2 * MAX_SIGMA / (len(values) - 1)
        return float(values[math.floor((nsig + MAX_SIGMA) / step + 0.5)])
    p = gaussian_cdf(nsig)
    if "P" in str(opt).upper():
        return _limit_of_quantiles(result, p, lower)
    dist = limit_distribution(result, lower)
    return 0.0 if dist is None else quantile(dist.GetSamplingDistribution(), p)


def _limit_of_quantiles(result: Any, p: float, lower: bool) -> float:
    from ..roofit.messages import INFO
    from ..roofit.printing import g

    xs, ys = [], []
    for i in sorted_order(result._x):
        dist = expected_p_value_dist(result, i)
        if dist is None:
            log(result, INFO, "Eval", "HypoTestInverterResult - cannot compute expected p value "
                f"distribution for point, x = {g(result.GetXValue(i))} skip it ")  # fmt: skip
            continue
        xs.append(result._x[i])
        ys.append(quantile(dist.GetSamplingDistribution(), p))
    if len(xs) < 2:
        log(result, ERROR, "Eval", "HypoTestInverterResult - cannot compute limits , not enough "
            f"points, n =  {len(xs)}")  # fmt: skip
        return 0.0
    return graph_x(result, xs, ys, 1 - result._cl, lower)[0]


def _band_quantiles(values: list[float], asymptotic: bool) -> list[float]:
    """The expected p-values at -2, -1, 0, 1 and 2 sigma."""
    from ..function.analytic import gaussian_cdf

    if asymptotic:
        step = 2 * MAX_SIGMA / (len(values) - 1)
        return [values[math.floor((s + MAX_SIGMA) / step + 0.5)] for s in (-2, -1, 0, 1, 2)]
    probs = [gaussian_cdf(-2.0), gaussian_cdf(-1.0), 0.5, gaussian_cdf(1.0), gaussian_cdf(2.0)]
    return [quantile(values, p) for p in probs]


def _band_at(result: Any, i: int, asymptotic: bool) -> Any:
    """The point's expected band - ``None`` if it has none, or not the asymptotic eleven."""
    dist = expected_p_value_dist(result, i)
    if dist is None:
        return None
    values = dist.GetSamplingDistribution()
    if len(values) != NUM_POINTS:
        log(result, ERROR, "Eval", "HypoTestInverterResult::ExclusionCleanup - invalid size of "
            "sampling distribution")  # fmt: skip
        return None
    return _band_quantiles(values, asymptotic)


def _dropped(result: Any, i: int, q: list[float], asymptotic: bool,
             previous: float) -> tuple[bool, float]:  # fmt: skip
    """Whether the point goes, and the ``CLs`` the next is compared with."""
    observed = result.CLs(i)
    later = i >= 1
    rises = asymptotic and later and observed > previous
    drop = rises or observed < 0 or (later and (observed >= 0.9999 or q[4] < result._cleanup))
    return drop, previous if drop and (rises or observed < 0) else observed


def _asymptotic(result: Any) -> bool:
    first = result._results[0] if result._results else None
    return first is not None and first.GetNullDistribution() is None and (
        first.GetAltDistribution() is None)  # fmt: skip


def exclusion_cleanup(result: Any) -> int:
    """``ExclusionCleanup``: drop the points whose ``CLs`` rises again - asymptotically - is one,
    or is negative, or whose expected +2 sigma ``CLs`` is below the threshold; then the lower
    limit found again. Returns how many were dropped."""
    asymptotic = _asymptotic(result)
    removed, previous, position = 0, 1.0, 0
    while position < len(result._x):
        i = result.FindIndex(result._x[position])
        q = _band_at(result, i, asymptotic)
        if q is None:
            break
        drop, previous = _dropped(result, i, q, asymptotic, previous)
        if drop:  # RooStats' erase-and-step passes over the next point too
            del result._x[i], result._results[i]
            removed += 1
        position += 1
    result._fitted = [False, False]
    find_interpolated_limit(result, 1 - result._cl, True)
    return removed
