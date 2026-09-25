"""``HFit::Fit``: a histogram or graph fitted the way ``TH1::Fit`` and ``TGraph::Fit`` fit it.

The steps are ROOT's, in ROOT's order. The options are read. The points are
chosen (:mod:`.data`). A built-in shape that is not to be fitted linearly
is started from the data (:mod:`.guesses`) unless ``B`` says otherwise, and
every parameter is set up from the function: fixed where ``FixParameter``
marked it, bounded where it has limits, and its first step its error if it
has one, a tenth of its range if bounded, 30% of its value otherwise. A
function linear in its parameters, fitted by least squares without an
option that needs Minuit, is solved exactly (:mod:`.linear`); anything else
goes to MIGRAD (:mod:`.minuit`) - a chi-square with an error definition of
one, a likelihood with a half. Where the data had no errors, or ``W`` set
them all to one, the errors are scaled by ``sqrt(chi2/ndf)`` afterwards, as
ROOT scales them.

What the fit found is written into the function as ROOT's ``TF1`` records
it - ``fChisquare``, ``fNDF``, ``fNpfits``, the parameters and ``fParErrors``
- and, unless ``N``, a copy of it is hung on what was fitted, replacing the
functions there unless ``+``: its range set to the one drawn, and sampled
into ``fSave`` over it as ``TF1::Save`` samples it, so a file written with
the histogram is one ROOT reads as fitted.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..fillrandom import STANDARD, first_last, standard_function
from ..function import Function
from ..function.saved import sample
from . import cost, guesses, linear, minuit
from .data import (
    COORD_ERROR,
    NO_ERROR,
    DataOptions,
    FitData,
    from_graphs,
    from_histogram,
)
from .options import FitOptions, parse
from .result import FitResult

__all__ = ["fit_object"]

#: ``TF1::kNotDraw``, which option ``0`` sets on the function it stores.
NOT_DRAW = 1 << 9
#: ``TFitResult``'s name for linear least squares.
LINEAR = "Linear"
#: The number of a ``polN``: 300 plus its degree.
POLYNOMIAL = 300


# -- what is fitted --------------------------------------------------------------------------


def _target(obj: Any) -> tuple[str, int]:
    """What is fitted - ``"histogram"``, ``"graph"`` or ``"multigraph"`` - and its dimension."""
    from ..graph import Graph
    from ..hist import Histogram
    from ..stacks import MultiGraph

    if isinstance(obj, Histogram):
        return "histogram", len(obj.axes)
    if isinstance(obj, (Graph, MultiGraph)):
        return ("graph" if isinstance(obj, Graph) else "multigraph"), 1
    raise TypeError(f"a fit is made to a histogram, a profile or a graph, not {type(obj).__name__}")


def _extent(obj: Any, kind: str) -> list[tuple[float, float]]:
    """The object's own span along each axis, which a formula given as text is defined over."""
    if kind == "histogram":
        return [(axis.low, axis.high) for axis in obj.axes]
    graphs = [obj] if kind == "graph" else list(obj)
    xs = np.concatenate([graph.x for graph in graphs] + [np.zeros(0)])
    return [(float(xs.min()), float(xs.max())) if len(xs) else (0.0, 1.0)]


def _model(obj: Any, kind: str, model: Any, npar: int | None) -> Function:
    """The function to fit: given, one of ROOT's standard ones, a formula, or Python code."""
    if isinstance(model, Function):
        return model
    span = _extent(obj, kind)
    if isinstance(model, str):
        if model in STANDARD:
            return standard_function(model)
        return Function(model, model, range=span)
    if callable(model):
        if npar is None:
            raise ValueError(
                "a Python model is fn(x, params), and how many parameters it takes cannot be "
                "read off it: give npar="
            )
        return Function.from_callable("fit", model, int(npar), dimensions=len(span), range=span)
    raise TypeError(
        f"a model is a shape's name, a formula, a Function or fn(x, params), not "
        f"{type(model).__name__}"
    )


def _by_parameter(function: Function, given: Any) -> list[tuple[int, Any]]:
    """``given`` as ``(index, value)`` pairs: a dictionary by name or index, or a sequence."""
    items = given.items() if isinstance(given, dict) else enumerate(given)
    return [(function._index(key), value) for key, value in items]


def _apply(function: Function, parameters: Any, limits: Any, fixed: Any) -> None:
    """``SetParameters``, ``SetParLimits`` and ``FixParameter`` from the keyword arguments."""
    if parameters is not None:
        for index, value in _by_parameter(function, parameters):
            function.set_parameters(**{function.parameter_names[index]: float(value)})
    for index, pair in _by_parameter(function, limits or ()):
        if pair is not None:
            function.set_limits(index, float(pair[0]), float(pair[1]))
    _fix(function, fixed)


def _fix(function: Function, fixed: Any) -> None:
    """``fixed``: a flag per parameter, the parameters by name or index, or ``{name: value}``."""
    if fixed is None:
        return
    if isinstance(fixed, dict):
        for index, value in _by_parameter(function, fixed):
            function.fix(index, None if value is None else float(value))
        return
    _fix_listed(function, list(fixed))


def _fix_listed(function: Function, flags: list[Any]) -> None:
    """A flag per parameter, or the parameters to fix by name or index."""
    if flags and all(isinstance(flag, (bool, np.bool_)) for flag in flags):
        function.fixed = [bool(flag) for flag in flags]
        return
    for parameter in flags:
        function.fix(parameter)


# -- choosing the points ---------------------------------------------------------------------


def _data_options(opts: FitOptions, number: int) -> DataOptions:
    """``HFit::Fit``'s ``DataOptions`` from the fit options."""
    found = DataOptions(integral=bool(opts.integral), exp_errors=bool(opts.pchi2))
    found.use_empty = bool(opts.like or opts.pchi2 or opts.w1 > 1)
    found.coord_errors = number != POLYNOMIAL and not opts.no_x_errors
    found.errors1 = bool(opts.w1) or opts.pchi2 == 1
    found.bin_volume = bool(opts.bin_volume)
    found.norm_bin_volume = opts.bin_volume == 2
    return found


def _spans(given: Any, ndim: int) -> list[Any]:
    """One ``(low, high)`` or ``None`` per axis from ``range``: a pair, or a pair per axis."""
    if given is None:
        return [None] * ndim
    pairs = [given] if _is_pair(given) else list(given)
    spans: list[Any] = [None] * ndim
    for axis, pair in enumerate(pairs[:ndim]):
        if pair is not None and float(pair[0]) < float(pair[1]):
            spans[axis] = (float(pair[0]), float(pair[1]))
    return spans


def _is_pair(given: Any) -> bool:
    """Whether ``given`` is one ``(low, high)``, rather than a pair per axis."""
    return len(given) == 2 and all(isinstance(end, (int, float, np.number)) for end in given)


def _with_function_range(spans: list[Any], function: Function) -> list[Any]:
    """Option ``R``: the function's range on every axis no range was given for."""
    ranges = function.range if function.dimensions > 1 else (function.range,)
    return [
        span if span is not None else (ranges[axis] if axis < len(ranges) else (0.0, 0.0))
        for axis, span in enumerate(spans)
    ]


def _points(obj: Any, kind: str, options: DataOptions, spans: list[Any]) -> FitData:
    if kind == "histogram":
        return from_histogram(obj, options, spans)
    graphs = [obj] if kind == "graph" else list(obj)
    return from_graphs(graphs, options, spans[0])


# -- the fit itself -----------------------------------------------------------------------------


def _is_linear(opts: FitOptions, data: FitData) -> bool:
    """Whether ROOT's options and the data leave the linear fitter free to be used."""
    excluded = (
        opts.bound,
        opts.like,
        opts.errors,
        opts.gradient,
        opts.more,
        opts.integral,
        opts.minuit,
        opts.pchi2,
        opts.bin_volume,
    )
    if any(excluded):
        return False
    return not (data.kind >= COORD_ERROR and (data.options.coord_errors or data.kind > COORD_ERROR))


def _settings(function: Function) -> dict[str, Any]:
    """Each parameter set up as ``HFit::Fit`` sets it up: fixed, bounded, and its first step."""
    low, high = function._per_parameter("fParMin"), function._per_parameter("fParMax")
    values, errors = function.parameters, function.parameter_errors
    steps = minuit.default_steps(values)
    limits: list[Any] = [None] * function.npar
    fixed = [bool(a >= b and a * b != 0) for a, b in zip(low, high)]
    for i in range(function.npar):
        if not fixed[i] and low[i] < high[i]:
            limits[i] = (float(low[i]), float(high[i]))
        steps[i] = _step(float(values[i]), float(errors[i]), limits[i], float(steps[i]))
    return {"errors": steps, "limits": limits, "fixed": fixed}


def _step(value: float, error: float, limits: Any, default: float) -> float:
    """The first step: the error, else a tenth of a finite range kept clear of its ends."""
    if error > 0:
        return error
    if limits is None or not (math.isfinite(limits[0]) and math.isfinite(limits[1])):
        return default
    low, high = float(limits[0]), float(limits[1])
    step = 0.1 * (high - low)
    if value < high and high - value < 2 * step:
        return (high - value) / 2
    if value > low and value - low < 2 * step:
        return (value - low) / 2
    return step


def _objective(opts: FitOptions, data: FitData, function: Function) -> tuple[Any, float]:
    """What MIGRAD minimises, and its error definition."""
    if opts.like:
        predict = cost.Predictor(function, data)
        return cost.poisson(data, predict, extended=(opts.like & 4) != 4), 0.5
    if data.kind >= COORD_ERROR:
        return cost.effective_chi2(data, function), 1.0
    return cost.chi2(data, cost.Predictor(function, data)), 1.0


def _by_minuit(opts: FitOptions, data: FitData, function: Function) -> FitResult:
    fcn, errordef = _objective(opts, data, function)
    weighted = (opts.like & 2) == 2
    found = minuit.minimize(
        fcn,
        function.parameters.copy(),
        names=function.parameter_names,
        errordef=errordef,
        minos=opts.errors and not weighted,
        hesse=opts.errors,
        improve=opts.more,
        **_settings(function),
    )
    if weighted and found.valid:
        _weight_corrected(found, data, function, opts)
    found.chi2 = 2 * found.fcn if opts.like else found.fcn
    return found


def _weight_corrected(
    found: FitResult, data: FitData, function: Function, opts: FitOptions
) -> None:
    """``ApplyWeightCorrection``: the covariance ``C H C``, ``H`` from the weights squared."""
    squares = cost.poisson(data, cost.Predictor(function, data), (opts.like & 4) != 4, True)
    settings = _settings(function)
    settings["errors"] = np.where(found.errors > 0, found.errors, settings["errors"])
    hessian = minuit.hessian(squares, found.parameters, names=function.parameter_names, **settings)
    corrected = found.covariance @ hessian @ found.covariance
    found.covariance = corrected
    found.errors = np.sqrt(np.abs(np.diag(corrected)))


def _by_least_squares(data: FitData, function: Function, basis: tuple[Any, Any]) -> FitResult:
    fixed = list(function.fixed)
    params, covariance, chi2 = linear.solve(data, basis, function.parameters, fixed)
    return FitResult(
        parameters=params,
        errors=np.sqrt(np.abs(np.diag(covariance))),
        covariance=covariance,
        names=function.parameter_names,
        fcn=chi2,
        fixed=fixed,
        minimizer=LINEAR,
    )


def _normalised(found: FitResult) -> None:
    """``FitResult::NormalizeErrors``: errors times ``sqrt(chi2/ndf)``, for data with none."""
    if found.ndf == 0 or found.chi2 <= 0:
        return
    found.errors = found.errors * math.sqrt(found.chi2 / found.ndf)
    found.covariance = found.covariance * (found.chi2 / found.ndf)


def _minimised(
    opts: FitOptions, data: FitData, function: Function, linear_allowed: bool
) -> FitResult:
    number = function.number
    basis = (
        linear.design(function, data.coordinates(), function.parameters) if linear_allowed else None
    )
    if basis is not None:
        return _by_least_squares(data, function, basis)
    if number and not opts.bound:
        guesses.initial(data, function, number)
    return _by_minuit(opts, data, function)


# -- recording and storing -----------------------------------------------------------------------


def _record(function: Function, found: FitResult, data: FitData) -> None:
    """What ``HFit::Fit`` writes back into the ``TF1``: the parameters and figures of merit."""
    function.set_parameters(*found.parameters)
    function.fit_result = {
        "chi2": found.chi2,
        "ndf": found.ndf,
        "npfits": data.size,
        "errors": found.errors,
    }


def _histogram_range(axis: Any, row: dict[str, Any]) -> tuple[float, float]:
    """``GetDrawingRange`` along one axis: from the first bin's low edge to the last's high."""
    first, last = first_last(row)
    if axis.even:
        width = (axis.high - axis.low) / axis.nbins
        return axis.low + (first - 1) * width, axis.low + (last - 1) * width + width
    edges = axis.edges()
    return float(edges[first - 1]), float(edges[last - 1] + (edges[last] - edges[last - 1]))


def _graph_ends(graph: Any) -> tuple[float, float]:
    """``ComputeRange`` in x: the points, and their bars for a graph that has them."""
    low, high = graph.x, graph.x
    bars = graph.xerr if graph.classname in ("TGraphErrors", "TGraphAsymmErrors") else None
    if bars is not None:
        low, high = graph.x - bars[0], graph.x + bars[1]
    return float(np.min(low)), float(np.max(high))


def _graph_range(graph: Any) -> tuple[float, float]:
    """``TGraph::GetHistogram``'s x axis: the graph's range, a tenth wider each side."""
    held = graph._core.get("fHistogram")
    if held is not None and hasattr(held, "axes"):
        return _histogram_range(held.axes[0], held._core["fXaxis"])
    low, high = _graph_ends(graph)
    if low == high:
        high += 1.0
    margin = 0.1 * (high - low)
    ulow = 0.0 if low - margin < 0 <= low else low - margin
    uhigh = 0.0 if high + margin > 0 >= high else high + margin
    bins = max(100, len(graph))
    width = (uhigh - ulow) / bins
    return ulow, ulow + (bins - 1) * width + width


def _drawing_range(obj: Any, kind: str, spans: list[Any]) -> list[tuple[float, float]]:
    """``HFit::GetDrawingRange``: the fit's range where there is one, else the object's."""
    if kind == "histogram":
        own = [
            _histogram_range(axis, obj._core[f"f{'XYZ'[i]}axis"]) for i, axis in enumerate(obj.axes)
        ]
    elif kind == "graph":
        own = [_graph_range(obj)]
    else:
        own = [_multigraph_range(obj)]
    return [span if span is not None else mine for span, mine in zip(spans, own)]


def _multigraph_range(multigraph: Any) -> tuple[float, float]:
    held = multigraph.members.get("fHistogram")
    if held is not None and hasattr(held, "axes"):
        return _histogram_range(held.axes[0], held._core["fXaxis"])
    ends = [_graph_ends(graph) for graph in multigraph]
    return min(low for low, _ in ends), max(high for _, high in ends)


def _save(function: Function, obj: Any, kind: str, span: tuple[float, float]) -> None:
    """``TF1::Save``: on a histogram at its bin centres over a wide log range, else ``fNpx + 1``."""
    low, high = span
    npx = function._npx()
    if kind == "histogram" and low > 0 and high > 0 and high / low > npx:
        axis = obj.axes[0]
        bins = np.arange(int(axis.find_bin(low)), int(axis.find_bin(high)) + 1)
        values = function.evaluate(axis.root_centers()[bins])
        function._f1["fSave"] = np.concatenate([values, [low, high, high]])
        return
    function._f1["fSave"] = sample(function.evaluate, low, high, npx)


def _replaced(held: list[Any], function: Function, plus: bool) -> bool:
    """Without ``+``, every function but this one taken off; whether this one was there."""
    reused = any(entry is function for entry in held)
    if not plus:
        held[:] = [entry for entry in held if entry is function or not isinstance(entry, Function)]
    return reused


def _store(obj: Any, kind: str, function: Function, opts: FitOptions, spans: list[Any]) -> Function:
    """``StoreAndDrawFitFunction``: a copy of the fitted function hung on what was fitted."""
    held = obj.functions
    reused = _replaced(held, function, opts.plus)
    stored = function if reused else function.copy()
    ranges = _drawing_range(obj, kind, spans)
    stored.range = ranges[0] if len(ranges) == 1 else tuple(ranges)
    if stored.dimensions == 1:
        _save(stored, obj, kind, ranges[0])
    if opts.nograph:
        named = stored._named()
        named["fBits"] = int(named.get("fBits", 0)) | NOT_DRAW
    if not reused:
        held.append(stored)
    return stored


# -- the entry point -------------------------------------------------------------------------------


def _check(function: Function, dimension: int, kind: str) -> None:
    if function.npar <= 0:
        raise ValueError(f"{function.name!r} has no parameters, and a fit is of parameters")
    if function.dimensions != dimension:
        raise UnsupportedFeatureError(
            f"{function.name!r} is a function of {function.dimensions} variables, and the "
            f"{kind} fitted has {dimension} axes; ROOT's fit of a function of one variable "
            f"fewer than the histogram is not here"
        )


def _weighted_likelihood(obj: Any, kind: str, opts: FitOptions) -> None:
    """``WL`` on a histogram without the squares of its weights is ROOT's plain ``L``."""
    import warnings

    if (opts.like & 2) == 2 and kind == "histogram" and obj._sumw2() is None:
        warnings.warn(
            "A weighted likelihood fit is requested but histogram is not weighted - do a "
            "standard Likelihood fit",
            RuntimeWarning,
            stacklevel=4,
        )
        opts.like = 1


def _finished(found: FitResult, data: FitData, opts: FitOptions) -> None:
    """The degrees of freedom, and the errors scaled where the data had none of their own."""
    nfree = sum(1 for flag in found.fixed if not flag)
    found.ndf = data.size - nfree if data.size > nfree else 0
    found.npoints = data.size
    least_squares = found.minimizer == LINEAR or not opts.like
    if least_squares and (data.kind == NO_ERROR or data.options.errors1):
        _normalised(found)


def _prepared(obj: Any, model: Any, option: str, keywords: dict[str, Any]) -> tuple[Any, ...]:
    """What is fitted, the options, and the function set up as the keywords ask."""
    kind, dimension = _target(obj)
    opts = parse(option, graph=kind != "histogram")
    function = _model(obj, kind, model, keywords["npar"])
    _apply(function, keywords["parameters"], keywords["limits"], keywords["fixed"])
    _check(function, dimension, kind)
    _weighted_likelihood(obj, kind, opts)
    return kind, dimension, opts, function


def fit_object(
    obj: Any,
    model: Any,
    option: str = "",
    range: Any = None,
    *,
    parameters: Any = None,
    limits: Any = None,
    fixed: Any = None,
    npar: int | None = None,
) -> FitResult:
    """``TH1::Fit`` or ``TGraph::Fit``: ``model`` fitted to ``obj`` with ROOT's ``option``."""
    keywords = {"parameters": parameters, "limits": limits, "fixed": fixed, "npar": npar}
    kind, dimension, opts, function = _prepared(obj, model, option, keywords)
    spans = _spans(range, dimension)
    if opts.range:
        spans = _with_function_range(spans, function)
    data = _points(obj, kind, _data_options(opts, function.number), spans)
    if not data.size:
        raise ValueError(f"there are no points to fit in {obj.name!r}: ROOT's Fit data is empty")
    found = _minimised(opts, data, function, _is_linear(opts, data))
    _finished(found, data, opts)
    _record(function, found, data)
    found.function = function if opts.nostore else _store(obj, kind, function, opts, spans)
    if not opts.quiet:
        print(found.summary(covariance=opts.verbose > 0))
    return found
