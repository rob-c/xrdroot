"""An unbinned maximum-likelihood fit: ``TTree::UnbinnedFit``, on arrays.

Each point contributes ``-log p(x)``, where ``p`` is the model divided by
its integral over the range, so the model need not be a normalised density:
that is ``LogLikelihoodFCN`` on ``UnBinData``, with ROOT's error definition
of a half. With ``extended`` the number of points is Poisson too, and the
model's integral is the expected number: ``nu - sum log f(x)``.

The normalisation is recomputed at every step - ``TF1::Integral`` in one
dimension, closed-form for the built-in shapes and adaptive Gauss-Kronrod
otherwise, and the 10-point Gauss rule on a 16-cell grid per axis in more,
which is good to far better than a fit's statistical error for any smooth
model. Points outside the range are left out, as ROOT leaves them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..filling import running
from ..function import Function
from .cost import GAUSS_NODES, GAUSS_WEIGHTS, eval_log
from .minuit import minimize
from .result import FitResult

__all__ = ["unbinned"]

#: The cells per axis the normalisation of a model of several variables is summed over.
CELLS = 16


def _points(data: Any, ndim: int, span: list[tuple[float, float]]) -> Any:
    """The points inside the range, ``(n,)`` for one variable and ``(n, ndim)`` for more."""
    points = np.asarray(data, dtype=np.float64)
    points = points.reshape(-1, 1) if points.ndim == 1 else points
    if points.shape[1] != ndim:
        raise ValueError(
            f"the model is a function of {ndim} variables, and the data are points of "
            f"{points.shape[1]}"
        )
    inside = np.ones(len(points), dtype=bool)
    for axis, (low, high) in enumerate(span):
        inside &= (points[:, axis] >= low) & (points[:, axis] <= high)
    kept = points[inside]
    return kept[:, 0] if ndim == 1 else kept


def _grid(span: list[tuple[float, float]]) -> tuple[Any, Any]:
    """Gauss points over a grid of cells covering the range, and their weights."""
    axes, weights = [], []
    for low, high in span:
        edges = np.linspace(low, high, CELLS + 1)
        half, centre = 0.5 * np.diff(edges), 0.5 * (edges[:-1] + edges[1:])
        axes.append((centre[:, None] + half[:, None] * GAUSS_NODES).ravel())
        weights.append((half[:, None] * GAUSS_WEIGHTS).ravel())
    grids = np.meshgrid(*axes, indexing="ij")
    weight = np.prod(np.meshgrid(*weights, indexing="ij"), axis=0).ravel()
    return np.stack([grid.ravel() for grid in grids], axis=1), weight


def _normaliser(function: Function, span: list[tuple[float, float]]) -> Any:
    """The model's integral over the range, for a set of parameters."""
    if function.dimensions == 1:
        low, high = span[0]

        def one(params: Any) -> float:
            saved = function.parameters.copy()
            function.parameters[:] = params
            try:
                return function._integrated(low, high, 1e-9)
            finally:
                function.parameters[:] = saved

        return one
    points, weight = _grid(span)
    return lambda params: float(np.dot(weight, function.evaluate(points, params)))


def _model(model: Any, values: Any, ndim: int, span: list[tuple[float, float]]) -> Function:
    """The model as a :class:`Function`: given as one, as a formula, or as Python code."""
    if isinstance(model, Function):
        if values is not None:
            model.set_parameters(*values)
        return model
    ranges: Any = span if ndim > 1 else span[0]
    if isinstance(model, str):
        return Function(model, model, range=ranges, parameters=values)
    return Function.from_callable(
        "unbinned", model, len(values), dimensions=ndim, range=ranges, parameters=values
    )


def _span(given: Any, points: Any) -> list[tuple[float, float]]:
    """The range: a ``(low, high)`` per variable as given, or the points' own extent."""
    if given is None:
        return [(float(low), float(high)) for low, high in zip(points.min(0), points.max(0))]
    pairs = [given] if np.ndim(given) == 1 else list(given)
    return [(float(low), float(high)) for low, high in pairs]


def _setup(data: Any, model: Any, parameters: Any, given: Any) -> tuple[Function, Any, Any]:
    """The model as a function, the range, and the points inside it."""
    probe = np.asarray(data, dtype=np.float64)
    table = probe.reshape(-1, 1) if probe.ndim == 1 else probe
    values = None if parameters is None else np.asarray(parameters, dtype=np.float64).ravel()
    if values is None and not isinstance(model, Function):
        raise ValueError("an unbinned fit of a formula or of Python code needs its parameters=")
    span = _span(given, table) if given is not None else None
    function = _model(model, values, table.shape[1], span or _span(None, table))
    span = span or _own_span(function)
    return function, span, _points(probe, function.dimensions, span)


def unbinned(
    data: Any,
    model: Any,
    *,
    parameters: Any = None,
    range: Any = None,
    extended: bool = False,
    limits: Any = None,
    fixed: Any = None,
) -> FitResult:
    """Fit ``model`` to the points ``data`` by maximum likelihood, without binning them.

        >>> r = unbinned(x, "gaus", parameters=[1, 0, 1], range=(-5, 5))   # doctest: +SKIP
        >>> r = unbinned(xy, lambda p, a: ..., parameters=[...], range=[(0, 1), (0, 1)])
        ... # doctest: +SKIP

    ``data`` is ``(n,)`` for one variable or ``(n, d)`` for ``d``;
    ``model`` a formula, a :class:`~xrdroot.Function`, or ``fn(x, params)``;
    ``range`` a ``(low, high)`` per variable, the model's own if not given.
    The model need not be normalised - that is what the range is for - and
    its overall scale is then no parameter the data can fix: hold it with
    ``fixed``, or pass ``extended=True`` to fit it to the number of points.
    """
    function, span, points = _setup(data, model, parameters, range)
    normaliser = _normaliser(function, span)
    count = len(points)

    def nll(params: Any) -> float:
        norm = normaliser(params)
        density = np.asarray(function.evaluate(points, params), dtype=np.float64) / norm
        total = -running(0.0, eval_log(density))
        return total + (norm - count * np.log(norm) if extended else 0.0)

    found = minimize(
        nll,
        function.parameters.copy(),
        names=function.parameter_names,
        errordef=0.5,
        limits=function.parameter_limits if limits is None else limits,
        fixed=function.fixed if fixed is None else fixed,
    )
    found.chi2 = -1.0  # an unbinned likelihood has no chi-square, and ROOT says so with -1
    found.npoints = count
    found.ndf = count - sum(1 for flag in found.fixed if not flag)
    function.set_parameters(*found.parameters)
    function.fit_result = {"chi2": 0.0, "ndf": found.ndf, "npfits": count, "errors": found.errors}
    found.function = function
    return found


def _own_span(function: Function) -> list[tuple[float, float]]:
    """The function's own range, one pair per variable."""
    ranges = function.range if function.dimensions > 1 else (function.range,)
    return [(float(low), float(high)) for low, high in ranges]
