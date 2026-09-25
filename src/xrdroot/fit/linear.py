"""The linear fitter: least squares solved exactly, for a function linear in its parameters.

``TH1::Fit`` does not hand a ``polN`` to Minuit. ``TLinearFitter`` writes
the chi-square of a function ``f = sum p_k g_k(x)`` as a quadratic in the
parameters, whose minimum is the solution of the normal equations, and
solves them: no starting values, no steps, no tolerance - the parameters,
the chi-square and the covariance are exact to rounding. That is the
answer MIGRAD converges towards, and this gives it for any formula or
model that is linear in its parameters, not only the ones ROOT recognises:
whether it is linear is asked of the function itself, by evaluating it.

The solution is by least squares on the weighted design matrix, which is
the normal equations' answer computed stably; a fixed parameter's term is
moved to the other side first, as ``TLinearFitter::FixParameter`` has it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .data import FitData

__all__ = ["design", "solve"]

#: How closely the function must equal its linear expansion to be taken as linear.
TOLERANCE = 1e-9


def design(function: Any, points: Any, values: Any) -> tuple[Any, Any] | None:
    """The offset ``f(x; 0)`` and the columns ``g_k(x)``, or ``None`` if it is not linear.

    Each column is the function with one parameter at one and the others
    at zero, less the offset; the function is linear if it equals the
    offset plus the columns weighted by two quite different parameter sets.
    """
    npar = len(values)
    with np.errstate(all="ignore"):
        offset = np.asarray(function.evaluate(points, np.zeros(npar)), dtype=np.float64)
        columns = np.stack(
            [function.evaluate(points, np.eye(npar)[k]) - offset for k in range(npar)], axis=1
        )
        trials = (np.asarray(values, dtype=np.float64), 0.75 + 1.25 * np.arange(npar))
        for trial in trials:
            found = np.asarray(function.evaluate(points, trial), dtype=np.float64)
            expected = offset + columns @ trial
            scale = np.max(np.abs(expected), initial=1.0)
            if not np.all(np.isfinite(found)) or not np.allclose(
                found, expected, rtol=TOLERANCE, atol=TOLERANCE * scale
            ):
                return None
    return offset, columns


def solve(data: FitData, basis: tuple[Any, Any], values: Any, fixed: Any) -> tuple[Any, Any, float]:
    """The parameters, their covariance and the chi-square at the least-squares minimum."""
    offset, columns = basis
    weights = np.ones(data.size) if data.error is None else 1.0 / data.error
    free = ~np.asarray(fixed, dtype=bool)
    params = np.asarray(values, dtype=np.float64).copy()
    target = data.y - offset - columns[:, ~free] @ params[~free]
    matrix = columns[:, free] * weights[:, None]
    solution, *_ = np.linalg.lstsq(matrix, target * weights, rcond=None)
    params[free] = solution
    inverse = np.linalg.pinv(matrix.T @ matrix)
    covariance = np.zeros((len(params), len(params)))
    covariance[np.ix_(free, free)] = inverse
    residual = (data.y - offset - columns @ params) * weights
    return params, covariance, float(np.add.accumulate(np.concatenate(([0.0], residual**2)))[-1])
