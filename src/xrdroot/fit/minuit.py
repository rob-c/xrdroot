"""Minuit, as ``ROOT::Math::Minimizer`` drives it: through iminuit, which is Minuit2.

ROOT's default minimiser is Minuit2's MIGRAD, and iminuit is the same C++,
so a fit here takes the steps a fit in ROOT takes when it is set up the
same way. This is where it is set up the same way: ROOT's tolerance of
0.01 - MIGRAD stops at an estimated distance to the minimum of
``0.002 * tolerance * errordef`` - strategy 1, one call to MIGRAD and no
SIMPLEX first, HESSE afterwards only when asked for, and ``Minuit2Minimizer
::ExamineMinimum``'s status codes.

iminuit is an optional dependency, ``pip install xrdroot[fit]``: a fit that
is linear in its parameters is solved exactly without it, and everything
else asks for it by name.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from .result import FitResult

__all__ = ["minimize", "iminuit"]

#: ``MinimizerOptions::DefaultTolerance``.
TOLERANCE = 0.01
#: ``MinimizerOptions::DefaultStrategy``.
STRATEGY = 1
#: ``FitConfig``'s first step for a parameter: 30% of its value, or 0.3 for a value of zero.
STEP_FRACTION = 0.3


def iminuit() -> Any:
    """The ``iminuit`` module, or a refusal saying how to get it."""
    try:
        import iminuit as module
    except ImportError:
        raise UnsupportedFeatureError(
            "this fit needs Minuit, which xrdroot reaches through iminuit, and iminuit is not "
            "installed: pip install xrdroot[fit]. A fit linear in its parameters - a polN, "
            "say - is solved without it."
        ) from None
    return module


def default_steps(values: Any) -> np.ndarray[Any, Any]:
    """``FitConfig::SetParamsSettings``' step sizes: 30% of each value, 0.3 for a zero."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(values == 0, STEP_FRACTION, STEP_FRACTION * np.abs(values))


def _status(fmin: Any) -> int:
    """``Minuit2Minimizer::ExamineMinimum``: the last of its checks that fails wins."""
    status = 0 if fmin.has_posdef_covar else 5
    for failed, code in (
        (fmin.has_made_posdef_covar, 1),
        (fmin.hesse_failed, 2),
        (fmin.is_above_max_edm, 3),
        (fmin.has_reached_call_limit, 4),
    ):
        if failed:
            status = code
    return 6 if status == 0 and not fmin.is_valid else status


def _limits(limits: Sequence[Any] | None, npar: int) -> list[tuple[float, float]]:
    """One ``(low, high)`` per parameter, infinite for none, as iminuit takes them."""
    found = [(-np.inf, np.inf)] * npar
    for index, pair in enumerate(limits or ()):
        if pair is not None:
            found[index] = (float(pair[0]), float(pair[1]))
    return found


def _configured(
    fcn: Callable[[Any], float], x0: Any, names: Sequence[str], options: dict[str, Any]
) -> Any:
    module = iminuit()
    minuit = module.Minuit(fcn, np.asarray(x0, dtype=np.float64), name=tuple(names))
    minuit.errordef = options["errordef"]
    minuit.tol = options["tolerance"]
    minuit.strategy = options["strategy"]
    minuit.print_level = 0
    npar = len(names)
    errors = options["errors"]
    minuit.errors = default_steps(x0) if errors is None else np.asarray(errors, dtype=np.float64)
    minuit.limits = _limits(options["limits"], npar)
    minuit.fixed = [bool(flag) for flag in (options["fixed"] or [False] * npar)]
    return minuit


def minimize(
    fcn: Callable[[Any], float],
    x0: Any,
    *,
    names: Sequence[str] | None = None,
    errors: Any = None,
    limits: Sequence[Any] | None = None,
    fixed: Sequence[bool] | None = None,
    errordef: float = 1.0,
    minos: bool = False,
    hesse: bool = False,
    improve: bool = False,
    tolerance: float = TOLERANCE,
    strategy: int = STRATEGY,
) -> FitResult:
    """Minimise ``fcn(params)`` from ``x0`` with MIGRAD, as ``ROOT::Math::Minimizer`` does.

        >>> r = minimize(lambda p: (p[0] - 1) ** 2 + (p[1] + 2) ** 2, [0, 0])  # doctest: +SKIP
        >>> r.parameters, r.errors                                            # doctest: +SKIP

    ``errors`` are the first step sizes - ROOT's 30% of each value unless
    given - ``limits`` a ``(low, high)`` or ``None`` per parameter,
    ``fixed`` a flag per parameter, and ``errordef`` 1 for a chi-square and
    0.5 for a negative log-likelihood. ``hesse`` runs HESSE after MIGRAD,
    ``minos`` MINOS for every free parameter, and ``improve`` - ROOT's
    option ``M`` - MIGRAD a second time from where the first stopped.
    """
    npar = len(np.atleast_1d(x0))
    labels = tuple(names) if names is not None else tuple(f"p{i}" for i in range(npar))
    options = {
        "errordef": errordef,
        "tolerance": tolerance,
        "strategy": strategy,
        "errors": errors,
        "limits": limits,
        "fixed": fixed,
    }
    minuit = _configured(fcn, x0, labels, options)
    minuit.migrad(iterate=1, use_simplex=False)
    if improve:
        minuit.migrad(iterate=1, use_simplex=False)
    if hesse or minos:
        minuit.hesse()
    return _result(minuit, labels, minos)


def hessian(
    fcn: Callable[[Any], float],
    values: Any,
    *,
    names: Sequence[str],
    errors: Any,
    limits: Sequence[Any] | None,
    fixed: Sequence[bool] | None,
    errordef: float = 0.5,
) -> np.ndarray[Any, Any]:
    """``Minimizer::Hesse`` then ``GetHessianMatrix`` at ``values``: the inverse covariance.

    A fixed parameter's row and column are zero, as Minuit2 gives them.
    """
    options = {
        "errordef": errordef,
        "tolerance": TOLERANCE,
        "strategy": STRATEGY,
        "errors": errors,
        "limits": limits,
        "fixed": fixed,
    }
    minuit = _configured(fcn, values, names, options)
    minuit.hesse()
    covariance = np.array(minuit.covariance, dtype=np.float64)
    free = ~np.asarray(minuit.fixed, dtype=bool)
    found = np.zeros_like(covariance)
    found[np.ix_(free, free)] = np.linalg.inv(covariance[np.ix_(free, free)])
    return found


def _minos(minuit: Any, labels: Sequence[str]) -> dict[str, tuple[float, float]]:
    """MINOS for every free parameter, where it succeeds, as ``Fitter`` keeps them."""
    found: dict[str, tuple[float, float]] = {}
    if not minuit.valid:
        return found
    minuit.minos()
    for label in labels:
        error = minuit.merrors.get(label)
        if error is not None and error.is_valid:
            found[label] = (float(error.lower), float(error.upper))
    return found


def _result(minuit: Any, labels: Sequence[str], minos: bool) -> FitResult:
    fixed = [bool(flag) for flag in minuit.fixed]
    found_minos = _minos(minuit, labels) if minos else {}
    covariance = (
        np.zeros((len(labels), len(labels)))
        if minuit.covariance is None
        else np.array(minuit.covariance, dtype=np.float64)
    )
    errors = np.where(fixed, 0.0, np.asarray(minuit.errors, dtype=np.float64))
    limits = [np.isfinite(low) or np.isfinite(high) for low, high in minuit.limits]
    return FitResult(
        parameters=np.asarray(minuit.values, dtype=np.float64),
        errors=errors,
        covariance=covariance,
        names=labels,
        fcn=float(minuit.fval),
        edm=float(minuit.fmin.edm),
        nfev=int(minuit.nfcn),
        status=_status(minuit.fmin),
        valid=bool(minuit.valid),
        minos=found_minos,
        fixed=fixed,
        bounded=[flag and not fix for flag, fix in zip(limits, fixed)],
    )
