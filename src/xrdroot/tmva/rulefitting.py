"""``RuleFitParams``: the coefficients of the rules and linear terms, by a gradient-directed path.

The model is ``F = a0 + sum a_k r_k + sum b_j l_j``; its risk is the ramp
loss ``(y - H(F))^2``, ``H`` clipping ``F`` to ``[-1, 1]`` and ``y`` being
+1 for signal and -1 for background. Starting from nothing, each step moves
the coefficients whose gradient is within ``tau`` of the largest along it
- ``tau = 0`` is ridge-like, ``tau = 1`` lasso-like - and the path stops
where the risk of the validation events is least. With ``GDTau=-1`` the
cut-off is chosen by scanning it for the least validation risk, first in
tenths, then to ``GDTauPrec`` around the best. This is Friedman and
Popescu's algorithm, which TMVA implements; its paths are not TMVA's own,
step for step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Path", "fit_path", "scan_tau"]

#: How many steps apart the validation risk is looked at.
CHECK = 100


@dataclass
class Path:
    """The end of a path: the coefficients, the offset, the least risk, and where it was found."""

    coefficients: Any
    offset: float
    risk: float
    step: int
    tau: float


def _risk(features: Any, target: Any, weights: Any, coefficients: Any, offset: float) -> float:
    fitted = np.clip(offset + features @ coefficients, -1.0, 1.0)
    return float(weights @ (target - fitted) ** 2 / weights.sum())


def fit_path(
    sample: tuple[Any, Any, Any],
    valid: tuple[Any, Any, Any],
    tau: float,
    options: dict[str, Any],
    steps: int | None = None,
) -> Path:
    """One gradient-directed path, stopped at its least validation risk."""
    features, target, weights = sample
    rate, total = float(options["GDStep"]), int(steps or options["GDNSteps"])
    scale = float(options["GDErrScale"])
    coefficients = np.zeros(features.shape[1])
    norm = weights / weights.sum()
    offset = float(norm @ target)
    best = Path(coefficients.copy(), offset, _risk(*valid, coefficients, offset), 0, tau)
    for step in range(1, total + 1):
        fitted = offset + features @ coefficients
        residual = np.where(np.abs(fitted) < 1.0, target - fitted, 0.0)
        gradient = features.T @ (norm * residual)
        largest = float(np.max(np.abs(gradient))) if len(gradient) else 0.0
        if largest <= 0.0:
            break
        moved = np.abs(gradient) >= tau * largest
        coefficients[moved] += rate * gradient[moved]
        offset = float(norm @ (target - features @ coefficients))
        if step % CHECK == 0:
            risk = _risk(*valid, coefficients, offset)
            if risk < best.risk:
                best = Path(coefficients.copy(), offset, risk, step, tau)
            elif risk > scale * best.risk:
                break
    return best


def scan_tau(
    sample: tuple[Any, Any, Any], valid: tuple[Any, Any, Any], options: dict[str, Any]
) -> Path:
    """``FindGDTau``: the cut-off of least validation risk, in tenths, then to ``GDTauPrec``."""
    given = float(options["GDTau"])
    if given >= 0:
        return fit_path(sample, valid, given, options)
    steps = max(CHECK, int(options["GDNSteps"]) // 10)
    found = [fit_path(sample, valid, tau / 10, options, steps) for tau in range(11)]
    best = min(found, key=lambda path: path.risk)
    precision = max(float(options["GDTauPrec"]), 1e-3)
    fine = np.arange(max(0.0, best.tau - 0.1), min(1.0, best.tau + 0.1) + 1e-12, precision * 5)
    for tau in fine:
        path = fit_path(sample, valid, float(tau), options, steps)
        if path.risk < best.risk:
            best = path
    return fit_path(sample, valid, best.tau, options)
