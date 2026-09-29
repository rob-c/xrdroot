"""What ``TSpectrum2Fit`` reports: every parameter's value and error, and each peak's volume.

A peak's volume is ``2 pi a sigmax sigmay sqrt(1 - ro*ro)``, and its error
is added up from the volume's derivatives by the amplitude and the shared
parameters, as ``TSpectrum2Fit`` adds it - with, for the derivatives by
the sigmas and ``ro``, the fitted value at the peak's own place among the
fitted ones, and the sums at the shared parameters' places among all of
them: ROOT's indexing, kept.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .fit import FitResult, _value_error
from .fit2 import PEAK, SHARED2, FitSettings2, Peak2Model, Peak2Setup
from .fitengine import Fitted, run

__all__ = ["fit2"]

Array = Any

#: ``TSpectrum2Fit``'s pi, to eleven figures.
PI = 3.1415926535


def _root(ro: float) -> float | None:
    """``sqrt(1 - ro*ro)``, or ``None`` where ROOT gives up and answers 0."""
    r = 1 - ro * ro
    return math.sqrt(r) if r > 0 else None


def volume(a: float, sx: float, sy: float, ro: float) -> float:
    """``TSpectrum2Fit::Volume``."""
    r = _root(ro)
    return 0.0 if r is None else ((((2 * a) * PI) * sx) * sy) * r


def _derivatives(a: float, q: dict[str, float]) -> dict[str, float]:
    """``Derpa2``, ``Derpsigmax``, ``Derpsigmay`` and ``Derpro``, with ``a`` for the last three."""
    sx, sy, ro = q["sigmax"], q["sigmay"], q["ro"]
    r = _root(ro)
    if r is None:
        return dict.fromkeys(("amp", "sigmax", "sigmay", "ro"), 0.0)
    return {
        "amp": (((2 * PI) * sx) * sy) * r,
        "sigmax": (((a * 2) * PI) * sy) * r,
        "sigmay": (((a * 2) * PI) * sx) * r,
        "ro": ((((((-a) * 2) * PI) * sx) * sy) * ro) / r,
    }


def _reciprocal(b: float) -> float:
    return 1.0 if b == 0 else 1 / b


def _volume_error(model: Peak2Model, fitted: Fitted, j: int, amp_free: bool,
                  chi_er: float) -> float:  # fmt: skip
    """The error of the volume of the peak whose parameters start at ``j`` among the fitted.

    Past the fitted parameters ROOT reads memory it never wrote; here that is 0.
    """
    size, shared = len(fitted.xk), 7 * model.count
    found = _derivatives(fitted.xk[j] if j < size else 0.0, model.shared())
    c = 0.0
    if amp_free:
        c = c + (found["amp"] * found["amp"]) * _reciprocal(fitted.temp_xk[j])
    free = {kind for _, kind in model.free}
    for name, k in (("sigmax", shared), ("sigmay", shared + 1), ("ro", shared + 2)):
        if name in free:
            a = found[name]
            c = c + (a * a) * _reciprocal(fitted.temp_xk[k] if k < size else 0.0)
    return math.sqrt(abs(chi_er * c))


def _volumes(model: Peak2Model, fitted: Fitted, chi_er: float) -> tuple[list[float], list[float]]:
    """Each peak's volume and its error: 0 for a volume not above 0."""
    q, amps = model.shared(), model.peaks().amp
    at = {index: j for j, (index, _) in enumerate(model.free)}
    volumes, errors = [], []
    for k in range(model.count):
        start = sum(1 for index, _ in model.free if index < 7 * k)
        v = volume(float(amps[k]), q["sigmax"], q["sigmay"], q["ro"])
        volumes.append(v)
        errors.append(_volume_error(model, fitted, start, 7 * k in at, chi_er) if v > 0 else 0.0)
    return volumes, errors


def _results(model: Peak2Model, setup: Peak2Setup, fitted: Fitted, chi_er: float) -> FitResult:
    """ROOT's ``fAmpCalc``, ``fPositionErrX``, ``fVolume`` and the rest."""
    found = _found(model, setup, fitted)
    values = {name: [v for v, _ in found[part]] for name, part in _parts(model.count)}
    errors = {name: [e for _, e in found[part]] for name, part in _parts(model.count)}
    volumes, volume_errors = _volumes(model, fitted, chi_er)
    values["volume"], errors["volume"] = volumes, list(volume_errors)
    return FitResult(values, errors, chi_er, None)


def _found(model: Peak2Model, setup: Peak2Setup,
           fitted: Fitted) -> list[tuple[float, float | None]]:  # fmt: skip
    """Every parameter's value and error, in ROOT's order, fitted or fixed."""
    at = {index: j for j, (index, _) in enumerate(model.free)}
    starts = [float(setup.peaks[kind][k]) for k in range(model.count) for kind in PEAK]
    starts += [float(setup.init[name]) for name in SHARED2]
    return [_value_error(fitted, at.get(n), start) for n, start in enumerate(starts)]


def _parts(count: int) -> list[tuple[str, slice]]:
    """Where each kind of parameter is among all of them, for ``count`` peaks."""
    shared = 7 * count
    parts = [(kind, slice(n, shared, 7)) for n, kind in enumerate(PEAK)]
    return parts + [(name, slice(shared + n, shared + n + 1)) for n, name in enumerate(SHARED2)]


def fit2(source: Array, setup: Peak2Setup, settings: FitSettings2, stiefel: bool) -> FitResult:
    """Fit the peaks ``setup`` describes to ``source[x][y]`` over the fitted channels.

    ``TSpectrum2Fit`` refuses nothing here, not even a fit of nothing free.
    """
    model = Peak2Model(setup, settings)
    window = np.asarray(source, dtype=np.float64)
    y = window[settings.xmin : settings.xmax + 1, settings.ymin : settings.ymax + 1].ravel()
    fitted = run(model, y, settings, stiefel)
    chi_er = fitted.chi_cel / (len(y) - len(model.free))
    result = _results(model, setup, fitted, chi_er)
    result.spectrum = model.shape().reshape(settings.xmax - settings.xmin + 1, -1)
    return result
