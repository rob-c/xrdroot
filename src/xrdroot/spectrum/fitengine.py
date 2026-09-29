"""Morhac's two fitting iterations - AWMI and Stiefel's - for any peak model.

``TSpectrumFit`` and ``TSpectrum2Fit`` run the same loop over different
peaks: from the parameters' gradient (AWMI, "algorithm without matrix
inversion", each parameter's own step) or from the normal equations solved
by Stiefel and Hestenes' conjugate gradients, a step is taken - halved
until the chi-square falls, or searched for in tenths - and at the end the
parameters' errors come from one last pass over the channels. The model
(:class:`Model`) says what the spectrum and its derivatives are at the
channels, and how each fitted parameter is kept in bounds; this module is
the loop, with every sum over the channels added channel by channel, in
ROOT's order, and ROOT's quirks kept: the step's ``iter += 1`` that
spends the outer loop's iterations too, and ``pmin`` remembered from one
iteration to the next.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..random import libm
from .fitpeaks import ourpowl

__all__ = ["FitSettings", "Fitted", "Model", "run"]

Array = Any

#: ``kFitOptimChiCounts``, ``kFitOptimChiFuncValues``, ``kFitOptimMaxLikelihood``.
CHI_COUNTS, CHI_FUNC_VALUES, MAX_LIKELIHOOD = 0, 1, 2

#: ``kFitAlphaHalving`` and ``kFitAlphaOptimal``.
ALPHA_HALVING, ALPHA_OPTIMAL = 0, 1

#: ``kFitTaylorOrderSecond``: a second-derivative correction to the position and sigma steps.
TAYLOR_SECOND = 1

#: ``kFitNumRegulCycles``: at most this many tries at a step that lowers the chi-square.
REGUL_CYCLES = 100


@dataclass
class FitSettings:
    """What ``SetFitParameters`` sets: the channels, iterations, and how to step."""

    xmin: int = 0
    xmax: int = 100
    iterations: int = 1
    alpha: float = 1.0
    statistic: int = CHI_COUNTS
    alpha_optim: int = ALPHA_HALVING
    power: int = 2
    taylor: int = 0


class Model(Protocol):
    """A peak model: its spectrum, its derivatives by the fitted parameters, and their bounds."""

    #: The fitted parameters' starting values, in ROOT's order.
    start: list[float]

    def shape(self) -> Array:
        """The model at every fitted channel, from the parameters as they now are."""

    def derivatives(self) -> Array:
        """A row per fitted parameter: its derivative at every channel."""

    def second(self) -> Array:
        """A row per fitted parameter: the second derivative a Taylor step adds, or 0."""

    def taylored(self) -> Array:
        """Whether each fitted parameter's step takes the second-order correction."""

    def apply(self, xk: list[float]) -> None:
        """Keep each of ``xk`` in its bounds, in place, and make it the parameters."""


@dataclass
class Fitted:
    """What the fit leaves: the fitted values, and the sums their errors come from."""

    xk: list[float]
    temp: list[float] = field(default_factory=list)
    der: list[float] = field(default_factory=list)
    temp_xk: list[float] = field(default_factory=list)
    chi_cel: float = 0.0


@dataclass
class _State:
    """What ROOT's fitting loop carries from one iteration to the next."""

    iteration: int = 0
    pmin: float = 0.0


def _total(terms: Array) -> float:
    """The channels' terms added one after another, as ROOT's ``+=`` adds them."""
    return float(np.cumsum(terms)[-1])


def _totals(rows: Array) -> Array:
    """:func:`_total` of each row."""
    return np.cumsum(rows, axis=1)[:, -1]


def _log_terms(y: Array, f: Array) -> Array:
    """The log-likelihood's ``y*log(f) - f``, where ``f > 0.00001``."""
    ok = f > 0.00001
    return np.where(ok, y * libm.log(np.where(ok, f, 1.0)) - f, 0.0)


def _chi_terms(y: Array, f: Array, weights: Array) -> Array:
    """``(y - f)**2 / w`` where the weight is not 0."""
    ok = weights != 0
    return np.where(ok, ((y - f) * (y - f)) / np.where(ok, weights, 1.0), 0.0)


def gradient_chi(y: Array, f: Array, statistic: int) -> float:
    """The statistic an iteration starts from - weighted by the counts, whatever the choice."""
    if statistic == MAX_LIKELIHOOD:
        return _total(_log_terms(y, f))
    return _total(_chi_terms(y, f, y))


def step_chi(y: Array, f: Array, statistic: int) -> float:
    """The statistic a step is judged by: here the function-values choice weights by ``f``."""
    if statistic == MAX_LIKELIHOOD:
        return _total(_log_terms(y, f))
    if statistic == CHI_FUNC_VALUES:
        return _total(_chi_terms(y, f, np.where(f < 0.00001, 0.00001, f)))
    return _total(_chi_terms(y, f, y))


def weights(y: Array, f: Array, statistic: int, likelihood_floor: float) -> Array:
    """Each channel's weight in the gradient: its counts, or the function held off zero."""
    if statistic == CHI_FUNC_VALUES:
        return np.where(f < 0.00001, 0.00001, f)
    if statistic == MAX_LIKELIHOOD:
        return np.where(f < likelihood_floor, likelihood_floor, f)
    return np.where(y == 0, 1.0, y)


def _taylor(model: Model, a: Array, y: Array, f: Array, ywm: Array) -> Array:
    """The derivatives with the second-order correction, where ROOT makes it."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (model.second() * np.abs(y - f)) / ((2 * a) * ywm)
    d = np.where(np.abs(a) > 0.00000001, d, 0.0)
    flip = ((a + d <= 0) & (a >= 0)) | ((a + d >= 0) & (a <= 0))
    return np.where(model.taylored()[:, None], a + np.where(flip, 0.0, d), a)


def _moments(a: Array, c: Array, y: Array, f: Array, ywm: Array, statistic: int) -> Array:
    """The AWMI gradient ``der`` and curvature ``temp`` of every parameter, summed over channels."""
    if statistic == CHI_FUNC_VALUES:
        der = ((a * (y * y - f * f)) / (ywm * ywm)) * c
        temp = (((a * a) * (4 * y - 2 * f)) / (ywm * ywm)) * c
    else:
        der = ((a * (y - f)) / ywm) * c
        temp = ((a * a) / ywm) * c
    return _totals(der), _totals(temp)


def awmi_gradient(model: Model, y: Array, settings: FitSettings) -> tuple[Array, Array, float]:
    """AWMI's step direction: each parameter's gradient over its own curvature.

    The power ``c = a**(power - 2)`` weights each channel by how much the
    parameter matters there, and it is taken before the Taylor correction.
    """
    f = model.shape()
    chi = gradient_chi(y, f, settings.statistic)
    ywm = weights(y, f, settings.statistic, 0.001)
    a = model.derivatives()
    c = ourpowl(a, settings.power - 2)
    if settings.taylor == TAYLOR_SECOND:
        a = _taylor(model, a, y, f, ywm)
    else:
        a = np.where(model.taylored()[:, None], a + 0.0, a)
    der, temp = _moments(a, c, y, f, ywm, settings.statistic)
    with np.errstate(divide="ignore", invalid="ignore"):
        der = np.where(np.abs(temp) > 0.000001, der / np.abs(temp), 0.0)
    return der, temp, chi


def _normal_matrix(a: Array, y: Array, f: Array, ywm: Array, statistic: int) -> Array:
    """``sum over channels of a_j * a_k / w``, each element added channel by channel."""
    rows = a.shape[0]
    matrix = np.zeros((rows, rows))
    for start in range(0, a.shape[1], 256):
        part = slice(start, start + 256)
        b = (a[:, None, part] * a[None, :, part]) / ywm[part]
        if statistic == CHI_FUNC_VALUES:
            b = (b * (4 * y[part] - 2 * f[part])) / ywm[part]
        matrix = np.cumsum(np.concatenate([matrix[:, :, None], b], axis=2), axis=2)[:, :, -1]
    return matrix


def stiefel_inversion(matrix: Array, rhs: Array) -> Array:
    """``TSpectrumFit::StiefelInversion``: conjugate gradients from 0 for ``matrix x = rhs``."""
    size = len(rhs)
    x, u = np.zeros(size), np.zeros(size)
    sk, normk_old, k = 0.0, 0.0, 0
    while True:
        r = np.cumsum(np.concatenate([-rhs[:, None], matrix * x[None, :]], axis=1), axis=1)[:, -1]
        normk = _total(r * r)
        if k != 0:
            sk = normk / normk_old
        u = (-r) + sk * u
        lambdak = _total(_totals(matrix * u[None, :]) * u)
        lambdak = normk / lambdak if abs(lambdak) > 1e-50 else 0.0
        x = x + lambdak * u
        normk_old, k = normk, k + 1
        if not (k < size and abs(normk) > 1e-50):
            return x


def stiefel_gradient(model: Model, y: Array, settings: FitSettings) -> tuple[Array, Array, float]:
    """Stiefel's step: the normal equations of the linearised fit, solved.

    Its ``temp`` - which the errors are divided by - is the matrix's diagonal.
    """
    f = model.shape()
    chi = gradient_chi(y, f, settings.statistic)
    ywm = weights(y, f, settings.statistic, 0.00001)
    a = model.derivatives()
    matrix = _normal_matrix(a, y, f, ywm, settings.statistic)
    if settings.statistic == CHI_FUNC_VALUES:
        b = (f * f - y * y) / (ywm * ywm)
    else:
        b = (f - y) / ywm
    rhs = _totals(-(b[None, :] * a))
    return stiefel_inversion(matrix, rhs), np.diagonal(matrix).copy(), chi


def _better(chi: float, best: float, statistic: int) -> bool:
    """A smaller chi-square, or a larger likelihood."""
    return chi > best if statistic == MAX_LIKELIHOOD else chi < best


def _moved(model: Model, xk: list[float], base: list[float], scale: float, der: Array) -> None:
    """``xk = base + scale * der``, kept in bounds and made the parameters."""
    xk[:] = [b + scale * float(d) for b, d in zip(base, der)]
    model.apply(xk)


def _search(model: Model, y: Array, xk: list[float], step: tuple[list[float], Array, float],
            chi2: float, state: _State, statistic: int) -> float:  # fmt: skip
    """``kFitAlphaOptimal``: the best of ``0.1, 0.2, ...`` steps, until one is worse."""
    base, der, alpha = step
    chi_min = 0.1 * chi2 if statistic == MAX_LIKELIHOOD else 10000 * chi2
    pi, chi = 0.1, 0.0
    while pi <= 100:
        _moved(model, xk, base, pi * alpha, der)
        chi2 = step_chi(y, model.shape(), statistic)
        found = _better(chi2, chi_min, statistic)
        if found:
            state.pmin, chi_min = pi, chi2
        if pi == 0.1:
            chi_min = chi2
        chi = chi_min
        if not found:
            break
        pi += 0.1
    if state.pmin != 0.1:
        _moved(model, xk, base, state.pmin * alpha, der)
        chi = chi_min
    return chi


def _stepped(model: Model, y: Array, xk: list[float], der: Array, chi2: float,
             settings: FitSettings, state: _State) -> None:  # fmt: skip
    """The step along ``der``, retried with a smaller ``alpha`` until the fit improves."""
    base = list(xk)
    alpha, regul, chi_opt = settings.alpha, 0, math.sqrt(abs(chi2))
    while True:
        if settings.alpha_optim == ALPHA_OPTIMAL:
            chi = _search(model, y, xk, (base, der, alpha), chi2, state, settings.statistic)
        else:
            _moved(model, xk, base, alpha, der)
            chi = step_chi(y, model.shape(), settings.statistic)
        chi2, chi = chi, math.sqrt(abs(chi))
        if settings.alpha_optim == ALPHA_HALVING and chi > 1e-6:
            alpha = (alpha * chi_opt) / (2 * chi)
        elif settings.alpha_optim == ALPHA_OPTIMAL:
            alpha = alpha / 10.0
        state.iteration += 1
        regul += 1
        if not (_better(chi_opt, chi, settings.statistic) and regul < REGUL_CYCLES):
            return


def _error_sums(model: Model, y: Array, power: int, fitted: Fitted) -> None:
    """The last pass over the channels, whose sums the errors are made of.

    A channel of no counts is weighted as one of one count, and each
    parameter's sums take the same ``a**(power - 2)`` as its gradient did.
    """
    f = model.shape()
    yw = np.where(y == 0, 1.0, y)
    chi = ((yw - f) * (yw - f)) / yw
    a = model.derivatives()
    c = ourpowl(a, power - 2)
    fitted.chi_cel = _total(chi)
    fitted.der = _totals(chi[None, :] * c).tolist()
    fitted.temp_xk = _totals(((a * a) / yw) * c).tolist()


def run(model: Model, y: Array, settings: FitSettings, stiefel: bool) -> Fitted:
    """Fit ``model`` to the counts ``y`` by AWMI or, with ``stiefel``, by Stiefel's method.

    Stiefel's error sums are AWMI's with no power, which is ``power`` 2.
    """
    gradient = stiefel_gradient if stiefel else awmi_gradient
    power = 2 if stiefel else settings.power
    fitted = Fitted(list(model.start))
    state = _State()
    while state.iteration < settings.iterations:
        der, temp, chi2 = gradient(model, y, settings)
        fitted.temp = temp.tolist()
        _stepped(model, y, fitted.xk, der, chi2, settings, state)
        _error_sums(model, y, power, fitted)
        state.iteration += 1
    return fitted
