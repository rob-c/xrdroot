"""``TSpectrumFit``: peaks of one common sigma fitted to a spectrum's channels.

The fitted parameters are, in ROOT's order, each peak's amplitude and
position, then sigma, the tail's ``t`` and ``b``, the step's ``s``, and the
background's ``a0``, ``a1``, ``a2`` - those not fixed. Each is held where
ROOT holds it as the fit steps: an amplitude not below 0, a position among
the fitted channels, sigma not below 0.001 and ``b`` not within 0.001 of 0.
:func:`fit` runs :mod:`.fitengine`'s loop over :class:`PeakModel` and
works out the errors and areas as ``TSpectrumFit`` does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import fitpeaks as peaks
from .fitengine import FitSettings, Fitted, run

__all__ = ["FitResult", "PeakModel", "PeakSetup", "fit"]

Array = Any

#: The parameters after the peaks', in ROOT's order.
SHARED = ("sigma", "t", "b", "s", "a0", "a1", "a2")


@dataclass
class PeakSetup:
    """What ``SetPeakParameters``, ``SetTailParameters`` and ``SetBackgroundParameters`` set.

    ``init`` and ``fixed`` hold sigma, ``t``, ``b``, ``s``, ``a0``, ``a1``
    and ``a2`` by those names; ROOT fits sigma and fixes the rest unless told.
    """

    positions: list[float]
    amplitudes: list[float]
    fix_positions: list[bool]
    fix_amplitudes: list[bool]
    init: dict[str, float] = field(
        default_factory=lambda: dict(sigma=2.0, t=0.0, b=1.0, s=0.0, a0=0.0, a1=0.0, a2=0.0)
    )
    fixed: dict[str, bool] = field(
        default_factory=lambda: dict(sigma=False, t=True, b=True, s=True, a0=True, a1=True, a2=True)
    )


def _free(setup: PeakSetup) -> list[tuple[int, str]]:
    """Each fitted parameter's place among all the parameters, and what it is."""
    free: list[tuple[int, str]] = []
    for k in range(len(setup.positions)):
        if not setup.fix_amplitudes[k]:
            free.append((2 * k, "amp"))
        if not setup.fix_positions[k]:
            free.append((2 * k + 1, "pos"))
    shared = 2 * len(setup.positions)
    free += [(shared + n, name) for n, name in enumerate(SHARED) if not setup.fixed[name]]
    return free


def _held(kind: str, value: float, channels: tuple[int, int]) -> float:
    """``value`` kept where ROOT keeps a parameter of this kind as the fit steps."""
    if kind == "amp":
        return 0.0 if value < 0 else value
    if kind == "pos":
        low, high = channels
        value = float(low) if value < low else value
        return float(high) if value > high else value
    if kind == "sigma":
        return 0.001 if value < 0.001 else value
    if kind == "b" and abs(value) < 0.001:
        return -0.001 if value < 0 else 0.001
    return value


class PeakModel:
    """``TSpectrumFit``'s spectrum: its peaks, their tails and steps, and the background."""

    def __init__(self, setup: PeakSetup, settings: FitSettings) -> None:
        count = len(setup.positions)
        self.count = count
        self.param = [0.0] * (2 * count + len(SHARED))
        self.param[0 : 2 * count : 2] = [float(a) for a in setup.amplitudes]
        self.param[1 : 2 * count : 2] = [float(p) for p in setup.positions]
        self.param[2 * count :] = [float(setup.init[name]) for name in SHARED]
        self.free = _free(setup)
        self.start = [self.param[index] for index, _ in self.free]
        self.channels = (settings.xmin, settings.xmax)
        self.i = np.arange(settings.xmin, settings.xmax + 1, dtype=np.float64)

    def shared(self) -> dict[str, float]:
        """Sigma, the tail, the step and the background, by name."""
        return dict(zip(SHARED, self.param[2 * self.count :]))

    def peaks(self) -> tuple[Array, Array]:
        """The peaks' amplitudes and positions."""
        amps = np.array(self.param[0 : 2 * self.count : 2])
        return amps, np.array(self.param[1 : 2 * self.count : 2])

    def shape(self) -> Array:
        amp, pos = self.peaks()
        q = self.shared()
        background = (q["a0"], q["a1"], q["a2"])
        return peaks.shape(self.i, amp, pos, q["sigma"], q["t"], q["s"], q["b"], background)

    def derivatives(self) -> Array:
        found = {kind: _DERIVATIVES[kind](self) for kind in {kind for _, kind in self.free}}
        return np.array([_pick(found[kind], index, kind) for index, kind in self.free])

    def second(self) -> Array:
        amp, pos = self.peaks()
        sigma = self.shared()["sigma"]
        by_position = peaks.derderi0(self.i, amp, pos, sigma)
        by_sigma = peaks.derdersigma(self.i, amp, pos, sigma)
        zero = np.zeros_like(self.i)
        rows = {"pos": lambda index: by_position[index // 2], "sigma": lambda index: by_sigma}
        return np.array([rows.get(kind, lambda index: zero)(index) for index, kind in self.free])

    def taylored(self) -> Array:
        return np.array([kind in ("pos", "sigma") for _, kind in self.free], dtype=bool)

    def apply(self, xk: list[float]) -> None:
        for n, (index, kind) in enumerate(self.free):
            xk[n] = _held(kind, xk[n], self.channels)
            self.param[index] = xk[n]


def _pick(rows: Array, index: int, kind: str) -> Array:
    """A peak's own row of a per-peak derivative; a shared parameter's is the whole."""
    return rows[index // 2] if kind in ("amp", "pos") else rows


def _by_amplitude(m: PeakModel) -> Array:
    q = m.shared()
    return peaks.deramp(m.i, m.peaks()[1], q["sigma"], q["t"], q["s"], q["b"])


def _by_position(m: PeakModel) -> Array:
    q = m.shared()
    return peaks.deri0(m.i, *m.peaks(), q["sigma"], q["t"], q["s"], q["b"])


def _by_sigma(m: PeakModel) -> Array:
    q = m.shared()
    return peaks.dersigma(m.i, *m.peaks(), q["sigma"], q["t"], q["s"], q["b"])


def _by_t(m: PeakModel) -> Array:
    q = m.shared()
    return peaks.dert(m.i, *m.peaks(), q["sigma"], q["b"])


def _by_b(m: PeakModel) -> Array:
    q = m.shared()
    return peaks.derb(m.i, *m.peaks(), q["sigma"], q["t"], q["b"])


def _by_s(m: PeakModel) -> Array:
    return peaks.ders(m.i, *m.peaks(), m.shared()["sigma"])


#: Each kind of parameter's derivative, ``TSpectrumFit``'s ``Deramp`` to ``Dera2``.
_DERIVATIVES = {
    "amp": _by_amplitude,
    "pos": _by_position,
    "sigma": _by_sigma,
    "t": _by_t,
    "b": _by_b,
    "s": _by_s,
    "a0": lambda m: np.ones_like(m.i),
    "a1": lambda m: m.i,
    "a2": lambda m: m.i * m.i,
}


@dataclass
class FitResult:
    """What a fit found: a value and an error for every parameter, the areas, and the spectrum.

    An error is ``None`` where ROOT leaves the one it had - a parameter whose
    curvature came to exactly 0.
    """

    values: dict[str, list[float]]
    errors: dict[str, list[float | None]]
    chi: float
    spectrum: Array


def _error(fitted: Fitted, j: int) -> float | None:
    """``sqrt(|der|) / sqrt(|temp|)``, unless ``temp`` is 0."""
    if fitted.temp[j] == 0:
        return None
    return math.sqrt(abs(fitted.der[j])) / math.sqrt(abs(fitted.temp[j]))


def _area_error(model: PeakModel, fitted: Fitted, j: int, chi_er: float) -> float:
    """A fitted amplitude's peak's area error, from ``Derpa`` and the last pass's sums."""
    q = model.shared()
    a = peaks.derpa(q["sigma"], q["t"], q["b"])
    b = fitted.temp_xk[j]
    b = 1.0 if b == 0 else 1 / b
    return math.sqrt(abs(((a * a) * b) * chi_er))


def _results(model: PeakModel, setup: PeakSetup, fitted: Fitted, chi_er: float) -> FitResult:
    """ROOT's ``fAmpCalc``, ``fPositionErr``, ``fArea`` and the rest, from the fitted values."""
    q = model.shared()
    amp = model.peaks()[0]
    areas = [peaks.area(float(a), q["sigma"], q["t"], q["b"]) for a in amp]
    initial = {"amp": setup.amplitudes, "pos": setup.positions}
    values: dict[str, list[float]] = {"amp": [], "pos": [], "area": areas}
    errors: dict[str, list[float | None]] = {"amp": [], "pos": [], "area": []}
    at = {index: j for j, (index, _) in enumerate(model.free)}
    for k in range(model.count):
        for kind, index in (("amp", 2 * k), ("pos", 2 * k + 1)):
            j = at.get(index)
            values[kind].append(fitted.xk[j] if j is not None else float(initial[kind][k]))
            errors[kind].append(_error(fitted, j) if j is not None else 0.0)
            if kind == "amp":
                wanted = j is not None and areas[k] > 0
                errors["area"].append(_area_error(model, fitted, j, chi_er) if wanted else 0.0)
    for n, name in enumerate(SHARED):
        j = at.get(2 * model.count + n)
        values[name] = [fitted.xk[j] if j is not None else setup.init[name]]
        errors[name] = [_error(fitted, j) if j is not None else 0.0]
    return FitResult(values, errors, chi_er, None)


def fit(source: Array, setup: PeakSetup, settings: FitSettings, stiefel: bool) -> FitResult | str:
    """Fit the peaks ``setup`` describes to ``source``'s channels ``xmin`` to ``xmax``.

    The answer is ROOT's refusal, as it words it, when every parameter is
    fixed or there are more of them than channels.
    """
    model = PeakModel(setup, settings)
    points = settings.xmax - settings.xmin + 1
    if not model.free:
        return "All parameters are fixed"
    if len(model.free) >= points:
        return "Number of fitted parameters is larger than # of fitted points"
    y = np.asarray(source, dtype=np.float64)[settings.xmin : settings.xmax + 1]
    fitted = run(model, y, settings, stiefel)
    chi_er = fitted.chi_cel / (points - len(model.free))
    result = _results(model, setup, fitted, chi_er)
    result.chi, result.spectrum = chi_er, model.shape()
    return result
