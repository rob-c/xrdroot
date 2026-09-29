"""``TSpectrumFit`` and ``TSpectrum2Fit``: Morhac's peak fitters, as ROOT's classes.

A fitter is told its channels and how to step (``SetFitParameters``), where
its peaks start and which of them stay put (``SetPeakParameters``, with the
tails' and background's own setters), and then fits a spectrum in place:
``FitAwmi(source)`` or ``FitStiefel(source)`` leave the fitted spectrum in
``source`` over the fitted channels, and the values found in the arrays
``GetPositions``, ``GetAmplitudes`` and ``GetAreas`` hand back - the
fitter's own, which a script may change as it changes ROOT's. The fitting
is :mod:`xrdroot.spectrum.fit`; what these classes add is ROOT's checks,
with its words for what they refuse.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...spectrum import fit as fit1
from ...spectrum.fitengine import FitSettings
from ..core.objects import TNamed
from ..core.refs import store
from .arrays import vector_in, vector_out

__all__ = ["TSpectrumFit"]

Array = Any


class _Options:
    """The fitters' shared enumeration, ``TSpectrumFit::kFitOptimChiCounts`` and the rest."""

    kFitOptimChiCounts = 0
    kFitOptimChiFuncValues = 1
    kFitOptimMaxLikelihood = 2
    kFitAlphaHalving = 0
    kFitAlphaOptimal = 1
    kFitPower2 = 2
    kFitPower4 = 4
    kFitPower6 = 6
    kFitPower8 = 8
    kFitPower10 = 10
    kFitPower12 = 12
    kFitTaylorOrderFirst = 0
    kFitTaylorOrderSecond = 1
    kFitNumRegulCycles = 100


def _settings_error(settings: FitSettings) -> str | None:
    """What ``SetFitParameters`` says of iterations, a step, or a choice it does not know."""
    checks = (
        (settings.iterations <= 0, "Invalid number of iterations, must be positive"),
        (not 0 < settings.alpha <= 1,
         "Invalid step coefficient alpha, must be > than 0 and <=1"),
        (settings.statistic not in (0, 1, 2), "Wrong type of statistic"),
        (settings.alpha_optim not in (0, 1), "Wrong optimization algorithm"),
        (settings.power not in (2, 4, 6, 8, 10, 12), "Wrong power"),
        (settings.taylor not in (0, 1), "Wrong order of Taylor development"),
    )  # fmt: skip
    return next((message for failed, message in checks if failed), None)


def _flags(values: Any, count: int) -> list[bool]:
    """A ``Bool_t *``'s first ``count`` flags."""
    return [bool(values[k]) for k in range(count)]


def _kept(old: Array, new: list[float | None]) -> Array:
    """``new``, but ``old`` where ROOT leaves an error as it was."""
    return np.array([o if n is None else n for o, n in zip(old.tolist(), new)], dtype=np.float64)


class TSpectrumFit(_Options, TNamed):
    """ROOT's ``TSpectrumFit``: peaks of one sigma, fitted by AWMI or Stiefel's method."""

    def __init__(self, numberPeaks: int | None = None) -> None:
        super().__init__("SpectrumFit", "Miroslav Morhac peak fitter")
        count = 0 if numberPeaks is None else int(numberPeaks)
        self._settings = FitSettings()
        self._setup = fit1.PeakSetup([], [], [], [])
        self._calc: dict[str, float] = dict.fromkeys(fit1.SHARED, 0.0)
        self._calc["sigma"] = 1.0
        self._err: dict[str, float] = dict.fromkeys(fit1.SHARED, 0.0)
        self.fChi = 0.0
        if numberPeaks is not None and count <= 0:
            self.Error("TSpectrumFit", "Invalid number of peaks, must be > than 0")
            count = 0
        self.fNPeaks = count
        self._setup = fit1.PeakSetup([0.0] * count, [0.0] * count, [False] * count, [False] * count)
        names = ("PositionCalc", "PositionErr", "AmpCalc", "AmpErr", "Area", "AreaErr")
        self._arrays = {name: np.zeros(count) for name in names}

    def SetFitParameters(self, xmin: int, xmax: int, numberIterations: int, alpha: float,
                         statisticType: int, alphaOptim: int, power: int,
                         fitTaylor: int) -> None:  # fmt: skip
        """``SetFitParameters``: the channels fitted, how many iterations, and how to step."""
        if xmin < 0 or xmax <= xmin:
            self.Error("SetFitParameters", "Wrong range")
            return
        chosen = (int(statisticType), int(alphaOptim), int(power), int(fitTaylor))
        settings = FitSettings(int(xmin), int(xmax), int(numberIterations), float(alpha), *chosen)
        refused = _settings_error(settings)
        if refused:
            self.Error("SetFitParameters", refused)
            return
        self._settings = settings

    def SetPeakParameters(self, sigma: float, fixSigma: bool, positionInit: Any, fixPosition: Any,
                          ampInit: Any, fixAmp: Any) -> None:  # fmt: skip
        """``SetPeakParameters``: sigma and each peak's starting position and amplitude."""
        if sigma <= 0:
            self.Error("SetPeakParameters", "Invalid sigma, must be > than 0")
            return
        count, low, high = self.fNPeaks, self._settings.xmin, self._settings.xmax
        positions, amplitudes = vector_in(positionInit, count), vector_in(ampInit, count)
        for position, amplitude in zip(positions.tolist(), amplitudes.tolist()):
            if not low <= int(position) <= high:
                self.Error("SetPeakParameters",
                           "Invalid peak position, must be in the range fXmin, fXmax")  # fmt: skip
                return
            if amplitude < 0:
                self.Error("SetPeakParameters", "Invalid peak amplitude, must be > than 0")
                return
        self._setup.init["sigma"], self._setup.fixed["sigma"] = float(sigma), bool(fixSigma)
        self._setup.positions, self._setup.amplitudes = positions.tolist(), amplitudes.tolist()
        self._setup.fix_positions = _flags(fixPosition, count)
        self._setup.fix_amplitudes = _flags(fixAmp, count)

    def _set(self, names: tuple[str, ...], values: tuple[Any, ...]) -> None:
        for n, name in enumerate(names):
            self._setup.init[name] = float(values[2 * n])
            self._setup.fixed[name] = bool(values[2 * n + 1])

    def SetBackgroundParameters(self, a0Init: float, fixA0: bool, a1Init: float, fixA1: bool,
                                a2Init: float, fixA2: bool) -> None:  # fmt: skip
        """``SetBackgroundParameters``: the background ``a0 + a1*x + a2*x*x`` to start from."""
        self._set(("a0", "a1", "a2"), (a0Init, fixA0, a1Init, fixA1, a2Init, fixA2))

    def SetTailParameters(self, tInit: float, fixT: bool, bInit: float, fixB: bool,
                          sInit: float, fixS: bool) -> None:  # fmt: skip
        """``SetTailParameters``: the tail's amplitude ``t`` and slope ``b``, the step's ``s``."""
        self._set(("t", "b", "s"), (tInit, fixT, bInit, fixB, sInit, fixS))

    def _fit(self, source: Any, stiefel: bool) -> None:
        """Fit ``source``, keep what was found, and leave the fitted spectrum in ``source``."""
        settings = self._settings
        spectrum = vector_in(source, settings.xmax + 1)
        found = fit1.fit(spectrum, self._setup, settings, stiefel)
        if isinstance(found, str):
            self.Error("FitAwmi", found)
            return
        arrays, values, errors = self._arrays, found.values, found.errors
        for name, kind in (("PositionCalc", "pos"), ("AmpCalc", "amp"), ("Area", "area")):
            arrays[name][:] = values[kind]
        for name, kind in (("PositionErr", "pos"), ("AmpErr", "amp"), ("AreaErr", "area")):
            arrays[name][:] = _kept(arrays[name], errors[kind])
        for name in fit1.SHARED:
            self._calc[name] = values[name][0]
            self._err[name] = _kept(np.array([self._err[name]]), errors[name])[0]
        self.fChi = found.chi
        spectrum[settings.xmin : settings.xmax + 1] = found.spectrum
        vector_out(source, spectrum)

    def FitAwmi(self, source: Any) -> None:
        """``FitAwmi``: fit by the algorithm without matrix inversion."""
        self._fit(source, stiefel=False)

    def FitStiefel(self, source: Any) -> None:
        """``FitStiefel``: fit by Stiefel and Hestenes' conjugate gradients."""
        self._fit(source, stiefel=True)

    def GetPositions(self) -> Array:
        return self._arrays["PositionCalc"]

    def GetPositionsErrors(self) -> Array:
        return self._arrays["PositionErr"]

    def GetAmplitudes(self) -> Array:
        return self._arrays["AmpCalc"]

    def GetAmplitudesErrors(self) -> Array:
        return self._arrays["AmpErr"]

    def GetAreas(self) -> Array:
        return self._arrays["Area"]

    def GetAreasErrors(self) -> Array:
        return self._arrays["AreaErr"]

    def GetChi(self) -> float:
        return self.fChi

    def _get(self, names: tuple[str, ...], targets: tuple[Any, ...]) -> tuple[float, ...]:
        """Each named value and its error, put where ``targets`` say and handed back too."""
        found = tuple(x for name in names for x in (self._calc[name], self._err[name]))
        for target, value in zip(targets, found):
            store(target, value)
        return found

    def GetSigma(self, sigma: Any = None, sigmaErr: Any = None) -> tuple[float, ...]:
        """``GetSigma(sigma, sigmaErr)``: the fitted sigma and its error, by reference."""
        return self._get(("sigma",), (sigma, sigmaErr))

    def GetBackgroundParameters(self, *targets: Any) -> tuple[float, ...]:
        """``GetBackgroundParameters(a0, a0Err, a1, a1Err, a2, a2Err)``, by reference."""
        return self._get(("a0", "a1", "a2"), targets)

    def GetTailParameters(self, *targets: Any) -> tuple[float, ...]:
        """``GetTailParameters(t, tErr, b, bErr, s, sErr)``, by reference."""
        return self._get(("t", "b", "s"), targets)
