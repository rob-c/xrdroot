"""``TSpectrum2Fit``: 2-D peaks and their ridges, fitted as ROOT's class fits them.

Its getters, like ROOT's, fill the arrays they are handed -
``GetPositions(x, y, x1, y1)`` - and put the shared parameters by reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...spectrum.fit2 import PEAK, SHARED2, FitSettings2, Peak2Setup
from ...spectrum.fit2result import fit2
from ..core.objects import TNamed
from ..core.refs import store
from .arrays import matrix_in, matrix_out, vector_in, vector_out
from .fits import _kept, _Options, _settings_error

__all__ = ["TSpectrum2Fit"]

#: What ``SetPeakParameters`` checks of each peak's starts, in its order, and says of one out.
_CHECKS = (
    ("x", "Invalid peak position, must be in the range fXmin, fXmax"),
    ("y", "Invalid peak position, must be in the range fYmin, fYmax"),
    ("x1", "Invalid ridge position, must be in the range fXmin, fXmax"),
    ("y1", "Invalid ridge position, must be in the range fYmin, fYmax"),
    ("amp", "Invalid peak amplitude, must be > than 0"),
    ("ampx", "Invalid x ridge amplitude, must be > than 0"),
    ("ampy", "Invalid y ridge amplitude, must be > than 0"),
)

#: The order ``SetPeakParameters`` takes each peak's starts and fixes in.
_TAKEN = ("x", "y", "x1", "y1", "amp", "ampx", "ampy")


class TSpectrum2Fit(_Options, TNamed):
    """ROOT's ``TSpectrum2Fit``: 2-D peaks with ridges, fitted by AWMI or Stiefel's method."""

    def __init__(self, numberPeaks: int | None = None) -> None:
        super().__init__("Spectrum2Fit", "Miroslav Morhac peak fitter")
        count = 0 if numberPeaks is None else int(numberPeaks)
        if numberPeaks is not None and count <= 0:
            self.Error("TSpectrum2Fit", "Invalid number of peaks, must be > than 0")
            count = 0
        self.fNPeaks = count
        self.fChi = 0.0
        self._settings = FitSettings2()
        starts, fixes = {k: [0.0] * count for k in PEAK}, {k: [False] * count for k in PEAK}
        self._setup = Peak2Setup(starts, fixes)
        self._calc = dict.fromkeys(SHARED2, 0.0)
        self._err = dict.fromkeys(SHARED2, 0.0)
        kinds = (*PEAK, "volume")
        self._values = {k: np.zeros(count) for k in kinds}
        self._errors = {k: np.zeros(count) for k in kinds}

    def SetFitParameters(self, xmin: int, xmax: int, ymin: int, ymax: int, numberIterations: int,
                         alpha: float, statisticType: int, alphaOptim: int, power: int,
                         fitTaylor: int) -> None:  # fmt: skip
        """``SetFitParameters``: the channels fitted in x and y, the iterations, the steps."""
        if xmin < 0 or xmax <= xmin or ymin < 0 or ymax <= ymin:
            self.Error("SetFitParameters", "Wrong range")
            return
        chosen = (int(statisticType), int(alphaOptim), int(power), int(fitTaylor))
        settings = FitSettings2(int(xmin), int(xmax), int(numberIterations), float(alpha),
                                *chosen, int(ymin), int(ymax))  # fmt: skip
        refused = _settings_error(settings)
        if refused:
            self.Error("SetFitParameters", refused)
            return
        self._settings = settings

    def _refusal(self, starts: dict[str, Any]) -> str | None:
        """What ``SetPeakParameters`` says of the first start out of its range."""
        s = self._settings
        ranges = {"x": (s.xmin, s.xmax), "y": (s.ymin, s.ymax), "x1": (s.xmin, s.xmax),
                  "y1": (s.ymin, s.ymax)}  # fmt: skip
        for k in range(self.fNPeaks):
            for kind, message in _CHECKS:
                low, high = ranges.get(kind, (0.0, np.inf))
                if not low <= starts[kind][k] <= high:
                    return message
        return None

    def SetPeakParameters(self, sigmaX: float, fixSigmaX: bool, sigmaY: float, fixSigmaY: bool,
                          ro: float, fixRo: bool, *peaks: Any) -> None:  # fmt: skip
        """``SetPeakParameters``: the sigmas and ``ro``, then each start and its fix.

        After ``ro`` come ROOT's fourteen arrays: the 2-D peaks' x and y
        positions, the ridges' x and y positions, and the amplitudes of the
        peaks and of their x and y ridges, each followed by its fixes.
        """
        if sigmaX <= 0 or sigmaY <= 0:
            self.Error("SetPeakParameters", "Invalid sigma, must be > than 0")
            return
        if ro < -1 or ro > 1:
            self.Error("SetPeakParameters", "Invalid ro, must be from region <-1,1>")
            return
        count = self.fNPeaks
        starts = {kind: vector_in(peaks[2 * n], count).tolist() for n, kind in enumerate(_TAKEN)}
        refused = self._refusal(starts)
        if refused:
            self.Error("SetPeakParameters", refused)
            return
        for n, kind in enumerate(_TAKEN):
            self._setup.peaks[kind] = starts[kind]
            self._setup.fix[kind] = [bool(peaks[2 * n + 1][k]) for k in range(count)]
        for name, value, fixed in (("sigmax", sigmaX, fixSigmaX), ("sigmay", sigmaY, fixSigmaY),
                                   ("ro", ro, fixRo)):  # fmt: skip
            self._setup.init[name], self._setup.fixed[name] = float(value), bool(fixed)

    def _set(self, names: tuple[str, ...], values: tuple[Any, ...]) -> None:
        for n, name in enumerate(names):
            self._setup.init[name] = float(values[2 * n])
            self._setup.fixed[name] = bool(values[2 * n + 1])

    def SetBackgroundParameters(self, a0Init: float, fixA0: bool, axInit: float, fixAx: bool,
                                ayInit: float, fixAy: bool) -> None:  # fmt: skip
        """``SetBackgroundParameters``: the background ``a0 + ax*x + ay*y`` to start from."""
        self._set(("a0", "ax", "ay"), (a0Init, fixA0, axInit, fixAx, ayInit, fixAy))

    def SetTailParameters(self, *values: Any) -> None:
        """``SetTailParameters(tInitXY, fixTxy, tInitX, fixTx, ...)``: the tails and steps.

        The order is ROOT's: the 2-D tail's and the ridges' amplitudes, the
        slopes in x and y, then the 2-D step's and the ridges' steps.
        """
        self._set(_TAILS, values)

    def _fit(self, source: Any, stiefel: bool) -> None:
        """Fit ``source``, keep what was found, and leave the fitted spectrum in ``source``."""
        s = self._settings
        spectrum = matrix_in(source, s.xmax + 1, s.ymax + 1)
        found = fit2(spectrum, self._setup, s, stiefel)
        for kind in (*PEAK, "volume"):
            self._values[kind][:] = found.values[kind]
            self._errors[kind][:] = _kept(self._errors[kind], found.errors[kind])
        for name in SHARED2:
            self._calc[name] = found.values[name][0]
            self._err[name] = _kept(np.array([self._err[name]]), found.errors[name])[0]
        self.fChi = found.chi
        spectrum[s.xmin : s.xmax + 1, s.ymin : s.ymax + 1] = found.spectrum
        matrix_out(source, spectrum)

    def FitAwmi(self, source: Any) -> None:
        """``FitAwmi``: fit by the algorithm without matrix inversion."""
        self._fit(source, stiefel=False)

    def FitStiefel(self, source: Any) -> None:
        """``FitStiefel``: fit by Stiefel and Hestenes' conjugate gradients."""
        self._fit(source, stiefel=True)

    @staticmethod
    def _fill(found: dict[str, Any], kinds: tuple[str, ...], targets: tuple[Any, ...]) -> None:
        """Each of ``kinds`` into the array the caller handed over for it, as ROOT fills them."""
        for kind, target in zip(kinds, targets):
            vector_out(target, found[kind])

    def GetPositions(self, *targets: Any) -> None:
        """``GetPositions(x, y, x1, y1)``: the peaks' and ridges' positions, into the arrays."""
        self._fill(self._values, ("x", "y", "x1", "y1"), targets)

    def GetPositionErrors(self, *targets: Any) -> None:
        """``GetPositionErrors(x, y, x1, y1)``."""
        self._fill(self._errors, ("x", "y", "x1", "y1"), targets)

    def GetAmplitudes(self, *targets: Any) -> None:
        """``GetAmplitudes(amplitudes, x1, y1)``: the peaks' and their ridges' amplitudes."""
        self._fill(self._values, ("amp", "ampx", "ampy"), targets)

    def GetAmplitudeErrors(self, *targets: Any) -> None:
        """``GetAmplitudeErrors(amplitudes, x1, y1)``."""
        self._fill(self._errors, ("amp", "ampx", "ampy"), targets)

    def GetVolumes(self, volumes: Any) -> None:
        """``GetVolumes``: each peak's volume, into the array."""
        self._fill(self._values, ("volume",), (volumes,))

    def GetVolumeErrors(self, volumeErrors: Any) -> None:
        """``GetVolumeErrors``."""
        self._fill(self._errors, ("volume",), (volumeErrors,))

    def GetChi(self) -> float:
        return self.fChi

    def _get(self, names: tuple[str, ...], targets: tuple[Any, ...]) -> tuple[float, ...]:
        """Each named value and its error, put where ``targets`` say and handed back too."""
        found = tuple(x for name in names for x in (self._calc[name], self._err[name]))
        for target, value in zip(targets, found):
            store(target, value)
        return found

    def GetSigmaX(self, *targets: Any) -> tuple[float, ...]:
        """``GetSigmaX(sigmaX, sigmaErrX)``, by reference."""
        return self._get(("sigmax",), targets)

    def GetSigmaY(self, *targets: Any) -> tuple[float, ...]:
        """``GetSigmaY(sigmaY, sigmaErrY)``, by reference."""
        return self._get(("sigmay",), targets)

    def GetRo(self, *targets: Any) -> tuple[float, ...]:
        """``GetRo(ro, roErr)``, by reference."""
        return self._get(("ro",), targets)

    def GetBackgroundParameters(self, *targets: Any) -> tuple[float, ...]:
        """``GetBackgroundParameters(a0, a0Err, ax, axErr, ay, ayErr)``, by reference."""
        return self._get(("a0", "ax", "ay"), targets)

    def GetTailParameters(self, *targets: Any) -> tuple[float, ...]:
        """``GetTailParameters(txy, txyErr, tx, txErr, ...)``, by reference, in ROOT's order."""
        return self._get(_TAILS, targets)


#: ``SetTailParameters``' and ``GetTailParameters``' order.
_TAILS = ("txy", "tx", "ty", "bx", "by", "sxy", "sx", "sy")
