"""``TCurlyLine`` and ``TCurlyArc``: a Feynman diagram's gluons and photons.

A curly line - a gluon - is a line of loops, a wavy one - a photon - a sine
wave, along a straight line or an arc. Their wave length and amplitude are
fractions of the pad's larger side, as ROOT's are; :mod:`xrdroot.canvas.curly`
builds the loops as ``TCurlyLine::Build`` does when the pad is drawn.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .drawn import Drawn

__all__ = ["TCurlyArc", "TCurlyLine"]

#: ``TCurlyLine``'s defaults: its wave length and amplitude, and whether a new one curls.
DEFAULTS: dict[str, Any] = {"wave": 0.02, "amplitude": 0.01, "curly": True}


class TCurlyLine(Drawn):
    """``TCurlyLine(x1, y1, x2, y2, wavelength, amplitude)``: loops from one point to another."""

    classname = "TCurlyLine"
    groups: ClassVar[tuple[str, ...]] = ("line", "fill")
    fields: ClassVar[dict[str, type]] = {"X1": float, "Y1": float, "X2": float, "Y2": float,
                                         "WaveLength": float, "Amplitude": float}  # fmt: skip

    def __init__(
        self, x1: float = 0.0, y1: float = 0.0, x2: float = 0.0, y2: float = 0.0,
        wavelength: float | None = None, amplitude: float | None = None,
    ) -> None:  # fmt: skip
        super().__init__(
            fX1=float(x1), fY1=float(y1), fX2=float(x2), fY2=float(y2),
            fWaveLength=float(DEFAULTS["wave"] if wavelength is None else wavelength),
            fAmplitude=float(DEFAULTS["amplitude"] if amplitude is None else amplitude),
            fIsCurly=bool(DEFAULTS["curly"]),
        )  # fmt: skip

    def SetCurly(self) -> None:
        """Loops, as a gluon is drawn."""
        self.members["fIsCurly"] = True

    def SetWavy(self) -> None:
        """A sine wave, as a photon is drawn."""
        self.members["fIsCurly"] = False

    def GetCurly(self) -> bool:
        return bool(self.members["fIsCurly"])

    IsCurly = GetCurly

    def SetStartPoint(self, x1: float, y1: float) -> None:
        self.members.update(fX1=float(x1), fY1=float(y1))

    def SetEndPoint(self, x2: float, y2: float) -> None:
        self.members.update(fX2=float(x2), fY2=float(y2))

    @staticmethod
    def SetDefaultWaveLength(wavelength: float) -> None:
        DEFAULTS["wave"] = float(wavelength)

    @staticmethod
    def SetDefaultAmplitude(amplitude: float) -> None:
        DEFAULTS["amplitude"] = float(amplitude)

    @staticmethod
    def SetDefaultIsCurly(curly: bool) -> None:
        DEFAULTS["curly"] = bool(curly)

    @staticmethod
    def GetDefaultWaveLength() -> float:
        return float(DEFAULTS["wave"])

    @staticmethod
    def GetDefaultAmplitude() -> float:
        return float(DEFAULTS["amplitude"])

    @staticmethod
    def GetDefaultIsCurly() -> bool:
        return bool(DEFAULTS["curly"])


class TCurlyArc(TCurlyLine):
    """``TCurlyArc(x, y, radius, phimin, phimax, wavelength, amplitude)``: loops round an arc,
    from ``phimin`` degrees to ``phimax``, anticlockwise."""

    classname = "TCurlyArc"
    fields: ClassVar[dict[str, type]] = {"X1": float, "Y1": float, "Phimin": float,
                                         "Phimax": float, "WaveLength": float,
                                         "Amplitude": float}  # fmt: skip

    def __init__(self, x1: float = 0.0, y1: float = 0.0, rad: float = 0.0, phimin: float = 0.0,
                 phimax: float = 360.0, wavelength: float | None = None,
                 amplitude: float | None = None) -> None:  # fmt: skip
        super().__init__(x1, y1, x1, y1, wavelength, amplitude)
        self.members.update(fR1=float(rad), fPhimin=float(phimin), fPhimax=float(phimax))

    def SetRadius(self, radius: float) -> None:
        self.members["fR1"] = float(radius)

    def GetRadius(self) -> float:
        return float(self.members["fR1"])

    def SetCenter(self, x1: float, y1: float) -> None:
        self.members.update(fX1=float(x1), fY1=float(y1))
