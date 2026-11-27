"""``TGeoPatternFinder``: the base of the pattern finders that divide a volume along an axis.

ROOT keeps this class beside ``TGeoPatternX``/``Y``/``Z`` (``geom/geom/inc/TGeoPatternFinder.h``),
and a geometry file's streamer list names it, so a file read here has to answer for it: a class the
namespace provides has a dictionary, which is what keeps ROOT's ``TClass::Init`` silent about it.

What the class holds is the division itself - a start, a step, an end, how many divisions, which
division comes first in the volume's node list, and the volume it applies to. The finders that
derive from it are the axis each divides along; this base's own ``CreateMatrix`` and ``MakeCopy``
are pure virtual in ROOT, so a program asking this one for a matrix is refused rather than answered.
"""

from __future__ import annotations

from typing import Any

from ..core.objects import TObject

__all__ = ["TGeoPatternFinder"]

#: ROOT's ``TGeoPatternFinder::kPatternReflected`` and ``kPatternSpacedOut`` - the two flags a
#: finder keeps in ``TObject``'s bits, 14 and 15.
REFLECTED = 14
SPACED_OUT = 15

#: ``TGeoPatternFinder::GetByteCount()``: how many bytes one division takes in a file.
BYTE_COUNT = 36


class TGeoPatternFinder(TObject):
    """``TGeoPatternFinder``: one axis of a volume divided into ``ndiv`` steps."""

    CLASS_TITLE = "patterns to divide volumes"

    def __init__(self, volume: Any = None, ndiv: int = 0) -> None:
        super().__init__()
        self._step = 0.0
        self._start = 0.0
        self._end = 0.0
        self._ndivisions = int(ndiv)
        self._div_index = 0
        self._volume = volume
        self._current = 0
        self._next = 0

    # -- what the division is ---------------------------------------------------------------

    def SetRange(self, start: float, step: float, ndivisions: int) -> None:
        """``SetRange``: the division's start, step and count - and the end it implies.

        ``fEnd = fStart + ndivisions * step``, as ROOT's own ``SetRange`` has it.
        """
        self._start = float(start)
        self._step = float(step)
        self._ndivisions = int(ndivisions)
        self._end = self._start + self._ndivisions * self._step

    def GetStart(self) -> float:
        return self._start

    def GetStep(self) -> float:
        return self._step

    def GetEnd(self) -> float:
        return self._end

    def GetNdiv(self) -> int:
        return self._ndivisions

    # -- which division, in which volume ----------------------------------------------------

    def GetDivIndex(self) -> int:
        return self._div_index

    def SetDivIndex(self, index: int) -> None:
        self._div_index = int(index)

    def GetVolume(self) -> Any:
        return self._volume

    def SetVolume(self, volume: Any) -> None:
        self._volume = volume

    def GetNodeOffset(self, idiv: int) -> Any:
        """The node of division ``idiv``: the volume's node after the first division's."""
        volume = self._volume
        if volume is None:
            return None
        return volume.GetNode(self._div_index + int(idiv)) if hasattr(volume, "GetNode") else None

    # -- the walk ROOT's ``CdNext`` does ---------------------------------------------------

    def cd(self, idiv: int) -> None:
        """``cd``: make division ``idiv`` current - the axis finders move the matrix here."""

    def GetCurrent(self) -> int:
        return self._current

    def GetNext(self) -> int:
        return self._next

    def SetNext(self, index: int) -> None:
        self._next = int(index)

    def CdNext(self) -> Any:
        """``CdNext``: make the next division current and answer its node, ``None`` at the end."""
        if self._next < 0:
            return None
        self.cd(self._next)
        self._current = self._next
        return self.GetNodeOffset(self._current)

    # -- the flags and the constants -------------------------------------------------------

    def GetByteCount(self) -> int:
        return BYTE_COUNT

    def GetDivAxis(self) -> int:
        """Which axis this finder divides along: ``1`` for the base, as ROOT's default has it."""
        return 1

    def IsReflected(self) -> bool:
        return bool(self.TestBit(REFLECTED))

    def Reflect(self, flag: bool = True) -> None:
        self.SetBit(REFLECTED, flag)

    def IsSpacedOut(self) -> bool:
        return bool(self.TestBit(SPACED_OUT))

    def SetSpacedOut(self, flag: bool) -> None:
        self.SetBit(SPACED_OUT, flag)

    def FindNode(self, point: Any, direction: Any = None) -> Any:
        """``FindNode``: ROOT's base answers nothing, and the axis finders are what look."""
        return None

    def IsOnBoundary(self, point: Any) -> bool:
        return False

    def UpdateMatrix(self, idiv: int, matrix: Any) -> None:
        """``UpdateMatrix``: the axis finders fill the matrix in; the base has no axis to fill."""

    # -- what only a derived finder can answer ---------------------------------------------

    def CreateMatrix(self) -> Any:
        """``CreateMatrix`` is pure virtual in ROOT: only an axis finder has a matrix to make."""
        from ...errors import UnsupportedFeatureError

        raise UnsupportedFeatureError(
            "TGeoPatternFinder::CreateMatrix is a pure virtual of ROOT's base class: the matrix "
            "of a division belongs to the axis finder that made it (TGeoPatternX, TGeoPatternY or "
            "TGeoPatternZ). Use one of those, which knows which axis it divides."
        )

    def MakeCopy(self, reflect: bool = False) -> Any:
        """``MakeCopy`` is pure virtual in ROOT, as ``CreateMatrix`` is."""
        from ...errors import UnsupportedFeatureError

        raise UnsupportedFeatureError(
            "TGeoPatternFinder::MakeCopy is a pure virtual of ROOT's base class: a copy of a "
            "division is the axis finder's own (TGeoPatternX, TGeoPatternY or TGeoPatternZ). Copy "
            "one of those instead."
        )
