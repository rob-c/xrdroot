"""What ``TSpectrum`` and ``TSpectrum2`` share: their peaks, their markers, their messages.

Both keep ``maxpositions`` places for peaks - ``GetPositionX`` hands back
the array itself, as ROOT hands back its pointer, so a macro indexes it -
and both hang a ``TPolyMarker`` of red triangles on a histogram they
search, replacing one already there, and draw the histogram with it.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..core.cformat import c_format
from ..core.objects import TNamed

__all__ = ["Peaks", "option_flags"]

#: ``kRed``: the colour of the markers a search hangs on its histogram.
MARKER_COLOR = 2
#: ``kFullTriangleDown``, drawn 1.3 times the usual size.
MARKER_STYLE, MARKER_SIZE = 23, 1.3


def option_flags(option: Any, *words: str) -> tuple[str, list[bool]]:
    """``option`` lowered with each of ``words`` taken out, and whether each was there."""
    text = str(option).lower()
    found = []
    for word in words:
        found.append(word in text)
        text = text.replace(word, "")
    return text, found


class Peaks(TNamed):
    """ROOT's ``Spectrum`` object: named so, and ``maxpositions`` places for peaks."""

    #: ``kBackIncreasingWindow`` and ``kBackDecreasingWindow``: the clipping window's direction.
    kBackIncreasingWindow: ClassVar[int] = 0
    kBackDecreasingWindow: ClassVar[int] = 1
    #: ``fgIterations`` and ``fgAverageWindow``: what ``Search`` deconvolves and smooths with.
    fgIterations: ClassVar[int] = 3
    fgAverageWindow: ClassVar[int] = 3

    def __init__(self, maxpositions: int = 100, resolution: float = 1) -> None:
        super().__init__("Spectrum", "Miroslav Morhac peak finder")
        self.fMaxPeaks = int(maxpositions)
        self.fPosition = np.zeros(max(self.fMaxPeaks, 0))
        self.fPositionX = np.zeros(max(self.fMaxPeaks, 0))
        self.fPositionY = np.zeros(max(self.fMaxPeaks, 0))
        self.fNPeaks = 0
        self.fHistogram: Any = None
        self.SetResolution(resolution)

    @classmethod
    def SetAverageWindow(cls, w: int = 3) -> None:
        """``SetAverageWindow``: the Markov window every ``Search`` smooths with."""
        cls.fgAverageWindow = int(w)

    @classmethod
    def SetDeconIterations(cls, n: int = 3) -> None:
        """``SetDeconIterations``: the deconvolution iterations every ``Search`` runs."""
        cls.fgIterations = int(n)

    def SetResolution(self, resolution: float = 1) -> None:
        """``SetResolution``: kept, as ROOT keeps it, and not used."""
        self.fResolution = float(resolution) if resolution > 1 else 1.0

    def GetHistogram(self) -> Any:
        return self.fHistogram

    def GetNPeaks(self) -> int:
        return self.fNPeaks

    def GetPositionX(self) -> np.ndarray[Any, Any]:
        """``GetPositionX``: the peaks' x, the array itself - ``fNPeaks`` of it filled."""
        return self.fPositionX

    def GetPositionY(self) -> np.ndarray[Any, Any]:
        return self.fPositionY

    def Print(self, option: str = "") -> None:
        """``Print``: the number of peaks and each one's x and y, in ROOT's words."""
        print(f"\nNumber of positions = {self.fNPeaks}")
        for i in range(self.fNPeaks):
            x, y = float(self.fPositionX[i]), float(self.fPositionY[i])
            print(c_format(" x[%d] = %g, y[%d] = %g", i, x, i, y))

    def _found(self, positions: list[float]) -> None:
        """Keep the peaks a search found: their x, and how many."""
        self.fPositionX[: len(positions)] = positions
        self.fNPeaks = len(positions)

    def _mark(self, hist: Any, count: int) -> None:
        """Hang the ``count`` peaks on ``hist`` as a ``TPolyMarker``, replacing the one there."""
        from ..graphics.shapes import TPolyMarker

        functions = hist.GetListOfFunctions()
        old = functions.FindObject("TPolyMarker")
        if old is not None:
            functions.Remove(old)
        marker = TPolyMarker(count, self.fPositionX, self.fPositionY)
        functions.Add(marker)
        marker.SetMarkerStyle(MARKER_STYLE)
        marker.SetMarkerColor(MARKER_COLOR)
        marker.SetMarkerSize(MARKER_SIZE)

