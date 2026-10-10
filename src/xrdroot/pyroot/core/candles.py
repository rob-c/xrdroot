"""``TCandle``: ROOT's settings for every candle plot, and one candle's numbers from a slice.

``TCandle::SetWhiskerRange``, ``SetBoxRange``, ``SetScaledCandle`` and
``SetScaledViolin`` are the statics a macro sets before drawing; a
``TCandle`` of its own, given a slice (a ``TH1D``) and an option, works
out the median, the box, the whiskers and the mean as ``Calculate`` does,
through :mod:`xrdroot.plot.candle`. ROOT's ``GetQ1`` hands back the box's
upper end and ``GetQ3`` its lower one, and so do these.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ...plot.candle import HORIZONTAL, SETTINGS, candle_of, parse_option, part
from .objects import TAttFill, TAttLine, TAttMarker
from .wrapping import unwrap

__all__ = ["TCandle"]


class TCandle(TAttLine, TAttFill, TAttMarker):
    """``TCandle()``, ``TCandle(option)``, or ``TCandle(position, width, slice)``."""

    CLASS_TITLE = "A candle"
    (kNoOption, kBox, kMedianLine, kMedianNotched, kMedianCircle, kMeanLine, kMeanCircle,
     kWhiskerAll, kWhisker15, kAnchor, kPointsOutliers, kPointsAll, kPointsAllScat,
     kHistoLeft, kHistoRight, kHistoViolin, kHistoZeroIndicator, kHorizontal) = (
        0, 1, 10, 20, 30, 100, 300, 1000, 2000, 10000, 100000, 200000, 300000, 1000000,
        2000000, 3000000, 10000000, 100000000)  # fmt: skip

    def __init__(self, *args: Any) -> None:
        self._option, self._spelling = 0, ""
        self._pos, self._width, self._proj = 0.0, 1.0, None
        self._found: Any = None
        self._set: dict[str, float] = {}
        if len(args) == 1:
            self.ParseOption(args[0])
        elif len(args) == 3:
            self._pos, self._width, self._proj = float(args[0]), float(args[1]), args[2]
        elif len(args) == 4:
            raise UnsupportedFeatureError(
                "a TCandle of raw points is not made here; fill a TH1D with them and hand "
                "it the histogram"
            )

    # -- the settings every candle shares ------------------------------------------------------

    @staticmethod
    def SetWhiskerRange(wrange: float) -> None:
        SETTINGS["whisker_range"] = float(wrange)

    @staticmethod
    def SetBoxRange(brange: float) -> None:
        SETTINGS["box_range"] = float(brange)

    @staticmethod
    def SetScaledCandle(cscale: bool = True) -> None:
        SETTINGS["scaled_candle"] = bool(cscale)

    @staticmethod
    def SetScaledViolin(vscale: bool = True) -> None:
        SETTINGS["scaled_violin"] = bool(vscale)

    @staticmethod
    def IsCandleScaled() -> bool:
        return bool(SETTINGS["scaled_candle"])

    @staticmethod
    def IsViolinScaled() -> bool:
        return bool(SETTINGS["scaled_violin"])

    # -- this candle's option -----------------------------------------------------------------

    def ParseOption(self, option: str) -> int:
        """``ParseOption``: the digits of ``CANDLEX2``, ``VIOLINY(112000000)`` and the rest."""
        self._spelling = str(option)
        self._option = parse_option(self._spelling)
        return self._option

    def GetOption(self) -> int:
        return self._option

    def SetOption(self, option: int) -> None:
        self._option = int(option)

    def GetDrawOption(self) -> str:
        return self._spelling

    def IsOption(self, option: int) -> bool:
        """``IsOption``: whether the option's digit at ``option``'s place is ``option``'s."""
        if option == self.kHorizontal:
            return self._option >= HORIZONTAL
        if option == 0:
            return self._option == 0
        place = int(10 ** (len(str(int(option))) - 1))
        return bool(part(self._option, place) == option // place)

    def IsHorizontal(self) -> bool:
        return self.IsOption(self.kHorizontal)

    def IsVertical(self) -> bool:
        return not self.IsHorizontal()

    # -- this candle's numbers ----------------------------------------------------------------

    def SetHistogram(self, proj: Any) -> None:
        self._proj, self._found = proj, None

    def SetAxisPosition(self, pos: float) -> None:
        self._pos = float(pos)

    def SetCandleWidth(self, width: float) -> None:
        self._width = float(width)

    def SetHistoWidth(self, width: float) -> None:
        """``SetHistoWidth``: the violin's width, which is the candle's here."""
        self._width = float(width)

    def SetLog(self, x: int, y: int, z: int) -> None:
        """``SetLog``: noted; a candle here is drawn on the pad's axes as they are."""

    def _candle(self) -> Any:
        """``Calculate``: the slice summed up, once, the first time a number is asked for."""
        if self._found is None and self._proj is not None:
            rng = np.random.default_rng(0)
            self._found = candle_of(unwrap(self._proj), self._pos, self._width, self._option, rng)
        return self._found

    def _number(self, name: str) -> float:
        """A number set by hand, else the slice's - or zero with no slice to take it from."""
        if name in self._set:
            return self._set[name]
        found = self._candle()
        return float(getattr(found, name)) if found is not None else 0.0

    def GetMean(self) -> float:
        return self._number("mean")

    def GetMedian(self) -> float:
        return self._number("median")

    def GetQ2(self) -> float:
        return self._number("median")

    def GetQ1(self) -> float:
        return self._number("box_up")

    def GetQ3(self) -> float:
        return self._number("box_down")

    def SetMean(self, mean: float) -> None:
        self._set["mean"] = float(mean)

    def SetMedian(self, median: float) -> None:
        self._set["median"] = float(median)

    def SetQ2(self, q2: float) -> None:
        self._set["median"] = float(q2)

    def SetQ1(self, q1: float) -> None:
        self._set["box_up"] = float(q1)

    def SetQ3(self, q3: float) -> None:
        self._set["box_down"] = float(q3)
