"""``TProfile``, ``TProfile2D``, ``TProfile3D``: the mean of a value in each bin.

A profile is a histogram whose bin content is a mean, so ``GetBinContent``
is ``sum(w*y) / sum(w)`` and ``GetBinError`` the error of that mean - or,
after ``SetErrorOption("s")``, the spread - as :class:`xrdroot.Profile`
computes them. The constructors are ROOT's: the axes, then optionally the
range of values averaged, then the error option.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from ...booking import ERROR_OPTIONS
from ...profile import ERROR_MODES
from .histcore import axis_specs
from .hists import TH1, TH2, TH3
from .wrapping import adopt, register, wrap

__all__ = ["TProfile", "TProfile2D", "TProfile3D"]


def _value_range(rest: tuple[Any, ...]) -> tuple[Any, str]:
    """The value range and error option after the axes: ``(ylow, yup[, opt])`` or ``(opt)``."""
    numbers = [value for value in rest if not isinstance(value, str)]
    option = next((value for value in rest if isinstance(value, str)), "")
    found = (float(numbers[0]), float(numbers[1])) if len(numbers) >= 2 else None
    return found, option


class _Profiled:
    """What the three profiles share over their histogram's methods."""

    _xrd: Any
    DIM: int

    def _booked(self, args: tuple[Any, ...]) -> Any:
        from ...profile import Profile

        if not args:
            return Profile.book("", *[(1, 0.0, 1.0)] * self.DIM)
        specs, rest = axis_specs(args[2:], self.DIM)
        values, option = _value_range(rest)
        return Profile.book(str(args[0]), *specs, title=str(args[1]), error_option=option.lower(),
                            value_range=values)  # fmt: skip

    def ClassName(self) -> str:
        return str(self._xrd.classname)

    def Fill(self, *args: Any) -> int:
        """``Fill(x[, y[, z]], value[, w])``: a value averaged in the bin of the coordinates."""
        count = self.DIM + 1
        given = [float(value) for value in args[:count]]
        weight = float(args[count]) if len(args) > count else 1.0
        self._xrd.fill(*given, weight=None if weight == 1.0 else weight)
        return int(self._xrd.find_bin(*given[: self.DIM]))

    def FillN(self, ntimes: int, x: Any, y: Any, w: Any = None, stride: int = 1) -> None:
        count = int(ntimes)
        weights = None if w is None else np.asarray(w, dtype=np.float64)[:count]
        xs = np.asarray(x, dtype=np.float64)[:count]
        self._xrd.fill(xs, np.asarray(y, dtype=np.float64)[:count], weight=weights)

    def _contents(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd.values(flow=True)).ravel(order="F")

    def _errors(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd.errors(flow=True)).ravel(order="F")

    def GetBinEntries(self, bin: int) -> float:
        return float(np.asarray(self._xrd.members["fBinEntries"])[int(bin)])

    def GetBinEffectiveEntries(self, bin: int) -> float:
        return float(np.asarray(self._xrd.counts(flow=True)).ravel(order="F")[int(bin)])

    def GetSumOfWeights(self) -> float:
        """``GetSumOfWeights``: of a profile, the sum of its bins' means, as ROOT adds them up."""
        return float(np.sum(self._xrd.values()))

    def SetErrorOption(self, option: str = "") -> None:
        """``SetErrorOption``: ``""`` the mean's error, ``"s"`` the spread, ``"i"``, ``"g"``."""
        chosen = str(option).lower()
        self._xrd.members["fErrorMode"] = ERROR_OPTIONS[chosen]
        self._xrd.error_mode = ERROR_MODES[ERROR_OPTIONS[chosen]]

    def GetErrorOption(self) -> str:
        return str(self._xrd.error_mode)

    def ProjectionX(self, name: str = "_px", option: str = "e") -> Any:
        """``ProjectionX``: a ``TH1D`` of the means - with ``"b"``, the entries - as ROOT's."""
        from ...hist import Histogram

        axis = self._xrd.axes[0]
        called = f"{self.GetName()}{name}" if name == "_px" else str(name)  # type: ignore[attr-defined]
        chosen = str(option).lower()
        values = self._xrd.bin_entries() if "b" in chosen else self._xrd.values()
        errors = None if "b" in chosen else self._xrd.errors()
        made = Histogram.new(called, axis.edges(), values, title=self.GetTitle(), errors=errors)  # type: ignore[attr-defined]
        made.members["TH1"]["fEntries"] = float(self.GetEntries())  # type: ignore[attr-defined]
        return wrap(made)

    def GetYmin(self) -> float:
        return float(self._xrd.members.get("fYmin", 0.0))

    def GetYmax(self) -> float:
        return float(self._xrd.members.get("fYmax", 0.0))


class TProfile(_Profiled, TH1):  # type: ignore[misc]  # ProjectionX takes a profile's arguments
    """``TProfile``: the mean of y in bins of x."""

    CLASS_TITLE = "Profile histogram class"
    DIM = 1
    KIND = "D"


class TProfile2D(_Profiled, TH2):  # type: ignore[misc]  # ProjectionX takes a profile's arguments
    """``TProfile2D``: the mean of z in bins of x and y."""

    CLASS_TITLE = "Profile2D histogram class"
    DIM = 2
    KIND = "D"


class TProfile3D(_Profiled, TH3):  # type: ignore[misc]  # ProjectionX takes a profile's arguments
    """``TProfile3D``: the mean of t in bins of x, y and z."""

    CLASS_TITLE = "Profile3D histogram class"
    DIM = 3
    KIND = "D"


for _cls in (TProfile, TProfile2D, TProfile3D):
    register(_cls.__name__, factory=partial(adopt, _cls))
