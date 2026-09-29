"""``PointSetInterval`` and ``ConfidenceBelt``: an interval as the points of a scan kept in it,
and the acceptance region found at each.

A Neyman construction tests each point of a scan of the parameters and keeps
those whose data statistic falls in the point's acceptance region; the
interval is that set of points - its limits in a parameter the least and
greatest kept - and the belt remembers each point's region.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..roofit.collections import RooArgSet, as_list
from ..roofit.messages import ERROR, INFO, log
from .intervals import ConfInterval, Named, same_parameters

__all__ = ["AcceptanceRegion", "ConfidenceBelt", "PointSetInterval"]


def _same_point(point: Any, row: Any) -> bool:
    return all(one.getVal() == row.getRealValue(one.GetName()) for one in as_list(point))


def _own_copy(data: Any) -> Any:
    """``data.Clone("PointsToTestForBelt")``: the points with variables of their own, so that a
    point read from the scan is not the row the belt reads to look it up."""
    import copy

    from ..roofit.data.store import copies_of

    found = copy.copy(data)
    found._vars = copies_of(as_list(data.get()))
    found.SetName("PointsToTestForBelt")
    return found


class PointSetInterval(ConfInterval):
    """The points of a scan that a construction accepted."""

    def __init__(self, name: Any = "", data: Any = None) -> None:
        super().__init__(name)
        self._points = data

    def GetParameterPoints(self) -> Any:
        return self._points

    def GetParameters(self) -> RooArgSet:
        return RooArgSet(list(self._points.get()))

    def CheckParameters(self, point: Any) -> bool:
        return same_parameters(point, self._points.get())

    def IsInInterval(self, point: Any) -> bool:
        """Whether ``point`` is one of the accepted points - for a histogram of them, a bin that
        weighs something."""
        if not self.CheckParameters(point):
            return False
        if self._points.ClassName() == "RooDataHist":
            return bool(self._points.weight(RooArgSet(as_list(point))) > 0)
        rows = range(self._points.numEntries())
        return any(_same_point(point, self._points.get(i)) for i in rows)

    def _range(self, param: Any) -> tuple[float, float]:
        """``RooDataSet::getRange``: the least and greatest kept - none, and said so, if none."""
        if not self._points.numEntries():
            log(None, ERROR, "InputArguments", f"RooDataSet::getRange({self._points.GetName()}) "
                "WARNING: empty dataset")  # fmt: skip
            return 0.0, 0.0
        column = self._points.column(param.GetName())
        return float(np.min(column)), float(np.max(column))

    def UpperLimit(self, param: Any) -> float:
        if self._points.ClassName() != "RooDataSet":
            return float(param.getMax())
        return self._range(param)[1]

    def LowerLimit(self, param: Any) -> float:
        if self._points.ClassName() != "RooDataSet":
            return float(param.getMin())
        return self._range(param)[0]


class AcceptanceRegion:
    """One point's region of the test statistic: ``[lower, upper]``."""

    def __init__(self, index: int = 0, lower: float = math.nan, upper: float = math.nan) -> None:
        self._index, self._lower, self._upper = int(index), float(lower), float(upper)

    def GetLowerLimit(self) -> float:
        return self._lower

    def GetUpperLimit(self) -> float:
        return self._upper

    def GetLookupIndex(self) -> int:
        return self._index


class ConfidenceBelt(Named):
    """The acceptance region of each point of a scan."""

    def __init__(self, name: Any = "", *rest: Any) -> None:
        title = rest[0] if rest and isinstance(rest[0], str) else None
        super().__init__(name, title)
        data = next((one for one in rest if hasattr(one, "numEntries")), None)
        self._points: Any = _own_copy(data) if data is not None else None
        self._regions: dict[int, AcceptanceRegion] = {}
        self._lookups: list[tuple[float, float]] = []

    def GetParameters(self) -> RooArgSet:
        return RooArgSet(list(self._points.get()))

    def CheckParameters(self, point: Any) -> bool:
        return same_parameters(point, self._points.get())

    def _index(self, point: Any) -> int:
        if self._points.ClassName() == "RooDataHist":
            columns = {one.GetName(): [one.getVal()] for one in as_list(point)}
            return int(self._points._bin_of(columns)[0])
        for index in range(self._points.numEntries()):
            if _same_point(point, self._points.get(index)):
                return index
        return int(self._points.numEntries())

    def AddAcceptanceRegion(self, point: Any, index: int, lower: float, upper: float,
                            cl: float = -1.0, leftside: float = -1.0) -> None:  # fmt: skip
        """The region of the ``index``-th point - the scan's bin, for a histogram of points."""
        if cl > 0 or leftside > 0:
            log(self, INFO, "Eval", "using default cl, leftside for now")
        if not self.CheckParameters(point):
            log(self, ERROR, "InputArguments", "problem with parameters")
        if (cl, leftside) not in self._lookups:
            self._lookups.append((cl, leftside))
            log(self, INFO, "Eval", f"lookup index = {self._lookups.index((cl, leftside))}")
        where = self._index(point) if self._points.ClassName() == "RooDataHist" else int(index)
        self._regions[where] = AcceptanceRegion(self._lookups.index((cl, leftside)), lower, upper)

    def GetAcceptanceRegion(self, point: Any, cl: float = -1.0, leftside: float = -1.0) -> Any:
        if cl > 0 or leftside > 0:
            log(self, INFO, "Eval", "using default cl, leftside for now")
        if not self.CheckParameters(point):
            log(self, ERROR, "InputArguments", "problem with parameters")
            return None
        region = self._regions.get(self._index(point))
        if region is None:
            raise RuntimeError(
                "ConfidenceBelt::GetAcceptanceRegion: Sampling summaries are not filled yet. "
                "Switch on NeymanConstruction::CreateConfBelt() or "
                "FeldmanCousins::CreateConfBelt()."
            )
        return region

    def GetAcceptanceRegionMin(self, point: Any, cl: float = -1.0, leftside: float = -1.0) -> float:
        region = self.GetAcceptanceRegion(point, cl, leftside)
        return region.GetLowerLimit() if region is not None else math.nan

    def GetAcceptanceRegionMax(self, point: Any, cl: float = -1.0, leftside: float = -1.0) -> float:
        region = self.GetAcceptanceRegion(point, cl, leftside)
        return region.GetUpperLimit() if region is not None else math.nan

    def ConfidenceLevels(self) -> list[float]:
        return []
