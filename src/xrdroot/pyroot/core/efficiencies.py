"""``TEfficiency``: what passed out of what was tried, bin by bin, with ROOT's intervals.

Beneath is :class:`xrdroot.Efficiency` and its two histograms; the
intervals are its, which are ROOT's formulas - Clopper-Pearson by default,
``SetStatisticOption`` for another - and bins are ROOT's global bins.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...efficiency import METHODS, Efficiency
from .histcore import axis_specs
from .objects import TAttFill, TAttLine, TAttMarker, TNamed
from .wrapping import adopt, register, remember, unwrap, wrap

__all__ = ["TEfficiency"]

#: ``EStatOption``, and the method each is in xrdroot's words.
OPTIONS = {"kFCP": 0, "kFNormal": 1, "kFWilson": 2, "kFAC": 3, "kFFC": 4, "kBJeffrey": 5,
           "kBUniform": 6, "kBBayesian": 7, "kMidP": 8}  # fmt: skip


class TEfficiency(TNamed, TAttLine, TAttFill, TAttMarker):
    """``TEfficiency``: an efficiency and its uncertainty in each bin."""

    CLASS_TITLE = "calculating efficiencies"
    kFCP, kFNormal, kFWilson, kFAC, kFFC, kBJeffrey, kBUniform, kBBayesian, kMidP = range(9)

    def __init__(self, *args: Any) -> None:
        TNamed.__init__(self)
        if args and hasattr(args[0], "_xrd"):
            self._xrd = Efficiency.from_histograms(unwrap(args[0]), unwrap(args[1]))
        elif len(args) > 2:
            dimensions = sum(1 for _ in _counted(args[2:]))
            specs, _rest = axis_specs(args[2:], dimensions)
            self._xrd = Efficiency.book(str(args[0]), *specs, title=str(args[1]))
        else:
            self._xrd = Efficiency.book("eff", (1, 0.0, 1.0))
        remember(self._xrd, self)

    def _adopted(self, xrd: Any) -> None:
        TNamed.__init__(self)
        self._xrd = xrd

    def _attribute_holder(self) -> dict[str, Any]:
        members: dict[str, Any] = self._xrd.members
        return members

    def GetName(self) -> str:
        return str(self._xrd.members["TNamed"]["fName"])

    def SetName(self, name: Any) -> None:
        self._xrd.members["TNamed"]["fName"] = str(name)

    def GetTitle(self) -> str:
        return str(self._xrd.members["TNamed"]["fTitle"])

    def SetTitle(self, title: Any = "") -> None:
        """``SetTitle("title;x;y")``: its title, and its histograms' axes' titles."""
        parts = str(title).split(";")
        self._xrd.members["TNamed"]["fTitle"] = parts[0]
        for histogram in (self._xrd.passed, self._xrd.total):
            for letter, label in zip("XYZ", parts[1:]):
                histogram._core[f"f{letter}axis"]["TNamed"]["fTitle"] = label

    # -- filling ---------------------------------------------------------------------------

    def Fill(self, bPassed: Any, x: float, y: Any = None, z: Any = None) -> None:
        coordinates = [value for value in (x, y, z) if value is not None][: self.GetDimension()]
        self._xrd.fill(bool(bPassed), *coordinates)

    def FillWeighted(
        self, bPassed: Any, weight: float, x: float, y: Any = None, z: Any = None
    ) -> None:
        coordinates = [value for value in (x, y, z) if value is not None][: self.GetDimension()]
        self._xrd.fill(bool(bPassed), *coordinates, weight=float(weight))

    def GetDimension(self) -> int:
        return len(self._xrd.axes)

    def GetGlobalBin(self, binx: int, biny: int = 0, binz: int = 0) -> int:
        return int(self.GetTotalHistogram().GetBin(binx, biny, binz))

    def FindFixBin(self, x: float, y: float = 0.0, z: float = 0.0) -> int:
        return int(self.GetTotalHistogram().FindBin(x, y, z))

    def GetPassedHistogram(self) -> Any:
        return wrap(self._xrd.passed)

    def GetTotalHistogram(self) -> Any:
        return wrap(self._xrd.total)

    def GetCopyPassedHisto(self) -> Any:
        return wrap(self._xrd.passed.copy())

    def GetCopyTotalHisto(self) -> Any:
        return wrap(self._xrd.total.copy())

    # -- the answers ---------------------------------------------------------------------------

    def _flat(self, values: Any) -> np.ndarray[Any, Any]:
        return np.asarray(values).ravel(order="F")

    def GetEfficiency(self, bin: int) -> float:
        return float(self._flat(self._xrd.values(flow=True))[int(bin)])

    def GetEfficiencyErrorLow(self, bin: int) -> float:
        return float(self._flat(self._xrd.errors(flow=True)[0])[int(bin)])

    def GetEfficiencyErrorUp(self, bin: int) -> float:
        return float(self._flat(self._xrd.errors(flow=True)[1])[int(bin)])

    def SetStatisticOption(self, option: int) -> None:
        self._xrd.members["fStatisticOption"] = int(option)

    def GetStatisticOption(self) -> int:
        return int(self._xrd.members.get("fStatisticOption", 0))

    def SetConfidenceLevel(self, level: float) -> None:
        self._xrd.members["fConfLevel"] = float(level)

    def GetConfidenceLevel(self) -> float:
        return float(self._xrd.level)

    def SetBetaAlpha(self, alpha: float) -> None:
        self._xrd.members["fBeta_alpha"] = float(alpha)

    def SetBetaBeta(self, beta: float) -> None:
        self._xrd.members["fBeta_beta"] = float(beta)

    def GetBetaAlpha(self, bin: int = -1) -> float:
        return float(self._xrd.members.get("fBeta_alpha", 1.0))

    def GetBetaBeta(self, bin: int = -1) -> float:
        return float(self._xrd.members.get("fBeta_beta", 1.0))

    def UsesBayesianStat(self) -> bool:
        return METHODS[self.GetStatisticOption()] in ("jeffreys", "uniform", "bayesian")

    def CreateGraph(self, option: str = "") -> Any:
        """``CreateGraph``: a ``TGraphAsymmErrors`` of the bins anything was tried in."""
        from ...graph import Graph

        return wrap(Graph.from_histogram(self._xrd))

    def CreateHistogram(self, option: str = "") -> Any:
        """``CreateHistogram``: a histogram of the efficiencies, the same binning as the counts."""
        made = self._xrd.total.copy(f"{self.GetName()}")
        made.reset()
        cells = made._cells()
        cells[:] = self._flat(self._xrd.values(flow=True))
        made._core["fEntries"], made._core["fTsumw"] = float(np.count_nonzero(cells)), 0.0
        return wrap(made)

    def Add(self, other: Any) -> None:
        """``Add``: another efficiency's counts added to these, bin by bin."""
        given = unwrap(other)
        self._xrd.passed.add(given.passed)
        self._xrd.total.add(given.total)

    def __iadd__(self, other: Any) -> TEfficiency:
        self.Add(other)
        return self

    @staticmethod
    def CheckConsistency(passed: Any, total: Any, option: str = "") -> bool:
        try:
            Efficiency.from_histograms(unwrap(passed), unwrap(total))
        except (ValueError, TypeError):
            return False
        return True


def _counted(args: tuple[Any, ...]) -> Any:
    """Each axis ROOT's constructor arguments give, to count them."""
    at = 0
    while (
        at < len(args)
        and isinstance(args[at], (int, np.integer))
        and not isinstance(args[at], bool)
    ):
        at += 2 if np.ndim(args[at + 1]) > 0 else 3
        yield at


register("TEfficiency", factory=lambda xrd: adopt(TEfficiency, xrd))
