"""``TGraph``, ``TGraphErrors``, ``TGraphAsymmErrors``: points, and the bars round them.

A graph grows a point at a time - ``SetPoint(n, x, y)`` past the end adds
one - so the points are kept here as arrays that grow, and the
:class:`xrdroot.Graph` in ``._xrd`` is made from them when it is asked for,
carrying the name, title, colours and fitted functions set here. What a
graph draws its axes with is ROOT's ``GetHistogram()``, a ``TH1F`` over the
points' range with a tenth more either side, made when first asked for.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...graph import Graph
from .cformat import c_format
from .objects import TAttFill, TAttLine, TAttMarker, TNamed
from .refs import store
from .wrapping import adopt, register, remember, unwrap, wrap

__all__ = ["TGraph", "TGraphErrors", "TGraphAsymmErrors", "TGraphBentErrors"]

#: Each class's error arrays, in the order its constructor takes them.
ERRORS: dict[str, tuple[str, ...]] = {
    "TGraph": (),
    "TGraphErrors": ("ex", "ey"),
    "TGraphAsymmErrors": ("exl", "exh", "eyl", "eyh"),
}


def _doubles(values: Any, count: int) -> np.ndarray[Any, Any]:
    """The first ``count`` of ``values`` as doubles, or zeros where there are none."""
    if values is None or (np.ndim(values) == 0 and not values):
        return np.zeros(count)
    return np.array(np.asarray(values, dtype=np.float64).reshape(-1)[:count])


def _grown(values: np.ndarray[Any, Any], count: int) -> np.ndarray[Any, Any]:
    """``values`` cut or padded with zeros to ``count``."""
    return np.concatenate([values[:count], np.zeros(max(count - len(values), 0))])


class TGraph(TNamed, TAttLine, TAttFill, TAttMarker):
    """``TGraph``: points, joined or marked."""

    CLASS_TITLE = "Graph graphics class"

    def __init__(self, *args: Any) -> None:
        TNamed.__init__(self, "Graph", "Graph")
        self._points = np.zeros(0), np.zeros(0)
        self._bars = {key: np.zeros(0) for key in ERRORS[self._root_class()]}
        self._functions: list[Any] = []
        self._extremes = [-1111.0, -1111.0]
        self._histogram: Any = None
        self._cached: Any = None
        self._construct(args)

    def _root_class(self) -> str:
        return next(base.__name__ for base in type(self).__mro__ if base.__name__ in ERRORS)

    def _construct(self, args: tuple[Any, ...]) -> None:
        """Every one of ROOT's constructors: ``(n)``, ``(n, x, y, errors...)``, ``(x, y)``, ``(h)``."""
        if not args:
            return
        first = args[0]
        if hasattr(first, "_xrd") and not isinstance(first, TGraph):
            self._from_histogram(first, args[1:])
            return
        if isinstance(first, TGraph):
            self._take(first)
            return
        if np.ndim(first) > 0:
            args = (len(first), *args)
        count = int(args[0])
        self._points = (_doubles(args[1] if len(args) > 1 else None, count),
                        _doubles(args[2] if len(args) > 2 else None, count))  # fmt: skip
        for at, key in enumerate(self._bars):
            given = args[3 + at] if len(args) > 3 + at else None
            self._bars[key] = _doubles(given, count)

    def _from_histogram(self, source: Any, rest: tuple[Any, ...]) -> None:
        """``TGraph(h)``: a point per bin - or for ``TGraphAsymmErrors(pass, total)``, the ratio."""
        from ...efficiency import Efficiency

        if rest and hasattr(rest[0], "_xrd"):
            made = Graph.from_histogram(Efficiency.from_histograms(unwrap(source), unwrap(rest[0])))
        else:
            made = Graph.from_histogram(unwrap(source))
        self._absorb(made)

    def _take(self, other: TGraph) -> None:
        self._points = tuple(np.array(part) for part in other._points)  # type: ignore[assignment]
        self._bars = {key: np.array(value) for key, value in other._bars.items() if key in self._bars}
        self.SetNameTitle(other.GetName(), other.GetTitle())

    def _adopted(self, xrd: Any) -> None:
        TGraph.__init__(self)
        self._absorb(xrd)
        self._cached = xrd

    def _absorb(self, made: Any) -> None:
        """Take the points, bars, name and look of an xrdroot graph."""
        self._points = (np.array(made.x), np.array(made.y))
        count = len(made.x)
        across, upward = made.xerr, (made.layers[0] if made.layers else None)
        sides = {"ex": across[1] if across else None, "ey": upward[1] if upward else None,
                 "exl": across[0] if across else None, "exh": across[1] if across else None,
                 "eyl": upward[0] if upward else None, "eyh": upward[1] if upward else None}  # fmt: skip
        for key in self._bars:
            self._bars[key] = _doubles(sides[key], count)
        self.SetNameTitle(made.name, made.title)
        for group in ("TAttLine", "TAttFill", "TAttMarker"):
            if group in made._core:
                self.__dict__.setdefault("_atts", {})[group] = dict(made._core[group])
        self._functions = list(made.functions)
        self._extremes = [float(made._core.get("fMinimum", -1111.0)), float(made._core.get("fMaximum", -1111.0))]

    # -- the xrdroot graph, made when asked ---------------------------------------------------

    @property
    def _xrd(self) -> Any:
        if self._cached is None:
            self._cached = self._built()
            remember(self._cached, self)
        return self._cached

    def _changed(self) -> None:
        self._cached = None

    def _built(self) -> Any:
        x, y = self._points
        bars = self._bars
        xerr: Any = None
        yerr: Any = None
        if "ex" in bars:
            xerr, yerr = bars["ex"], bars["ey"]
        elif "exl" in bars:
            xerr, yerr = (bars["exl"], bars["exh"]), (bars["eyl"], bars["eyh"])
        made = Graph.new(self.GetName(), x, y, title=self.GetTitle(), xerr=xerr, yerr=yerr)
        for group, values in self.__dict__.get("_atts", {}).items():
            made._core[group] = values
        made._core["fFunctions"] = self._functions
        made._core["fMinimum"], made._core["fMaximum"] = self._extremes
        return made

    def ClassName(self) -> str:
        return type(self).__name__

    def SetName(self, name: Any) -> None:
        TNamed.SetName(self, name)
        self._changed()

    def SetTitle(self, title: Any = "") -> None:
        """``SetTitle("title;x;y")``: the graph's title, then its axes' titles."""
        parts = str(title).split(";")
        TNamed.SetTitle(self, parts[0])
        if len(parts) > 1:
            self.GetXaxis().SetTitle(parts[1])
        if len(parts) > 2:
            self.GetYaxis().SetTitle(parts[2])
        self._changed()

    def _attribute_holder(self) -> None:
        """A graph keeps its look itself, and hands it to each xrdroot graph made."""
        self._changed()

    # -- the points --------------------------------------------------------------------------

    def GetN(self) -> int:
        return len(self._points[0])

    def Set(self, n: int) -> None:
        """``Set(n)``: ``n`` points, the new ones at zero, the ones past ``n`` gone."""
        count = int(n)
        self._points = (_grown(self._points[0], count), _grown(self._points[1], count))
        self._bars = {key: _grown(value, count) for key, value in self._bars.items()}
        self._changed()

    def SetPoint(self, i: int, x: float, y: float) -> None:
        """``SetPoint(i, x, y)``: point ``i`` moved, or made - with any before it - past the end."""
        if i < 0:
            return
        if i >= self.GetN():
            self.Set(i + 1)
        self._points[0][i], self._points[1][i] = float(x), float(y)
        self._changed()

    def AddPoint(self, x: float, y: float) -> None:
        self.SetPoint(self.GetN(), x, y)

    def SetPointX(self, i: int, x: float) -> None:
        self.SetPoint(i, x, self.GetPointY(i) if i < self.GetN() else 0.0)

    def SetPointY(self, i: int, y: float) -> None:
        self.SetPoint(i, self.GetPointX(i) if i < self.GetN() else 0.0, y)

    def GetPoint(self, i: int, x: Any = None, y: Any = None) -> int:
        """``GetPoint(i, x, y)``: into ``x`` and ``y``; ``i``, or ``-1`` for no such point."""
        if not 0 <= i < self.GetN():
            return -1
        store(x, float(self._points[0][i]))
        store(y, float(self._points[1][i]))
        return int(i)

    def GetPointX(self, i: int) -> float:
        return float(self._points[0][i]) if 0 <= i < self.GetN() else -1.0

    def GetPointY(self, i: int) -> float:
        return float(self._points[1][i]) if 0 <= i < self.GetN() else -1.0

    def RemovePoint(self, i: int) -> int:
        if not 0 <= i < self.GetN():
            return -1
        self._points = (np.delete(self._points[0], i), np.delete(self._points[1], i))
        self._bars = {key: np.delete(value, i) for key, value in self._bars.items()}
        self._changed()
        return int(i)

    def GetX(self) -> np.ndarray[Any, Any]:
        """``GetX``: the x of every point, to read or change in place."""
        self._changed()
        return self._points[0]

    def GetY(self) -> np.ndarray[Any, Any]:
        self._changed()
        return self._points[1]

    def _bar(self, key: str) -> np.ndarray[Any, Any] | None:
        found = self._bars.get(key)
        if found is not None:
            self._changed()
        return found

    def GetEX(self) -> Any:
        return self._bar("ex")

    def GetEY(self) -> Any:
        return self._bar("ey")

    def GetEXlow(self) -> Any:
        return self._bar("exl")

    def GetEXhigh(self) -> Any:
        return self._bar("exh")

    def GetEYlow(self) -> Any:
        return self._bar("eyl")

    def GetEYhigh(self) -> Any:
        return self._bar("eyh")

    def _error(self, i: int, *keys: str) -> float:
        """A point's bar: ``ex`` or ``ey``, or the root of the mean square of the two sides."""
        if not self._bars or not 0 <= i < self.GetN():
            return -1.0
        found = [self._bars[key][i] for key in keys]
        return float(found[0]) if len(found) == 1 else float(np.sqrt(np.mean(np.square(found))))

    def GetErrorX(self, i: int) -> float:
        return self._error(i, "ex") if "ex" in self._bars else self._error(i, "exl", "exh")

    def GetErrorY(self, i: int) -> float:
        return self._error(i, "ey") if "ey" in self._bars else self._error(i, "eyl", "eyh")

    def GetErrorXlow(self, i: int) -> float:
        return self._error(i, "ex") if "ex" in self._bars else self._error(i, "exl")

    def GetErrorXhigh(self, i: int) -> float:
        return self._error(i, "ex") if "ex" in self._bars else self._error(i, "exh")

    def GetErrorYlow(self, i: int) -> float:
        return self._error(i, "ey") if "ey" in self._bars else self._error(i, "eyl")

    def GetErrorYhigh(self, i: int) -> float:
        return self._error(i, "ey") if "ey" in self._bars else self._error(i, "eyh")

    def SetPointError(self, i: Any, *errors: float) -> None:
        """``SetPointError(i, ex, ey)`` - or ``(i, exl, exh, eyl, eyh)`` for asymmetric bars."""
        if not self._bars:
            return
        if i >= self.GetN():
            self.Set(i + 1)
        for key, value in zip(self._bars, errors):
            self._bars[key][i] = float(value)
        self._changed()

    def Sort(self, *args: Any) -> None:
        """``Sort``: the points in increasing x, their bars with them."""
        order = np.argsort(self._points[0], kind="stable")
        self._points = (self._points[0][order], self._points[1][order])
        self._bars = {key: value[order] for key, value in self._bars.items()}
        self._changed()

    def Print(self, option: str = "") -> None:
        """``Print``: a line per point, ``x[i]=..., y[i]=...`` and its bars, as ROOT prints them."""
        for i in range(self.GetN()):
            parts = [c_format("x[%d]=%g, y[%d]=%g", i, self._points[0][i], i, self._points[1][i])]
            parts += [c_format(f"{key}[%d]=%g", i, value[i]) for key, value in self._bars.items()]
            print(", ".join(parts))

    # -- what is worked out from it ---------------------------------------------------------------

    def Eval(self, x: float, spline: Any = None, option: str = "") -> float:
        """``Eval``: along straight lines between the points, extrapolated past the ends."""
        return float(self._xrd.eval(float(x)))

    def Integral(self, first: int = 0, last: int = -1) -> float:
        return float(self._xrd.integral(first, last))

    def GetMean(self, axis: int = 1) -> float:
        return float(self._xrd.mean(axis - 1))

    def GetRMS(self, axis: int = 1) -> float:
        return float(self._xrd.rms(axis - 1))

    GetStdDev = GetRMS

    def GetCovariance(self) -> float:
        x, y = self._points
        return float(np.mean(x * y) - np.mean(x) * np.mean(y)) if len(x) else 0.0

    def GetCorrelationFactor(self) -> float:
        spread = self.GetRMS(1) * self.GetRMS(2)
        return self.GetCovariance() / spread if spread else 0.0

    def Fit(self, f1: Any, option: str = "", goption: str = "", rxmin: float = 0.0, rxmax: float = 0.0) -> Any:
        """``Fit``: as ``TGraph::Fit``, the function hung on the graph afterwards."""
        from .fits import fit

        made = self._xrd
        found = fit(self, f1, option, goption, (rxmin, rxmax))
        self._functions = list(made.functions)
        return found

    def GetFunction(self, name: Any) -> Any:
        found = [item for item in self._functions if getattr(item, "name", None) == str(name)]
        return wrap(found[0]) if found else None

    def GetListOfFunctions(self) -> Any:
        from .collections import TList

        made = TList()
        for item in self._functions:
            made.Add(wrap(item))
        return made

    # -- the frame it is drawn in ----------------------------------------------------------------

    def ComputeRange(self) -> tuple[float, float, float, float]:
        """``ComputeRange``: the least and greatest x and y, bars included."""
        x, y = self._points
        if not len(x):
            return 0.0, 0.0, 0.0, 0.0
        left = self._bars.get("ex", self._bars.get("exl", np.zeros(len(x))))
        right = self._bars.get("ex", self._bars.get("exh", np.zeros(len(x))))
        down = self._bars.get("ey", self._bars.get("eyl", np.zeros(len(x))))
        up = self._bars.get("ey", self._bars.get("eyh", np.zeros(len(x))))
        return (float(np.min(x - left)), float(np.min(y - down)),
                float(np.max(x + right)), float(np.max(y + up)))  # fmt: skip

    def GetHistogram(self) -> Any:
        """``GetHistogram``: the frame - a ``TH1F`` a tenth wider than the points - made once."""
        if self._histogram is None:
            self._histogram = self._frame()
        return self._histogram

    def _frame(self) -> Any:
        from .hists import TH1F

        xmin, ymin, xmax, ymax = self.ComputeRange()
        xmax, ymax = (xmax + 1.0 if xmin == xmax else xmax), (ymax + 1.0 if ymin == ymax else ymax)
        dx, dy = 0.1 * (xmax - xmin), 0.1 * (ymax - ymin)
        low, high = xmin - dx, xmax + dx
        minimum = self._extremes[0] if self._extremes[0] != -1111.0 else ymin - dy
        maximum = self._extremes[1] if self._extremes[1] != -1111.0 else ymax + dy
        low = 0.0 if low < 0 <= xmin else low
        high = 0.0 if high > 0 >= xmax else high
        minimum = 0.9 * ymin if minimum < 0 <= ymin else minimum
        TH1F.AddDirectory(False)
        try:
            made = TH1F(self.GetName() or "Graph", self.GetTitle(), max(100, self.GetN()), low, high)
        finally:
            TH1F.AddDirectory(True)
        made.SetMinimum(minimum)
        made.SetMaximum(maximum)
        made.SetStats(False)
        made.GetYaxis().SetLimits(minimum, maximum)
        return made

    def GetXaxis(self) -> Any:
        return self.GetHistogram().GetXaxis()

    def GetYaxis(self) -> Any:
        return self.GetHistogram().GetYaxis()

    def SetHistogram(self, h: Any) -> None:
        self._histogram = h

    def SetMinimum(self, minimum: float = -1111.0) -> None:
        self._extremes[0] = float(minimum)
        if self._histogram is not None:
            self._histogram.SetMinimum(minimum)
        self._changed()

    def SetMaximum(self, maximum: float = -1111.0) -> None:
        self._extremes[1] = float(maximum)
        if self._histogram is not None:
            self._histogram.SetMaximum(maximum)
        self._changed()

    def GetMinimum(self) -> float:
        return self._extremes[0]

    def GetMaximum(self) -> float:
        return self._extremes[1]

    def SetEditable(self, editable: bool = True) -> None:
        """``SetEditable``: whether a pad lets the points be dragged; there is no pad to drag in."""

    def Clone(self, newname: str = "") -> Any:
        import copy

        made = copy.deepcopy(self)
        made._cached = None
        if newname:
            made.SetName(newname)
        return made


class TGraphErrors(TGraph):
    """``TGraphErrors``: points with a bar each way along x and along y."""

    CLASS_TITLE = "A graph with error bars"


class TGraphAsymmErrors(TGraph):
    """``TGraphAsymmErrors``: points with bars of their own length on each side."""

    CLASS_TITLE = "A graph with asymmetric error bars"

    def Divide(self, passed: Any, total: Any, option: str = "cp") -> None:
        """``Divide(pass, total)``: the efficiency in each bin, with its interval."""
        self._from_histogram(passed, (total,))
        self._changed()


TGraphBentErrors = TGraphAsymmErrors

for _cls in (TGraph, TGraphErrors, TGraphAsymmErrors):
    register(_cls.__name__, factory=lambda xrd, cls=_cls: adopt(cls, xrd))
