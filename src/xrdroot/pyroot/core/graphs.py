"""``TGraph``, ``TGraphErrors``, ``TGraphAsymmErrors``: points, and the bars round them.

A graph grows a point at a time - ``SetPoint(n, x, y)`` past the end adds
one - so the points are kept here as arrays that grow, and the
:class:`xrdroot.Graph` in ``._xrd`` is made from them when it is asked for,
carrying the name, title, colours and fitted functions set here. What a
graph draws its axes with is ROOT's ``GetHistogram()``, a ``TH1F`` over the
points' range with a tenth more either side, made when first asked for.
"""

from __future__ import annotations

from functools import partial
from os import PathLike, fspath
from pathlib import Path
from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ...graph import TEXT_ORDERS, Graph, text_delimiter, text_lines
from ...graph import read_text as read_graph_text
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
    "TGraphBentErrors": ("exl", "exh", "eyl", "eyh", "exld", "exhd", "eyld", "eyhd"),
}

#: How many columns of a text file each class's file constructor reads: the point, then the
#: bars its own constructor takes - a bent graph's four distances are not among them, so it
#: reads the six its base class does.
READ_COLUMNS = {"TGraph": 2, "TGraphErrors": 4, "TGraphAsymmErrors": 6, "TGraphBentErrors": 6}

#: The bars each ``GetErrorX``-style name reads, in the order it tries them: the one bar a
#: point has both sides of it, or the two sides whose root mean square is the bar.
ERROR_KEYS = {"ex": ("ex", "exl", "exh"), "ey": ("ey", "eyl", "eyh"),
              "exl": ("exl", "ex"), "exh": ("exh", "ex"),
              "eyl": ("eyl", "ey"), "eyh": ("eyh", "ey")}  # fmt: skip


def _doubles(values: Any, count: int) -> np.ndarray[Any, Any]:
    """The first ``count`` of ``values`` as doubles, or zeros where there are none."""
    if values is None or (np.ndim(values) == 0 and not values):
        return np.zeros(count)
    return np.array(np.asarray(values, dtype=np.float64).reshape(-1)[:count])


def _grown(values: np.ndarray[Any, Any], count: int) -> np.ndarray[Any, Any]:
    """``values`` cut or padded with zeros to ``count``."""
    return np.concatenate([values[:count], np.zeros(max(count - len(values), 0))])


def _sides(made: Any) -> dict[str, Any]:
    """Every bar an xrdroot graph has, by the names each pyroot class keeps them under."""
    across = made.xerr or (None, None)
    upward = made.layers[0] if made.layers else (None, None)
    return {"ex": across[1], "ey": upward[1], "exl": across[0], "exh": across[1],
            "eyl": upward[0], "eyh": upward[1]}  # fmt: skip


def _frame_histogram(
    name: Any, title: Any, count: int, low: float, high: float, down: float, up: float
) -> Any:
    """The ``TH1F`` a graph's or a multigraph's axes are drawn in.

    ``count`` bins over ``low`` to ``high`` in x, ``down`` to ``up`` in y, and no
    statistics box - ROOT's ``kNoStats``. The frame is nobody's directory entry, so it
    is made while ``AddDirectory`` says not to file it.
    """
    from .hists import TH1F

    kept = TH1F.AddDirectoryStatus()
    TH1F.AddDirectory(False)
    try:
        made = TH1F(name, title, count, low, high)
    finally:
        TH1F.AddDirectory(kept)
    made.SetMinimum(down)
    made.SetMaximum(up)
    made.SetStats(False)
    made.GetYaxis().SetLimits(down, up)
    return made


class TGraph(TNamed, TAttLine, TAttFill, TAttMarker):
    """``TGraph``: points, joined or marked."""

    CLASS_TITLE = "Graph graphics class"
    kIsHighlight = 1 << 20

    def __init__(self, *args: Any) -> None:
        # ROOT's default constructor leaves the name and title empty; the others name it "Graph".
        TNamed.__init__(self, *(("Graph", "Graph") if args else ("", "")))
        self._points = np.zeros(0), np.zeros(0)
        self._bars = {key: np.zeros(0) for key in ERRORS[self._root_class()]}
        self._functions: list[Any] = []
        self._extremes = [-1111.0, -1111.0]
        self._histogram: Any = None
        self._cached: Any = None
        # ROOT's graphs are made with TAttFill(0, 1000): no colour, but a solid fill once given one.
        self.__dict__["_atts"] = {"TAttFill": {"fFillColor": 0, "fFillStyle": 1000}}
        self._construct(args)

    def SetHighlight(self, set: bool = True) -> None:
        """``SetHighlight``: mark the graph for highlighting its point under the mouse, which on
        a batch canvas never comes."""
        self.SetBit(self.kIsHighlight, bool(set))

    def IsHighlight(self) -> bool:
        return self.TestBit(self.kIsHighlight)

    def _root_class(self) -> str:
        return next(base.__name__ for base in type(self).__mro__ if base.__name__ in ERRORS)

    def _construct(self, args: tuple[Any, ...]) -> None:
        """ROOT's constructors: ``(n)``, ``(n, x, y, errors...)``, ``(x, y)``, ``(h)``, and
        ``(filename, format, option)`` for the points a text file holds."""
        if not args:
            return
        first = args[0]
        if isinstance(first, (str, PathLike)):  # a TString is one, as ROOT's names are
            self._from_text(fspath(first), args[1:])
        elif isinstance(first, TGraph):
            self._take(first)
        elif hasattr(first, "_xrd"):
            self._from_histogram(first, args[1:])
        else:
            self._from_arrays((len(first), *args) if np.ndim(first) > 0 else args)

    def _from_text(self, filename: str, rest: tuple[Any, ...]) -> None:
        """``TGraph("points.csv")``: the points of a text file whose columns the extension
        names - a comma for ``.csv``, a tab for ``.tsv``, a space for ``.txt``, and a comma
        for any other name, which is the format ROOT falls back to.

        ROOT names the graph ``Graph`` and titles it with the file it came from, and a file
        it cannot open is a graph of no points after an error line rather than a refusal.
        """
        self._refuse_text_format(filename, rest)
        self.SetNameTitle("Graph", filename)
        count = READ_COLUMNS[self._root_class()]
        columns = read_graph_text(filename, count, text_delimiter(filename) or ",")
        if columns is None:
            classname = self._root_class()
            self.Error(classname, "Cannot open file: %s, %s is Zombie", filename, classname)
            return
        self._points = (columns[0], columns[1])
        for key, values in zip(self._bars, columns[2:], strict=False):
            self._bars[key] = values

    @staticmethod
    def _refuse_text_format(filename: str, rest: tuple[Any, ...]) -> None:
        """ROOT reads a graph's text file with a ``%lg`` format or with an option naming the
        delimiters, which is a tokeniser this kit has not written: refusing by name keeps a
        file of another shape from being read as if it were one of ROOT's own."""
        given = [str(value) for value in rest[:2] if value]
        if not given:
            return
        raise UnsupportedFeatureError(
            f"reading the points of {filename} with {given[0]!r} is not supported: this kit "
            "reads the text files ROOT writes, whose columns the extension names - a comma "
            "for .csv, a tab for .tsv and a space for .txt - with an empty format and no "
            "option. Write the file with TGraph::SaveAs or name the columns in that format"
        )

    def _from_arrays(self, args: tuple[Any, ...]) -> None:
        """``(n, x, y, bars...)``: the first ``n`` of each array, zeros for any not given."""
        count = int(args[0])
        given = [*args[1:], *([None] * (2 + len(self._bars)))]
        self._points = (_doubles(given[0], count), _doubles(given[1], count))
        for at, key in enumerate(self._bars):
            self._bars[key] = _doubles(given[2 + at], count)

    def _from_histogram(self, source: Any, rest: tuple[Any, ...]) -> None:
        """``TGraph(h)``: a point per bin - or for ``TGraphAsymmErrors(pass, total)``, the ratio."""
        from ...efficiency import Efficiency

        if rest and hasattr(rest[0], "_xrd"):
            made = Graph.from_histogram(Efficiency.from_histograms(unwrap(source), unwrap(rest[0])))
        else:
            made = Graph.from_histogram(unwrap(source))
        self._absorb(made)

    def _take(self, other: TGraph) -> None:
        self._points = (np.array(other._points[0]), np.array(other._points[1]))
        self._bars = {
            key: np.array(value) for key, value in other._bars.items() if key in self._bars
        }
        self.SetNameTitle(other.GetName(), other.GetTitle())

    def _adopted(self, xrd: Any) -> None:
        TGraph.__init__(self)
        self._absorb(xrd)
        self._cached = xrd

    def _absorb(self, made: Any) -> None:
        """Take the points, bars, name and look of an xrdroot graph."""
        self._points = (np.array(made.x), np.array(made.y))
        sides = _sides(made)
        for key in self._bars:
            self._bars[key] = _doubles(sides[key], len(made.x))
        self.SetNameTitle(made.name, made.title)
        for group in ("TAttLine", "TAttFill", "TAttMarker"):
            if group in made._core:
                self.__dict__.setdefault("_atts", {})[group] = dict(made._core[group])
        self._functions = list(made.functions)
        self._extremes = [
            float(made._core.get("fMinimum", -1111.0)),
            float(made._core.get("fMaximum", -1111.0)),
        ]

    # -- the xrdroot graph, made when asked ---------------------------------------------------

    @property
    def _xrd(self) -> Any:
        """The xrdroot graph of the points as they are, sharing this graph's look."""
        if self._cached is None:
            self._cached = self._built()
            remember(self._cached, self)
        for group, values in self.__dict__.get("_atts", {}).items():
            self._cached._core[group] = values
        if self._histogram is not None:  # the frame a script reached, which draws it as it is
            self._cached._core["fHistogram"] = self._histogram._xrd
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
        if self._histogram is not None:  # the frame's title is what is drawn, as ROOT keeps it
            self._histogram.SetTitle(parts[0])
        if len(parts) > 1:
            self.GetXaxis().SetTitle(parts[1])
        if len(parts) > 2:
            self.GetYaxis().SetTitle(parts[2])
        self._changed()

    def _attribute_holder(self) -> None:
        """A graph keeps its look itself, and shares it with each xrdroot graph made."""

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

    def Apply(self, f: Any) -> None:
        """``Apply(f)``: each point's ``y`` made ``f(x, y)`` - a ``TF1`` of ``x``, a ``TF2`` of
        both - and its y error bars carried through ``f`` as ROOT carries them."""
        xs, ys = self._points
        for i in range(self.GetN()):
            x, y = float(xs[i]), float(ys[i])
            self._apply_bars(f, i, x, y)
            ys[i] = float(f.Eval(x, y))
        self._changed()

    def _apply_bars(self, f: Any, i: int, x: float, y: float) -> None:
        """``TGraphErrors::Apply``'s half the spread of ``f`` across the bar; for asymmetric
        bars, ``TGraphAsymmErrors``'s distances to ``f`` at each end, low end the lower."""
        bars = self._bars
        if "ey" in bars:
            spread = float(f.Eval(x, y + bars["ey"][i])) - float(f.Eval(x, y - bars["ey"][i]))
            bars["ey"][i] = abs(spread) / 2.0
        elif "eyl" in bars:
            at = float(f.Eval(x, y))
            low, high = y - bars["eyl"][i], y + bars["eyh"][i]
            ends = sorted((float(f.Eval(x, low)), float(f.Eval(x, high))))
            bars["eyl"][i], bars["eyh"][i] = at - ends[0], ends[1] - at

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

    def IsInside(self, x: float, y: float) -> int:
        """``TGraph::IsInside``: whether ``(x, y)`` is inside the polygon the points make,
        ``TMath::IsInside``'s crossings counted."""
        from ..graphics.cutg import is_inside

        return int(is_inside(self._points[0], self._points[1], x, y))

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

    def _set_bar(self, key: str, i: int, value: float) -> None:
        """One side of one point's bar - ``SetPointEYlow`` and its kin - the graph grown to it."""
        if key not in self._bars:
            return
        if i >= self.GetN():
            self.Set(i + 1)
        self._bars[key][i] = float(value)
        self._changed()

    def SetPointEXlow(self, i: int, value: float) -> None:
        self._set_bar("exl", i, value)

    def SetPointEXhigh(self, i: int, value: float) -> None:
        self._set_bar("exh", i, value)

    def SetPointEYlow(self, i: int, value: float) -> None:
        self._set_bar("eyl", i, value)

    def SetPointEYhigh(self, i: int, value: float) -> None:
        self._set_bar("eyh", i, value)

    def SetPointError(self, i: Any, *errors: float) -> None:
        """``SetPointError(i, ex, ey)`` - or ``(i, exl, exh, eyl, eyh)`` for asymmetric bars."""
        if not self._bars:
            return
        if i >= self.GetN():
            self.Set(i + 1)
        for key, value in zip(self._bars, errors, strict=False):
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

    # -- written out as text, which is what a spreadsheet reads --------------------------------

    def SaveAs(self, filename: str = "", option: str = "") -> None:
        """``TGraph::SaveAs``: the points as text for a ``.csv``, ``.tsv`` or ``.txt`` name, and
        the ROOT file every other name means, which is ``TObject::SaveAs``.

        The columns are ROOT's: a graph keeping no errors writes the two of each point, one
        with a bar either way writes the point and its two, and an asymmetric one writes the
        two sides it has. The option says which to write whatever the class is - ``errors``,
        ``asymmerrors``, ``noerrors`` - and ``asroot`` (or ``native``) puts the point before
        its errors as ROOT's own files have them, where a spreadsheet's order is x, its
        errors, y and its errors. ``title`` adds the header line naming the columns.
        """
        delimiter = text_delimiter(filename)
        if delimiter is None:
            super().SaveAs(filename, option)
            return
        mode = self._text_mode(option)
        lines = text_lines(
            self._text_values(mode),
            mode,
            delimiter,
            as_root=self._as_root(option),
            titles=self._axis_titles(),
            header="title" in str(option).lower(),
        )
        Path(filename).write_text("".join([line + "\n" for line in lines]), encoding="utf-8")
        self.Info("SaveAs", "%s file: %s has been generated", str(filename)[-3:], filename)

    def _text_mode(self, option: str) -> str:
        """Which of ``TGraph::SaveAs``'s three column sets ``option`` asks for. ROOT reads the
        option before the class is asked: ``noerrors`` first, then ``asymmerrors``, which is
        asked before ``errors`` because it holds that word."""
        text = str(option).lower()
        if "noerrors" in text:
            return "none"
        if "asymmerrors" in text:
            return "asymm"
        if "errors" in text:
            return "even"
        if isinstance(self, TGraphAsymmErrors):
            return "asymm"
        return "even" if isinstance(self, TGraphErrors) else "none"

    @staticmethod
    def _as_root(option: str) -> bool:
        """Does ``option`` ask for ROOT's own column order, the point before its errors."""
        text = str(option).lower()
        return "asroot" in text or "native" in text

    def _axis_titles(self) -> tuple[str, str]:
        """The titles of the frame the graph is drawn in, or the ``x`` and ``y`` ROOT writes
        where the frame has none."""
        if self._histogram is None:
            return ("x", "y")
        return (
            self._histogram.GetXaxis().GetTitle() or "x",
            self._histogram.GetYaxis().GetTitle() or "y",
        )

    def _error_column(self, name: str) -> np.ndarray[Any, Any]:
        """A whole column of bars as ``GetErrorX`` and its kin answer for one point, and -1
        for a graph keeping none - which is what ROOT writes for a graph without them."""
        found = [self._bars[key] for key in ERROR_KEYS[name] if key in self._bars]
        if not found:
            return np.full(self.GetN(), -1.0)
        return found[0] if len(found) == 1 else np.sqrt(np.mean(np.square(found), axis=0))

    def _text_values(self, mode: str) -> dict[str, Any]:
        """The columns ``mode`` names, each read off the bars the graph keeps."""
        values: dict[str, Any] = {"x": self._points[0], "y": self._points[1]}
        for name in TEXT_ORDERS[mode][0]:
            if name not in values:
                values[name] = self._error_column(name)
        return values

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

    def Fit(
        self, f1: Any, option: str = "", goption: str = "", rxmin: float = 0.0, rxmax: float = 0.0
    ) -> Any:
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
        """``GetListOfFunctions``: the fits and whatever else is hung on the graph."""
        from .collections import FunctionList

        self._changed()
        return FunctionList(self._functions, self.__dict__.setdefault("_extras", []))

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

    def _point_range(self) -> tuple[float, float, float, float]:
        """``TGraph::ComputeRange`` over the points themselves, the bars left out.

        ``TMultiGraph::GetHistogram`` merges this, one graph at a time, and pads the
        result its own way; :meth:`ComputeRange` is the other question, how far a drawn
        graph reaches, and answers it with the bars. ROOT's log-axis rule - the least
        positive value rather than the least - keeps the minimum whenever that minimum
        is positive, and changes nothing when it is not, so the plain minimum is what it
        comes to. The caller leaves a graph of no points out.
        """
        x, y = self._points
        return float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))

    def GetHistogram(self) -> Any:
        """``GetHistogram``: the frame - a ``TH1F`` a tenth wider than the points - made once."""
        if self._histogram is None:
            self._histogram = self._frame()
        return self._histogram

    def _frame(self) -> Any:
        low, high, minimum, maximum = self._frame_range()
        return _frame_histogram(
            self.GetName() or "Graph", self.GetTitle(), max(100, self.GetN()),
            low, high, minimum, maximum,
        )  # fmt: skip

    def _frame_range(self) -> tuple[float, float, float, float]:
        """``GetHistogram``'s range: the points', a tenth wider, kept off zero's other side."""
        xmin, ymin, xmax, ymax = self.ComputeRange()
        xmax += 1.0 if xmin == xmax else 0.0
        ymax += 1.0 if ymin == ymax else 0.0
        dx, dy = 0.1 * (xmax - xmin), 0.1 * (ymax - ymin)
        low = max(xmin - dx, 0.0) if xmin >= 0 else xmin - dx
        high = min(xmax + dx, 0.0) if xmax <= 0 else xmax + dx
        minimum = self._extremes[0] if self._extremes[0] != -1111.0 else ymin - dy
        maximum = self._extremes[1] if self._extremes[1] != -1111.0 else ymax + dy
        minimum = 0.9 * ymin if minimum < 0 <= ymin else minimum
        return low, high, minimum, maximum

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
        """``Divide(pass, total, option)``: the efficiency - or ratio - in each bin, with its
        interval, as :mod:`.divide` works them out."""
        from .divide import divide

        points = divide(passed, total, option)
        if points is None:
            return
        columns = [np.array(column, dtype=np.float64) for column in zip(*points, strict=False)] or [
            np.zeros(0) for _ in range(6)
        ]
        self._points = (columns[0], columns[1])
        self._bars = dict(zip(("exl", "exh", "eyl", "eyh"), columns[2:], strict=False))
        self._changed()
        if "v" in str(option).lower():
            self.Print()


class TGraphBentErrors(TGraphAsymmErrors):
    """``TGraphBentErrors``: bars whose ends are drawn apart from the points, each end's
    distance along each axis a number of its own - ROOT's ``fEXlowd`` and its kin."""

    CLASS_TITLE = "A graph with bent, asymmetric error bars"

    def GetEXlowd(self) -> Any:
        return self._bar("exld")

    def GetEXhighd(self) -> Any:
        return self._bar("exhd")

    def GetEYlowd(self) -> Any:
        return self._bar("eyld")

    def GetEYhighd(self) -> Any:
        return self._bar("eyhd")


for _cls in (TGraph, TGraphErrors, TGraphAsymmErrors, TGraphBentErrors):
    register(_cls.__name__, factory=partial(adopt, _cls))
