"""Graphs, as points rather than as the arrays they are written as.

A graph in a ROOT file is two arrays of the same length and, if whoever wrote
it kept them, two or four more holding the error bars. This turns that into
the thing it is - points, and what the bars round them are - while leaving
every member it was written with in reach under :attr:`Graph.members`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from xrdclient._compat import zip_strict

from . import graphmath
from .draw import axes
from .efficiency import Efficiency
from .errors import FormatError, UnsupportedFeatureError
from .function.attached import listed
from .hist import FILL, LINE, MARKER, Histogram

__all__ = ["GRAPHS", "Graph"]

#: The graph classes this reads: points, points with error bars, points whose
#: bars are a different length each side, and points whose y errors are kept
#: in more than one layer.
GRAPHS = ("TGraph", "TGraphErrors", "TGraphAsymmErrors", "TGraphMultiErrors")


def _core(row: dict[str, Any]) -> dict[str, Any] | None:
    """The ``TGraph`` part of a graph, however far down it is inherited."""
    if "fNpoints" in row:
        return row
    for value in row.values():
        if isinstance(value, dict):
            found = _core(value)
            if found is not None:
                return found
    return None


class Graph:
    """A graph: its points, and the error bars round them.

        >>> for x, y in graph:                     # doctest: +SKIP
        ...     print(x, y)

    ``x`` and ``y`` are NumPy arrays of one value per point. :attr:`xerr` and :attr:`yerr`
    are the bars either side of each point, or ``None`` for a graph written
    without them; a graph keeping its errors in layers has them in
    :attr:`layers`.
    """

    __slots__ = ("classname", "members", "x", "y", "_core")

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        core = _core(members)
        if core is None or "fX" not in core or "fY" not in core:
            raise FormatError(f"a {classname} was written without its points")
        points = int(core["fNpoints"])
        if len(core["fX"]) < points or len(core["fY"]) < points:
            raise FormatError(
                f"a {classname} of {points} points holds only "
                f"{min(len(core['fX']), len(core['fY']))} of them"
            )
        #: The class the file says this is, such as ``TGraphErrors``.
        self.classname = classname
        #: Every member, as it was written, for whatever is not here by name.
        self.members = members
        self._core = core
        #: Where each point is, one value per point.
        self.x: np.ndarray[Any, Any] = _doubles(core["fX"][:points])
        self.y: np.ndarray[Any, Any] = _doubles(core["fY"][:points])

    @property
    def name(self) -> str:
        """What the graph is called, which is the key it was written under."""
        return str(self._core["TNamed"]["fName"])

    @property
    def title(self) -> str:
        """The title it is drawn with, which is usually a sentence about it."""
        return str(self._core["TNamed"]["fTitle"])

    @property
    def functions(self) -> list[Any]:
        """``GetListOfFunctions``: the fits and functions attached, written with it."""
        return listed(self._core)

    def attach(self, function: Any) -> None:
        """Hang ``function`` - a :class:`~xrdroot.Function`, a fit - on this graph."""
        self.functions.append(function)

    def fit(
        self,
        model: Any,
        option: str = "",
        range: Any = None,
        *,
        parameters: Any = None,
        limits: Any = None,
        fixed: Any = None,
        npar: int | None = None,
    ) -> Any:
        """``TGraph::Fit``: fit ``model`` to the points, their error bars deciding the chi-square.

            >>> r = graph.fit("pol1")            # gr->Fit("pol1")          # doctest: +SKIP
            >>> r = graph.fit("gaus", "EX0")     # ignoring the x errors    # doctest: +SKIP

        Errors in y weigh the points; errors in x as well make it the
        effective-variance chi-square, and asymmetric ones take the side
        facing the function; a graph with no errors is fitted with errors
        of one, scaled afterwards. See :mod:`xrdroot.fit`.
        """
        from .fit import fit_object

        return fit_object(
            self, model, option, range, parameters=parameters, limits=limits, fixed=fixed, npar=npar
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[float, float]:
        return (float(self.x[index]), float(self.y[index]))

    def __iter__(self) -> Any:
        return iter(zip_strict(self.x.tolist(), self.y.tolist()))

    def points(self) -> list[tuple[float, float]]:
        """Every point as a pair, which is what a graph is a picture of."""
        return list(self)

    def _pair(self, low: Any, high: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Two arrays of bars, cut to the points the graph says it has."""
        points = len(self)
        return (_doubles(low[:points]), _doubles(high[:points]))

    def _bars(
        self, axis: str, *spellings: tuple[str, str]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
        """The bars either side along one axis, or ``None`` if there are none.

        Each spelling is the pair of names one class gives the low and the
        high bar; a graph whose bars are the same length both sides keeps a
        single array instead, and that array is both sides of the point.
        """
        for below, above in spellings:
            low, high = self.members.get(below), self.members.get(above)
            if low is not None and high is not None:
                return self._pair(low, high)
        same = self.members.get(f"fE{axis}")
        return None if same is None else self._pair(same, same)

    @property
    def xerr(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
        """The bars left and right of each point, or ``None`` if none were kept."""
        return self._bars("X", ("fEXlow", "fEXhigh"), ("fExL", "fExH"))

    @property
    def layers(self) -> tuple[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]], ...]:
        """The bars below and above each point, one pair per layer of them.

        A graph told to keep its statistical and its systematic errors apart
        keeps a layer of each, in the order they were added. An ordinary graph
        keeps one layer, and a graph written without y errors keeps none.
        """
        low, high = self.members.get("fEyL"), self.members.get("fEyH")
        if low is None or high is None:
            bars = self._bars("Y", ("fEYlow", "fEYhigh"))
            return () if bars is None else (bars,)
        return tuple(self._pair(a, b) for a, b in zip_strict(low, high))

    @property
    def yerr(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None:
        """The bars below and above each point, or ``None`` if none were kept.

        A graph keeping more than one layer of them refuses here rather than
        answer with one of the layers or with a sum of them it made up: which
        of those you want is yours to say, and :attr:`layers` has them all.
        """
        layers = self.layers
        if len(layers) > 1:
            raise UnsupportedFeatureError(
                f"this {self.classname} keeps its y errors in {len(layers)} layers, "
                f"which are one pair of bars only once you have said how to add them "
                f"up; they are each of them in .layers"
            )
        return layers[0] if layers else None

    @classmethod
    def new(
        cls,
        name: str,
        x: Any,
        y: Any,
        *,
        title: str = "",
        xerr: Any = None,
        yerr: Any = None,
    ) -> Graph:
        """A graph built from Python numbers, ready to write.

            >>> g = Graph.new("scan", [1, 2, 3], [2.0, 3.9, 6.1], yerr=[0.1, 0.2, 0.2])
            >>> g.classname
            'TGraphErrors'

        ``xerr`` and ``yerr`` are each either one bar per point, or a
        ``(low, high)`` pair of runs for bars of different lengths each side.
        The class picks itself: plain points make a ``TGraph``, even bars a
        ``TGraphErrors``, and any uneven pair a ``TGraphAsymmErrors`` with
        the even ones carried on both sides.
        """
        xs, ys = _doubles(x), _doubles(y)
        if len(xs) != len(ys):
            raise ValueError(f"{len(xs)} x values and {len(ys)} y values are not points")
        count = len(xs)
        across = _sides(xerr, count, "xerr")
        upward = _sides(yerr, count, "yerr")
        core = _graph_core(name, title, xs, ys)
        if across is None and upward is None:
            return cls("TGraph", core)
        zeros = np.zeros(count)
        if _symmetric(across, upward):
            return cls("TGraphErrors", _even_members(core, across, upward, zeros))
        return cls("TGraphAsymmErrors", _uneven_members(core, across, upward, zeros))

    @classmethod
    def from_histogram(cls, histogram: Any, name: str | None = None) -> Graph:
        """A point per bin of a one-dimensional histogram, profile or efficiency.

            >>> Graph.from_histogram(h)              # TGraphErrors(h)      # doctest: +SKIP
            >>> Graph.from_histogram(eff)            # eff->CreateGraph()   # doctest: +SKIP

        Each point is at its bin's centre, half a bin wide either way, as
        ROOT's ``TGraphErrors(const TH1*)`` puts it: a histogram's or a
        profile's height is its content and the bar its error. An
        efficiency's is the efficiency, with the bars of its interval - so
        the graph is a ``TGraphAsymmErrors`` - and a bin nothing was tried in
        has no point at all, as ``TEfficiency::CreateGraph`` leaves it out.
        """
        if isinstance(histogram, Efficiency):
            return _efficiency_graph(cls, histogram, name)
        if not isinstance(histogram, Histogram) or len(histogram.axes) != 1:
            raise ValueError(
                "a graph is made from a histogram, profile or efficiency of one axis, a point "
                f"per bin, and a {getattr(histogram, 'classname', type(histogram).__name__)} of "
                f"{len(getattr(histogram, 'axes', ()))} axes is not one"
            )
        axis = histogram.axes[0]
        return cls.new(
            name or histogram.name,
            axis.centers(),
            histogram.values(),
            title=histogram.title,
            xerr=axis.widths() / 2,
            yerr=histogram.errors(),
        )

    def eval(self, x: Any) -> Any:
        """``Eval``: the graph at ``x`` along straight lines between points, past the ends too.

            >>> Graph.new("g", [0, 1, 2], [0, 10, 40]).eval([0.5, 3.0]).tolist()
            [5.0, 70.0]

        As ROOT's ``Eval`` does it without a spline: the points need not be
        in order, and past either end it extrapolates along the last two.
        """
        return graphmath.evaluate(self, x)

    def integral(self, first: int = 0, last: int = -1) -> float:
        """``Integral``: the area of the polygon the points make, from ``first`` to ``last``."""
        return graphmath.integral(self, int(first), int(last))

    def sort(self) -> None:
        """``Sort``: the points put in increasing ``x``, in place, error bars and all."""
        graphmath.sort(self)

    def mean(self, axis: int = 0) -> float:
        """``GetMean``: the mean of the points' ``x``, or ``y`` for ``axis=1``."""
        return graphmath.mean(self, axis)

    def rms(self, axis: int = 0) -> float:
        """``GetRMS``: the spread of the points' ``x``, or ``y`` for ``axis=1``."""
        return graphmath.rms(self, axis)

    def plot(self, ax: Any = None, **options: Any) -> Any:
        """Draw onto matplotlib axes, made fresh unless ``ax`` brings some.

        Points with their error bars; a graph keeping layers of them draws
        every layer over the same points. The axes come back, so styling and
        saving carry on where this left off - and matplotlib not being
        installed refuses with the two ways out by name.
        """
        if ax is None:
            ax = axes()
        style: dict[str, Any] = {"fmt": "o", "markersize": 4}
        style.update(options)
        for index, bars in enumerate(self.layers or (None,)):
            ax.errorbar(self.x, self.y, yerr=bars, xerr=self.xerr if index == 0 else None, **style)
        if self.title:
            ax.set_title(self.title)
        return ax

    def text(self, width: int = 60, height: int = 16) -> str:
        """The graph as a grid of points, for a terminal or a log file."""
        if not len(self.x):
            return "(a graph of no points)"
        xlo, xhi = float(self.x.min()), float(self.x.max())
        ylo, yhi = float(self.y.min()), float(self.y.max())
        xspan, yspan = (xhi - xlo) or 1.0, (yhi - ylo) or 1.0
        grid = [[" "] * width for _ in range(height)]
        for x, y in self:
            column = round((x - xlo) / xspan * (width - 1))
            line = round((y - ylo) / yspan * (height - 1))
            grid[height - 1 - line][column] = "*"
        lines = []
        for index, cells in enumerate(grid):
            label = _axis_label(index, height, ylo, yhi)
            lines.append(f"{label:>10} |{''.join(cells)}|")
        left, right = f"{xlo:g}", f"{xhi:g}"
        lines.append(f"{'':>10}  {left:<{width - len(right)}}{right}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<{self.classname} {self.name!r} of {len(self)} points>"


def _doubles(values: Any) -> np.ndarray[Any, Any]:
    """A run of numbers as a one-dimensional array of doubles of its own."""
    return np.array(values, dtype=np.float64).reshape(-1)


def _sides(
    err: Any, count: int, label: str
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], bool] | None:
    """The bars along one axis: low, high, and whether they were given uneven.

    One bar per point is both sides of it; a pair of runs is a side each,
    which is only tellable from two points' worth of bars because runs are
    not numbers.
    """
    if err is None:
        return None
    given = list(err)
    if _is_side_pair(given):
        return _uneven_sides(given, count, label)
    return _even_sides(given, count, label)


def _is_side_pair(given: list[Any]) -> bool:
    return len(given) == 2 and all(np.ndim(side) == 1 for side in given)


def _uneven_sides(
    given: list[Any], count: int, label: str
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], bool]:
    low, high = _doubles(given[0]), _doubles(given[1])
    if len(low) != count or len(high) != count:
        raise ValueError(f"{label} has {len(low)} low and {len(high)} high bars for {count} points")
    return low, high, True


def _even_sides(
    given: list[Any], count: int, label: str
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], bool]:
    bars = _doubles(given)
    if len(bars) != count:
        raise ValueError(
            f"{label} has {len(bars)} bars for {count} points: give one per "
            f"point, or a (low, high) pair of runs"
        )
    return (bars, bars, False)


def _efficiency_graph(cls: type[Graph], efficiency: Efficiency, name: str | None) -> Graph:
    """``TEfficiency::CreateGraph``: the bins anything was tried in, with their intervals."""
    if len(efficiency.axes) != 1:
        raise ValueError(
            f"{efficiency.name!r} has {len(efficiency.axes)} axes, and a graph is made from an "
            f"efficiency of one"
        )
    axis = efficiency.axes[0]
    tried = efficiency.total.values() != 0
    low, high = efficiency.errors()
    half = axis.widths()[tried] / 2
    return cls.new(
        name or efficiency.name,
        axis.centers()[tried],
        efficiency.values()[tried],
        title=efficiency.title,
        xerr=(half, half),
        yerr=(low[tried], high[tried]),
    )


def _graph_core(
    name: str, title: str, xs: np.ndarray[Any, Any], ys: np.ndarray[Any, Any]
) -> dict[str, Any]:
    return {
        "TNamed": {"fName": str(name), "fTitle": str(title)},
        "TAttLine": dict(LINE),
        "TAttFill": dict(FILL),
        "TAttMarker": dict(MARKER),
        "fNpoints": len(xs),
        "fX": xs,
        "fY": ys,
        "fFunctions": None,
        "fHistogram": None,
        "fMinimum": -1111.0,
        "fMaximum": -1111.0,
    }


def _symmetric(across: Any, upward: Any) -> bool:
    return (across is None or not across[2]) and (upward is None or not upward[2])


def _even_members(core: dict[str, Any], across: Any, upward: Any, zeros: Any) -> dict[str, Any]:
    return {
        "TGraph": core,
        "fEX": across[0] if across is not None else zeros,
        "fEY": upward[0] if upward is not None else zeros,
    }


def _uneven_members(core: dict[str, Any], across: Any, upward: Any, zeros: Any) -> dict[str, Any]:
    return {
        "TGraph": core,
        "fEXlow": across[0] if across is not None else zeros,
        "fEXhigh": across[1] if across is not None else zeros,
        "fEYlow": upward[0] if upward is not None else zeros,
        "fEYhigh": upward[1] if upward is not None else zeros,
    }


def _axis_label(index: int, height: int, low: float, high: float) -> str:
    if index == 0:
        return f"{high:g}"
    if index == height - 1:
        return f"{low:g}"
    return ""
