"""``TH2Poly``: a 2-D histogram whose bins are polygons, drawn as ``THistPainter`` draws one;
``TProfile2Poly``, its profile.

Bins are added one by one - a rectangle, a polygon, a ``TGraph`` or the
graphs of a ``TMultiGraph`` - or a honeycomb of hexagons at once, and a fill
goes to the first bin holding its point (:mod:`xrdroot.polybins`). Drawn,
the histogram is its frame and axes, then each bin as its option says:
filled in the palette's colour for its content (``COL``, with the palette
beside it for ``Z``), outlined (``L``), its content written in it
(``TEXT``), marked (``P``). What it paints is made afresh each time its pad
is painted. The frame carries no stats box: a ``TH2Poly``'s moments are
kept, and asked of it, but not drawn.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..hist import Histogram
from ..polybins import REGIONS, PolyBins
from ._pythonization._uhi import UnifiedHistogram
from ._pythonization._uhi.indexing import _compute_common_index
from ._pythonization._uhi.plotting import _grid_by_copy
from .core.objects import TAttFill, TAttLine, TAttMarker, TNamed

__all__ = ["TH2Poly", "TProfile2Poly", "kERRORMEAN", "kERRORSPREAD", "kERRORSPREADI",
           "kERRORSPREADG"]  # fmt: skip

#: ``EErrorType``, the option a profile's ``SetErrorOption`` takes, as ``TProfile.h``
#: declares it. A ``TProfile2Poly`` keeps the spread of the means for ``kERRORSPREAD``
#: and the error on each mean for anything else, as ROOT's ``GetBinError`` does.
kERRORMEAN = 0
kERRORSPREAD = 1
kERRORSPREADI = 2
kERRORSPREADG = 3

#: The contents ``GetMinimum`` and ``GetMaximum`` answer when none were set.
UNSET = -1111.0
#: ``TH1::kNoStats``: the frame draws no statistics box.
NO_STATS = 1 << 9
#: The moments ``GetStats`` reports, per axis: where x's, y's and z's sum of weights is.
AXIS_SUM = (2, 4, 7)
#: The sums an empty profile bin holds: ``fSumw``, ``fSumvw``, ``fSumwv2`` and ``fSumw2``.
_EMPTY = [0.0, 0.0, 0.0, 0.0]


def _points(given: Any) -> list[tuple[Any, Any]]:
    """The polygons of a ``TGraph``, a ``TMultiGraph``, or a wrapper of either."""
    graphs = given.GetListOfGraphs() if hasattr(given, "GetListOfGraphs") else [given]
    return [(np.array(g.GetX()), np.array(g.GetY())) for g in graphs]


def _hexagon(x0: float, y0: float, a: float, vertical: bool) -> tuple[list[float], list[float]]:
    """One of ``Honeycomb``'s hexagons, corners in ROOT's order from ``(x0, y0)``."""
    h = a * math.sqrt(3) / 2
    if vertical:
        return ([x0, x0, x0 + h, x0 + 2 * h, x0 + 2 * h, x0 + h],
                [y0, y0 + a, y0 + 1.5 * a, y0 + a, y0, y0 - a / 2])  # fmt: skip
    return ([x0, x0 + a / 2, x0 + 1.5 * a, x0 + 2 * a, x0 + 1.5 * a, x0 + a / 2],
            [y0, y0 + h, y0 + h, y0, y0 - h, y0 - h])  # fmt: skip


class TH2Poly(TNamed, TAttLine, TAttFill, TAttMarker):
    """``TH2Poly()``, or ``TH2Poly(name, title, xlow, xup, ylow, yup)``: bins of any shape."""

    #: The cells of the fixed grid ROOT gives a ``TH2Poly``'s x and y axes.
    GRID = 100

    #: The bins, what they hold, and the regions round the range.
    _xrd_bins: PolyBins

    def __init__(self, name: Any = "NoName", title: Any = "NoTitle", *ranges: float) -> None:
        source = name if isinstance(name, TH2Poly) else None
        super().__init__(*self._called(source, name, title))
        #: ``TH2Poly(h2p)`` copies the bins, as ROOT's copy constructor copies them.
        self._xrd_bins = source._xrd_bins.copy() if source else PolyBins(*self._booked(ranges))
        self._extremes, self._stats = [UNSET, UNSET], True
        self._made: list[tuple[Any, str]] = []
        self._pad: Any = None
        self._copied(source)

    @staticmethod
    def _called(source: TH2Poly | None, name: Any, title: Any) -> tuple[str, str]:
        """The name and title of a new histogram, or those of the one being copied."""
        if source is None:
            return str(name), str(title)
        return str(source.GetName()), str(source.GetTitle())

    @staticmethod
    def _booked(ranges: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[float, ...], bool]:
        """The ranges and whether they float, from ``TH2Poly(name, title, xlow, ...)``."""
        given = [float(v) for v in ranges]
        if len(given) >= 6:  # (name, title, nx, xlow, xup, ny, ylow, yup)
            given = [given[1], given[2], given[4], given[5]]
        return tuple(given[:2] or (0.0, 0.0)), tuple(given[2:4] or (0.0, 0.0)), not given

    def _copied(self, source: TH2Poly | None) -> None:
        """What a subclass keeps beyond the bins, carried over by ``TH2Poly(h)``.

        ROOT's copy constructor copies the whole object, so a subclass that keeps
        sums of its own puts them back here, once the bins are in place.
        """

    # -- bins ----------------------------------------------------------------------------

    def AddBin(self, *args: Any) -> int:
        """``AddBin(x1, y1, x2, y2)``, ``AddBin(n, x, y)`` or ``AddBin(graph)``: its number."""
        if len(args) == 4:
            x1, y1, x2, y2 = (float(v) for v in args)
            return self._xrd_bins.add([([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1])])
        if len(args) == 3:
            n = int(args[0])
            return self._xrd_bins.add([(np.asarray(args[1])[:n], np.asarray(args[2])[:n])])
        number = self._xrd_bins.add(_points(args[0]))
        name = args[0].GetName() if hasattr(args[0], "GetName") else ""
        if name:
            self.SetBinName(number, name)
        return number

    def Honeycomb(self, xstart: float, ystart: float, a: float, k: int, s: int,
                  option: str = "v") -> None:  # fmt: skip
        """``Honeycomb``: ``s`` rows of hexagons of side ``a``, ``k`` and ``k - 1`` by turns."""
        vertical = "v" in str(option).lower()
        step, shift = (a * math.sqrt(3), a * math.sqrt(3) / 2)
        along, across = (xstart, ystart + a / 2) if vertical else (ystart + a * math.sqrt(3) / 2,
                                                                    xstart)  # fmt: skip
        for row in range(int(s)):
            for i in range(int(k) if row % 2 == 0 else int(k) - 1):
                x0, y0 = (along + i * step, across) if vertical else (across, along + i * step)
                self._xrd_bins.add([_hexagon(x0, y0, a, vertical)])
            along += shift if row % 2 == 0 else -shift
            across += 1.5 * a

    # -- filling and asking --------------------------------------------------------------

    def Fill(self, x: Any, y: float = 1.0, w: float = 1.0) -> int:
        """``Fill(x, y[, w])``, or ``Fill(name, w)`` into the bin of that name: its number."""
        if isinstance(x, str):
            return self._fill_named(x, float(y))
        return self._xrd_bins.fill(float(x), float(y), float(w))

    def _fill_named(self, name: str, w: float) -> int:
        names = self.__dict__.get("_names", {})
        found = names.get(name)
        if found is None:
            return 0
        self._xrd_bins.contents[found - 1] += w
        self._xrd_bins.sums[6] += 1  # ROOT's ``Fill(name, w)`` counts an entry, no moments
        if self._xrd_bins.squares is not None:
            self._xrd_bins.squares[self._xrd_bins.slot(int(found))] += w * w
        return int(found)

    def Add(self, other: Any, c: float = 1.0) -> bool:
        """``Add(h, c)``: ``c`` times a ``TH2Poly`` of the same bins added, bin by bin.

        The moments add as ROOT's do - the sum of squared weights by ``c``
        squared, the others by ``|c|`` - and the entries by ``c``.
        """
        theirs = other._xrd_bins
        if len(theirs.contents) != len(self._xrd_bins.contents):
            self.Error("Add", "Attempt to add histograms with different number of bins")
            return False
        mine, c = self._xrd_bins, float(c)
        mine.contents += c * theirs.contents
        mine.overflow += c * theirs.overflow
        factors = [abs(c), c * c, abs(c), abs(c), abs(c), abs(c), 0.0, abs(c), abs(c), abs(c)]
        entries = abs(mine.sums[6] + c * theirs.sums[6])
        for at, (factor, value) in enumerate(zip(factors, theirs.sums, strict=True)):
            mine.sums[at] += factor * value
        mine.sums[6] = entries
        if mine.squares is not None or theirs.squares is not None:
            # ``TH2Poly::Add`` keeps the squared weights of both, times ``c`` squared.
            mine.sumw2()
            theirs.sumw2()
            mine.squares = mine.squares + c * c * theirs.squares
        return True

    def SetDirectory(self, directory: Any) -> None:
        """``SetDirectory``: a ``TH2Poly`` here belongs to no directory, so there is none to
        leave or join."""

    def SetBinName(self, number: int, name: str) -> None:
        self.__dict__.setdefault("_names", {})[str(name)] = int(number)

    def GetNumberOfBins(self) -> int:
        return len(self._xrd_bins.polygons)

    # -- the grid ROOT draws it on -------------------------------------------------------
    #
    # A ``TH2Poly`` has no axis of its own: ROOT keeps a fixed 100-cell x and y axis and a
    # one-cell z, only so that the frame, the range and ``TAxis::kAxisRange`` have something
    # to be asked of. The bins are *not* these cells - a bin is a polygon by number.

    def GetDimension(self) -> int:
        return 2

    def GetXaxis(self) -> Any:
        from .core.axes import TAxis

        return TAxis(self.GRID, *self._xrd_bins.xrange)

    def GetYaxis(self) -> Any:
        from .core.axes import TAxis

        return TAxis(self.GRID, *self._xrd_bins.yrange)

    def GetZaxis(self) -> Any:
        from .core.axes import TAxis

        return TAxis(1, 0.0, 1.0)

    def GetNcells(self) -> int:
        """ROOT's cell count: the bins, then the nine under/overflow regions round them."""
        return self.GetNumberOfBins() + REGIONS

    def GetBinContent(self, number: int) -> float:
        """A bin's content by number from 1, or a region's, -1 to -9."""
        return self._xrd_bins.content(int(number))

    def SetBinContent(self, number: int, content: float) -> None:
        number = int(number)
        if number < 0:
            self._xrd_bins.overflow[-number - 1] = float(content)
        elif 0 < number <= len(self._xrd_bins.contents):
            self._xrd_bins.contents[number - 1] = float(content)

    def GetBinError(self, number: int) -> float:
        """``GetBinError``: the root of the squared weights kept, or of the content.

        Bins are numbered from 1 and regions -1 to -9, as ``GetBinContent`` numbers them;
        0, a bin beyond the last and a region beyond the ninth have no error.
        """
        number = int(number)
        if number == 0 or number > len(self._xrd_bins.contents) or number < -REGIONS:
            return 0.0
        return self._xrd_bins.error(number)

    def SetBinError(self, number: int, error: float) -> None:
        """``SetBinError``: the squared error kept for that bin, which starts the sums.

        Setting a content afresh does not set its error, as ROOT's does not: a histogram
        that keeps squared weights keeps the ones it was given.
        """
        number = int(number)
        if number == 0 or number > len(self._xrd_bins.contents) or number < -REGIONS:
            return
        self._xrd_bins.sumw2()[self._xrd_bins.slot(number)] = float(error) ** 2

    def Sumw2(self, flag: bool = True) -> None:
        """``Sumw2``: keep the sum of squared weights per bin; ``Sumw2(false)`` drops them."""
        if not flag:
            self._xrd_bins.squares = None
        else:
            self._xrd_bins.sumw2()

    def GetSumw2N(self) -> int:
        """The length of the array of squared weights, zero while none is kept."""
        kept = self._xrd_bins.squares
        return 0 if kept is None else len(kept)

    def ClearBinContents(self) -> None:
        self._xrd_bins.contents[:] = 0.0
        self._xrd_bins.overflow[:] = 0.0

    Reset = ClearBinContents

    def GetEntries(self) -> float:
        return float(self._xrd_bins.sums[6])

    def Integral(self, option: str = "") -> float:
        return float(self._xrd_bins.contents.sum())

    def GetMean(self, axis: int = 1) -> float:
        """``GetMean(axis)``: 1, 2, 3 for x, y, z; 11, 12, 13 for the error on each."""
        if not 1 <= axis <= 3 and not 11 <= axis <= 13:
            return 0.0
        if axis > 10:
            return self.GetMeanError(axis - 10)
        found = self._stats_array()
        return float(found[AXIS_SUM[axis - 1]] / found[0]) if found[0] else 0.0

    def GetStdDev(self, axis: int = 1) -> float:
        """``GetStdDev(axis)``: the spread, never below zero, and the error on it past 10."""
        if not 1 <= axis <= 3 and not 11 <= axis <= 13:
            return 0.0
        found = self._stats_array()
        if not found[0]:
            return 0.0
        at = AXIS_SUM[axis % 10 - 1]
        variance = max(found[at + 1] / found[0] - (found[at] / found[0]) ** 2, 0.0)
        if axis < 10:
            return math.sqrt(variance)
        count = self.GetEffectiveEntries()
        return math.sqrt(variance / (2 * count)) if count > 0 else 0.0

    GetRMS = GetStdDev

    def GetMeanError(self, axis: int = 1) -> float:
        """``GetMeanError``: the spread over the root of the effective entries."""
        count = self.GetEffectiveEntries()
        return self.GetStdDev(axis) / math.sqrt(count) if count > 0 else 0.0

    def GetStdDevError(self, axis: int = 1) -> float:
        return self.GetStdDev(axis + 10)

    def GetEffectiveEntries(self) -> float:
        """``GetEffectiveEntries``: ``(sum w) ** 2 / sum w ** 2``: the count the weights hold."""
        found = self._stats_array()
        return float(found[0] * found[0] / found[1]) if found[1] else abs(float(found[0]))

    def _stats_array(self) -> np.ndarray[Any, Any]:
        """``GetStats``' numbers from the sums: the moments, the cross term, the value's."""
        sums = self._xrd_bins.sums
        return np.array([*sums[:6], sums[7], sums[8], sums[9]])

    def GetMinimum(self, *args: Any) -> float:
        contents = self._xrd_bins.contents
        low = self._extremes[0]
        return low if low != UNSET else float(contents.min()) if len(contents) else 0.0

    def GetMaximum(self, *args: Any) -> float:
        contents = self._xrd_bins.contents
        high = self._extremes[1]
        return high if high != UNSET else float(contents.max()) if len(contents) else 0.0

    def SetMinimum(self, minimum: float = UNSET) -> None:
        self._extremes[0] = float(minimum)

    def SetMaximum(self, maximum: float = UNSET) -> None:
        self._extremes[1] = float(maximum)

    def SetStats(self, stats: bool = True) -> None:
        self._stats = bool(stats)

    def GetBinCenter(self, number: int) -> tuple[float, float]:
        """The middle of a bin's box: where its content is written."""
        x1, x2, y1, y2 = self._xrd_bins.boxes[int(number) - 1]
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    # -- drawing -------------------------------------------------------------------------

    def Draw(self, option: str = "") -> None:
        """``Draw``: into the current pad - cleared first unless ``same``."""
        from .graphics.canvas import default_canvas
        from .graphics.pads import current

        self._pad = current() or default_canvas()
        if "SAME" not in str(option).upper():
            self._pad.Clear()
        self._pad.add(self, str(option))

    def _frame(self, option: str) -> tuple[Any, str]:
        """The frame the bins are drawn in: the axes' range, the palette's for ``COLZ``."""
        bins = self._xrd_bins
        frame = Histogram.book(f"{self.GetName()}_frame", (1, *bins.xrange), (1, *bins.yrange),
                               title=self.GetTitle())  # fmt: skip
        low, high = self.GetMinimum(), self.GetMaximum()
        frame._core.update(fMinimum=low, fMaximum=high)
        frame._cells()[:] = low - 1.0  # below the palette: no cell of the frame is coloured
        named = frame._core["TNamed"]
        named["fBits"] = int(named.get("fBits", 0) or 0) | NO_STATS
        palette = "COLZ" if "COL" in option and "Z" in option else "COL" if "COL" in option else ""
        return frame, palette or "AXIS"

    def _colour(self, z: float, low: float, high: float) -> int:
        """``PaintTH2PolyColorLevels``' palette colour for a content ``z``."""
        from .graphics.style import gStyle

        ncolors, ndiv = gStyle.GetNumberOfColors(), gStyle.GetNumberContours()
        level = int(0.01 + (z - low) * ndiv / (high - low)) if high > low else 0
        return int(gStyle.GetColorPalette(min(int((level + 0.99) * ncolors / ndiv), ncolors - 1)))

    def _outline(self, xs: Any, ys: Any, fill: int | None) -> tuple[Any, str]:
        from .graphics.shapes import TPolyLine

        line = TPolyLine(len(xs), xs, ys, "f" if fill is not None else "")
        if fill is None:
            look: Any = self  # its own line is the outline's
            line.SetLineColor(look.GetLineColor())
            line.SetLineStyle(look.GetLineStyle())
            line.SetLineWidth(look.GetLineWidth())
        else:
            line.SetFillColor(fill)
            line.SetFillStyle(1001)
        return line, "f" if fill is not None else ""

    def _pieces(self, option: str) -> list[tuple[Any, str]]:
        """What the option paints of each bin: colour, outline, content, marker."""
        from .graphics.text import TLatex

        bins, made = self._xrd_bins, []
        low, high = self.GetMinimum(), self.GetMaximum()
        for number, (polygons, z) in enumerate(zip(bins.polygons, bins.contents, strict=False), 1):
            if "COL" in option and z >= low:
                made += [self._outline(x, y, self._colour(z, low, high)) for x, y in polygons]
            if "L" in option.replace("COL", ""):
                made += [self._outline(x, y, None) for x, y in polygons]
            if "TEXT" in option:
                text = TLatex(*self.GetBinCenter(number), f"{z:g}")
                text.SetTextAlign(22)
                look: Any = self  # its marker's colour and size are the text's
                text.SetTextColor(look.GetMarkerColor())
                text.SetTextSize(0.02 * look.GetMarkerSize())
                made.append((text, ""))
        return made

    def paint_pad(self) -> None:
        """The frame, then each bin as the option says, in place of what was painted last."""
        from .graphics.polar import replace_made

        option = next((opt for obj, opt in self._pad.primitives if obj is self), "").upper()
        replace_made(self._pad, self, [self._frame(option), *self._pieces(option)])


class TProfile2Poly(UnifiedHistogram, TH2Poly):
    """``TProfile2Poly``: a ``TH2Poly`` whose bins are the means of a value filled into them.

    ``Fill(x, y, value[, w])`` averages ``value`` in the bin holding the
    point; a fill missing every bin is not kept. Like every ``TH2`` it also
    lays a grid of 100 by 100 bins over its range, and UHI's protocols see it
    through that grid, as ROOT's do: a cell's ROOT number names a polygon
    only by chance, so through the grid its bins are empty, and it cannot be
    sliced, set or serialized.
    """

    #: The bins ``TH2Poly`` lays along each axis of its range.
    GRID = 100

    def __init__(self, name: Any = "NoName", title: Any = "NoTitle", *ranges: float) -> None:
        #: Each bin's sum of weights, of weighted values, of weighted squared values and
        #: of squared weights - ROOT's ``fSumw, fSumvw, fSumwv2, fSumw2``.
        self._sums: dict[int, list[float]] = {}
        self._error_mode = 0
        super().__init__(name, title, *ranges)

    def _copied(self, source: TH2Poly | None) -> None:
        """Each bin's sums too: a copy of a profile is a profile of the same fills."""
        if isinstance(source, TProfile2Poly):
            self._sums = {number: list(sums) for number, sums in source._sums.items()}
            self._error_mode = source._error_mode

    def ClassName(self) -> str:
        return "TProfile2Poly"

    def Fill(  # type: ignore[override]  # a profile's fill takes the value averaged too
        self, x: Any, y: float = 1.0, value: float = 1.0, w: float = 1.0
    ) -> int:
        """``Fill(x, y, value[, w])``: ``value`` averaged into the bin holding ``(x, y)``.

        A fill in one of the regions round the range, or within it in no bin, is kept
        too, as ROOT keeps it: the mean of the values that missed every bin is the
        content of that region. Only a fill inside the axes adds to the moments, as
        ``TProfile2D``'s does - which is what the two are compared over.
        """
        x, y, value, w = float(x), float(y), float(value), float(w)
        bins = self._xrd_bins
        found = bins.find(x, y)
        self._filled(found, value, w)
        if found > 0 or found == -5:
            bins.moments(w, x, y)
            sums = bins.sums
            sums[7] += w * x * y
            sums[8] += w * value
            sums[9] += w * value * value
        return found

    def _filled(self, number: int, value: float, weight: float) -> None:
        """One fill of one bin's - or one region's - sums, and the mean it now holds."""
        sums = self._sums.setdefault(number, [0.0, 0.0, 0.0, 0.0])
        sums[0] += weight
        sums[1] += weight * value
        sums[2] += weight * value * value
        sums[3] += weight * weight
        mean = sums[1] / sums[0] if sums[0] else 0.0
        bins = self._xrd_bins
        if number > 0:
            bins.contents[number - 1] = mean
        else:
            bins.overflow[-number - 1] = mean

    def Reset(self, option: str = "") -> None:
        """``Reset``: the bins' contents and sums, and the moments, back to nothing."""
        self._sums.clear()
        self._xrd_bins.sums = [0.0] * 10
        self.ClearBinContents()

    def Merge(self, collection: Any) -> int:
        """``Merge``: every profile of ``collection`` added bin by bin into this one.

        A profile is its bins' sums, so those add - the mean a bin then holds is
        recomputed - and so do the moments and the regions' means.
        """
        others = [other for other in collection if isinstance(other, TProfile2Poly)]
        for other in others:
            if other.GetNumberOfBins() != self.GetNumberOfBins():
                self.Error("Merge", "Bin numbers of TProfile2Polys to be merged differ")
                return 0
            for at, value in enumerate(other._xrd_bins.sums):
                self._xrd_bins.sums[at] += value
        self._merged(others)
        return len(others)

    def _merged(self, others: list[TProfile2Poly]) -> None:
        """The sums themselves merged, then every content put back to the mean it now has."""
        for other in others:
            for number, sums in other._sums.items():
                mine = self._sums.setdefault(number, [0.0, 0.0, 0.0, 0.0])
                for at, value in enumerate(sums):
                    mine[at] += value
        for number, sums in self._sums.items():
            mean = sums[1] / sums[0] if sums[0] else 0.0
            if number > 0:
                self._xrd_bins.contents[number - 1] = mean
            else:
                self._xrd_bins.overflow[-number - 1] = mean

    # -- the grid every TH2 has ---------------------------------------------------------

    def GetDimension(self) -> int:
        return 2

    def GetXaxis(self) -> Any:
        from .core.axes import TAxis

        return TAxis(self.GRID, *self._xrd_bins.xrange)

    def GetYaxis(self) -> Any:
        from .core.axes import TAxis

        return TAxis(self.GRID, *self._xrd_bins.yrange)

    def GetNbinsX(self) -> int:
        return self.GRID

    GetNbinsY = GetNbinsX

    def GetNcells(self) -> int:
        return (self.GRID + 2) ** 2

    def GetSumw2N(self) -> int:
        return 0

    def GetBin(self, binx: int, biny: int = 0, binz: int = 0) -> int:
        """``TH2::GetBin``: the grid cell's ROOT number, x running fastest."""
        return int(binx) + (self.GRID + 2) * int(biny)

    def FindBin(self, x: float, y: float = 0.0, z: float = 0.0) -> int:
        """The bin holding ``(x, y)``, or the region round the range it fell in."""
        return self._xrd_bins.find(float(x), float(y))

    def GetBinContent(self, *args: Any) -> float:
        """``GetBinContent(bin)``, or of the grid cell ``(binx, biny)``: a bin's mean."""
        number = self.GetBin(*args) if len(args) > 1 else int(args[0])
        return TH2Poly.GetBinContent(self, number)

    def GetBinEntries(self, bin: int) -> float:
        return self._sums.get(int(bin), _EMPTY)[0]

    def GetBinEffectiveEntries(self, bin: int) -> float:
        """``GetBinEffectiveEntries``: ``(sum w) ** 2 / sum w ** 2`` in that bin."""
        weight, _, _, squares = self._sums.get(int(bin), _EMPTY)
        return weight * weight / squares if squares else 0.0

    def GetBinError(self, *args: Any) -> float:
        """The error on a bin's mean, or on the spread of its values, as the option says.

        ROOT's ``kERRORSPREAD`` (``SetErrorOption("s")``) is the root of the mean of the
        squared deviations - the spread of what was filled - and the default is that over
        the root of the bin's effective entries, the error on the mean. A bin nothing was
        filled into has none, rather than ROOT's division by zero.
        """
        number = self.GetBin(*args) if len(args) > 1 else int(args[0])
        weight, total, squares, _ = self._sums.get(number, _EMPTY)
        if weight == 0:
            return 0.0
        spread = math.sqrt(max(squares / weight - (total / weight) ** 2, 0.0))
        if self._error_mode == kERRORSPREAD:
            return spread
        count = self.GetBinEffectiveEntries(number)
        return spread / math.sqrt(count) if count > 0 else 0.0

    def SetErrorOption(self, option: Any) -> None:
        """``SetErrorOption``: ``kERRORSPREAD``, or ``"s"``, for the spread of the means;
        anything else - ``kERRORMEAN``, ``""`` - for the error on each mean."""
        if isinstance(option, str):
            self._error_mode = kERRORSPREAD if option.strip().lower() == "s" else kERRORMEAN
        else:
            self._error_mode = kERRORSPREAD if int(option) == kERRORSPREAD else kERRORMEAN

    def GetErrorOption(self) -> str:
        return "s" if self._error_mode == kERRORSPREAD else ""

    # -- UHI, through the grid -----------------------------------------------------------

    def variances(self, flow: bool = False) -> Any:
        return _grid_by_copy(self, lambda *cell: self.GetBinError(*cell) ** 2, flow)

    def counts(self, flow: bool = False) -> Any:
        return _grid_by_copy(self, lambda *cell: self.GetBinEntries(self.GetBin(*cell)), flow)

    def __getitem__(self, index: Any) -> Any:
        cells = _compute_common_index(self, index)
        if not all(isinstance(cell, int) for cell in cells):
            raise UnsupportedFeatureError(
                "A TProfile2Poly cannot be sliced: its bins are polygons, not the grid a "
                "slice would cut. Index one bin at a time, or ask GetBinContent."
            )
        return self.GetBinContent(*cells)

    def __setitem__(self, index: Any, value: Any) -> None:
        raise UnsupportedFeatureError(
            "A TProfile2Poly's bins cannot be set: each is the mean of what was filled into "
            "it, which a mean alone does not say. Fill it instead."
        )

    def _to_uhi_(self) -> dict[str, Any]:
        raise UnsupportedFeatureError(
            "A TProfile2Poly cannot be written as UHI's histogram: its bins are polygons, "
            "which UHI's serialization has no axis for."
        )
