"""``TH2Poly``: a 2-D histogram whose bins are polygons, drawn as ``THistPainter`` draws one.

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

from ..hist import Histogram
from ..polybins import PolyBins
from .core.objects import TAttFill, TAttLine, TAttMarker, TNamed

__all__ = ["TH2Poly"]

#: The contents ``GetMinimum`` and ``GetMaximum`` answer when none were set.
UNSET = -1111.0
#: ``TH1::kNoStats``: the frame draws no statistics box.
NO_STATS = 1 << 9


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

    def __init__(self, name: Any = "NoName", title: Any = "NoTitle", *ranges: float) -> None:
        super().__init__(str(name), str(title))
        given = [float(v) for v in ranges]
        if len(given) >= 6:  # (name, title, nx, xlow, xup, ny, ylow, yup)
            given = [given[1], given[2], given[4], given[5]]
        self._xrd_bins = PolyBins(given[:2] or (0.0, 0.0), given[2:4] or (0.0, 0.0), not given)
        self._extremes, self._stats = [UNSET, UNSET], True
        self._made: list[tuple[Any, str]] = []
        self._pad: Any = None

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
        return int(found)

    def SetBinName(self, number: int, name: str) -> None:
        self.__dict__.setdefault("_names", {})[str(name)] = int(number)

    def GetNumberOfBins(self) -> int:
        return len(self._xrd_bins.polygons)

    def GetBinContent(self, number: int) -> float:
        """A bin's content by number from 1, or a region's, -1 to -9."""
        number = int(number)
        if number < 0:
            return float(self._xrd_bins.overflow[-number - 1])
        bins = self._xrd_bins.contents
        return float(bins[number - 1]) if 0 < number <= len(bins) else 0.0

    def SetBinContent(self, number: int, content: float) -> None:
        number = int(number)
        if number < 0:
            self._xrd_bins.overflow[-number - 1] = float(content)
        elif 0 < number <= len(self._xrd_bins.contents):
            self._xrd_bins.contents[number - 1] = float(content)

    def ClearBinContents(self) -> None:
        self._xrd_bins.contents[:] = 0.0
        self._xrd_bins.overflow[:] = 0.0

    Reset = ClearBinContents

    def GetEntries(self) -> float:
        return float(self._xrd_bins.sums[6])

    def Integral(self, option: str = "") -> float:
        return float(self._xrd_bins.contents.sum())

    def GetMean(self, axis: int = 1) -> float:
        sums = self._xrd_bins.sums
        return float(sums[2 if axis == 1 else 4] / sums[0]) if sums[0] else 0.0

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
