"""A ``TLegend``: its box, and a row per entry of a symbol and a label.

``TLegend::PaintPrimitives`` splits the box into rows, one per entry (the
header one of its own), and columns of ``fNColumns``; each entry's symbol
is drawn in the ``fMargin`` of its column - a filled box for ``f``, a line
for ``l``, an error bar for ``e``, a marker for ``p``, in the attributes of
the object it stands for - and its label after it. Text left at size 0 is
sized to fit: ``1 - fEntrySeparation`` of a row, less if the tallest label
or the widest row of labels would not fit.
"""

from __future__ import annotations

import math
from typing import Any

from .latex import formula_form, paint_latex
from .model import Primitive, lookup
from .paves import draw_box
from .scene import Scene

__all__ = ["LEGEND", "legend"]

#: ``gStyle``'s ``fEndErrorSize``: how wide an error bar's end is, in tenths of the box.
END_ERROR_SIZE = 2.0


class _Rows:
    """A legend's geometry: its corners in NDC, rows and columns, and the room for symbols."""

    def __init__(self, scene: Scene, prim: Primitive, entries: list[Any], corners: tuple[float, ...]) -> None:
        self.scene = scene
        self.x1, self.y1, self.x2, self.y2 = corners
        self.columns = max(int(prim.get("fNColumns", 1) or 1), 1)
        header = 1 if any("h" in str(entry.get("fOption", "")).lower() for entry in entries[:1]) else 0
        self.rows = header + math.ceil((len(entries) - header) / self.columns)
        self.margin = float(prim.get("fMargin", 0.25)) * (self.x2 - self.x1) / self.columns
        self.space = (self.y2 - self.y1) / self.rows
        self.separation = float(prim.get("fEntrySeparation", 0.1))

    def width(self, text: str, size: float, font: int) -> float:
        form = formula_form(text, size, font, self.scene.whole, self.scene.height)
        return form.width / self.scene.pixels[0]

    def height(self, text: str, size: float, font: int) -> float:
        form = formula_form(text, size, font, self.scene.whole, self.scene.height)
        return form.height / self.scene.pixels[1]


def _font(entry: Any, prim: Primitive, autosize: bool) -> int:
    font = int(entry.get("fTextFont", 0) or prim.get("fTextFont", 42) or 42)
    return font - 1 if autosize and font % 10 == 3 else font


def _autosize(rows: _Rows, prim: Primitive, entries: list[Any]) -> float:
    """The labels' size when the legend left it at 0: the row, less if a label would not fit."""
    size = (1 - rows.separation) * rows.space
    tallest, widths, widest = 0.0, [0.0] * rows.columns, 0.0
    column = 0
    for entry in entries:
        label, font = str(entry.get("fLabel", "")), _font(entry, prim, True)
        tallest = max(tallest, rows.height(label, size, font))
        if "h" in str(entry.get("fOption", "")).lower():
            widest = max(widest, rows.width(label, size, font))
        else:
            widths[column] = max(widths[column], rows.width(label, size, font))
            column = (column + 1) % rows.columns
        widest = max(widest, sum(widths))
    size = min(size, tallest)
    share = 1.0 - float(prim.get("fMargin", 0.25)) - (float(prim.get("fColumnSeparation", 0.0)) if rows.columns > 1 else 0.0)
    return min(size, size * (rows.x2 - rows.x1) * share / widest) if widest else size


def legend(scene: Scene, prim: Primitive, _option: str) -> None:
    """``TLegend::Paint``: the box, then each entry's label and symbol in its row."""
    corners = draw_box(scene, prim)
    entries = list(prim.get("fPrimitives") or [])
    if not entries:
        return
    rows = _Rows(scene, prim, entries, corners)
    size = float(prim.get("fTextSize", 0.0) or 0.0)
    autosize = not size
    if autosize:
        size = _autosize(rows, prim, entries)
    at_y = rows.y2 + 0.5 * rows.space
    for index, entry in enumerate(entries):
        column = index % rows.columns if rows.columns > 1 else 0
        if column == 0:
            at_y -= rows.space
        _entry(rows, prim, entry, at_y, size, autosize)


def _entry(rows: _Rows, prim: Primitive, entry: Any, at_y: float, size: float, autosize: bool) -> None:
    """One entry: its label after the margin, and the symbol of what it stands for in the margin."""
    option = str(entry.get("fOption", "")).lower()
    align = int(entry.get("fTextAlign", 0) or prim.get("fTextAlign", 12) or 12)
    across, up = divmod(align, 10)
    margin = rows.margin / 10 if "h" in option else rows.margin
    x1, x2 = rows.x1, rows.x2
    x = {1: x1 + margin, 2: 0.5 * (x1 + margin + x2)}.get(across, x2 - margin / 10)
    half = rows.space / 2
    y = {1: at_y - (1 - rows.separation) * half, 3: at_y + (1 - rows.separation) * half}.get(up, at_y)
    font = _font(entry, prim, autosize)
    if up == 2 and half < size:
        align, y = 10 * across + 1, at_y - (1 - rows.separation) * half / 2
    own = float(entry.get("fTextSize", 0.0) or 0.0)
    attributes = rows.scene.attributes(entry, prim, font=font, size=own or size, align=align)
    paint_latex(rows.scene, str(entry.get("fLabel", "")), rows.scene.pixel(x, y), attributes)
    _symbol(rows, entry, option, (x1 + rows.margin / 2, at_y))


def _source(entry: Any) -> Any:
    found = entry.get("fObject")
    return found if found is not None else entry


def _segments(rows: _Rows, style: Any, segments: list[tuple[float, float, float, float]]) -> None:
    """``PaintSegmentsNDC``: each segment of a symbol, in the entry's object's line attributes."""
    from .raster import add_line

    scene = rows.scene
    width = int(lookup(style, "fLineWidth", 1) or 0)
    color = scene.colors.rgb(lookup(style, "fLineColor", 1))
    for a, b, c, d in segments if width > 0 else []:
        add_line(scene, [scene.pixel(a, b), scene.pixel(c, d)], color, width, lookup(style, "fLineStyle", 1))


def _symbol(rows: _Rows, entry: Any, option: str, centre: tuple[float, float]) -> None:
    """``f``, ``l``, ``e`` and ``p``: the fill, line, error bar and marker of the entry's object."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    from .shapes import patch_style

    source = _source(entry)
    x, y = centre
    box = rows.margin * 0.35
    tall = rows.space * 0.35
    if "f" in option:
        corners = [(x - box, y - tall), (x + box, y - tall), (x + box, y + tall), (x - box, y + tall)]
        rows.scene.ax.add_artist(Polygon(corners, closed=True, transform=rows.scene.ndc, clip_on=False,
                                         zorder=rows.scene.layer(), **patch_style(rows.scene, source, outline=False)))
    segments: list[tuple[float, float, float, float]] = []
    if "f" in option and "l" not in option:
        segments += [(x - box, y + tall, x + box, y + tall), (x - box, y - tall, x + box, y - tall),
                     (x + box, y - tall, x + box, y + tall), (x - box, y - tall, x - box, y + tall)]
    elif "l" in option:
        segments.append((x - box, y, x + box, y))
    if "e" in option:
        segments += _error_mark(rows, source, option, centre, box)
    _segments(rows, source, segments)
    if "p" in option:
        rows.scene.ax.add_artist(Line2D([x], [y], linestyle="none", transform=rows.scene.ndc, clip_on=False,
                                        zorder=rows.scene.layer(), **rows.scene.marker(source)))


def _error_mark(rows: _Rows, source: Any, option: str, centre: tuple[float, float], box: float) -> list[tuple[float, float, float, float]]:
    """An error bar through the symbol, broken round its marker, with ends for a graph's or ``E1``'s."""
    x, y = centre
    reach = rows.space * 0.3
    if "p" not in option:
        marks = [(x, y - reach, x, y + reach)]
    else:
        style = int(lookup(source, "fMarkerStyle", 1) or 1)
        size = float(lookup(source, "fMarkerSize", 1) or 1) if style >= 5 else 0.0
        gap = 0.5 * int(size * 8) / rows.scene.pixels[1]
        marks = [(x, y + gap, x, y + reach), (x, y - gap, x, y - reach)]
    kind = type(source).__name__
    if "Graph" in kind or ("e1" in option and "Histogram" in kind):
        bar = box * 0.1 * END_ERROR_SIZE
        marks += [(x - bar, y + reach, x + bar, y + reach), (x - bar, y - reach, x + bar, y - reach)]
    return marks


#: How this class draws.
LEGEND = {"TLegend": legend}
