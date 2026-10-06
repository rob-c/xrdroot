"""A ``TLegend``: its box, and a row per entry of a symbol and a label.

``TLegend::PaintPrimitives`` splits the box into rows, one per entry (the
header one of its own), and columns of ``fNColumns``; each entry's symbol
is drawn in the ``fMargin`` of its column - a filled box for ``f``, a line
for ``l``, an error bar for ``e``, a marker for ``p``, in the attributes of
the object it stands for - and its label after it. Text left at size 0 is
sized to fit: ``1 - fEntrySeparation`` of a row, less if the tallest label
or the widest row of labels would not fit. Each column is as wide as its
widest label's share of the room left by the margins, and its margin.
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

Segment = tuple[float, float, float, float]


def _option(entry: Any) -> str:
    return str(entry.get("fOption", "")).lower()


class _Rows:
    """A legend's geometry: its corners in NDC, rows and columns, and the room for symbols."""

    def __init__(self, scene: Scene, prim: Primitive, entries: list[Any], corners: Any) -> None:
        self.scene = scene
        self.x1, self.y1, self.x2, self.y2 = corners
        self.columns = max(int(prim.get("fNColumns", 1) or 1), 1)
        header = 1 if any("h" in _option(entry) for entry in entries[:1]) else 0
        self.rows = header + math.ceil((len(entries) - header) / self.columns)
        self.margin = float(prim.get("fMargin", 0.25)) * (self.x2 - self.x1) / self.columns
        self.space = (self.y2 - self.y1) / self.rows
        self.separation = float(prim.get("fEntrySeparation", 0.1))
        #: The share of the box's width the labels have: less the margins and the columns' gaps.
        self.share = 1.0 - float(prim.get("fMargin", 0.25))
        if self.columns > 1:
            self.share -= float(prim.get("fColumnSeparation", 0.0) or 0.0)

    def width(self, text: str, size: float, font: int) -> float:
        form = formula_form(text, size, font, self.scene.whole, self.scene.height)
        return form.width / self.scene.pixels[0]

    def height(self, text: str, size: float, font: int) -> float:
        form = formula_form(text, size, font, self.scene.whole, self.scene.height)
        return form.height / self.scene.pixels[1]


def _font(entry: Any, prim: Primitive, autosize: bool) -> int:
    """An entry's font, or the legend's; one sized in pixels is not when the legend sizes it."""
    font = int(entry.get("fTextFont", 0) or prim.get("fTextFont", 42) or 42)
    return font - 1 if autosize and font % 10 == 3 else font


def _widths(rows: _Rows, entries: list[Any], sized: Any) -> tuple[list[float], float]:
    """Each column's widest label, and the header's width: ``sized`` gives each entry's size."""
    widths, header, column = [0.0] * rows.columns, 0.0, 0
    for entry in entries:
        size, font = sized(entry)
        width = rows.width(str(entry.get("fLabel", "")), size, font)
        if "h" in _option(entry):
            header = max(header, width)
            continue
        widths[column] = max(widths[column], width)
        column = (column + 1) % rows.columns
    return widths, header


def _autosize(rows: _Rows, prim: Primitive, entries: list[Any]) -> float:
    """The labels' size when the legend left it at 0: the row, less if a label would not fit."""
    size = (1 - rows.separation) * rows.space
    labels = [(str(entry.get("fLabel", "")), _font(entry, prim, True)) for entry in entries]
    tallest = max(rows.height(label, size, font) for label, font in labels)
    widths, header = _widths(rows, entries, lambda entry: (size, _font(entry, prim, True)))
    widest = max(header, sum(widths))
    size = min(size, tallest)
    return float(min(size, size * (rows.x2 - rows.x1) * rows.share / widest) if widest else size)


def _column_widths(rows: _Rows, prim: Primitive, entries: list[Any], size: float,
                   autosize: bool) -> list[float]:  # fmt: skip
    """Each column's width in NDC: its widest label's share of the labels' room, and the margin."""

    def sized(entry: Any) -> tuple[float, int]:
        return float(entry.get("fTextSize", 0.0) or 0.0) or size, _font(entry, prim, autosize)

    widths, _header = _widths(rows, entries, sized)
    total = sum(widths) / rows.share
    across = rows.x2 - rows.x1
    return [(width / total if total else 0.0) * across + rows.margin for width in widths]


def legend(scene: Scene, prim: Primitive, _option_: str) -> None:
    """``TLegend::Paint``: the box, then each entry's label and symbol in its row and column."""
    corners = draw_box(scene, prim)
    entries = list(prim.get("fPrimitives") or [])
    if not entries:
        return
    rows = _Rows(scene, prim, entries, corners)
    size = float(prim.get("fTextSize", 0.0) or 0.0)
    autosize = not size
    if autosize:
        size = _autosize(rows, prim, entries)
    widths = _column_widths(rows, prim, entries, size, autosize)
    for entry, place in zip(entries, _places(rows, prim, entries, widths), strict=False):
        _entry(rows, prim, entry, place, (size, autosize))


def _places(rows: _Rows, prim: Primitive, entries: list[Any], widths: list[float]) -> list[Any]:
    """Each entry's span across and its row's middle: a header all the width, the rest a column."""
    separation = float(prim.get("fColumnSeparation", 0.0) or 0.0)
    gap = separation * (rows.x2 - rows.x1) / max(rows.columns - 1, 1)
    at_y = rows.y2 + 0.5 * rows.space
    column, places = 0, []
    for entry in entries:
        if column == 0:
            at_y -= rows.space
        span = (rows.x1, rows.x2)
        if "h" not in _option(entry) and rows.columns > 1:
            left = rows.x1 + sum(widths[:column]) + column * gap
            span, column = (left, left + widths[column]), (column + 1) % rows.columns
        places.append((span, at_y))
    return places


def _entry(rows: _Rows, prim: Primitive, entry: Any, place: Any, sizing: Any) -> None:
    """One entry: its label after the margin, and the symbol of what it stands for in the margin."""
    (x1, x2), at_y = place
    size, autosize = sizing
    align = int(entry.get("fTextAlign", 0) or prim.get("fTextAlign", 12) or 12)
    across, up = divmod(align, 10)
    margin = rows.margin / 10 if "h" in _option(entry) else rows.margin
    x = {1: x1 + margin, 2: 0.5 * (x1 + margin + x2)}.get(across, x2 - margin / 10)
    half = (1 - rows.separation) * rows.space / 2
    y = {1: at_y - half, 3: at_y + half}.get(up, at_y)
    if up == 2 and rows.space / 2 < size:
        align, y = 10 * across + 1, at_y - half / 2
    own = float(entry.get("fTextSize", 0.0) or 0.0)
    font = _font(entry, prim, autosize)
    attributes = rows.scene.attributes(entry, prim, font=font, size=own or size, align=align)
    paint_latex(rows.scene, str(entry.get("fLabel", "")), rows.scene.pixel(x, y), attributes)
    _symbol(rows, entry, (x1 + rows.margin / 2, at_y))


def _source(entry: Any) -> Any:
    """What an entry stands for, whose attributes its symbol is drawn in: itself if nothing."""
    found = entry.get("fObject")
    return found if found is not None else entry


def _segments(rows: _Rows, style: Any, segments: list[Segment]) -> None:
    """``PaintSegmentsNDC``: each segment of a symbol, in the entry's object's line attributes."""
    from .raster import add_line

    scene = rows.scene
    width = int(lookup(style, "fLineWidth", 1) or 0)
    color = scene.colors.rgb(lookup(style, "fLineColor", 1))
    dashes = lookup(style, "fLineStyle", 1)
    for a, b, c, d in segments if width > 0 else []:
        add_line(scene, [scene.pixel(a, b), scene.pixel(c, d)], color, width, dashes)


def _filled(rows: _Rows, source: Any, centre: tuple[float, float], lined: bool) -> list[Segment]:
    """``f``: a box in the object's fill, outlined in its line unless ``l`` draws a line instead."""
    from matplotlib.patches import Polygon

    from .shapes import patch_style

    (x, y), box, tall = centre, rows.margin * 0.35, rows.space * 0.35
    corners = [(x - box, y - tall), (x + box, y - tall), (x + box, y + tall), (x - box, y + tall)]
    style = patch_style(rows.scene, source, outline=False)
    shape = Polygon(corners, closed=True, transform=rows.scene.ndc, clip_on=False, **style)
    shape.set_zorder(rows.scene.layer())
    rows.scene.ax.add_artist(shape)
    if lined:
        return []
    left, right, low, high = x - box, x + box, y - tall, y + tall
    return [(left, high, right, high), (left, low, right, low),
            (right, low, right, high), (left, low, left, high)]  # fmt: skip


def _symbol(rows: _Rows, entry: Any, centre: tuple[float, float]) -> None:
    """``f``, ``l``, ``e`` and ``p``: the fill, line, error bar and marker of the entry's object."""
    from .marks import draw_markers

    option, source = _option(entry), _source(entry)
    x, y = centre
    segments: list[Segment] = []
    if "f" in option:
        segments += _filled(rows, source, centre, "l" in option)
    if "l" in option:
        segments.append((x - rows.margin * 0.35, y, x + rows.margin * 0.35, y))
    if "e" in option:
        segments += _error_mark(rows, source, option, centre)
    _segments(rows, source, segments)
    if "p" in option:
        style = int(lookup(source, "fMarkerStyle", 1) or 1)
        size = float(lookup(source, "fMarkerSize", 1) or 1)
        color = rows.scene.colors.rgb(lookup(source, "fMarkerColor", 1))
        draw_markers(rows.scene, [rows.scene.pixel(x, y)], style, size, color)


def _error_mark(rows: _Rows, source: Any, option: str, centre: tuple[float, float],
                ) -> list[Segment]:  # fmt: skip
    """An error bar through the symbol, broken round its marker, ended for a graph or ``E1``."""
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
        marks += _ends(x, y, rows.margin * 0.35 * 0.1 * END_ERROR_SIZE, reach)
    return marks


def _ends(x: float, y: float, bar: float, reach: float) -> list[Segment]:
    """An error bar's ends, ``bar`` either side of it, ``reach`` above and below its middle."""
    return [(x - bar, y + reach, x + bar, y + reach), (x - bar, y - reach, x + bar, y - reach)]


#: How this class draws.
LEGEND = {"TLegend": legend}
