"""The layers a candle or violin plot is drawn as, one candle per bin of a 2-D histogram.

``THistPainter::PaintCandlePlot`` stands a candle in each bin along x (or
along y for ``CANDLEY``), ``fBarWidth`` of the bin wide and ``fBarOffset``
of it off centre, and ``TCandle::Paint`` draws what the option's digits ask
for in the histogram's line, fill and marker attributes: a dashed line for
the mean, a circle for a median or mean drawn as one, the violin as the
slice's own histogram mirrored about the candle's axis.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .candle import (
    ANCHOR,
    BOX,
    HISTO,
    HORIZONTAL,
    MEAN,
    MEDIAN,
    POINTS,
    SETTINGS,
    WHISKER,
    ZERO,
    Candle,
    candle_of,
    parse_option,
    part,
)  # fmt: skip
from .model import Area, Boxes, Curve, Look, Points
from .request import Request, styled

__all__ = ["candle_layers"]

#: ``TCandle``'s circle for a median or a mean drawn as one, and the dashes of a mean line.
CIRCLE, DASHED = 24, 2


class _Maker:
    """The layers of one candle in one look, swapping x and y for a horizontal one."""

    def __init__(self, look: Look, horizontal: bool) -> None:
        self.look, self.horizontal = look, horizontal

    def _xy(self, across: Any, along: Any) -> tuple[Any, Any]:
        across, along = np.asarray(across, float), np.asarray(along, float)
        return (along, across) if self.horizontal else (across, along)

    def line(self, across: Any, along: Any, dashed: bool = False) -> Curve:
        look = self.look._replace(dash="dashed", line_style=DASHED) if dashed else self.look
        return Curve(*self._xy(across, along), look)

    def box(self, c: Candle, half: float) -> list[Any]:
        """The box: filled as the histogram is, and outlined in its line."""
        x0, x1, y0, y1 = c.pos - half, c.pos + half, c.box_down, c.box_up
        if self.horizontal:
            x0, x1, y0, y1 = y0, y1, x0, x1
        filled = [Boxes([x0], [x1], [y0], [y1], self.look)] if self.look.fill is not None else []
        outline = self.line([c.pos - half, c.pos + half, c.pos + half, c.pos - half, c.pos - half],
                            [c.box_down, c.box_down, c.box_up, c.box_up, c.box_down])  # fmt: skip
        return [*filled, outline]

    def notched(self, c: Candle, half: float) -> list[Any]:
        """``kMedianNotched``: the box pinched to half its width at the median's notch."""
        across = [c.pos - half, c.pos + half, c.pos + half, c.pos + half / 2, c.pos + half,
                  c.pos + half, c.pos - half, c.pos - half, c.pos - half / 2, c.pos - half,
                  c.pos - half]  # fmt: skip
        along = [c.box_down, c.box_down, c.median - c.median_err, c.median,
                 c.median + c.median_err, c.box_up, c.box_up, c.median + c.median_err,
                 c.median, c.median - c.median_err, c.box_down]  # fmt: skip
        filled = [Area(*self._xy(across, along), self.look)] if self.look.fill is not None else []
        return [*filled, self.line(across, along)]

    def marker(self, across: Any, along: Any, style: int | None = None) -> Points:
        look = self.look if style is None else self.look._replace(
            marker="circle", hollow=True, marker_style=style)
        x, y = self._xy(across, along)
        zeros = np.zeros_like(x)
        return Points(x, y, zeros, zeros, zeros, zeros, look)

    def violin(self, c: Candle, code: int, half: float, top: float) -> list[Any]:
        """``kHistoLeft``, ``kHistoRight`` and ``kHistoViolin``: the slice's histogram, each
        bin's content as a width out from the axis, on one side or mirrored on both."""
        reach = half * np.asarray(c.contents, float) / (top or 1.0)
        along = np.repeat(c.edges, 2)[1:-1]
        out = np.repeat(reach, 2)
        side = part(code, HISTO)
        left = c.pos - out if side in (1, 3) else np.full(len(out), c.pos)
        right = c.pos + out if side in (2, 3) else np.full(len(out), c.pos)
        across = np.concatenate([left, right[::-1], left[:1]])
        both = np.concatenate([along, along[::-1], along[:1]])
        filled = [Area(*self._xy(across, both), self.look)] if self.look.fill is not None else []
        return [*filled, self.line(across, both)]


def _one(c: Candle, code: int, maker: _Maker, scale: float, top: float) -> list[Any]:
    """Every layer the option's digits ask for, of one candle: the violin first, under the
    box, then what crosses it, then the whiskers and the points."""
    half = c.width * scale / 2
    layers: list[Any] = maker.violin(c, code, c.width / 2, top) if part(code, HISTO) else []
    if part(code, BOX) == 1:
        layers += maker.notched(c, half) if part(code, MEDIAN) == 2 else maker.box(c, half)
    layers += _across(c, code, maker, half)
    return layers + _outward(c, code, maker, half)


def _across(c: Candle, code: int, maker: _Maker, half: float) -> list[Any]:
    """The median and the mean: a line across the box, or a circle on its axis."""
    layers: list[Any] = []
    for place, value in ((MEDIAN, c.median), (MEAN, c.mean)):
        digit = part(code, place)
        if digit == 3:
            layers.append(maker.marker([c.pos], [value], CIRCLE))
        elif digit in (1, 2):
            dashed = place == MEAN
            layers.append(maker.line([c.pos - half, c.pos + half], [value, value], dashed))
    return layers


def _outward(c: Candle, code: int, maker: _Maker, half: float) -> list[Any]:
    """The whiskers and their anchors, the points, and the zero line."""
    layers: list[Any] = []
    if part(code, WHISKER):
        layers.append(maker.line([c.pos, c.pos], [c.box_up, c.whisker_up]))
        layers.append(maker.line([c.pos, c.pos], [c.box_down, c.whisker_down]))
    if part(code, ANCHOR):
        for end in (c.whisker_up, c.whisker_down):
            layers.append(maker.line([c.pos - half / 2, c.pos + half / 2], [end, end]))
    if part(code, POINTS) and len(c.points[0]):
        layers.append(maker.marker(c.points[0], c.points[1]))
    if part(code, ZERO):
        layers.append(maker.line([c.pos - half, c.pos + half], [0.0, 0.0], dashed=True))
    return layers


#: The seed of the generator that scatters points, the same every time a picture is drawn.
SEED = 4357


def _slices(histogram: Any, horizontal: bool) -> list[tuple[float, float, Any]]:
    """Each bin along the candles' axis: its centre, its width and the slice in it."""
    edges = np.asarray(histogram.axes[1 if horizontal else 0].edges(), dtype=float)
    found = []
    for at in range(len(edges) - 1):
        bins = (at + 1, at + 1)
        proj = (histogram.projection_x(y_range=bins) if horizontal
                else histogram.projection_y(x_range=bins))  # fmt: skip
        found.append((0.5 * (edges[at] + edges[at + 1]), edges[at + 1] - edges[at], proj))
    return found


def candle_layers(histogram: Any, request: Request) -> list[Any]:
    """A candle per bin, as the option's digits and ``TCandle``'s settings draw them."""
    code = parse_option(request.chosen.candle)
    horizontal = code >= HORIZONTAL
    core = getattr(histogram, "_core", {})  # fBarWidth and fBarOffset, in thousandths
    bar_width = float(core.get("fBarWidth", 1000)) / 1000
    bar_offset = float(core.get("fBarOffset", 0)) / 1000
    rng = np.random.default_rng(SEED)
    candles = []
    for centre, width, proj in _slices(histogram, horizontal):
        made = candle_of(proj, centre + bar_offset * width, bar_width * width, code, rng)
        if made is not None:
            candles.append(made)
    if not candles:
        return []
    most = max(c.entries for c in candles)
    tallest = max(float(np.max(c.contents)) for c in candles)
    maker = _Maker(styled(histogram.members, request, markers=True), horizontal)
    layers: list[Any] = []
    for c in candles:
        scale = c.entries / most if SETTINGS["scaled_candle"] else 1.0
        top = tallest if SETTINGS["scaled_violin"] else float(np.max(c.contents))
        layers += _one(c, code, maker, scale, top)
    return layers
