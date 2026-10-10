"""A pad's frame: the axes its data is drawn in, and how they are dressed.

A pad whose first histogram, or a graph drawn with ``A``, draws axes has a
frame: the pad less its margins. Its extent is what ROOT last drew it with
(``fUxmin`` to ``fUymax``) when the pad was drawn before it was saved, and
otherwise what ROOT would work out drawing it now - a histogram's axis and
its highest bin with five percent above it, a graph's points with a tenth
of their spread either side. A pad with no frame is one axes over the whole
of it, its range ``fX1`` to ``fY2``, with no axis lines at all.

The dressing is ROOT's: ticks inside the frame, on the top and right as
well when ``fTickx`` and ``fTicky`` say so, a dotted grid for ``fGridx`` and
``fGridy``, logarithmic scales for ``fLogx`` and ``fLogy``, and the axis
titles at the far end of each axis, sized as their ``TAttAxis`` says.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..efficiency import Efficiency
from ..errors import ROOTError
from ..function import Function
from ..graph import Graph
from ..hist import Histogram
from ..stacks import MultiGraph, Stack
from . import styles
from .model import Pad, lookup
from .options import ERRORS, axisless, histogram_option
from .scene import Scene

__all__ = ["open_axes", "dress", "extent", "owner", "shown_bins"]

#: How much room ROOT leaves above a histogram's highest bin, as ``gStyle->GetHistTopMargin()``.
TOP_MARGIN = 0.05
#: How much room ROOT leaves round a graph's points, as a fraction of their spread.
GRAPH_MARGIN = 0.1
#: What ``TAttAxis`` gives an axis never styled: label and title size, tick length.
LABEL_SIZE, TITLE_SIZE, TICK_LENGTH = 0.035, 0.035, 0.03
#: ``TH1::kNoTitle``: the bit of a histogram drawn with no title.
NO_TITLE = 1 << 17
#: Where ``gStyle`` puts a title a pad was saved without: its middle top, and its size.
TITLE_X, TITLE_Y, TITLE_SIZE_PAD = 0.5, 0.995, 0.05

#: How far below the top a logarithmic axis starts when its bottom is not above zero.
LOG_FLOOR = 1e-3

Extent = tuple[float, float, float, float]


def owner(pad: Pad) -> tuple[Any, str] | None:
    """What draws a pad's frame: the first histogram not drawn ``SAME``, or graph drawn ``A``."""
    for obj, option in pad.primitives:
        upper = option.upper()
        if isinstance(obj, (Graph, MultiGraph)):
            if "A" in upper.replace("SAME", ""):
                return obj, option
        elif isinstance(obj, (Histogram, Stack, Function, Efficiency)) and "SAME" not in upper:
            return obj, option
    return None


def _histogram_y(values: np.ndarray[Any, Any], log: bool) -> tuple[float, float]:
    """The y extent ROOT gives bins: from zero, or below the lowest, to above the highest."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return (0.1, 10.0) if log else (0.0, 1.0)
    low, high = float(finite.min()), float(finite.max())
    if log:
        positive = finite[finite > 0]
        bottom = float(positive.min()) * 0.5 if positive.size else 0.1
        return bottom, max(high, bottom) * 2.0
    low = 0.0 if low >= 0 else low
    span = (high - low) or 1.0
    return low - (TOP_MARGIN * span if low < 0 else 0.0), high + TOP_MARGIN * span


def _limit(obj: Any, name: str, fallback: float) -> float:
    """``fMinimum`` or ``fMaximum`` if the histogram was given one, or else ``fallback``.

    ROOT keeps ``-1111`` in them to mean unset, so the frame fits the bins.
    """
    value = lookup(obj, name)
    return fallback if value is None or float(value) == -1111 else float(value)


def shown_bins(h: Any, axis: int = 0) -> tuple[int, int]:
    """The bins of an axis that are drawn, from zero and past the last: all, or its range.

    ``TAxis::SetRange`` and ``SetRangeUser`` keep the first and last bin to
    draw, counted from one, as ``fFirst`` and ``fLast``; both zero, or a
    range outside the axis, is the whole of it.
    """
    count = h.axes[axis].nbins
    members = lookup(h, f"f{'XYZ'[axis]}axis") or {}
    first, last = int(lookup(members, "fFirst", 0) or 0), int(lookup(members, "fLast", 0) or 0)
    if 1 <= first <= last <= count:
        return first - 1, last
    return 0, count


def _ends(h: Any, axis: int) -> tuple[float, float]:
    """Where an axis starts and ends, as far as its range reaches."""
    first, last = shown_bins(h, axis)
    edges = h.axes[axis].edges()
    return float(edges[first]), float(edges[last])


def _histogram_extent(h: Histogram, option: str, log: bool) -> Extent:
    """A histogram's frame: its axes' ranges, and in one dimension the bins' heights round them.

    Error bars, drawn or implied by weights, reach the frame too, as
    ``THistPainter`` counts them - but not for a histogram drawn ``HIST``,
    which draws none.
    """
    (xlow, xhigh) = _ends(h, 0)
    if len(h.axes) > 1:
        ylow, yhigh = _ends(h, 1)
        return xlow, ylow, xhigh, yhigh
    first, last = shown_bins(h)
    values, errors = h.values()[first:last], h.errors()[first:last]
    words = histogram_option(option)
    if words & ERRORS or (h.weighted and "HIST" not in words):
        values = np.concatenate([values - errors, values + errors])
    if log:
        low, high = _log_y(values, _limit(h, "fMinimum", -1111), _limit(h, "fMaximum", -1111))
        return xlow, low, xhigh, high
    low, high = _histogram_y(values, log)
    low, high = _crossed(_limit(h, "fMinimum", low), _limit(h, "fMaximum", high))
    return xlow, low, xhigh, high


def _log_y(values: np.ndarray[Any, Any], minimum: float, maximum: float) -> tuple[float, float]:
    """``PaintInit`` on a logarithmic scale: the lowest positive bin and the highest, the
    histogram's own minimum and maximum over them - and room below and above, where it has none.

    A minimum at or above the maximum is a thousandth of it - as a
    ``RooPlot``'s frame has, whose one bin holds the maximum it is given.
    """
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    if not positive.size:
        return _histogram_y(finite, True)
    low = float(positive.min()) if minimum == -1111 else minimum
    high = float(finite.max()) if maximum == -1111 else maximum
    if low >= high and high > 0:
        low = 0.001 * high
    if low <= 0 or high <= 0:
        return _histogram_y(finite, True)
    return _log_room(low, high, minimum == -1111, maximum == -1111)


def _log_room(low: float, high: float, below: bool, above: bool) -> tuple[float, float]:
    """A logarithmic range with room - a factor of two - at the ends the histogram left open."""
    return low * (0.5 if below else 1.0), high * (2.0 if above else 1.0)


def _crossed(low: float, high: float) -> tuple[float, float]:
    """``PaintInit``'s answer to a minimum at or above the maximum - the one a plot given a
    maximum of -1 and a minimum of 0 has: ``[0, 1]``, or doubled away from zero."""
    if low < high:
        return low, high
    if low > 0:
        return 0.0, 2.0 * high
    if low < 0:
        return 2.0 * low, 0.0
    return 0.0, 1.0


def _efficiency_extent(e: Efficiency, pad: Pad) -> Extent:
    """An efficiency's frame: its axes, or its axis and its points' intervals round them."""
    axis = e.axes[0]
    if len(e.axes) > 1:
        return axis.low, e.axes[1].low, axis.high, e.axes[1].high
    low, high = e.intervals()
    y0, y1 = _spread(np.concatenate([low, high]), pad.logy, floor=True)
    return axis.low, y0, axis.high, y1


def _log_spread(values: np.ndarray[Any, Any], high: float) -> tuple[float, float]:
    """A logarithmic axis's range: half its lowest positive value to twice its highest.

    With nothing positive to show it starts at a tenth, which a log scale can draw.
    """
    positive = values[values > 0]
    low = float(positive.min()) if positive.size else 0.1
    return low * 0.5, max(high, low) * 2.0


def _spread(values: np.ndarray[Any, Any], log: bool, floor: bool) -> tuple[float, float]:
    """A tenth of the spread of ``values`` either side, not below zero if none are."""
    if not values.size:
        return 0.0, 1.0
    low, high = float(values.min()), float(values.max())
    if log:
        return _log_spread(values, high)
    if high == low:
        high = low + 1.0
    margin = GRAPH_MARGIN * (high - low)
    bottom, top = low - margin, high + margin
    if bottom < 0 and low >= 0:
        bottom = 0.9 * low  # TGraphPainter keeps a range of positive values positive
    if not floor and top > 0 and high <= 0:
        top = 0.0
    return bottom, top


def _graph_points(graphs: list[Graph]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Every graph's points reached out to the ends of their error bars, which the frame spans."""
    xs, ys = [], []
    for g in graphs:
        xlow, xhigh = g.xerr if g.xerr is not None else (0.0, 0.0)
        ylow, yhigh = g.yerr if g.yerr is not None else (0.0, 0.0)
        xs += [g.x - xlow, g.x + xhigh]
        ys += [g.y - ylow, g.y + yhigh]
    return np.concatenate(xs) if xs else np.zeros(0), np.concatenate(ys) if ys else np.zeros(0)


def _graphs_extent(graphs: list[Graph], pad: Pad) -> Extent:
    """The frame ``TGraphPainter`` makes for graphs: their points, and a tenth more round them.

    An x range of values none above zero ends at zero rather than past it; the
    y range is left its margin.
    """
    xs, ys = _graph_points(graphs)
    x0, x1 = _spread(xs, pad.logx, floor=False)
    y0, y1 = _spread(ys, pad.logy, floor=True)
    return x0, y0, x1, y1


def _limited_ends(h: Histogram) -> tuple[float, float]:
    """Where a frame histogram's x axis starts and ends: its limits as they are now.

    ``TAxis::SetLimits`` moves ``fXmin`` and ``fXmax`` and leaves the bins as
    many; each edge is ``GetBinLowEdge``'s, ``fXmin`` and a whole number of widths.
    """
    members = lookup(h, "fXaxis") or {}
    first, last = shown_bins(h)
    low, high = float(lookup(members, "fXmin", 0.0)), float(lookup(members, "fXmax", 1.0))
    width = (high - low) / h.axes[0].nbins
    return low + first * width, low + last * width


def _graph_extent(g: Graph, pad: Pad) -> Extent:
    """A graph's frame: the histogram it was given to draw it in, or else its points'.

    ``TGraphPainter`` paints the frame histogram a graph already has - one a
    script reached through ``GetXaxis()`` and set limits and a range on - as it
    is, its ``fMinimum`` and ``fMaximum`` its height; and the graph's own
    ``SetMinimum`` and ``SetMaximum`` override either.
    """
    framing = lookup(g, "fHistogram")
    if isinstance(framing, Histogram):
        x0, x1 = _limited_ends(framing)
        yaxis = lookup(framing, "fYaxis") or {}
        y0 = _limit(framing, "fMinimum", float(lookup(yaxis, "fXmin", 0.0)))
        y1 = _limit(framing, "fMaximum", float(lookup(yaxis, "fXmax", 1.0)))
    else:
        x0, y0, x1, y1 = _graphs_extent([g], pad)
    return x0, _limit(g, "fMinimum", y0), x1, _limit(g, "fMaximum", y1)


def _function_extent(f: Function, log: bool) -> Extent:
    """A function's frame: its range, and the heights it reaches, sampled as ``TF1`` draws it."""
    if f.dimensions != 1:
        return 0.0, 0.0, 1.0, 1.0  # it is not drawn, and says so when it is not
    low, high = (float(end) for end in f.range[:2])
    try:
        values = np.asarray(f(np.linspace(low, high, 101)), dtype=float)
    except ROOTError:  # a function that will not draw says so when it is drawn
        values = np.zeros(0)
    y0, y1 = _histogram_y(values, log)
    return low, y0, high, y1


def extent(obj: Any, option: str, pad: Pad) -> Extent:
    """The extent ROOT would draw ``obj``'s frame with: ``xmin, ymin, xmax, ymax``."""
    if isinstance(obj, Histogram):
        return _histogram_extent(obj, option, pad.logy)
    if isinstance(obj, Graph):
        return _graph_extent(obj, pad)
    if isinstance(obj, MultiGraph):
        return _graphs_extent(list(obj), pad)
    if isinstance(obj, Stack) and len(obj):
        first = obj[0].axes[0]
        total = np.sum([h.values() for h in obj], axis=0)
        low, high = _histogram_y(np.asarray(total), pad.logy)
        return first.low, low, first.high, high
    if isinstance(obj, Function):
        return _function_extent(obj, pad.logy)
    if isinstance(obj, Efficiency):
        return _efficiency_extent(obj, pad)
    return 0.0, 0.0, 1.0, 1.0


def open_axes(scene: Scene) -> None:
    """The pad's axes: over its frame, scaled and ranged, or over all of it, bare."""
    pad = scene.pad
    scene.owner = owner(pad)
    x, y, w, h = scene.box
    if scene.owner is None:
        rect = (x, y, w, h)
    else:
        left, right, bottom, top = pad.margins
        rect = (x + left * w, y + bottom * h, w * (1 - left - right), h * (1 - bottom - top))
    ax = scene.figure.add_axes(rect, label=pad.name or "pad")
    ax.patch.set_visible(False)
    scene.ax = ax
    if scene.owner is None:
        x1, y1, x2, y2 = _usable(pad.range, (0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax.set_axis_off()
        return
    if pad.logx:
        ax.set_xscale("log")
    if pad.logy:
        ax.set_yscale("log")
    ax.xaxis.set_visible(False)  # ROOT's axes are painted over the frame, as ``TGaxis`` paints them
    ax.yaxis.set_visible(False)
    worked_out = extent(*scene.owner, pad)
    xmin, ymin, xmax, ymax = _usable(pad.frame, worked_out) if pad.painted else worked_out
    ax.set_xlim(*_positive(xmin, xmax, pad.logx))
    ax.set_ylim(*_positive(ymin, ymax, pad.logy))


def _usable(ends: Extent, otherwise: Extent) -> Extent:
    """``ends``, unless they enclose nothing - a range never set - when ``otherwise``."""
    x1, y1, x2, y2 = ends
    return ends if x1 != x2 and y1 != y2 else otherwise


def _positive(low: float, high: float, log: bool) -> tuple[float, float]:
    """An axis's ends, the lower above zero on a logarithmic one, as ROOT moves it.

    ROOT takes a thousandth of the upper end when the lower is not above
    zero - an axis from zero, drawn logarithmically, starts somewhere.
    """
    if log and low <= 0:
        return high * LOG_FLOOR, high
    return low, high


def _axes_of(obj: Any) -> Any:
    """What holds the axes' styles: a histogram, or the one a graph draws its frame with."""
    if isinstance(obj, Histogram):
        return obj
    framing = lookup(obj, "fHistogram")
    return framing if isinstance(framing, Histogram) else None


#: The frame's attributes, as a ``TFrame`` keeps them and as its pad does.
FRAME_MEMBERS = {
    "fFillStyle": ("fFrameFillStyle", 1001),
    "fFillColor": ("fFrameFillColor", 0),
    "fLineColor": ("fFrameLineColor", 1),
    "fLineWidth": ("fFrameLineWidth", 1),
}


def _frame_style(pad: Pad) -> dict[str, Any]:
    """The frame's fill and line: the ``TFrame`` it was drawn with, or else its pad's."""
    frames = [obj for obj, _option in pad.primitives if getattr(obj, "classname", "") == "TFrame"]
    if frames:
        return {
            name: lookup(frames[0], name, default) for name, (_, default) in FRAME_MEMBERS.items()
        }
    return {name: pad.get(member, default) for name, (member, default) in FRAME_MEMBERS.items()}


def _frame(scene: Scene) -> None:
    """The frame's fill behind the data, and its outline in the frame's line style."""
    from matplotlib.patches import Rectangle

    from .gradient import shade

    style = _frame_style(scene.pad)
    fills, _hatch, alpha = styles.fill(style["fFillStyle"])
    if fills:
        backdrop = Rectangle((0, 0), 1, 1, transform=scene.ax.transAxes, zorder=-50,
                             facecolor=scene.colors.rgba(style["fFillColor"], alpha),
                             edgecolor="none")  # fmt: skip
        scene.ax.add_artist(backdrop)
        shade(scene, backdrop, style["fFillColor"])
    from .raster import add_line, frame_clip

    for spine in scene.ax.spines.values():
        spine.set_visible(False)
    x0, y0, x1, y1 = frame_clip(scene)
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    add_line(scene, corners, scene.colors.rgb(style["fLineColor"]), int(style["fLineWidth"]))




def dress(scene: Scene) -> None:
    """The frame's fill and outline, and its axes painted over them, once everything is drawn.

    A lego or surface plot has neither: its box and axes are its own.
    """
    if scene.owner is None or scene.solid:
        return
    from .dressing import dress_axes

    _frame(scene)
    obj, option = scene.owner
    if not axisless(option, isinstance(obj, (Graph, MultiGraph))):
        dress_axes(scene, _axes_of(obj))


def _painted_title(obj: Any) -> str:
    """The title painted for ``obj``: a graph's frame histogram's, if it has one, else its own."""
    framing = lookup(obj, "fHistogram") if isinstance(obj, (Graph, MultiGraph)) else None
    if isinstance(framing, Histogram) and framing.title:
        return str(framing.title)
    return str(getattr(obj, "title", "") or "")


def default_title(scene: Scene) -> None:
    """The title ``gStyle`` draws for a pad saved without its own ``title`` pave."""
    if scene.owner is None or scene.pad.painted:
        return
    obj = scene.owner[0]
    title = _painted_title(obj)
    if not title or int(lookup(obj, "fBits", 0) or 0) & NO_TITLE:
        return
    from .latex import paint_latex

    attributes = {"font": 42, "size": TITLE_SIZE_PAD, "color": 1, "align": 23, "angle": 0.0}
    paint_latex(scene, title, scene.pixel(TITLE_X, TITLE_Y), attributes)
