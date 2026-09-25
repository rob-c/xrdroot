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

from ..errors import ROOTError
from ..function import Function
from ..graph import Graph
from ..hist import Histogram
from ..stacks import MultiGraph, Stack
from . import styles
from .model import Pad, lookup
from .options import histogram_option
from .scene import Scene

__all__ = ["open_axes", "dress"]

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

Extent = tuple[float, float, float, float]


def owner(pad: Pad) -> tuple[Any, str] | None:
    """What draws a pad's frame: the first histogram not drawn ``SAME``, or graph drawn ``A``."""
    for obj, option in pad.primitives:
        upper = option.upper()
        if isinstance(obj, (Graph, MultiGraph)):
            if "A" in upper.replace("SAME", ""):
                return obj, option
        elif isinstance(obj, (Histogram, Stack, Function)) and "SAME" not in upper:
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
    value = lookup(obj, name)
    return fallback if value is None or float(value) == -1111 else float(value)


def _histogram_extent(h: Histogram, option: str, log: bool) -> Extent:
    axis = h.axes[0]
    if len(h.axes) > 1:
        return axis.low, h.axes[1].low, axis.high, h.axes[1].high
    values = h.values()
    if histogram_option(option) & {"E", "E0", "E1", "E2", "E3", "E4"} or h.weighted:
        values = np.concatenate([values - h.errors(), values + h.errors()])
    low, high = _histogram_y(values, log)
    return axis.low, _limit(h, "fMinimum", low), axis.high, _limit(h, "fMaximum", high)


def _spread(values: np.ndarray[Any, Any], log: bool, floor: bool) -> tuple[float, float]:
    """A tenth of the spread of ``values`` either side, not below zero if none are."""
    if not values.size:
        return 0.0, 1.0
    low, high = float(values.min()), float(values.max())
    if log:
        positive = values[values > 0]
        low = float(positive.min()) if positive.size else 0.1
        return low * 0.5, max(high, low) * 2.0
    margin = GRAPH_MARGIN * ((high - low) or abs(high) or 1.0)
    bottom = low - margin
    if floor and low >= 0 and bottom < 0:
        bottom = 0.0
    return bottom, high + margin


def _graph_points(graphs: list[Graph]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    xs, ys = [], []
    for g in graphs:
        xlow, xhigh = g.xerr if g.xerr is not None else (0.0, 0.0)
        ylow, yhigh = g.yerr if g.yerr is not None else (0.0, 0.0)
        xs += [g.x - xlow, g.x + xhigh]
        ys += [g.y - ylow, g.y + yhigh]
    return np.concatenate(xs) if xs else np.zeros(0), np.concatenate(ys) if ys else np.zeros(0)


def _graphs_extent(graphs: list[Graph], pad: Pad) -> Extent:
    xs, ys = _graph_points(graphs)
    x0, x1 = _spread(xs, pad.logx, floor=False)
    y0, y1 = _spread(ys, pad.logy, floor=True)
    return x0, y0, x1, y1


def _function_extent(f: Function, log: bool) -> Extent:
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
        return _graphs_extent([obj], pad)
    if isinstance(obj, MultiGraph):
        return _graphs_extent(list(obj), pad)
    if isinstance(obj, Stack) and len(obj):
        first = obj[0].axes[0]
        total = np.sum([h.values() for h in obj], axis=0)
        low, high = _histogram_y(np.asarray(total), pad.logy)
        return first.low, low, first.high, high
    if isinstance(obj, Function):
        return _function_extent(obj, pad.logy)
    return 0.0, 0.0, 1.0, 1.0


def _logarithmic(axis: Any) -> None:
    """An axis scaled logarithmically, labelled with plain numbers as ROOT labels one."""
    from matplotlib.ticker import LogFormatter

    axis.set_major_formatter(LogFormatter())
    axis.set_minor_formatter(LogFormatter(minor_thresholds=(1, 0.4)))


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
        x1, y1, x2, y2 = pad.range
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax.set_axis_off()
        return
    for scale, axis, log in (
        (ax.set_xscale, ax.xaxis, pad.logx),
        (ax.set_yscale, ax.yaxis, pad.logy),
    ):
        if log:
            scale("log")
            _logarithmic(axis)
    xmin, ymin, xmax, ymax = pad.frame if pad.painted else extent(*scene.owner, pad)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


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

    style = _frame_style(scene.pad)
    fills, _hatch, alpha = styles.fill(style["fFillStyle"])
    if fills:
        scene.ax.add_artist(
            Rectangle(
                (0, 0), 1, 1, transform=scene.ax.transAxes, zorder=-50,
                facecolor=scene.colors.rgba(style["fFillColor"], alpha), edgecolor="none",
            )
        )  # fmt: skip
    for spine in scene.ax.spines.values():
        spine.set_color(scene.colors.rgb(style["fLineColor"]))
        spine.set_linewidth(styles.points(float(style["fLineWidth"])))


def _label(scene: Scene, axis: Any, which: str) -> None:
    """One axis's title, at its far end, and its tick labels, sized as ``TAttAxis`` says."""
    title = str(lookup(axis, "fTitle", "") or "")
    label_size = scene.text_points(
        lookup(axis, "fLabelSize", LABEL_SIZE), lookup(axis, "fLabelFont", 42)
    )
    title_size = scene.text_points(
        lookup(axis, "fTitleSize", TITLE_SIZE), lookup(axis, "fTitleFont", 42)
    )
    color = scene.colors.rgb(lookup(axis, "fLabelColor", 1))
    scene.ax.tick_params(axis=which, labelsize=label_size, labelcolor=color)
    from .latex import translate

    setter = scene.ax.set_xlabel if which == "x" else scene.ax.set_ylabel
    where = {"loc": "right"} if which == "x" else {"loc": "top"}
    setter(
        translate(title), fontsize=title_size,
        color=scene.colors.rgb(lookup(axis, "fTitleColor", 1)), **where,
    )  # fmt: skip


def _ticks(scene: Scene, source: Any) -> None:
    """Ticks inside the frame, on the far sides too when the pad asks for them."""
    tickx, ticky = scene.pad.ticks
    frame_w = scene.ax.get_position().width * scene.figure.get_figwidth() * styles.DPI
    frame_h = scene.ax.get_position().height * scene.figure.get_figheight() * styles.DPI
    xaxis, yaxis = lookup(source, "fXaxis"), lookup(source, "fYaxis")
    xlength = styles.points(float(lookup(xaxis, "fTickLength", TICK_LENGTH)) * frame_h)
    ylength = styles.points(float(lookup(yaxis, "fTickLength", TICK_LENGTH)) * frame_w)
    scene.ax.minorticks_on()
    scene.ax.tick_params(axis="x", which="major", direction="in", length=xlength, top=bool(tickx))
    scene.ax.tick_params(
        axis="x", which="minor", direction="in", length=xlength / 2, top=bool(tickx)
    )
    scene.ax.tick_params(axis="y", which="major", direction="in", length=ylength, right=bool(ticky))
    scene.ax.tick_params(
        axis="y", which="minor", direction="in", length=ylength / 2, right=bool(ticky)
    )


def dress(scene: Scene) -> None:
    """The frame's fill, outline, ticks, grid and axis titles, once everything is drawn."""
    if scene.owner is None:
        return
    source = _axes_of(scene.owner[0])
    _frame(scene)
    _ticks(scene, source)
    gridx, gridy = scene.pad.grid
    if gridx:
        scene.ax.grid(True, axis="x", which="major", linestyle=":", color="black", linewidth=0.5)
    if gridy:
        scene.ax.grid(True, axis="y", which="major", linestyle=":", color="black", linewidth=0.5)
    _label(scene, lookup(source, "fXaxis") if source is not None else None, "x")
    _label(scene, lookup(source, "fYaxis") if source is not None else None, "y")


def default_title(scene: Scene) -> None:
    """The title ``gStyle`` draws for a pad saved without its own ``title`` pave."""
    if scene.owner is None or scene.pad.painted:
        return
    obj = scene.owner[0]
    title = str(getattr(obj, "title", "") or "")
    if not title or int(lookup(obj, "fBits", 0) or 0) & NO_TITLE:
        return
    from .latex import translate
    from .shapes import draw_text

    style = scene.text(None, None, scene.text_points(TITLE_SIZE_PAD))
    style["ha"], style["va"] = "center", "top"
    draw_text(scene, translate(title), TITLE_X, TITLE_Y, style, ndc=True)
