"""Pictures drawn onto matplotlib axes, which come back for the caller to keep styling.

Each layer is one matplotlib call - ``stairs`` for steps, ``errorbar`` for
points, ``pcolorfast`` for a shaded grid, which draws a million cells in the
time ``pcolormesh`` takes over a few thousand - with ROOT's look translated
into its keywords. ``LEGO`` and ``SURF`` need axes with depth, and are drawn
on new ones made that way; a 3-D histogram has no matplotlib picture here
and says to draw it with plotly.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from .. import colors
from ..model import (
    Band,
    Bars,
    Boxes,
    Cloud,
    Contour,
    Curve,
    Frame,
    Labels,
    Look,
    Mesh,
    Picture,
    Points,
    Steps,
    Surface,
)
from ..smooth import smoothed

__all__ = ["joined", "panels", "render", "svg"]

#: The marker shapes, in matplotlib's marks; ``(6, 2, 0)`` is a six-armed asterisk.
MARKS: dict[str, Any] = {
    "point": ".", "plus": "+", "asterisk": (6, 2, 0), "circle": "o", "x": "x",
    "square": "s", "triangle-up": "^", "triangle-down": "v", "diamond": "D",
    "cross": "P", "star": "*",
}  # fmt: skip

#: ROOT's dashes, as matplotlib spells them.
DASHES = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}

#: A ROOT marker of size one is eight pixels, which is about six points.
POINTS_PER_SIZE = 6.0


def _pyplot() -> Any:
    from matplotlib import pyplot

    return pyplot


def _fresh(deep: bool) -> Any:
    """New axes, with depth if the picture needs it."""
    if deep:
        return _pyplot().figure().add_subplot(projection="3d")
    return _pyplot().subplots()[1]


def _cmap(name: Any) -> Any:
    """ROOT's palette as a colour map, or the name for matplotlib to look up."""
    listed = colors.palette(name)
    if listed is None:
        return name
    from matplotlib.colors import ListedColormap

    return ListedColormap(listed, name=str(name))


def _norm(frame: Frame) -> Any:
    if not frame.logz:
        return None
    from matplotlib.colors import LogNorm

    return LogNorm()


def _line(look: Look) -> dict[str, Any]:
    made = {
        "color": look.color,
        "linewidth": look.width,
        "linestyle": DASHES[look.dash],
    }
    if look.label is not None:
        made["label"] = look.label
    return made


def _marker(look: Look) -> dict[str, Any]:
    """Points are always marked: ROOT's default marker is a dot, which is still a marker."""
    shape = look.marker or "point"
    size = 2.0 if shape == "point" else POINTS_PER_SIZE * look.marker_size
    return {
        "marker": MARKS[shape],
        "markersize": size,
        "markeredgecolor": look.marker_color,
        "markerfacecolor": "none" if look.hollow else look.marker_color,
    }


# -- one function per kind of layer ---------------------------------------------------------


def _steps(ax: Any, layer: Steps, frame: Frame, native: dict[str, Any]) -> Any:
    look = layer.look
    options: dict[str, Any] = {**_line(look), **native}
    if look.fill is None:
        baseline = 0 if layer.baseline is None else None
        return ax.stairs(layer.values, layer.edges, baseline=baseline, **options)
    options.pop("color")
    baseline = 0 if layer.baseline is None else layer.baseline
    return ax.stairs(
        layer.values,
        layer.edges,
        baseline=baseline,
        fill=True,
        facecolor=look.fill,
        edgecolor=look.color,
        alpha=look.alpha,
        hatch=look.hatch,
        **options,
    )


def _bars(ax: Any, layer: Bars, frame: Frame, native: dict[str, Any]) -> Any:
    look = layer.look
    options = {**_line(look), **native}
    options.pop("color")
    return ax.bar(
        layer.left,
        layer.values,
        width=layer.right - layer.left,
        align="edge",
        facecolor=look.fill or "none",
        edgecolor=look.color,
        hatch=look.hatch,
        **options,
    )


def _either(low: Any, high: Any) -> Any:
    """Bars as matplotlib wants them, or ``None`` when there are none to draw."""
    if not (np.any(low) or np.any(high)):
        return None
    return np.vstack([low, high])


def _points(ax: Any, layer: Points, frame: Frame, native: dict[str, Any]) -> Any:
    look = layer.look
    options = {**_line(look), **_marker(look), **native}
    return ax.errorbar(
        layer.x,
        layer.y,
        xerr=_either(layer.xlow, layer.xhigh),
        yerr=_either(layer.ylow, layer.yhigh),
        capsize=3 if layer.caps else 0,
        **{**options, "linestyle": "none", "elinewidth": look.width},
    )


def _polygons(x0: Any, x1: Any, y0: Any, y1: Any) -> Any:
    return np.stack(
        [np.stack([x0, y0], -1), np.stack([x1, y0], -1), np.stack([x1, y1], -1),
         np.stack([x0, y1], -1)],
        axis=1,
    )  # fmt: skip


def _boxes(ax: Any, layer: Boxes, frame: Frame, native: dict[str, Any]) -> Any:
    from matplotlib.collections import PolyCollection

    look = layer.look
    boxes = PolyCollection(
        _polygons(layer.x0, layer.x1, layer.y0, layer.y1),
        facecolors=look.fill or "none",
        edgecolors=look.color if look.fill is None else "none",
        alpha=look.alpha,
        hatch=look.hatch,
        label=look.label,
        **native,
    )
    ax.add_collection(boxes)
    ax.autoscale_view()
    return boxes


def _band(ax: Any, layer: Band, frame: Frame, native: dict[str, Any]) -> Any:
    look = layer.look
    x, low, high = layer.x, layer.low, layer.high
    if layer.smooth:
        (x, low), (_, high) = smoothed(x, low), smoothed(layer.x, high)
    label = {} if look.label is None else {"label": look.label}
    return ax.fill_between(
        x, low, high, color=look.fill, alpha=look.alpha, hatch=look.hatch, linewidth=0,
        **label, **native,
    )  # fmt: skip


def _curve(ax: Any, layer: Curve, frame: Frame, native: dict[str, Any]) -> Any:
    x, y = smoothed(layer.x, layer.y) if layer.smooth else (layer.x, layer.y)
    return ax.plot(x, y, **{**_line(layer.look), **native})


def _labels(ax: Any, layer: Labels, frame: Frame, native: dict[str, Any]) -> Any:
    return [
        ax.text(x, y, text, rotation=layer.angle, ha="center", va="bottom",
                color=layer.color, **native)
        for x, y, text in zip(layer.x.tolist(), layer.y.tolist(), layer.texts)
    ]  # fmt: skip


def _scale(ax: Any, artist: Any, wanted: bool, frame: Frame) -> Any:
    if wanted:
        ax.figure.colorbar(artist, ax=ax, label=frame.zlabel or None)
    return artist


def _mesh(ax: Any, layer: Mesh, frame: Frame, native: dict[str, Any]) -> Any:
    values = np.ma.masked_invalid(layer.values.T)
    if frame.logz:
        values = np.ma.masked_less_equal(values, 0)
    artist = ax.pcolorfast(
        layer.xedges, layer.yedges, values, cmap=_cmap(layer.palette), norm=_norm(frame),
        **native,
    )  # fmt: skip
    ax.set_xlim(layer.xedges[0], layer.xedges[-1])
    ax.set_ylim(layer.yedges[0], layer.yedges[-1])
    return _scale(ax, artist, layer.scale, frame)


def _contour(ax: Any, layer: Contour, frame: Frame, native: dict[str, Any]) -> Any:
    draw = ax.contourf if layer.filled else ax.contour
    artist = draw(
        layer.x, layer.y, layer.values.T, levels=layer.levels, cmap=_cmap(layer.palette),
        norm=_norm(frame), **native,
    )  # fmt: skip
    return _scale(ax, artist, layer.scale, frame)


def _surface(ax: Any, layer: Surface, frame: Frame, native: dict[str, Any]) -> Any:
    if getattr(ax, "name", "") != "3d":
        raise UnsupportedFeatureError(
            "LEGO and SURF draw in three dimensions, and these axes have two: pass "
            "ax=figure.add_subplot(projection='3d'), or no ax at all"
        )
    cmap = _cmap(layer.palette)
    if layer.lego:
        return _lego(ax, layer, cmap, native)
    xs = (layer.xedges[:-1] + layer.xedges[1:]) / 2
    ys = (layer.yedges[:-1] + layer.yedges[1:]) / 2
    grid_x, grid_y = np.meshgrid(xs, ys)
    artist = ax.plot_surface(grid_x, grid_y, layer.values.T, cmap=cmap, **native)
    return _scale(ax, artist, layer.scale, frame)


def _lego(ax: Any, layer: Surface, cmap: Any, native: dict[str, Any]) -> Any:
    """``LEGO``: a block per bin with anything in it, coloured by its height."""
    from matplotlib import colormaps

    x0, y0 = np.meshgrid(layer.xedges[:-1], layer.yedges[:-1], indexing="ij")
    dx, dy = np.meshgrid(np.diff(layer.xedges), np.diff(layer.yedges), indexing="ij")
    keep = layer.values != 0
    heights = layer.values[keep]
    top = float(np.max(heights)) if heights.size else 1.0
    shades = (colormaps[cmap] if isinstance(cmap, str) else cmap)(heights / (top or 1.0))
    return ax.bar3d(
        x0[keep], y0[keep], np.zeros_like(heights), dx[keep], dy[keep], heights,
        color=shades, **native,
    )  # fmt: skip


def _cloud(ax: Any, layer: Cloud, frame: Frame, native: dict[str, Any]) -> Any:
    raise UnsupportedFeatureError(
        "a three-dimensional histogram has no matplotlib picture here: draw it with "
        "backend='plotly', which shows it as markers or ISO surfaces you can turn round"
    )


#: Each kind of layer, against what draws it.
DRAWN: dict[type, Callable[[Any, Any, Frame, dict[str, Any]], Any]] = {
    Steps: _steps, Bars: _bars, Points: _points, Boxes: _boxes, Band: _band,
    Curve: _curve, Labels: _labels, Mesh: _mesh, Contour: _contour, Surface: _surface,
    Cloud: _cloud,
}  # fmt: skip


# -- the frame --------------------------------------------------------------------------------


def _titles(ax: Any, frame: Frame) -> None:
    for text, setter in (
        (frame.title, "set_title"),
        (frame.xlabel, "set_xlabel"),
        (frame.ylabel, "set_ylabel"),
        (frame.zlabel, "set_zlabel"),
    ):
        if text and hasattr(ax, setter) and not (setter == "set_zlabel" and ax.name != "3d"):
            getattr(ax, setter)(text)  # fmt: skip


def _scales(ax: Any, frame: Frame) -> None:
    if frame.logx:
        ax.set_xscale("log")
    if frame.logy:
        ax.set_yscale("log")
    if frame.xlim is not None:
        ax.set_xlim(*frame.xlim)
    if frame.ylim is not None:
        ax.set_ylim(*frame.ylim)
    if frame.grid:
        ax.grid(True)


def _legend(ax: Any, picture: Picture) -> None:
    labelled = any(getattr(getattr(layer, "look", None), "label", None) for layer in picture.layers)
    if picture.frame.legend or (picture.frame.legend is None and labelled):
        ax.legend()


def draw(ax: Any, picture: Picture) -> Any:
    """Every layer of ``picture`` onto ``ax``, then its frame; ``ax`` comes back."""
    for index, layer in enumerate(picture.layers):
        DRAWN[type(layer)](ax, layer, picture.frame, picture.native if index == 0 else {})
    _titles(ax, picture.frame)
    _scales(ax, picture.frame)
    _legend(ax, picture)
    return ax


def render(picture: Picture, target: Any = None, last: Any = None) -> Any:
    """``picture`` drawn onto ``target``, or ``last`` for ``SAME``, or new axes."""
    if target is None and picture.same:
        target = last
    if target is None:
        target = _fresh(picture.deep)
    return draw(target, picture)


def panels(frame: Frame | None = None) -> tuple[Any, Any, Any]:
    """A ratio plot's two panels, the upper three times the lower, sharing x."""
    figure, (upper, lower) = _pyplot().subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    return upper, lower, figure


def joined(upper: Any, lower: Any, whole: Any) -> Any:
    """What a ratio plot gives back in matplotlib: its two axes, upper first."""
    upper.set_xlabel("")
    return upper, lower


def svg(picture: Picture, width: float = 4.5, height: float = 3.0) -> str:
    """``picture`` as a small SVG, drawn on a figure pyplot never hears of.

    That is what a notebook shows for an object: made without pyplot, it is
    not shown a second time when the cell ends, and leaves no figure open.
    """
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height))
    ax = figure.add_subplot(projection="3d" if picture.deep else None)
    draw(ax, picture)
    figure.tight_layout()
    out = io.StringIO()
    figure.savefig(out, format="svg")
    return out.getvalue()
