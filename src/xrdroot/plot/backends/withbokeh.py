"""Pictures drawn as bokeh glyphs, on a ``bokeh.plotting.figure`` that comes back.

A bokeh figure fixes whether its axes are logarithmic when it is made, so a
figure made here is made with the scales the picture asks for, and one the
caller brings that is made otherwise is refused rather than drawn wrong.
Bokeh has no depth, so ``LEGO``, ``SURF`` and 3-D histograms are refused
with plotly named as the backend that draws them; error bars are drawn as
segments, without ``E1``'s ticks at their ends.
"""

from __future__ import annotations

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

__all__ = ["joined", "panels", "render"]

#: The marker shapes as bokeh's.
MARKERS = {
    "point": "dot", "plus": "cross", "asterisk": "asterisk", "circle": "circle", "x": "x",
    "square": "square", "triangle-up": "triangle", "triangle-down": "inverted_triangle",
    "diamond": "diamond", "cross": "plus", "star": "star",
}  # fmt: skip

#: A ROOT marker of size one, in bokeh's pixels.
PIXELS_PER_SIZE = 7.0


def _figure(frame: Frame, **options: Any) -> Any:
    from bokeh.plotting import figure

    return figure(
        x_axis_type="log" if frame.logx else "linear",
        y_axis_type="log" if frame.logy else "linear",
        **options,
    )


def _palette(name: Any) -> Any:
    """ROOT's palette, or one of bokeh's by name - ``"Viridis256"`` - or a refusal."""
    listed = colors.palette(name)
    if listed is not None:
        return list(listed)
    from bokeh import palettes

    found = getattr(palettes, str(name), None)
    if found is None:
        raise ValueError(
            f"palette={name!r} is neither one of ROOT's (bird, viridis) nor one of bokeh's, "
            f"which are named as bokeh.palettes names them, such as 'Viridis256'"
        )
    return found


def _mapper(name: Any, values: Any, frame: Frame) -> Any:
    from bokeh.models import LinearColorMapper, LogColorMapper

    finite = values[np.isfinite(values) & ((values > 0) if frame.logz else True)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    kind = LogColorMapper if frame.logz else LinearColorMapper
    return kind(palette=_palette(name), low=low, high=high, nan_color="rgba(0,0,0,0)")


def _legend(look: Look) -> dict[str, Any]:
    return {} if look.label is None else {"legend_label": look.label}


def _line(look: Look) -> dict[str, Any]:
    return {"line_color": look.color, "line_width": look.width, "line_dash": look.dash}


def _fill(look: Look) -> dict[str, Any]:
    return {
        "fill_color": look.fill,
        "fill_alpha": look.alpha if look.fill else 0.0,
        "hatch_pattern": look.hatch,
        "hatch_color": look.fill,
    }


def _held(values: Any) -> Any:
    return np.append(values, values[-1:])


# -- one function per kind of layer ---------------------------------------------------------


def _steps(fig: Any, layer: Steps, frame: Frame, native: dict[str, Any]) -> Any:
    look = layer.look
    if look.fill is not None:
        below = np.zeros_like(layer.values) if layer.baseline is None else layer.baseline
        fig.varea_step(
            x=layer.edges, y1=_held(below), y2=_held(layer.values), step_mode="after",
            **_fill(look),
        )  # fmt: skip
    return fig.step(
        x=layer.edges, y=_held(layer.values), mode="after", **_line(look), **_legend(look),
        **native,
    )  # fmt: skip


def _bars(fig: Any, layer: Bars, frame: Frame, native: dict[str, Any]) -> Any:
    look = layer.look
    return fig.quad(
        left=layer.left, right=layer.right, bottom=0, top=layer.values, **_fill(look),
        **_line(look), **_legend(look), **native,
    )  # fmt: skip


def _marker(look: Look) -> dict[str, Any]:
    """Points are always marked: ROOT's default marker is a dot, which is still a marker."""
    shape = look.marker or "point"
    return {
        "marker": MARKERS[shape],
        "size": 3.0 if shape == "point" else PIXELS_PER_SIZE * look.marker_size,
        "line_color": look.marker_color,
        "fill_color": look.marker_color,
        "fill_alpha": 0.0 if look.hollow else 1.0,
    }


def _points(fig: Any, layer: Points, frame: Frame, native: dict[str, Any]) -> Any:
    look = layer.look
    x, y = layer.x, layer.y
    line = {"line_color": look.color, "line_width": look.width}
    if np.any(layer.ylow) or np.any(layer.yhigh):
        fig.segment(x0=x, y0=y - layer.ylow, x1=x, y1=y + layer.yhigh, **line)
    if np.any(layer.xlow) or np.any(layer.xhigh):
        fig.segment(x0=x - layer.xlow, y0=y, x1=x + layer.xhigh, y1=y, **line)
    return fig.scatter(x=x, y=y, **_marker(look), **_legend(look), **native)


def _boxes(fig: Any, layer: Boxes, frame: Frame, native: dict[str, Any]) -> Any:
    look = layer.look
    line = _line(look) if look.fill is None else {"line_color": None}
    return fig.quad(
        left=layer.x0, right=layer.x1, bottom=layer.y0, top=layer.y1, **_fill(look), **line,
        **_legend(look), **native,
    )  # fmt: skip


def _band(fig: Any, layer: Band, frame: Frame, native: dict[str, Any]) -> Any:
    from ..smooth import smoothed

    look = layer.look
    x, low, high = layer.x, layer.low, layer.high
    if layer.smooth:
        (x, low), (_, high) = smoothed(x, low), smoothed(layer.x, high)
    return fig.varea(x=x, y1=low, y2=high, **_fill(look), **_legend(look), **native)


def _curve(fig: Any, layer: Curve, frame: Frame, native: dict[str, Any]) -> Any:
    from ..smooth import smoothed

    x, y = smoothed(layer.x, layer.y) if layer.smooth else (layer.x, layer.y)
    return fig.line(x=x, y=y, **_line(layer.look), **_legend(layer.look), **native)


def _labels(fig: Any, layer: Labels, frame: Frame, native: dict[str, Any]) -> Any:
    return fig.text(
        x=layer.x, y=layer.y, text=list(layer.texts), angle=layer.angle, angle_units="deg",
        text_align="center", text_baseline="bottom", text_color=layer.color, **native,
    )  # fmt: skip


def _scale(fig: Any, mapper: Any, wanted: bool, frame: Frame) -> None:
    if wanted:
        from bokeh.models import ColorBar

        fig.add_layout(ColorBar(color_mapper=mapper, title=frame.zlabel or None), "right")


def _even(edges: Any) -> bool:
    widths = np.diff(edges)
    return bool(np.allclose(widths, widths[0]))


def _mesh(fig: Any, layer: Mesh, frame: Frame, native: dict[str, Any]) -> Any:
    """An image when the cells are all one size, which is quick; a quad per cell when not."""
    mapper = _mapper(layer.palette, layer.values, frame)
    xs, ys = layer.xedges, layer.yedges
    if _even(xs) and _even(ys):
        drawn = fig.image(
            image=[layer.values.T], x=xs[0], y=ys[0], dw=xs[-1] - xs[0], dh=ys[-1] - ys[0],
            color_mapper=mapper, **native,
        )  # fmt: skip
    else:
        drawn = _cells(fig, layer, mapper, native)
    _scale(fig, mapper, layer.scale, frame)
    return drawn


def _cells(fig: Any, layer: Mesh, mapper: Any, native: dict[str, Any]) -> Any:
    from bokeh.models import ColumnDataSource
    from bokeh.transform import transform

    x0, y0 = np.meshgrid(layer.xedges[:-1], layer.yedges[:-1], indexing="ij")
    x1, y1 = np.meshgrid(layer.xedges[1:], layer.yedges[1:], indexing="ij")
    source = ColumnDataSource(
        {"left": x0.ravel(), "right": x1.ravel(), "bottom": y0.ravel(), "top": y1.ravel(),
         "value": layer.values.ravel()}
    )  # fmt: skip
    return fig.quad(
        left="left", right="right", bottom="bottom", top="top", source=source,
        fill_color=transform("value", mapper), line_color=None, **native,
    )  # fmt: skip


def _contour(fig: Any, layer: Contour, frame: Frame, native: dict[str, Any]) -> Any:
    finite = layer.values[np.isfinite(layer.values)]
    levels = np.linspace(float(finite.min()), float(finite.max()), layer.levels + 1)
    shades = _palette(layer.palette)
    chosen = [shades[round(at)] for at in np.linspace(0, len(shades) - 1, len(levels))]
    x, y = np.meshgrid(layer.x, layer.y)
    colouring = (
        {"fill_color": chosen[:-1], "line_color": None}
        if layer.filled
        else {"fill_color": None, "line_color": chosen}
    )
    drawn = fig.contour(x, y, layer.values.T, levels, **colouring, **native)
    if layer.scale:
        fig.add_layout(drawn.construct_color_bar(title=frame.zlabel or None), "right")
    return drawn


def _deep(fig: Any, layer: Any, frame: Frame, native: dict[str, Any]) -> Any:
    raise UnsupportedFeatureError(
        "bokeh draws in two dimensions only: LEGO, SURF and three-dimensional histograms "
        "are drawn with backend='plotly'"
    )


#: Each kind of layer, against what draws it.
DRAWN: dict[type, Callable[[Any, Any, Frame, dict[str, Any]], Any]] = {
    Steps: _steps, Bars: _bars, Points: _points, Boxes: _boxes, Band: _band,
    Curve: _curve, Labels: _labels, Mesh: _mesh, Contour: _contour, Surface: _deep,
    Cloud: _deep,
}  # fmt: skip


# -- the frame --------------------------------------------------------------------------------


def _check_scales(fig: Any, frame: Frame) -> None:
    from bokeh.models import LogScale

    for wanted, scale, name in ((frame.logx, fig.x_scale, "x"), (frame.logy, fig.y_scale, "y")):
        if wanted and not isinstance(scale, LogScale):
            raise ValueError(
                f"a bokeh figure's {name} axis is logarithmic or not from when it is made, "
                f"and this one was made linear: make it with {name}_axis_type='log', or let "
                f"plot() make it"
            )


def _framed(fig: Any, picture: Picture) -> None:
    frame = picture.frame
    if frame.title:
        fig.title.text = frame.title
    if frame.xlabel:
        fig.xaxis.axis_label = frame.xlabel
    if frame.ylabel:
        fig.yaxis.axis_label = frame.ylabel
    if frame.xlim is not None:
        fig.x_range.start, fig.x_range.end = frame.xlim
    if frame.ylim is not None:
        fig.y_range.start, fig.y_range.end = frame.ylim
    fig.xgrid.visible = fig.ygrid.visible = frame.grid
    if frame.legend is False and fig.legend:
        fig.legend.visible = False


def render(picture: Picture, target: Any = None, last: Any = None) -> Any:
    """``picture``'s glyphs drawn on a figure, which comes back."""
    if picture.deep:
        _deep(None, None, picture.frame, {})
    if target is None and picture.same:
        target = last
    fig = _figure(picture.frame) if target is None else target
    _check_scales(fig, picture.frame)
    for index, layer in enumerate(picture.layers):
        DRAWN[type(layer)](fig, layer, picture.frame, picture.native if index == 0 else {})
    _framed(fig, picture)
    return fig


def panels(frame: Frame | None = None) -> tuple[Any, Any, Any]:
    """A ratio plot's two figures, the lower a third the height and sharing the upper's x."""
    frame = frame or Frame()
    upper = _figure(frame, height=360, width=600)
    lower = _figure(frame._replace(logy=False), height=160, width=600, x_range=upper.x_range)
    return upper, lower, None


def joined(upper: Any, lower: Any, whole: Any) -> Any:
    """What a ratio plot gives back in bokeh: a column of its two figures."""
    from bokeh.layouts import column

    return column(upper, lower)
