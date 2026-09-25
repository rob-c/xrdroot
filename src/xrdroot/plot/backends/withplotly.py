"""Pictures drawn as plotly traces, on a ``plotly.graph_objects.Figure`` that comes back.

Plotly draws what matplotlib cannot turn round: ``LEGO`` and ``SURF`` are
surfaces you can rotate, and a 3-D histogram is a cloud of markers or, with
``ISO``, the surfaces of equal content. Everything else is the same picture
as matplotlib's, trace for layer. A target may also be ``(figure, row)``, a
row of a figure made with ``make_subplots``, which is how the two panels of
a ratio plot are drawn.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

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

__all__ = ["html", "joined", "panels", "render"]

#: The marker shapes as plotly's symbols; the line-only ones exist only open.
SYMBOLS = {
    "point": "circle", "plus": "cross-thin-open", "asterisk": "asterisk-open",
    "circle": "circle", "x": "x-thin-open", "square": "square", "triangle-up": "triangle-up",
    "triangle-down": "triangle-down", "diamond": "diamond", "cross": "cross", "star": "star",
}  # fmt: skip

#: ROOT's dashes, as plotly spells them.
DASHES = {"solid": "solid", "dashed": "dash", "dotted": "dot", "dashdot": "dashdot"}

#: A ROOT marker of size one, in plotly's pixels.
PIXELS_PER_SIZE = 7.0


def _go() -> Any:
    import plotly.graph_objects as go

    return go


def _colorscale(name: Any) -> Any:
    """ROOT's palette as a plotly colour scale, or the name for plotly to look up."""
    listed = colors.palette(name)
    if listed is None:
        return name
    last = len(listed) - 1
    return [[index / last, shade] for index, shade in enumerate(listed)]


def _named(look: Look) -> dict[str, Any]:
    if look.label is None:
        return {"showlegend": False}
    return {"name": look.label, "showlegend": True}


def _line(look: Look) -> dict[str, Any]:
    return {"color": look.color, "width": look.width, "dash": DASHES[look.dash]}


def _marker(look: Look) -> dict[str, Any]:
    """Points are always marked: ROOT's default marker is a dot, which is still a marker."""
    shape = look.marker or "point"
    symbol = SYMBOLS[shape]
    if look.hollow and not symbol.endswith("-open"):
        symbol += "-open"
    size = 3.0 if shape == "point" else PIXELS_PER_SIZE * look.marker_size
    return {"symbol": symbol, "size": size, "color": look.marker_color}


def _held(values: Any) -> Any:
    """A bin's value again at its high edge, so a step line reaches the last edge."""
    return np.append(values, values[-1:])


# -- one function per kind of layer ---------------------------------------------------------


def _steps(layer: Steps, frame: Frame) -> list[Any]:
    go, look = _go(), layer.look
    line = {"x": layer.edges, "line_shape": "hv", "mode": "lines"}
    if look.fill is None:
        return [go.Scatter(y=_held(layer.values), line=_line(look), **line, **_named(look))]
    below = []
    fill = "tozeroy"
    if layer.baseline is not None:
        below = [go.Scatter(y=_held(layer.baseline), line={"width": 0}, showlegend=False, **line)]
        fill = "tonexty"
    top = go.Scatter(
        y=_held(layer.values), line=_line(look), fill=fill, fillcolor=look.fill,
        opacity=look.alpha, fillpattern={"shape": look.hatch or ""}, **line, **_named(look),
    )  # fmt: skip
    return [*below, top]


def _bars(layer: Bars, frame: Frame) -> list[Any]:
    look = layer.look
    return [
        _go().Bar(
            x=(layer.left + layer.right) / 2, y=layer.values, width=layer.right - layer.left,
            marker={"color": look.fill or "rgba(0,0,0,0)", "line": {"color": look.color,
                    "width": look.width}}, **_named(look),
        )
    ]  # fmt: skip


def _bars_of(low: Any, high: Any, caps: bool, look: Look) -> dict[str, Any]:
    return {
        "type": "data", "symmetric": False, "array": high, "arrayminus": low,
        "visible": bool(np.any(low) or np.any(high)), "width": 4 if caps else 0,
        "color": look.color, "thickness": look.width,
    }  # fmt: skip


def _points(layer: Points, frame: Frame) -> list[Any]:
    look = layer.look
    return [
        _go().Scatter(
            x=layer.x, y=layer.y, mode="markers", marker=_marker(look),
            error_x=_bars_of(layer.xlow, layer.xhigh, layer.caps, look),
            error_y=_bars_of(layer.ylow, layer.yhigh, layer.caps, look), **_named(look),
        )
    ]  # fmt: skip


def _outlines(x0: Any, x1: Any, y0: Any, y1: Any) -> tuple[Any, Any]:
    """Every rectangle as one closed path, the paths apart by gaps: one trace for them all."""
    gap = np.full_like(x0, np.nan)
    xs = np.stack([x0, x1, x1, x0, x0, gap], -1).ravel()
    ys = np.stack([y0, y0, y1, y1, y0, gap], -1).ravel()
    return xs, ys


def _boxes(layer: Boxes, frame: Frame) -> list[Any]:
    look = layer.look
    xs, ys = _outlines(layer.x0, layer.x1, layer.y0, layer.y1)
    filled = {"fill": "toself", "fillcolor": look.fill} if look.fill else {}
    line = {"width": 0} if look.fill else _line(look)
    return [
        _go().Scatter(x=xs, y=ys, mode="lines", line=line, opacity=look.alpha, **filled,
                      **_named(look))
    ]  # fmt: skip


def _band(layer: Band, frame: Frame) -> list[Any]:
    go, look = _go(), layer.look
    shape = "spline" if layer.smooth else "linear"
    edge = {"x": layer.x, "mode": "lines", "line": {"width": 0, "shape": shape}}
    return [
        go.Scatter(y=layer.low, showlegend=False, **edge),
        go.Scatter(y=layer.high, fill="tonexty", fillcolor=look.fill, opacity=look.alpha,
                   **edge, **_named(look)),
    ]  # fmt: skip


def _curve(layer: Curve, frame: Frame) -> list[Any]:
    look = layer.look
    line = {**_line(look), "shape": "spline" if layer.smooth else "linear"}
    return [_go().Scatter(x=layer.x, y=layer.y, mode="lines", line=line, **_named(look))]


def _labels(layer: Labels, frame: Frame) -> list[Any]:
    return [
        _go().Scatter(
            x=layer.x, y=layer.y, text=list(layer.texts), mode="text",
            textposition="top center", textfont={"color": layer.color}, showlegend=False,
        )
    ]  # fmt: skip


def _shaded(values: Any, frame: Frame) -> tuple[Any, dict[str, Any]]:
    """The values to colour by: their logarithm for ``logz``, which plotly has no scale for."""
    if not frame.logz:
        return values, {"title": {"text": frame.zlabel}}
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.where(values > 0, np.log10(values), np.nan)
    return logged, {"title": {"text": f"log10 {frame.zlabel}".strip()}}


def _mesh(layer: Mesh, frame: Frame) -> list[Any]:
    values, bar = _shaded(layer.values.T, frame)
    return [
        _go().Heatmap(
            x=layer.xedges, y=layer.yedges, z=values, colorscale=_colorscale(layer.palette),
            showscale=layer.scale, colorbar=bar, zsmooth=False,
        )
    ]  # fmt: skip


def _contour(layer: Contour, frame: Frame) -> list[Any]:
    values, bar = _shaded(layer.values.T, frame)
    return [
        _go().Contour(
            x=layer.x, y=layer.y, z=values, colorscale=_colorscale(layer.palette),
            showscale=layer.scale, colorbar=bar, ncontours=layer.levels,
            contours={"coloring": "fill" if layer.filled else "lines"},
        )
    ]  # fmt: skip


def _blocks(edges: Any) -> Any:
    """Each edge twice, so a surface over them rises in steps: ``LEGO``'s blocks."""
    return np.repeat(edges, 2)[1:-1]


def _surface(layer: Surface, frame: Frame) -> list[Any]:
    values, bar = _shaded(layer.values.T, frame)
    if layer.lego:
        x, y = _blocks(layer.xedges), _blocks(layer.yedges)
        values = np.repeat(np.repeat(values, 2, axis=0), 2, axis=1)
    else:
        x = (layer.xedges[:-1] + layer.xedges[1:]) / 2
        y = (layer.yedges[:-1] + layer.yedges[1:]) / 2
    return [
        _go().Surface(x=x, y=y, z=values, colorscale=_colorscale(layer.palette),
                      showscale=layer.scale, colorbar=bar)
    ]  # fmt: skip


def _cloud(layer: Cloud, frame: Frame) -> list[Any]:
    go = _go()
    scale = _colorscale(layer.palette)
    if layer.iso:
        return [go.Isosurface(x=layer.x, y=layer.y, z=layer.z, value=layer.values,
                              colorscale=scale)]  # fmt: skip
    top = float(np.max(np.abs(layer.values))) if layer.values.size else 1.0
    sizes = 2 + 10 * np.abs(layer.values) / (top or 1.0)
    return [
        go.Scatter3d(x=layer.x, y=layer.y, z=layer.z, mode="markers",
                     marker={"size": sizes, "color": layer.values, "colorscale": scale})
    ]  # fmt: skip


#: Each kind of layer, against what makes its traces.
TRACES: dict[type, Callable[[Any, Frame], list[Any]]] = {
    Steps: _steps, Bars: _bars, Points: _points, Boxes: _boxes, Band: _band,
    Curve: _curve, Labels: _labels, Mesh: _mesh, Contour: _contour, Surface: _surface,
    Cloud: _cloud,
}  # fmt: skip


# -- the frame --------------------------------------------------------------------------------


def _given(**settings: Any) -> dict[str, Any]:
    """Only the settings made, so drawing ``SAME`` leaves what an earlier call set."""
    return {name: value for name, value in settings.items() if value is not None}


def _range(limits: Any, log: bool) -> list[float] | None:
    """An axis's limits as plotly takes them: in decades, on a log axis."""
    if limits is None:
        return None
    return [float(np.log10(end)) if log else float(end) for end in limits]


def _axes(figure: Any, frame: Frame, row: int | None) -> None:
    where = {} if row is None else {"row": row, "col": 1}
    for update, label, log, limits in (
        (figure.update_xaxes, frame.xlabel, frame.logx, frame.xlim),
        (figure.update_yaxes, frame.ylabel, frame.logy, frame.ylim),
    ):
        update(
            showgrid=frame.grid,
            **_given(title_text=label or None, type="log" if log else None,
                     range=_range(limits, log)),
            **where,
        )  # fmt: skip


def _scene(figure: Any, frame: Frame) -> None:
    figure.update_layout(
        scene={"xaxis_title": frame.xlabel, "yaxis_title": frame.ylabel,
               "zaxis_title": frame.zlabel}
    )  # fmt: skip


def _framed(figure: Any, picture: Picture, row: int | None) -> None:
    frame = picture.frame
    if frame.title and row in (None, 1):
        figure.update_layout(title_text=frame.title)
    if picture.deep:
        _scene(figure, frame)
    else:
        _axes(figure, frame, row)
    if frame.legend is not None:
        figure.update_layout(showlegend=frame.legend)


def _target(target: Any, last: Any, same: bool) -> tuple[Any, int | None]:
    """The figure and row to draw on: given, the last for ``SAME``, or a new figure."""
    if target is None and same:
        target = last
    if target is None:
        target = _go().Figure(layout={"template": "plotly_white"})
    if isinstance(target, tuple):
        return target[0], int(target[1])
    return target, None


def render(picture: Picture, target: Any = None, last: Any = None) -> Any:
    """``picture``'s traces added to a figure, which comes back."""
    figure, row = _target(target, last, picture.same)
    where = {} if row is None else {"row": row, "col": 1}
    for index, layer in enumerate(picture.layers):
        for trace in TRACES[type(layer)](layer, picture.frame):
            if index == 0:
                trace.update(picture.native)
            figure.add_trace(trace, **where)
    _framed(figure, picture, row)
    return figure


def panels(frame: Frame | None = None) -> tuple[Any, Any, Any]:
    """A ratio plot's figure of two rows sharing x, the upper three times the lower."""
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03
    )
    figure.update_layout(template="plotly_white")
    return (figure, 1), (figure, 2), figure


def joined(upper: Any, lower: Any, whole: Any) -> Any:
    """What a ratio plot gives back in plotly: the one figure holding both rows."""
    return whole


def html(picture: Picture) -> str:
    """``picture`` as a piece of HTML, with plotly's script fetched from its CDN."""
    figure = render(picture)
    figure.update_layout(width=480, height=320, margin={"l": 40, "r": 20, "t": 40, "b": 40})
    return str(figure.to_html(full_html=False, include_plotlyjs="cdn"))
