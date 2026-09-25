"""The layers of anything drawn as points: a graph, a multigraph's graphs, an efficiency.

``TGraphPainter`` reads ``P`` as markers, ``L`` a line through the points,
``C`` a smooth one, ``B`` a bar at each; ``2`` draws each point's errors as
a box, ``3`` as a band through their ends and ``4`` a smoothed band. Error
bars come with the markers unless ``X`` says not, with ticks on their ends
unless ``Z`` does. A graph keeping its errors in layers draws every layer.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from .model import Band, Bars, Boxes, Curve, Look, Points
from .request import Request

__all__ = ["Scatter", "scatter_layers"]


class Scatter(NamedTuple):
    """Points, the bars either side of them in x, and one or more layers in y."""

    x: Any
    y: Any
    xlow: Any
    xhigh: Any
    #: ``(below, above)`` pairs, one per layer; none for points without y errors.
    layers: tuple[tuple[Any, Any], ...]
    look: Look


def _first(points: Scatter) -> tuple[Any, Any]:
    zeros = np.zeros_like(points.x)
    return points.layers[0] if points.layers else (zeros, zeros)


def _shaded(look: Look) -> Look:
    """What a box or band of errors is filled with: the fill, or the line faintly."""
    if look.fill:
        return look
    return look._replace(fill=look.color, alpha=0.35)


def _bars(points: Scatter, request: Request) -> list[Any]:
    if not request.chosen.has("B"):
        return []
    spacing = np.diff(np.sort(points.x))
    half = 0.4 * (float(spacing[spacing > 0].min()) if np.any(spacing > 0) else 1.0)
    return [Bars(points.x - half, points.x + half, points.y, points.look)]


def _boxes(points: Scatter, request: Request) -> list[Any]:
    if not request.chosen.has("2"):
        return []
    below, above = _first(points)
    return [
        Boxes(
            points.x - points.xlow,
            points.x + points.xhigh,
            points.y - below,
            points.y + above,
            _shaded(points.look),
        )
    ]


def _band(points: Scatter, request: Request) -> list[Any]:
    chosen = request.chosen
    if not chosen.has("3", "4"):
        return []
    below, above = _first(points)
    order = np.argsort(points.x, kind="stable")
    return [
        Band(
            points.x[order],
            (points.y - below)[order],
            (points.y + above)[order],
            _shaded(points.look),
            smooth=chosen.has("4"),
        )
    ]


def _line(points: Scatter, request: Request) -> list[Any]:
    chosen = request.chosen
    if not chosen.has("L", "C"):
        return []
    return [Curve(points.x, points.y, points.look, smooth=chosen.has("C"))]


def _markers(points: Scatter, request: Request) -> list[Any]:
    """The points, with a set of bars per layer unless ``X`` asks for none."""
    chosen = request.chosen
    if not chosen.has("P", "*"):
        return []
    zeros = np.zeros_like(points.x)
    layers = points.layers if points.layers and not chosen.has("X") else ((zeros, zeros),)
    xlow, xhigh = (zeros, zeros) if chosen.has("X") else (points.xlow, points.xhigh)
    caps = not chosen.has("Z")
    made = []
    for index, (below, above) in enumerate(layers):
        across = (xlow, xhigh) if index == 0 else (zeros, zeros)
        look = points.look if index == 0 else points.look._replace(label=None)
        made.append(Points(points.x, points.y, *across, below, above, look, caps=caps))
    return made


#: ``TGraphPainter``'s order: bars, boxes and bands underneath, then lines, then markers.
ORDER = (_bars, _boxes, _band, _line, _markers)


def scatter_layers(points: Scatter, request: Request) -> list[Any]:
    """Every layer the option asks for, bottom first."""
    return [layer for make in ORDER for layer in make(points, request)]
