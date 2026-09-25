"""The layers of anything drawn as a grid: a 2-D histogram, a ``TF2``, a 2-D efficiency.

``COL`` shades each cell by its value, ``BOX`` draws a box as big as it,
``CONT`` the lines or bands of equal value, ``LEGO`` and ``SURF`` the
values as heights, and ``TEXT`` writes them in; ``Z`` adds the colour scale.
Each is made here from edges and a grid of values, shaped x first, so every
kind of grid goes through the same code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from .binned import number
from .model import Boxes, Contour, Labels, Look, Mesh, Surface
from .request import Request

__all__ = ["Grid", "grid_layers"]


class Grid(NamedTuple):
    """A grid of values, x first, with the edges of its cells."""

    xedges: Any
    yedges: Any
    values: Any
    look: Look


def _centres(edges: Any) -> Any:
    return (edges[:-1] + edges[1:]) / 2


def _painted(values: Any) -> Any:
    """The values with the empty cells made NaN, which ROOT leaves unpainted."""
    return np.where(values != 0, values, np.nan)


def _mesh(grid: Grid, request: Request) -> list[Any]:
    scale = request.chosen.has("Z")
    return [Mesh(grid.xedges, grid.yedges, _painted(grid.values), request.palette, scale)]


def _boxes(grid: Grid, request: Request) -> list[Any]:
    """``BOX``: a box per cell whose sides are the cell's in proportion to its value."""
    top = np.max(np.abs(grid.values)) if grid.values.size else 0.0
    share = np.abs(grid.values) / top if top else np.zeros_like(grid.values)
    xs, ys = np.meshgrid(_centres(grid.xedges), _centres(grid.yedges), indexing="ij")
    wide, high = np.meshgrid(np.diff(grid.xedges), np.diff(grid.yedges), indexing="ij")
    keep = share > 0
    half_x, half_y = (share * wide / 2)[keep], (share * high / 2)[keep]
    x, y = xs[keep], ys[keep]
    return [Boxes(x - half_x, x + half_x, y - half_y, y + half_y, grid.look)]


def _contour(filled: bool) -> Callable[[Grid, Request], list[Any]]:
    def made(grid: Grid, request: Request) -> list[Any]:
        return [
            Contour(
                _centres(grid.xedges),
                _centres(grid.yedges),
                grid.values,
                request.palette,
                request.chosen.has("Z"),
                request.levels,
                filled,
            )
        ]

    return made


def _surface(lego: bool) -> Callable[[Grid, Request], list[Any]]:
    def made(grid: Grid, request: Request) -> list[Any]:
        scale = request.chosen.has("Z")
        return [Surface(grid.xedges, grid.yedges, grid.values, request.palette, scale, lego)]

    return made


def _labels(grid: Grid, request: Request) -> list[Any]:
    xs, ys = np.meshgrid(_centres(grid.xedges), _centres(grid.yedges), indexing="ij")
    keep = grid.values != 0
    texts = tuple(number(float(value)) for value in grid.values[keep])
    return [Labels(xs[keep], ys[keep], texts, request.chosen.angle, grid.look.marker_color)]


#: Each word and the layers it makes, in the order they are drawn.
ORDER = (
    ("COL", _mesh),
    ("BOX", _boxes),
    ("CONT", _contour(True)),
    ("CONTL", _contour(False)),
    ("LEGO", _surface(True)),
    ("SURF", _surface(False)),
    ("TEXT", _labels),
)


def grid_layers(grid: Grid, request: Request) -> list[Any]:
    """Every layer the option asks for, bottom first."""
    return [
        layer for word, make in ORDER if request.chosen.has(word) for layer in make(grid, request)
    ]
