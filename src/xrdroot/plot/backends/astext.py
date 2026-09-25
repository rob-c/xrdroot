"""Pictures drawn in characters, for a terminal, a log file or a CI transcript.

Nothing needs to be installed. Binned layers are a line per bin - its edges,
a bar and the value; points, lines and bands are a grid of stars with the
ends of the axes written beside it; anything drawn as a grid of colours is
a grid of shades with y upward. The numbers of ``TEXT`` are in the lines
already, and a 3-D histogram has no flat picture in characters either.

What comes back is a string. Given one as its target - or, with ``SAME``,
the last one it made - the new picture is written after it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ...draw import bar, shade
from ...errors import UnsupportedFeatureError
from ..model import (
    Band,
    Bars,
    Boxes,
    Cloud,
    Contour,
    Curve,
    Frame,
    Labels,
    Mesh,
    Picture,
    Points,
    Steps,
    Surface,
)

__all__ = ["joined", "panels", "render"]

#: How wide a bar may grow, and how big a grid of stars is.
WIDTH = 50
HEIGHT = 12


def _binned(lows: Any, highs: Any, values: Any) -> str:
    """A line per bin - or per bar - with its ends, a bar as long as its value, and the value."""
    top = max((float(value) for value in values if value > 0), default=0.0)
    return "\n".join(
        f"[{low:g}, {high:g})".rjust(24) + f" {bar(value / top if top else 0.0, WIDTH):<{WIDTH}} "
        f"{value:g}"
        for low, high, value in zip(lows.tolist(), highs.tolist(), values.tolist())
    )


def _steps(layer: Steps) -> str:
    edges = np.asarray(layer.edges)
    return _binned(edges[:-1], edges[1:], np.asarray(layer.values))


def _bars(layer: Bars) -> str:
    return _binned(np.asarray(layer.left), np.asarray(layer.right), np.asarray(layer.values))


def _end(index: int, low: float, high: float) -> str:
    """The value written beside the top row and the bottom one."""
    if index == 0:
        return f"{high:g}"
    if index == HEIGHT - 1:
        return f"{low:g}"
    return ""


def stars(x: Any, y: Any) -> str:
    """Points as a grid of stars, the ends of each axis written beside it."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if not len(x):
        return "(no points)"
    xlo, xhi, ylo, yhi = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    grid = [[" "] * WIDTH for _ in range(HEIGHT)]
    columns = np.round((x - xlo) / ((xhi - xlo) or 1.0) * (WIDTH - 1)).astype(int)
    rows = np.round((y - ylo) / ((yhi - ylo) or 1.0) * (HEIGHT - 1)).astype(int)
    for column, row in zip(columns.tolist(), rows.tolist()):
        grid[HEIGHT - 1 - row][column] = "*"
    lines = [f"{_end(index, ylo, yhi):>10} |{''.join(cells)}|" for index, cells in enumerate(grid)]
    left, right = f"{xlo:g}", f"{xhi:g}"
    lines.append(f"{'':>10}  {left:<{WIDTH - len(right)}}{right}")
    return "\n".join(lines)


def shades(values: Any) -> str:
    """A grid, x across and y upward, each cell shaded by how full it is."""
    cells = np.nan_to_num(np.asarray(values, dtype=np.float64))
    top = float(cells.max()) if cells.size else 0.0
    share = cells / top if top > 0 else np.zeros_like(cells)
    return "\n".join(
        "".join(shade(float(value)) for value in share[:, row])
        for row in reversed(range(share.shape[1]))
    )


def _deep(layer: Any) -> str:
    raise UnsupportedFeatureError(
        "a three-dimensional histogram has no honest flat picture in characters; take "
        ".values() and slice it down to the two dimensions you want to see"
    )


#: Each kind of layer, against what writes it; ``TEXT``'s numbers are in the lines already.
WRITTEN: dict[type, Callable[[Any], str]] = {
    Steps: _steps,
    Bars: _bars,
    Points: lambda layer: stars(layer.x, layer.y),
    Curve: lambda layer: stars(layer.x, layer.y),
    Band: lambda layer: stars(layer.x, (layer.low + layer.high) / 2),
    Boxes: lambda layer: stars((layer.x0 + layer.x1) / 2, (layer.y0 + layer.y1) / 2),
    Labels: lambda layer: "",
    Mesh: lambda layer: shades(layer.values),
    Contour: lambda layer: shades(layer.values),
    Surface: lambda layer: shades(layer.values),
    Cloud: _deep,
}


def _heading(frame: Frame) -> list[str]:
    axes = " against ".join(label for label in (frame.ylabel, frame.xlabel) if label)
    return [line for line in (frame.title, axes) if line]


def render(picture: Picture, target: Any = None, last: Any = None) -> str:
    """``picture`` in characters, after ``target`` - or the last picture, for ``SAME``."""
    if target is None and picture.same:
        target = last
    blocks = [WRITTEN[type(layer)](layer) for layer in picture.layers]
    text = "\n\n".join(["\n".join(_heading(picture.frame))] + [b for b in blocks if b]).strip("\n")
    return f"{target}\n\n{text}" if target else text


def panels(frame: Frame | None = None) -> tuple[Any, Any, Any]:
    """A ratio plot in characters is its two pictures, one after the other."""
    return None, None, None


def joined(upper: Any, lower: Any, whole: Any) -> str:
    return f"{upper}\n\n{lower}"
