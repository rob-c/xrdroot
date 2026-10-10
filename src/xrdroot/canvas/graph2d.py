"""A ``TGraph2D`` on a pad, as ``TGraph2DPainter`` paints one.

What a graph of points in space draws is its histogram - the Delaunay
surface sampled on ``fNpx`` by ``fNpy`` bins - for any option a
histogram takes, ``SURF``, ``LEGO``, ``COL`` and the rest. The options of
its own draw in the box of that histogram's bins, empty: ``TRI`` the
triangles, outlined, ``TRI1`` filled by the palette colour of their height
and outlined, ``TRI2`` filled only; ``P`` a marker at each point, ``P0`` a
hollow circle, ``PCOL`` a marker coloured by its height; ``LINE`` a line
through the points in order, ``ERR`` each point's error bars. Drawn
``SAME``, they go in the box of what the pad drew before.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .marks import draw_markers
from .model import lookup
from .raster import add_line
from .scene import Scene

__all__ = ["paint_graph2d"]

#: Option words with a ``P`` in them that are not ``P``: polar, spherical and
#: pseudo-rapidity systems.
NOT_MARKERS = ("pol", "sph", "psr")
#: ``P0``'s marker: a hollow circle.
CIRCLE = 24
#: How a triangle's ``TRI`` digit draws it: outlined, filled and outlined, filled.
OUTLINED, FILLED = {"", "0", "1"}, {"1", "2"}


def _flags(option: str) -> dict[str, Any]:
    """Which of its own things the option asks the graph to draw."""
    opt = option.lower()
    tri = re.search(r"tri(\d?)", opt)
    letters = opt
    for word in NOT_MARKERS:
        letters = letters.replace(word, "")
    return {
        "tri": tri.group(1) if tri else None,
        "markers": "p" in letters,
        "p0": "p0" in letters,
        "pcol": "pcol" in letters,
        "line": "line" in opt,
        "err": "err" in opt or re.search(r"(^|[^a-z])e\d?($|[^a-z])", opt) is not None,
        "same": "same" in opt,
    }


def paint_graph2d(scene: Scene, g: Any, option: str) -> None:
    """A ``TGraph2D``: its histogram, or its own triangles, points, line and bars in a box."""
    from .data import paint_histogram

    flags = _flags(option)
    own = flags["tri"] is not None or flags["markers"] or flags["line"] or flags["err"]
    if not own:
        paint_histogram(scene, g.histogram(), option)
        return
    if not flags["same"] or scene.view3d is None:
        from .legoplot import paint_three_d

        scene.solid = True
        paint_three_d(scene, g.histogram(empty=True), "SURF", cells=False)
    _own(scene, g, flags)


def _own(scene: Scene, g: Any, flags: dict[str, Any]) -> None:
    """The graph's own things, in the box the pad has: triangles, line, bars, markers."""
    view, pad = scene.view3d
    if flags["tri"] is not None:
        _triangles(scene, g, view, pad, flags["tri"])
    if flags["line"]:
        add_line(scene, _pixels(view, pad, g.x, g.y, g.z), _rgb(scene, g, "fLineColor"))
    if flags["err"] and g.errors is not None:
        _bars(scene, g, view, pad)
    if flags["markers"]:
        _markers(scene, g, view, pad, flags)


def _pixels(view: Any, pad: Any, xs: Any, ys: Any, zs: Any) -> np.ndarray[Any, Any]:
    """Points in space as the canvas's pixels, through the pad's view."""
    found = [pad.pixel(*view.to_ndc((float(x), float(y), float(z)))[:2])
             for x, y, z in zip(xs, ys, zs, strict=False)]  # fmt: skip
    return np.asarray(found, dtype=float).reshape(-1, 2)


def _rgb(scene: Scene, g: Any, member: str) -> Any:
    return scene.colors.rgb(lookup(g.members, member, 1))


def _inside(g: Any, view: Any) -> np.ndarray[Any, Any]:
    """Which points are inside the box - what ``TGraph2DPainter`` draws of them."""
    low, high = view.rmin, view.rmax
    columns = (g.x, g.y, g.z)
    keep = np.ones(len(g), dtype=bool)
    for axis, values in enumerate(columns):
        keep &= (values >= low[axis]) & (values <= high[axis])
    return keep


def _markers(scene: Scene, g: Any, view: Any, pad: Any, flags: dict[str, Any]) -> None:
    """A marker at each point: the graph's, a hollow circle for ``P0``, coloured for ``PCOL``."""
    keep = _inside(g, view)
    pixels = _pixels(view, pad, g.x[keep], g.y[keep], g.z[keep])
    style = CIRCLE if flags["p0"] else int(lookup(g.members, "fMarkerStyle", 1))
    size = float(lookup(g.members, "fMarkerSize", 1))
    if not flags["pcol"]:
        draw_markers(scene, pixels, style, size, _rgb(scene, g, "fMarkerColor"))
        return
    colours = _heights(scene, g.z[keep], view)
    for pixel, colour in zip(pixels, colours, strict=False):
        draw_markers(scene, pixel.reshape(1, 2), style, size, colour)


def _heights(scene: Scene, zs: Any, view: Any) -> list[Any]:
    """The palette's colour of each height, over the box's z range."""
    cmap = scene.colors.colormap()
    low, high = view.rmin[2], view.rmax[2]
    span = high - low if high > low else 1.0
    return [cmap(min(max((float(z) - low) / span, 0.0), 1.0)) for z in zs]


def _triangles(scene: Scene, g: Any, view: Any, pad: Any, digit: str) -> None:
    """The Delaunay triangles, farthest first so the nearer cover them: outlined or filled."""
    from matplotlib.patches import Polygon

    tri = g.delaunay.triangles
    if not len(tri):
        return
    corners = [_pixels(view, pad, g.x[t], g.y[t], g.z[t]) for t in tri]
    depth = [sum(view.to_ndc((g.x[i], g.y[i], g.z[i]))[2] for i in t) for t in tri]
    colours = _heights(scene, g.z[tri].mean(axis=1), view)
    outline = _rgb(scene, g, "fLineColor")
    for at in np.argsort(depth):
        if digit in FILLED:
            scene.ax.add_patch(Polygon(corners[at], closed=True, transform=scene.display,
                                       zorder=scene.layer(), facecolor=colours[at],
                                       edgecolor="none", linewidth=0.0))  # fmt: skip
        if digit in OUTLINED:
            add_line(scene, np.vstack([corners[at], corners[at][:1]]), outline)


def _bars(scene: Scene, g: Any, view: Any, pad: Any) -> None:
    """Each point's three error bars, one along each axis, in the line's colour."""
    keep = _inside(g, view)
    ex, ey, ez = (bars[keep] for bars in g.errors)
    x, y, z = g.x[keep], g.y[keep], g.z[keep]
    lines = []
    for at in range(len(x)):
        for dx, dy, dz in ((ex[at], 0, 0), (0, ey[at], 0), (0, 0, ez[at])):
            ends = _pixels(view, pad, [x[at] - dx, x[at] + dx], [y[at] - dy, y[at] + dy],
                           [z[at] - dz, z[at] + dz])  # fmt: skip
            lines.append(ends)
    if lines:
        add_line(scene, lines, _rgb(scene, g, "fLineColor"), many=True)
