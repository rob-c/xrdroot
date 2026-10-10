"""A ``TScatter`` or ``TScatter2D`` on a pad: its markers, each in its colour and size.

In a plane the points are the picture :mod:`xrdroot.plot` makes, painted as
any data is, with the colour scale's bar beside the frame as ``TScatter``
draws a ``TPaletteAxis``. In space they stand in the box a ``TGraph2D``
would be drawn in, each marker coloured and sized as in the plane.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..plot.colors import PALETTES, palette
from .graph2d import _pixels
from .marks import draw_markers
from .scene import Scene

__all__ = ["paint_scatterplot", "colormap", "shades"]


def colormap(given: Any) -> Any:
    """A matplotlib colour map of a palette's name, or the map it already is."""
    from matplotlib.colors import ListedColormap

    if callable(given):
        return given
    return ListedColormap(list(palette(given) or PALETTES["bird"]), str(given))


def shades(fractions: Any, given: Any) -> tuple[str, ...]:
    """Each point's colour from the palette, where its fraction lies along it, as hex."""
    from matplotlib.colors import to_hex

    cmap = colormap(given)
    return tuple(to_hex(cmap(float(t)), keep_alpha=True) for t in np.asarray(fractions))


def paint_scatterplot(scene: Scene, obj: Any, option: str) -> None:
    """A scatter plot: in a plane as its picture, in space in a box with its markers."""
    from .data import _paint

    if obj.classname != "TScatter2D":
        _paint(scene, obj, option)
        return
    from .legoplot import paint_three_d

    if "SAME" not in option.upper() or scene.view3d is None:
        scene.solid = True
        paint_three_d(scene, obj.frame3d(), "SURF", cells=False)
    view, pad = scene.view3d
    pixels = _pixels(view, pad, obj.x, obj.y, obj.z)
    fractions = obj.colour_fractions()
    look = obj.members["TAttMarker"]
    colours = (shades(fractions, "bird") if fractions is not None
               else (scene.colors.hexed(look["fMarkerColor"]),) * len(pixels))  # fmt: skip
    for pixel, colour, size in zip(pixels, colours, obj.marker_sizes(), strict=True):
        draw_markers(scene, pixel.reshape(1, 2), int(look["fMarkerStyle"]), float(size), colour)
