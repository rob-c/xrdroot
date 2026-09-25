"""A canvas drawn with matplotlib: a figure its size, and an axes per pad.

The figure is the canvas's ``fCw`` by ``fCh`` pixels at :data:`~.styles.DPI`,
so a ROOT pixel is a figure pixel and every size ROOT gives in pixels -
line widths, borders, text in a font of precision 3 - comes out the size it
was. Each pad is drawn where it sits, its background and border first, then
each of its primitives in the order ROOT drew them, pads inside it drawn as
they are met, so that what ROOT drew over something is drawn over it here.

What this does not draw - a class it has no painter for, or one the file
held but this reader could not decode - is left out, and the figure comes
with a warning naming every one of them, never silently.
"""

from __future__ import annotations

import warnings
from typing import Any

from ..errors import UnsupportedFeatureError
from . import styles
from .colors import Colors
from .data import paint_data
from .frame import default_title, dress, open_axes
from .model import Canvas, Pad, Primitive
from .paves import PAVES
from .scene import Scene
from .shapes import SHAPES

__all__ = ["CanvasWarning", "paint"]

#: The classes a pad holds that draw nothing of their own: the frame, which
#: dresses the axes, and the palette and legend entries their owners draw.
QUIET = frozenset({"TFrame", "TPaletteAxis", "TLegendEntry", "TColor"})
#: How much lighter and darker a raised border's two sides are than the pad.
BEVEL = 0.4

#: Every drawing class this draws, and how.
PAINTERS = {**SHAPES, **PAVES}


class CanvasWarning(UserWarning):
    """A canvas held something this does not draw, which was left out."""


def _placed(
    parent: tuple[float, float, float, float], pad: Pad
) -> tuple[float, float, float, float]:
    """Where a pad is on the figure, from where it is in the pad it is in."""
    x, y, w, h = parent
    px, py, pw, ph = pad.place
    return x + px * w, y + py * h, pw * w, ph * h


def _bevel(scene: Scene, mode: int, size: int, face: tuple[float, float, float]) -> None:
    """A raised (``fBorderMode`` 1) or sunken (-1) border, lit from the top left."""
    from matplotlib.patches import Polygon

    bw, bh = size / scene.pixels[0], size / scene.pixels[1]
    light = tuple(channel + (1 - channel) * BEVEL for channel in face)
    dark = tuple(channel * (1 - BEVEL) for channel in face)
    upper = [(0, 0), (bw, bh), (bw, 1 - bh), (1 - bw, 1 - bh), (1, 1), (0, 1)]
    lower = [(0, 0), (bw, bh), (1 - bw, bh), (1 - bw, 1 - bh), (1, 1), (1, 0)]
    for corner, color in (
        (upper, light if mode > 0 else dark),
        (lower, dark if mode > 0 else light),
    ):
        scene.ax.add_artist(
            Polygon(
                corner,
                closed=True,
                transform=scene.ndc,
                clip_on=False,
                zorder=-90,
                facecolor=color,
                edgecolor="none",
            )
        )


def _background(scene: Scene) -> None:
    """The pad's fill, under everything drawn in it, and its border."""
    from matplotlib.patches import Rectangle

    pad = scene.pad
    fills, _hatch, alpha = styles.fill(pad.get("fFillStyle", 1001))
    face = scene.colors.rgb(pad.get("fFillColor", 0))
    if fills:
        scene.ax.add_artist(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=scene.ndc,
                clip_on=False,
                zorder=-100,
                facecolor=(*face, alpha),
                edgecolor="none",
            )
        )
    mode, size = int(pad.get("fBorderMode", 0)), int(pad.get("fBorderSize", 0))
    if mode and size > 0:
        _bevel(scene, mode, size, face)


def _paint_one(scene: Scene, obj: Any, option: str, canvas: Canvas) -> None:
    """One primitive of a pad: another pad, data, a drawing class, or something skipped."""
    if isinstance(obj, Pad):
        _paint_pad(scene.figure, obj, scene.box, scene.colors, canvas, scene.skipped)
        return
    if paint_data(scene, obj, option):
        return
    classname = getattr(obj, "classname", None)
    if isinstance(obj, Primitive) and classname in PAINTERS:
        PAINTERS[classname](scene, obj, option)
    elif classname not in QUIET:
        scene.skipped.append(_described(obj))


def _described(obj: Any) -> str:
    """How something skipped is named in the warning: its class, and its name if it has one."""
    if isinstance(obj, str):
        return f"{obj} (a class this file does not describe)"
    classname = getattr(obj, "classname", type(obj).__name__)
    name = getattr(obj, "name", "")
    return f"{classname} {name!r}" if name else str(classname)


def _paint_pad(
    figure: Any,
    pad: Pad,
    parent: tuple[float, float, float, float],
    colors: Colors,
    canvas: Canvas,
    skipped: list[str],
) -> Scene:
    """One pad, and every pad inside it."""
    box = _placed(parent, pad)
    scene = Scene(figure, pad, box, colors, (canvas.width, canvas.height), skipped)
    open_axes(scene)
    _background(scene)
    for obj, option in pad.primitives:
        _paint_one(scene, obj, option, canvas)
    default_title(scene)
    dress(scene)
    return scene


def paint(canvas: Canvas, figure: Any = None) -> Any:
    """``canvas`` drawn onto ``figure``, or onto a new figure the canvas's size."""
    try:
        from matplotlib.figure import Figure
    except ImportError:
        raise UnsupportedFeatureError(
            "drawing a canvas needs matplotlib, which is not installed: pip install matplotlib"
        ) from None
    if figure is None:
        figure = Figure(
            figsize=(canvas.width / styles.DPI, canvas.height / styles.DPI), dpi=styles.DPI
        )
    colors = canvas.palette()
    figure.set_facecolor(colors.rgb(canvas.get("fFillColor", 0)))
    skipped: list[str] = []
    _paint_pad(figure, canvas, (0.0, 0.0, 1.0, 1.0), colors, canvas, skipped)
    if skipped:
        warnings.warn(
            f"canvas {canvas.name!r} holds {len(skipped)} things this does not draw, "
            f"which are left out of the picture: {', '.join(skipped)}",
            CanvasWarning,
            stacklevel=3,
        )
    return figure
