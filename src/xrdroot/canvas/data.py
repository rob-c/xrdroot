"""The things a pad draws that are data: histograms, graphs, stacks and functions.

Each is drawn the way :mod:`xrdroot.plot` draws it - the same option read
the same way, the same attributes read off the object, the same layers -
by asking it for the :func:`~xrdroot.plot.picture` of the object and the
option the pad kept, and drawing the layers onto the pad's axes. The frame
is the pad's, not the picture's: its range, scales and titles are set
from the pad, so what a histogram drawn ``SAME`` would say about them does
not change the frame the first one drew.

Round that, a canvas adds what ROOT's pad adds. A ``COLZ`` scale goes where
its ``TPaletteAxis`` was, or in the pad's right margin, rather than taking
room from the frame. The stats box a histogram saved is drawn with the lines
it was saved with, and one saved without is given ``gStyle``'s. A
multigraph draws each graph by the option it was added with. The colours a
canvas saved are the ones its data is drawn in.

What the picture refuses - an option ROOT takes that is not drawn here,
``SCAT`` or ``LEGO`` on axes without depth - is drawn as the object would
be without it, and said in the canvas's warning; a three-dimensional
histogram, or a function that cannot be worked out here, is left out and
said too.
"""

from __future__ import annotations

from typing import Any

from ..errors import ROOTError
from ..function import Function
from ..graph import Graph
from ..hist import Histogram
from ..plot import picture
from ..plot.backends.withmatplotlib import DRAWN
from ..plot.model import Frame, Mesh, Picture
from ..stacks import MultiGraph, Stack
from .model import lookup
from .options import strip_same
from .paves import stats_box
from .scene import Scene
from .statbox import default_stats, shows_stats

__all__ = ["DATA", "paint_data"]

#: Where a histogram's colour scale goes, without a saved ``TPaletteAxis``: how
#: far right of the frame it starts, and how wide it is, as fractions of the pad.
PALETTE_GAP, PALETTE_WIDTH = 0.005, 0.05
#: The attributes whose colour a canvas may have saved, and the keyword each is.
COLOURED = (("fLineColor", "color"), ("fMarkerColor", "markercolor"))


# -- a picture, drawn onto the pad -------------------------------------------------


def _saved_colours(scene: Scene, obj: Any) -> dict[str, Any]:
    """The keywords that draw ``obj`` in the colours its canvas saved, where it saved any."""
    saved = scene.colors.saved
    style: dict[str, Any] = {}
    for member, keyword in COLOURED:
        index = lookup(obj, member)
        if index is not None and int(index) in saved:
            style[keyword] = scene.colors.hexed(index)
    fill = lookup(obj, "fFillColor")
    if fill is not None and int(fill) in saved and int(lookup(obj, "fFillStyle", 0) or 0):
        style["fill"] = scene.colors.hexed(fill)
    if scene.colors.palette:
        style["palette"] = scene.colors.colormap()
    return style


def _pictured(scene: Scene, obj: Any, option: str) -> Picture | None:
    """The picture of ``obj`` drawn with ``option``, or as it would be without it.

    An option the picture refuses is said in the warning, and the object is
    drawn as it is with no option but ``SAME``; one that cannot be drawn at
    all is left out, and said.
    """
    style = _saved_colours(scene, obj)
    try:
        return picture(obj, option, style)
    except ValueError as why:
        scene.skipped.append(f"{_named(obj)} drawn without its option {option!r} ({why})")
    except ROOTError as why:
        scene.skipped.append(f"{_named(obj)} ({why})")
        return None
    return picture(obj, _plain(option), style)


def _plain(option: str) -> str:
    """What an object is drawn with when its own option cannot be: nothing but ``SAME``."""
    return "SAME" if "SAME" in option.upper() else ""


def _named(obj: Any) -> str:
    return f"{obj.classname} {obj.name!r}"


def _draw(scene: Scene, obj: Any, drawn: Picture) -> None:
    """The picture's layers onto the pad's axes, whose range and scales stay the pad's."""
    ax = scene.ax
    limits = ax.get_xlim(), ax.get_ylim()
    frame = Frame(logz=scene.pad.logz)
    for index, layer in enumerate(drawn.layers):
        scale = isinstance(layer, Mesh) and layer.scale
        if isinstance(layer, Mesh):
            layer = layer._replace(scale=False)  # the pad places it, not matplotlib
        artist = DRAWN[type(layer)](ax, layer, frame, drawn.native if index == 0 else {})
        if scale:
            _palette(scene, artist, obj)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])


def _paint(scene: Scene, obj: Any, option: str) -> None:
    """``obj`` by ``option``, as :mod:`xrdroot.plot` pictures it, if it can be drawn at all."""
    drawn = _pictured(scene, obj, option)
    if drawn is None:
        return
    if drawn.deep and option != _plain(option):  # LEGO and SURF, on the flat axes of a pad
        scene.skipped.append(
            f"{_named(obj)} drawn without its option {option!r} (it draws in three "
            f"dimensions, and a pad's axes have two)"
        )
        drawn = picture(obj, _plain(option), _saved_colours(scene, obj))
    _draw(scene, obj, drawn)


def _palette(scene: Scene, mesh: Any, h: Any) -> None:
    """The colour scale of ``COLZ``, where its ``TPaletteAxis`` was, or in the right margin."""
    saved = [one for one in h.functions if getattr(one, "classname", "") == "TPaletteAxis"]
    _left, right, bottom, top = scene.pad.margins
    corners = (1 - right + PALETTE_GAP, bottom, 1 - right + PALETTE_GAP + PALETTE_WIDTH, 1 - top)
    if saved:
        corners = tuple(
            float(saved[0].get(name, 0.0)) for name in ("fX1NDC", "fY1NDC", "fX2NDC", "fY2NDC")
        )  # type: ignore[assignment]
    x, y, w, height = scene.box
    rect = (
        x + corners[0] * w,
        y + corners[1] * height,
        (corners[2] - corners[0]) * w,
        (corners[3] - corners[1]) * height,
    )
    cax = scene.figure.add_axes(rect, label=f"{scene.pad.name} palette")
    scene.figure.colorbar(mesh, cax=cax)
    axis = lookup(h, "fZaxis") or {}
    cax.tick_params(direction="in", labelsize=scene.text_points(lookup(axis, "fLabelSize", 0.035)))


# -- what ROOT hangs on what it draws -------------------------------------------------


def _stats(scene: Scene, obj: Any, option: str) -> None:
    """The stats box saved with ``obj``, or ``gStyle``'s for a histogram saved with none."""
    saved = [one for one in obj.functions if getattr(one, "classname", "") == "TPaveStats"]
    for box in saved:
        stats_box(scene, box, "")
    if not saved and _stats_made(scene, obj, option):
        default_stats(scene, obj)


def _stats_made(scene: Scene, obj: Any, option: str) -> bool:
    """Whether drawing ``obj`` now would make a stats box, as ROOT does for a histogram.

    A pad drawn before it was saved kept the box it made, if any, so this
    is only for one saved without being drawn.
    """
    return isinstance(obj, Histogram) and not scene.pad.painted and shows_stats(obj, option)


# -- each kind of data ----------------------------------------------------------------


def paint_histogram(scene: Scene, h: Histogram, option: str) -> None:
    """A histogram of one or two dimensions, by its draw option, with its stats box."""
    if len(h.axes) > 2:
        scene.skipped.append(f"{_named(h)} (three dimensions have no flat picture)")
        return
    _paint(scene, h, option)
    _stats(scene, h, option)


def paint_graph(scene: Scene, g: Graph, option: str) -> None:
    """A graph, by the letters of its draw option, with the stats box of its fit."""
    _paint(scene, g, option)
    _stats(scene, g, option)


def paint_multigraph(scene: Scene, mg: MultiGraph, option: str) -> None:
    """A ``TMultiGraph``: each graph by its own option, or the multigraph's without its axes.

    ROOT draws a graph added with an option of its own by that option, and
    one added with none by the multigraph's; the fits made to all of them
    together are drawn over them.
    """
    shared = strip_same(option).replace("A", "")
    listed = mg.members.get("fGraphs") or []
    options = list(getattr(listed, "options", [""] * len(mg)))
    for graph, own in zip(mg, options):
        paint_graph(scene, graph, (strip_same(own).replace("A", "") or shared) + " SAME")
    for function in mg.functions:
        if isinstance(function, Function):
            paint_function(scene, function, "SAME")


def paint_stack(scene: Scene, stack: Stack, option: str) -> None:
    """A ``THStack``: stacked, the top drawn first so each fill shows, or ``NOSTACK``."""
    _paint(scene, stack, option)


def paint_function(scene: Scene, f: Function, option: str) -> None:
    """A ``TF1`` as a line over its range, or a ``TF2`` as its contours."""
    _paint(scene, f, option)


#: How each kind of data draws, by the Python class it comes back as.
DATA: tuple[tuple[type, Any], ...] = (
    (Histogram, paint_histogram),
    (Graph, paint_graph),
    (MultiGraph, paint_multigraph),
    (Stack, paint_stack),
    (Function, paint_function),
)


def paint_data(scene: Scene, obj: Any, option: str) -> bool:
    """Draw ``obj`` if it is data, and say whether it was."""
    for kind, paint in DATA:
        if isinstance(obj, kind):
            paint(scene, obj, option)
            return True
    return False
