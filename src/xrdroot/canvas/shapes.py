"""The simple drawing classes: text, lines, arrows, boxes, ellipses and markers.

Each is placed in the units of the pad's axes unless it says otherwise:
text, a line or a marker by a bit of its ``fBits`` (``SetNDC``), and never a
box or an ellipse, which ROOT only ever places in the axes' units. None of
them is clipped to the frame - ROOT clips them to the pad, which the figure
does anyway.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import styles
from .latex import paint_latex
from .model import Primitive, lookup
from .scene import Scene
from .text import glyphs, pixel_size

__all__ = ["SHAPES", "canvas_point", "write"]

#: The points an ellipse's outline is drawn with, round a whole turn.
ELLIPSE_POINTS = 181
#: How long an arrow's head is when ``fArrowSize`` was never set, as a fraction of the pad.
ARROW_SIZE = 0.05
#: How much bigger than ``fArrowSize`` of the pad's height matplotlib's arrow must be
#: scaled for its head to be as long as ``TArrow::PaintArrowNDC`` draws one.
HEAD = 3.0


def canvas_point(scene: Scene, x: float, y: float, ndc: bool) -> tuple[float, float]:
    """A point of the pad - in NDC, or in its axes' units - as the canvas's pixel."""
    u, v = (x, y) if ndc else scene.to_ndc(x, y)
    return scene.pixel(u, v)


def write(scene: Scene, text: str, at: tuple[float, float], attributes: dict[str, Any], latex: bool = True) -> None:
    """``text`` at canvas pixel ``at``: a ``TLatex``'s formula, or a ``TText``'s string as it is."""
    if latex:
        paint_latex(scene, text, at, attributes)
        return
    font = int(attributes["font"])
    glyphs(scene, text, at, font, pixel_size(scene, float(attributes["size"]), font),
           scene.colors.rgb(attributes["color"]), int(attributes["align"]), float(attributes["angle"]))  # fmt: skip


def _text(scene: Scene, prim: Primitive, latex: bool) -> None:
    at = canvas_point(scene, float(prim.get("fX", 0.0)), float(prim.get("fY", 0.0)), prim.ndc)
    write(scene, str(prim.get("fTitle", "")), at, scene.attributes(prim), latex)


def text(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TText``: the string as it is."""
    _text(scene, prim, latex=False)


def latex(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TLatex``: the string, with ROOT's ``#`` mathematics translated."""
    _text(scene, prim, latex=True)


def _ends(prim: Primitive) -> tuple[list[float], list[float]]:
    return (
        [float(prim.get("fX1", 0.0)), float(prim.get("fX2", 0.0))],
        [float(prim.get("fY1", 0.0)), float(prim.get("fY2", 0.0))],
    )


def _line_of(scene: Scene, prim: Any, points: list[tuple[float, float]], style: Any = None) -> None:
    """A line through canvas ``points`` in ``prim``'s colour, width and style (or ``style``)."""
    from .raster import add_line

    width = int(lookup(prim, "fLineWidth", 1) or 0)
    if width > 0:
        chosen = lookup(prim, "fLineStyle", 1) if style is None else style
        add_line(scene, points, scene.colors.rgb(lookup(prim, "fLineColor", 1)), width, chosen)


def line(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TLine``, from ``(fX1, fY1)`` to ``(fX2, fY2)``."""
    xs, ys = _ends(prim)
    _line_of(scene, prim, [canvas_point(scene, x, y, prim.ndc) for x, y in zip(xs, ys)])


def _head(tip: tuple[float, float], along: tuple[float, float], length: float, half: float) -> list[tuple[float, float]]:
    """An arrow's head at ``tip``, pointing along ``along``: its two back corners about the tip."""
    (x, y), (cos, sin) = tip, along
    return [(x - length * cos - sin * half, y - length * sin + cos * half), (x, y),
            (x - length * cos + sin * half, y - length * sin - cos * half)]  # fmt: skip


def _heads(scene: Scene, prim: Primitive, option: str, ends: tuple[Any, Any], sizes: tuple[float, float]) -> None:
    """``TArrow::PaintArrow``'s heads: an open ``>``, or a ``|>`` filled and outlined."""
    from matplotlib.patches import Polygon

    (start, end), (length, half) = ends, sizes
    dx, dy = end[0] - start[0], end[1] - start[1]
    span = math.hypot(dx, dy) or 1.0
    cos, sin = dx / span, dy / span
    for mark, closed, tip, sign in ((">", "|>", end, 1.0), ("<", "<|", start, -1.0)):
        if mark not in option:
            continue
        corners = _head(tip, (sign * cos, sign * sin), length, half)
        if closed in option:
            if int(lookup(prim, "fFillColor", 0) or 0):
                scene.ax.add_artist(Polygon([(x, y) for x, y in corners], closed=True, transform=scene.display,
                                            clip_on=False, zorder=scene.layer(), linewidth=0.0, edgecolor="none",
                                            facecolor=scene.colors.rgb(lookup(prim, "fFillColor", 0))))  # fmt: skip
            corners = corners + corners[:1]
        _line_of(scene, prim, corners, 1)


def arrow(scene: Scene, prim: Primitive, _option: str) -> None:
    """``TArrow::PaintArrow``: the shaft, then a head at either end or both, by its ``fOption``.

    ROOT sizes the head in units of the canvas's longer side: ``0.7`` of
    ``fArrowSize`` long, as wide as ``fAngle`` (60 degrees unless set) opens.
    """
    xs, ys = _ends(prim)
    start, end = (canvas_point(scene, x, y, prim.ndc) for x, y in zip(xs, ys))
    option = str(prim.get("fOption", "") or "|>")
    size = float(prim.get("fArrowSize", 0.0) or 0.0) or ARROW_SIZE
    length = 0.7 * size * max(scene.canvas)
    half = length * math.tan(math.pi * float(prim.get("fAngle", 60.0) or 60.0) / 360)
    dx, dy = end[0] - start[0], end[1] - start[1]
    span = math.hypot(dx, dy) or 1.0
    shaft_start, shaft_end = start, end
    if "|>" in option and "-|>-" not in option:
        shaft_end = (end[0] - dx / span * length, end[1] - dy / span * length)
    if "<|" in option and "-<|-" not in option:
        shaft_start = (start[0] + dx / span * length, start[1] + dy / span * length)
    _line_of(scene, prim, [shaft_start, shaft_end])
    _heads(scene, prim, option, (start, end), (length, half))


def patch_style(scene: Scene, prim: Any, outline: bool = True) -> dict[str, Any]:
    """The keywords a filled shape is drawn with: its fill, and its outline if it has one."""
    style = scene.line(prim)
    filled = scene.fill(prim)
    made: dict[str, Any] = {
        "facecolor": "none",
        "edgecolor": style["color"] if outline else "none",
        "linewidth": style["linewidth"] if outline else 0.0,
        "linestyle": style["linestyle"],
    }
    if filled is not None:
        made["facecolor"] = filled["facecolor"]
        if filled["hatch"]:
            made["hatch"] = filled["hatch"]
            made[_HATCH_COLOUR] = filled["hatchcolor"]
    return made


def _hatch_colour() -> str:
    """The keyword a hatch takes its colour from in this matplotlib.

    ``hatchcolor`` came in matplotlib 3.10; before it a hatch is drawn in the
    patch's edge colour, which is where ROOT's fill colour goes instead.
    """
    try:
        from matplotlib.patches import Patch
    except ImportError:
        return "hatchcolor"
    return "hatchcolor" if hasattr(Patch, "set_hatchcolor") else "edgecolor"


#: What a patch's hatch colour is called here.
_HATCH_COLOUR = _hatch_colour()


def box(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TBox``, filled and outlined as its attributes say."""
    from matplotlib.patches import Rectangle

    xs, ys = _ends(prim)
    scene.ax.add_artist(
        Rectangle(
            (min(xs), min(ys)),
            abs(xs[1] - xs[0]),
            abs(ys[1] - ys[0]),
            transform=scene.ax.transData,
            clip_on=False,
            zorder=scene.layer(),
            **patch_style(scene, prim, outline=scene.fill(prim) is None),
        )
    )


def _outline(prim: Primitive) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """The points round an ellipse, or round the slice of one it is limited to."""
    low, high = float(prim.get("fPhimin", 0.0)), float(prim.get("fPhimax", 360.0))
    turn = np.radians(np.linspace(low, high, ELLIPSE_POINTS))
    across, up = (
        float(prim.get("fR1", 0.0)) * np.cos(turn),
        float(prim.get("fR2", 0.0)) * np.sin(turn),
    )
    tilt = math.radians(float(prim.get("fTheta", 0.0)))
    xs = across * math.cos(tilt) - up * math.sin(tilt)
    ys = across * math.sin(tilt) + up * math.cos(tilt)
    if high - low < 360.0:
        xs, ys = np.append(xs, 0.0), np.append(ys, 0.0)  # a slice closes on its centre
    return xs + float(prim.get("fX1", 0.0)), ys + float(prim.get("fY1", 0.0))


def ellipse(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TEllipse``: radii, a tilt, and the angles it runs between."""
    from matplotlib.patches import Polygon

    xs, ys = _outline(prim)
    scene.ax.add_artist(
        Polygon(
            np.column_stack([xs, ys]),
            closed=True,
            transform=scene.ax.transData,
            clip_on=False,
            zorder=scene.layer(),
            **patch_style(scene, prim),
        )
    )


def marker(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TMarker``: one marker, at ``(fX, fY)``."""
    from matplotlib.lines import Line2D

    scene.ax.add_artist(
        Line2D(
            [float(prim.get("fX", 0.0))],
            [float(prim.get("fY", 0.0))],
            linestyle="none",
            transform=scene.where(prim.ndc),
            clip_on=False,
            zorder=scene.layer(),
            **scene.marker(prim),
        )
    )


def _points(prim: Primitive) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """The ``fN`` points of a polyline or polymarker, ``fX`` and ``fY``."""
    count = int(prim.get("fN", 0) or 0)
    xs = np.asarray(_or_none(prim.get("fX")), dtype=float)[:count]
    ys = np.asarray(_or_none(prim.get("fY")), dtype=float)[:count]
    return xs, ys


def _or_none(values: Any) -> Any:
    return [] if values is None else values


def polyline(scene: Scene, prim: Primitive, option: str) -> None:
    """A ``TPolyLine``: its points joined, or an area when it is drawn with ``f``."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    xs, ys = _points(prim)
    where = scene.where(prim.ndc)
    if "F" in (str(prim.get("fOption", "")) + option).upper():
        scene.ax.add_artist(
            Polygon(
                np.column_stack([xs, ys]),
                closed=True,
                transform=where,
                clip_on=False,
                zorder=scene.layer(),
                **patch_style(scene, prim, outline=False),
            )
        )
        return
    scene.ax.add_artist(
        Line2D(xs, ys, transform=where, clip_on=False, zorder=scene.layer(), **scene.line(prim))
    )


def polymarker(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TPolyMarker``: a marker at each of its points."""
    from matplotlib.lines import Line2D

    xs, ys = _points(prim)
    scene.ax.add_artist(
        Line2D(
            xs,
            ys,
            linestyle="none",
            transform=scene.where(prim.ndc),
            clip_on=False,
            zorder=scene.layer(),
            **scene.marker(prim),
        )
    )


def crown(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TCrown``: the ring between two radii, or the slice of it between two angles."""
    from matplotlib.patches import Polygon

    turn = np.radians(
        np.linspace(float(prim.get("fPhimin", 0.0)), float(prim.get("fPhimax", 360.0)), 91)
    )
    inner, outer = float(prim.get("fR1", 0.0)), float(prim.get("fR2", 0.0))
    xs = np.concatenate([outer * np.cos(turn), inner * np.cos(turn[::-1])])
    ys = np.concatenate([outer * np.sin(turn), inner * np.sin(turn[::-1])])
    scene.ax.add_artist(
        Polygon(
            np.column_stack([xs + float(prim.get("fX1", 0.0)), ys + float(prim.get("fY1", 0.0))]),
            closed=True,
            transform=scene.ax.transData,
            clip_on=False,
            zorder=scene.layer(),
            **patch_style(scene, prim),
        )
    )


#: How each of these classes draws.
SHAPES = {
    "TText": text,
    "TLatex": latex,
    "TLine": line,
    "TArrow": arrow,
    "TBox": box,
    "TWbox": box,
    "TEllipse": ellipse,
    "TArc": ellipse,
    "TCrown": crown,
    "TMarker": marker,
    "TPolyLine": polyline,
    "TPolyMarker": polymarker,
}
