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

from .latex import translate
from .model import Primitive
from .scene import Scene

__all__ = ["SHAPES", "draw_text"]

#: The points an ellipse's outline is drawn with, round a whole turn.
ELLIPSE_POINTS = 181
#: How long an arrow's head is when ``fArrowSize`` was never set, as a fraction of the pad.
ARROW_SIZE = 0.05


def draw_text(scene: Scene, text: str, x: float, y: float, style: dict[str, Any], ndc: bool) -> Any:
    """One string at ``(x, y)``, already translated, in the style given."""
    return scene.ax.text(
        x, y, text, transform=scene.where(ndc), clip_on=False, zorder=scene.layer(), **style
    )


def _text(scene: Scene, prim: Primitive, latex: bool) -> None:
    title = str(prim.get("fTitle", ""))
    shown = translate(title) if latex else title.replace("$", r"\$")
    draw_text(
        scene, shown, float(prim.get("fX", 0.0)), float(prim.get("fY", 0.0)),
        scene.text(prim), prim.ndc,
    )  # fmt: skip


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


def line(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TLine``, from ``(fX1, fY1)`` to ``(fX2, fY2)``."""
    from matplotlib.lines import Line2D

    xs, ys = _ends(prim)
    scene.ax.add_artist(
        Line2D(
            xs,
            ys,
            transform=scene.where(prim.ndc),
            clip_on=False,
            zorder=scene.layer(),
            **scene.line(prim),
        )
    )


def _arrowstyle(shape: str) -> str:
    """ROOT's ``"|>"``, ``"<|>"``, ``"->-"`` and the rest, as matplotlib's arrow styles."""
    filled = "|" in shape
    core = shape.strip("-")  # an arrow drawn in the middle, "->-", heads the same way
    start = ("<|" if filled else "<") if core.startswith("<") else ""
    end = ("|>" if filled else ">") if core.endswith(">") else ""
    return f"{start}-{end}"


def arrow(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TArrow``: a line with a head at either end, or both, by its ``fOption``."""
    from matplotlib.patches import FancyArrowPatch

    xs, ys = _ends(prim)
    style = scene.line(prim)
    size = float(prim.get("fArrowSize", 0.0)) or ARROW_SIZE
    filled = scene.fill(prim)
    patch = FancyArrowPatch(
        (xs[0], ys[0]),
        (xs[1], ys[1]),
        arrowstyle=_arrowstyle(str(prim.get("fOption", "|>"))),
        mutation_scale=size * scene.pixels[1] * 0.5,
        transform=scene.where(prim.ndc),
        clip_on=False,
        zorder=scene.layer(),
        edgecolor=style["color"],
        linewidth=style["linewidth"],
        linestyle=style["linestyle"],
        facecolor=filled["facecolor"] if filled else style["color"],
    )
    scene.ax.add_artist(patch)


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
            made["hatchcolor"] = filled["hatchcolor"]
    return made


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
            **patch_style(scene, prim),
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


#: How each of these classes draws.
SHAPES = {
    "TText": text,
    "TLatex": latex,
    "TLine": line,
    "TArrow": arrow,
    "TBox": box,
    "TWbox": box,
    "TEllipse": ellipse,
    "TMarker": marker,
}
