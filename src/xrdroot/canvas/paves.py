"""Paves: the boxes of text a pad draws over its frame.

A ``TPave`` is a box with a border and a shadow, placed in NDC when its
``fOption`` says so (``"brNDC"``, the usual) and in the axes' units when it
does not; the letters before ``NDC`` say which sides the shadow falls on.
A ``TPaveText`` stacks lines of text in it, a ``TPaveStats`` is one of
those whose lines are a histogram's statistics - two columns, a name and a
value - and a ``TLegend`` is a pave of entries, each a symbol drawn the way
the thing it stands for is drawn, and a label.

ROOT sizes text left at size 0 to fit: a line of a pave takes most of the
height a line has, and a legend's label most of the height of a row. That
is done here the same way.
"""

from __future__ import annotations

import math
from typing import Any

from . import styles
from .latex import translate
from .model import Primitive, lookup
from .scene import Scene
from .shapes import draw_text, patch_style

__all__ = ["PAVES", "pave_box", "draw_lines"]

#: How much of a line's height text sized to fit takes, as ROOT's ``TPaveText`` does.
FIT = 0.85
#: How much of a row of a legend its label takes, when sized to fit.
LEGEND_FIT = 0.6
#: The room kept either side of a line of text in a pave, as a fraction of its width.
MARGIN = 0.05
#: How wide a character of text is, roughly, as a fraction of its size.
CHARACTER = 0.5
#: How far a legend's symbol reaches either side of its middle, as a fraction of the room it has.
SYMBOL = 0.35


def pave_box(scene: Scene, prim: Any) -> tuple[float, float, float, float]:
    """A pave's corners as fractions of the pad: ``x1, y1, x2, y2``."""
    if "NDC" in str(lookup(prim, "fOption", "")).upper():
        return tuple(
            float(lookup(prim, name, 0.0)) for name in ("fX1NDC", "fY1NDC", "fX2NDC", "fY2NDC")
        )  # type: ignore[return-value]
    x1, y1 = scene.to_ndc(float(lookup(prim, "fX1", 0.0)), float(lookup(prim, "fY1", 0.0)))
    x2, y2 = scene.to_ndc(float(lookup(prim, "fX2", 0.0)), float(lookup(prim, "fY2", 0.0)))
    return x1, y1, x2, y2


def _shadow(scene: Scene, prim: Any, corners: tuple[float, float, float, float]) -> None:
    """The shadow of a border wider than a pixel, on the sides ``fOption`` names."""
    from matplotlib.patches import Rectangle

    border = int(lookup(prim, "fBorderSize", 0))
    if border <= 1:
        return
    option = str(lookup(prim, "fOption", "br")).lower().replace("ndc", "") or "br"
    dx, dy = border / scene.pixels[0], border / scene.pixels[1]
    x1, y1, x2, y2 = corners
    across = dx if "r" in option else -dx
    up = -dy if "b" in option else dy
    scene.ax.add_artist(
        Rectangle(
            (x1 + across, y1 + up),
            x2 - x1,
            y2 - y1,
            transform=scene.ndc,
            clip_on=False,
            zorder=scene.layer(),
            facecolor=scene.colors.rgb(lookup(prim, "fShadowColor", 1)),
            edgecolor="none",
        )
    )


def draw_box(scene: Scene, prim: Any) -> tuple[float, float, float, float]:
    """A pave's box, shadow and border, and where it is."""
    from matplotlib.patches import Rectangle

    corners = pave_box(scene, prim)
    _shadow(scene, prim, corners)
    x1, y1, x2, y2 = corners
    outlined = int(lookup(prim, "fBorderSize", 0)) > 0
    scene.ax.add_artist(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            transform=scene.ndc,
            clip_on=False,
            zorder=scene.layer(),
            **patch_style(scene, prim, outline=outlined),
        )
    )
    return corners


def pave(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TPave``: a box and nothing in it."""
    draw_box(scene, prim)


def _fitted(scene: Scene, height: float, width: float, longest: int, fit: float) -> float:
    """The pixels text sized to fit a line ``height`` by ``width`` of the pad takes."""
    size = fit * height * scene.pixels[1]
    if longest:
        size = min(size, (1 - 2 * MARGIN) * width * scene.pixels[0] / (CHARACTER * longest))
    return size


def _line_size(scene: Scene, line: Any, holder: Any, fitted: float) -> float:
    """The size of one line of a pave, in points: its own, the pave's, or fitted."""
    for source in (line, holder):
        size = float(lookup(source, "fTextSize", 0.0) or 0.0)
        if size:
            return scene.text_points(size, lookup(source, "fTextFont", 42) or 42)
    return styles.points(fitted)


def _line_x(corners: tuple[float, float, float, float], across: str) -> float:
    x1, _y1, x2, _y2 = corners
    width = x2 - x1
    return {"left": x1 + MARGIN * width, "center": (x1 + x2) / 2}.get(across, x2 - MARGIN * width)


def draw_lines(
    scene: Scene, holder: Any, lines: list[Any], corners: tuple[float, float, float, float]
) -> None:
    """The lines of a pave, stacked from the top, each a row of the pave's height."""
    x1, y1, x2, y2 = corners
    texts = [line for line in lines if getattr(line, "classname", "") in ("TText", "TLatex")]
    if not texts:
        return
    step = (y2 - y1) / len(lines)
    longest = max(len(str(line.get("fTitle", ""))) for line in texts)
    fitted = _fitted(scene, step, x2 - x1, longest, FIT)
    for index, line in enumerate(lines):
        if line in texts:
            _pave_line(scene, holder, line, corners, y2 - (index + 0.5) * step, fitted)


def _pave_line(
    scene: Scene,
    holder: Any,
    line: Primitive,
    corners: tuple[float, float, float, float],
    y: float,
    fitted: float,
) -> None:
    """One line of text in a pave, where its own place says or in its row."""
    style = scene.text(line, holder, _line_size(scene, line, holder, fitted))
    if not lookup(line, "fTextAlign", 0) and not lookup(holder, "fTextAlign", 0):
        style["ha"], style["va"] = "left", "center"
    style["va"] = "center" if style["va"] == "bottom" else style["va"]
    x1, y1, x2, y2 = corners
    x, placed_y = float(line.get("fX", 0.0)), float(line.get("fY", 0.0))
    at_x = x1 + x * (x2 - x1) if x else _line_x(corners, style["ha"])
    at_y = y1 + placed_y * (y2 - y1) if placed_y else y
    draw_text(scene, translate(str(line.get("fTitle", ""))), at_x, at_y, style, ndc=True)


def pave_text(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TPaveText``: a box, and its lines of text stacked in it."""
    corners = draw_box(scene, prim)
    draw_lines(scene, prim, list(prim.get("fLines") or []), corners)


def pave_label(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TPaveLabel``: a box with one label in the middle of it."""
    x1, y1, x2, y2 = draw_box(scene, prim)
    label = str(prim.get("fLabel", ""))
    style = scene.text(prim, None, styles.points(_fitted(scene, y2 - y1, x2 - x1, len(label), FIT)))
    if float(prim.get("fTextSize", 0.0) or 0.0):
        style["fontsize"] = scene.text_points(prim.get("fTextSize"), prim.get("fTextFont", 42))
    style["ha"], style["va"] = "center", "center"
    draw_text(scene, translate(label), (x1 + x2) / 2, (y1 + y2) / 2, style, ndc=True)


def columns(
    scene: Scene,
    holder: Any,
    rows: list[tuple[str, str]],
    corners: tuple[float, float, float, float],
) -> None:
    """Lines of two columns - a name on the left, its value on the right - under a title.

    That is how a stats box is drawn: the first line alone, centred, and
    every one after split at its ``=``.
    """
    x1, y1, x2, y2 = corners
    step = (y2 - y1) / max(len(rows), 1)
    longest = max((len(left) + len(right) + 2 for left, right in rows), default=0)
    size = _line_size(scene, None, holder, _fitted(scene, step, x2 - x1, longest, FIT))
    for index, (left, right) in enumerate(rows):
        y = y2 - (index + 0.5) * step
        style = scene.text(holder, None, size)
        style["va"] = "center"
        if not right:
            style["ha"] = "center"
            draw_text(scene, translate(left), (x1 + x2) / 2, y, style, ndc=True)
            continue
        style["ha"] = "left"
        draw_text(scene, translate(left), _line_x(corners, "left"), y, style, ndc=True)
        style["ha"] = "right"
        draw_text(scene, translate(right), _line_x(corners, "right"), y, style, ndc=True)


def split(text: str) -> tuple[str, str]:
    """A line of a stats box as its name and its value."""
    name, equals, value = text.partition("=")
    return (name.strip(), value.strip()) if equals else (text.strip(), "")


def stats_box(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TPaveStats`` saved with its lines: the lines as they were drawn."""
    corners = draw_box(scene, prim)
    lines = [
        str(line.get("fTitle", "")) for line in prim.get("fLines") or [] if hasattr(line, "get")
    ]
    rows = [(lines[0].strip(), "")] + [split(line) for line in lines[1:]] if lines else []
    columns(scene, prim, rows, corners)


# -- legends -------------------------------------------------------------------


def _symbol(scene: Scene, entry: Any, option: str, cell: tuple[float, float, float, float]) -> None:
    """A legend entry's symbol: the fill, line, error bar and marker it asks for."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    x, y, room, height = cell
    source = entry.get("fObject") if entry.get("fObject") is not None else entry
    if "f" in option:
        scene.ax.add_artist(
            Rectangle(
                (x - SYMBOL * room, y - SYMBOL * height),
                2 * SYMBOL * room,
                2 * SYMBOL * height,
                transform=scene.ndc,
                clip_on=False,
                zorder=scene.layer(),
                **patch_style(scene, source, outline="l" in option),
            )
        )
    lines = []
    if "l" in option and "f" not in option:
        lines.append(([x - SYMBOL * room, x + SYMBOL * room], [y, y]))
    if "e" in option:
        lines.append(([x, x], [y - SYMBOL * height, y + SYMBOL * height]))
    for xs, ys in lines:
        scene.ax.add_artist(
            Line2D(
                xs,
                ys,
                transform=scene.ndc,
                clip_on=False,
                zorder=scene.layer(),
                **scene.line(source),
            )
        )
    if "p" in option:
        scene.ax.add_artist(
            Line2D(
                [x],
                [y],
                linestyle="none",
                transform=scene.ndc,
                clip_on=False,
                zorder=scene.layer(),
                **scene.marker(source),
            )
        )


def _cell(
    corners: tuple[float, float, float, float], index: int, columns_: int, rows: int, margin: float
) -> tuple[float, float, float, float]:
    """Where one entry of a legend goes: its symbol's middle, and the room it has."""
    x1, y1, x2, y2 = corners
    width, height = (x2 - x1) / columns_, (y2 - y1) / rows
    column, row = index % columns_, index // columns_
    room = margin * width
    return x1 + column * width + room / 2, y2 - (row + 0.5) * height, room / 2, height / 2


def legend(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TLegend``: its box, and each entry's symbol and label in rows and columns."""
    corners = draw_box(scene, prim)
    entries = list(prim.get("fPrimitives") or [])
    if not entries:
        return
    columns_ = max(int(prim.get("fNColumns", 1) or 1), 1)
    rows = math.ceil(len(entries) / columns_)
    margin = float(prim.get("fMargin", 0.25) or 0.25)
    x1, y1, x2, y2 = corners
    longest = max(len(str(entry.get("fLabel", ""))) for entry in entries)
    fitted = _fitted(
        scene, (y2 - y1) / rows, (x2 - x1) * (1 - margin) / columns_, longest, LEGEND_FIT
    )
    for index, entry in enumerate(entries):
        _entry(scene, prim, entry, _cell(corners, index, columns_, rows, margin), fitted)


def _entry(
    scene: Scene,
    holder: Primitive,
    entry: Any,
    cell: tuple[float, float, float, float],
    fitted: float,
) -> None:
    """One entry of a legend: a header across the whole row, or a symbol and a label."""
    option = str(entry.get("fOption", "")).lower()
    x, y, room, _height = cell
    style = scene.text(entry, holder, _line_size(scene, entry, holder, fitted))
    style["va"] = "center"
    label = translate(str(entry.get("fLabel", "")))
    if "h" in option:
        style["ha"] = "left"
        draw_text(scene, label, x - room, y, style, ndc=True)
        return
    _symbol(scene, entry, option, cell)
    style["ha"] = "left"
    draw_text(scene, label, x + room * 1.2, y, style, ndc=True)


#: How each of these classes draws.
PAVES = {
    "TPave": pave,
    "TPaveText": pave_text,
    "TPavesText": pave_text,
    "TPaveLabel": pave_label,
    "TPaveStats": stats_box,
    "TLegend": legend,
}
