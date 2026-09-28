"""Paves: the boxes of text a pad draws over its frame, as ROOT paints them.

A ``TPave`` is a box with a border and a shadow (``TPave::PaintPave``),
placed in NDC when its ``fOption`` says so (``"brNDC"``, the usual) and in
the axes' units when it does not; the letters before ``NDC`` say which
sides the shadow falls on. A ``TPaveText`` stacks lines of text in it, each
a row of the box's height (``TPaveText::PaintPrimitives``), a ``TPaveLabel``
is one label as big as fits, a ``TPaveStats`` is a pave of a histogram's
statistics - a title, then names on the left and values on the right - and
a ``TLegend`` is a pave of entries, each a symbol drawn the way the thing it
stands for is drawn, and a label (:mod:`.legend`).

Text left at size 0 is sized as ROOT sizes it: most of a row's height, and
smaller if the widest line would not fit, measured as ``TLatex`` measures
it.
"""

from __future__ import annotations

from typing import Any

from . import styles
from .latex import formula_form, paint_latex
from .model import Primitive, lookup
from .scene import Scene
from .shapes import patch_style, write

__all__ = ["PAVES", "draw_box", "pave_box", "text_width"]

#: How much of a row text sized to fit takes in a ``TPaveText``, and in a ``TPaveStats``.
FIT, STATS_FIT = 0.85, 0.92
#: How much of the box the widest line may take in each.
WIDEST, STATS_WIDEST = 0.92, 0.98
#: ``TPave``'s ``fMargin`` when it was never set: the room either side of a line.
MARGIN = 0.05

Corners = tuple[float, float, float, float]


def pave_box(scene: Scene, prim: Any) -> Corners:
    """A pave's corners as fractions of the pad: ``x1, y1, x2, y2``."""
    if "NDC" in str(lookup(prim, "fOption", "")).upper():
        return tuple(
            float(lookup(prim, name, 0.0)) for name in ("fX1NDC", "fY1NDC", "fX2NDC", "fY2NDC")
        )  # type: ignore[return-value]
    x1, y1 = scene.to_ndc(float(lookup(prim, "fX1", 0.0)), float(lookup(prim, "fY1", 0.0)))
    x2, y2 = scene.to_ndc(float(lookup(prim, "fX2", 0.0)), float(lookup(prim, "fY2", 0.0)))
    return x1, y1, x2, y2


def text_width(scene: Scene, text: str, size: float, font: int) -> float:
    """``TLatex::GetXsize`` as a fraction of the pad: how wide ``text`` is laid out."""
    return formula_form(text, size, font, scene.whole, scene.height).width / scene.pixels[0]


def _polygon(scene: Scene, points: list[tuple[float, float]], **style: Any) -> None:
    """A filled polygon through points of the pad in NDC, over what the pad drew before."""
    from matplotlib.patches import Polygon

    shape = Polygon(points, closed=True, transform=scene.ndc, clip_on=False, **style)
    shape.set_zorder(scene.layer())
    scene.ax.add_artist(shape)


def _outline(scene: Scene, prim: Any, points: list[tuple[float, float]]) -> None:
    """A line through points of the pad in NDC, in ``prim``'s line attributes."""
    from .raster import add_line

    width = int(lookup(prim, "fLineWidth", 1) or 0)
    if width > 0:
        color = scene.colors.rgb(lookup(prim, "fLineColor", 1))
        pixels = [scene.pixel(u, v) for u, v in points]
        add_line(scene, pixels, color, width, lookup(prim, "fLineStyle", 1))


def _box(scene: Scene, prim: Any, corners: Corners, outlined: bool) -> None:
    """``TPad::PaintBox``: the fill its style asks for, and the outline if asked or hollow."""
    x1, y1, x2, y2 = corners
    fill_style = int(lookup(prim, "fFillStyle", 1001) or 0)
    fills, _hatch, _alpha = styles.fill(fill_style)
    square = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
    if fills:
        made = patch_style(scene, prim, outline=False)
        _polygon(scene, square, **made)
    if outlined or 0 <= fill_style < 1000:
        _outline(scene, prim, [*square, square[0]])


#: ``TPave::PaintPave``'s shadows: for the sides ``fOption`` names, the six corners, each
#: a corner of the box (0 for ``x1`` or ``y1``, 1 for ``x2`` or ``y2``) moved by so many
#: of the border's widths across and up. Bottom right unless another is named.
SHADOWS = {
    "tr": ((0, 1.5, 1, 0), (0, 1.5, 1, 1), (1, 1, 1, 1), (1, 1, 0, 1.5), (1, 0, 0, 1.5),
           (1, 0, 1, 0)),
    "tl": ((0, -1, 0, 1.5), (0, -1, 1, 1), (1, -1.5, 1, 1), (1, -1.5, 1, 0), (0, 0, 1, 0),
           (0, 0, 0, 1.5)),
    "bl": ((0, -1, 1, -1.5), (0, -1, 0, -1), (1, -1.5, 0, -1), (1, -1.5, 0, 0), (0, 0, 0, 0),
           (0, 0, 1, -1.5)),
    "br": ((0, 1.5, 0, 0), (0, 1.5, 0, -1), (1, 1, 0, -1), (1, 1, 1, -1.5), (1, 0, 1, -1.5),
           (1, 0, 0, 0)),
}  # fmt: skip


def _shadow(scene: Scene, prim: Any, corners: Corners, border: int) -> list[tuple[float, float]]:
    """The six corners of a pave's shadow, on the sides ``fOption`` names, within the pad."""
    x1, y1, x2, y2 = corners
    wx, wy = border / scene.pixels[0], border / scene.pixels[1]
    option = str(lookup(prim, "fOption", "br")).lower()
    sides = next((name for name in ("tr", "tl", "bl") if name in option), "br")
    xs, ys = (x1, x2), (y1, y2)
    points = [(xs[i] + across * wx, ys[j] + up * wy) for i, across, j, up in SHADOWS[sides]]
    return [(min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)) for x, y in points]


def draw_box(scene: Scene, prim: Any) -> Corners:
    """``TPave::PaintPave``: a pave's box, and for a border over a pixel its shadow and outline."""
    corners = pave_box(scene, prim)
    border = int(lookup(prim, "fBorderSize", 0) or 0)
    if border <= 0 and int(lookup(prim, "fFillStyle", 1001) or 0) <= 0:
        return corners
    _box(scene, prim, corners, border == 1)
    if border <= 1 or "nb" in str(lookup(prim, "fOption", "")).lower():
        return corners
    shade = scene.colors.rgb(lookup(prim, "fShadowColor", 1))
    shadow = _shadow(scene, prim, corners, border)
    _polygon(scene, shadow, facecolor=shade, edgecolor="none", linewidth=0.0)
    x1, y1, x2, y2 = corners
    _outline(scene, prim, [(x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1)])
    return corners


def pave(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TPave``: a box and nothing in it."""
    draw_box(scene, prim)


# -- a pave of lines ---------------------------------------------------------------------------


def _is_text(line: Any) -> bool:
    return getattr(line, "classname", "") in ("TText", "TLatex")


def _fitted(scene: Scene, holder: Any, lines: list[Any], row: float, width: float) -> float:
    """A ``TPaveText``'s text size left at 0: 0.85 of a row, less if a line is too wide."""
    size = FIT * row
    font = int(lookup(holder, "fTextFont", 42) or 42)
    widths = [
        text_width(scene, str(line.get("fTitle", "")), size, int(line.get("fTextFont", 0) or font))
        for line in lines
        if getattr(line, "classname", "") == "TLatex" and not float(line.get("fTextSize", 0) or 0)
    ]
    longest = max(widths, default=0.0)
    if longest > WIDEST * width:
        size *= WIDEST * width / longest
    return size


#: ``TAttText``'s own colour, for a pave that never had one set.
UNSET = {"fTextColor": 1}


def _own(line: Any, holder: Any, name: str, size: float) -> Any:
    """A line's attribute, or its pave's where the line left it at 0."""
    value = lookup(line, name, 0) or 0
    if value:
        return value
    return size if name == "fTextSize" else lookup(holder, name, UNSET.get(name, 0))


def _line_attributes(holder: Any, line: Any, size: float) -> dict[str, Any]:
    """A pave's line's text attributes: its own, or its pave's where it left them at 0."""
    return {
        "font": int(_own(line, holder, "fTextFont", size) or 42),
        "size": float(_own(line, holder, "fTextSize", size)),
        "color": int(_own(line, holder, "fTextColor", size)),
        "align": int(_own(line, holder, "fTextAlign", size) or 22),
        "angle": float(lookup(line, "fTextAngle", 0.0) or 0.0),
        "line": int(lookup(line, "fLineWidth", 2) or 2),
    }


def _line_x(holder: Any, line: Any, corners: Corners, across: int) -> float:
    """Where a line of a pave starts: its own place in the box, or the margin its alignment says."""
    x1, _y1, x2, _y2 = corners
    xl = float(line.get("fX", 0.0) or 0.0)
    if 0 < xl < 1:
        return x1 + xl * (x2 - x1)
    margin = float(lookup(holder, "fMargin", MARGIN) or 0.0) * (x2 - x1)
    return {1: x1 + margin, 2: 0.5 * (x1 + x2)}.get(across, x2 - margin)


def _text_line(scene: Scene, holder: Any, line: Any, corners: Corners, at_y: float,
               size: float) -> None:  # fmt: skip
    """One ``TText`` or ``TLatex`` line of a pave, where its own place says or in its row."""
    _x1, y1, _x2, y2 = corners
    attributes = _line_attributes(holder, line, size)
    x = _line_x(holder, line, corners, attributes["align"] // 10)
    yl = float(line.get("fY", 0.0) or 0.0)
    y = y1 + yl * (y2 - y1) if 0 < yl < 1 else at_y
    latex = line.classname == "TLatex"
    write(scene, str(line.get("fTitle", "")), scene.pixel(x, y), attributes, latex)


def _rule(scene: Scene, line: Any, corners: Corners, at_y: float) -> None:
    """A ``TLine`` in a pave: across it at its row, or where its ends say, as fractions of it."""
    x1, y1, x2, y2 = corners
    ends = [float(line.get(name, 0.0) or 0.0) for name in ("fX1", "fX2", "fY1", "fY2")]
    xs = [x1 + ends[0] * (x2 - x1) if ends[0] else x1, x1 + ends[1] * (x2 - x1) if ends[1] else x2]
    ys = [y1 + ends[2] * (y2 - y1) if ends[2] else at_y,
          y1 + ends[3] * (y2 - y1) if ends[3] else at_y]  # fmt: skip
    _outline(scene, line, list(zip(xs, ys)))


def draw_lines(scene: Scene, holder: Any, lines: list[Any], corners: Corners) -> None:
    """``TPaveText::PaintPrimitives``: the pave's lines, each a row from the top."""
    x1, y1, x2, y2 = corners
    rows = len(lines) or 5
    row = (y2 - y1) / rows
    size = float(lookup(holder, "fTextSize", 0.0) or 0.0)
    size = size or _fitted(scene, holder, lines, row, x2 - x1)
    at_y = y2 + 0.5 * row
    for line in lines:
        if getattr(line, "classname", "") == "TLine":
            _rule(scene, line, corners, at_y)
        elif _is_text(line):
            at_y -= row
            _text_line(scene, holder, line, corners, at_y, size)


def pave_text(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TPaveText``: a box, and its lines of text stacked in it."""
    corners = draw_box(scene, prim)
    draw_lines(scene, prim, list(prim.get("fLines") or []), corners)


# -- a label -----------------------------------------------------------------------------------

#: How much of a character ``TPaveLabel`` does not count each of these as.
SPECIALS = {"!": 1.0, "?": 1.5, "#": 1.0, "`": 1.0, "^": 1.5, "~": 1.0, "&": 2.0, "\\": 3.0}


def _extent(scene: Scene, text: str, size: float, font: int) -> tuple[int, int]:
    """``TText::GetTextExtent``: how wide and tall ``text`` is in pixels, at ``size`` of the pad."""
    from . import fonts
    from .text import symbol_text

    found = fonts.extent(symbol_text(text, font), font, fonts.measure_em(size * min(scene.whole)))
    return found.width, found.ascent + found.descent


def _label_rows(scene: Scene, size: float, corners: Corners) -> float:
    """A label's size as a share of its box's rows of pixels, and of the pad's width if narrower."""
    _x1, y1, _x2, y2 = corners
    wide, high = scene.whole
    rows = abs(round(high * (1 - y1)) - round(high * (1 - y2)))
    return size * rows / high * (high / wide if wide < high else 1.0)


def _label_fitted(scene: Scene, label: str, size: float, font: int, corners: Corners) -> float:
    """An automatic label's size: as tall as it is drawn, narrowed until it fits 0.99 of the box."""
    x1, _y1, x2, _y2 = corners
    wide, high = scene.whole
    width, tall = _extent(scene, label, size, font)
    size = tall / high
    across = abs(round(wide * x2) - round(wide * x1))
    last = width
    while width > 0.99 * across:
        size *= 0.99 * across / width
        width, tall = _extent(scene, label, size, font)
        if width == last:
            break
        last = width
    return size


def _label_size(scene: Scene, prim: Any, label: str, corners: Corners) -> float:
    """``TPaveLabel``'s text size: its own, or for 0 (or 0.99) as big as the box has room for."""
    font = int(prim.get("fTextFont", 42) or 42)
    size = float(prim.get("fTextSize", 0.0) or 0.0)
    if font % 10 > 2:
        return size
    automatic = size == 0 or abs(size - 0.99) < 0.001
    size = _label_rows(scene, size or 0.99, corners)
    return _label_fitted(scene, label, size, font, corners) if automatic else size


def pave_label(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TPaveLabel``: a box with one label in it, as big as fits unless sized."""
    corners = draw_box(scene, prim)
    label = str(prim.get("fLabel", ""))
    if len(label) - int(sum(SPECIALS.get(char, 0.0) for char in label) + 0.5) <= 0:
        return
    if not _extent(scene, label, 0.99, int(prim.get("fTextFont", 42) or 42))[0]:
        return
    x1, y1, x2, y2 = corners
    align = int(prim.get("fTextAlign", 22) or 22)
    across, up = divmod(align, 10)
    x = {1: x1 + 0.02 * (x2 - x1), 3: x2 - 0.02 * (x2 - x1)}.get(across, 0.5 * (x1 + x2))
    y = {1: y1 + 0.02 * (y2 - y1), 3: y2 - 0.02 * (y2 - y1)}.get(up, 0.5 * (y1 + y2))
    attributes = scene.attributes(prim, size=_label_size(scene, prim, label, corners), align=align)
    paint_latex(scene, label, scene.pixel(x, y), attributes)


# -- statistics ----------------------------------------------------------------------------------


def _shrunk(size: float, long: float, room: float) -> float:
    """``size``, less by as much as ``long`` is longer than the ``room`` it has."""
    return size * (room / long if long > room else 1.0)


def _stats_sizes(scene: Scene, prim: Any, lines: list[Any], row: float,
                 width: float) -> tuple[float, float]:  # fmt: skip
    """``TPaveStats``'s text size and title size when left at 0: most of a row, less to fit."""
    size = STATS_FIT * row
    title, tokens = size, [0.0, 0.0]
    margin = float(lookup(prim, "fMargin", MARGIN) or 0.0) * width
    named = int(lookup(prim, "fOptStat", 0) or 0) % 10
    font = int(lookup(prim, "fTextFont", 42) or 42)
    for line in lines:
        text, own = str(line.get("fTitle", "")), int(line.get("fTextFont", 0) or font)
        if "=" in text and named == 0:
            _widest_tokens(scene, text, (size, own), tokens)
        elif "|" not in text:
            named = 0
            long = text_width(scene, text, title, own) + 2 * margin
            title = _shrunk(title, long, STATS_WIDEST * width)
    return _shrunk(size, tokens[0] + tokens[1] + 2 * margin, STATS_WIDEST * width), title


def _widest_tokens(scene: Scene, text: str, sized: tuple[float, int], tokens: list[float]) -> None:
    """``tokens`` widened to a line's name and value, if either is wider than any before."""
    size, font = sized
    for index, token in enumerate([one for one in text.split("=") if one][:2]):
        tokens[index] = max(tokens[index], text_width(scene, token, size, font))


def _stats_attributes(prim: Any, line: Any, size: float) -> dict[str, Any]:
    """A statistics line's text attributes: its own, or its box's where it left them at 0."""
    return {
        "font": int(_own(line, prim, "fTextFont", size) or 42),
        "size": float(_own(line, prim, "fTextSize", size)),
        "color": int(_own(line, prim, "fTextColor", size)),
        "angle": float(lookup(line, "fTextAngle", 0.0) or 0.0),
        "line": 2,
    }


def _stats_pair(scene: Scene, text: str, at: tuple[float, float, float],
                base: dict[str, Any]) -> None:  # fmt: skip
    """A line of statistics: its name at the left, its value at the right, a minus a minus sign."""
    left, right, y = at
    for index, token in enumerate([one for one in text.split("=") if one]):
        if index == 0:
            paint_latex(scene, token, scene.pixel(left, y), dict(base, align=12))
        else:
            shown = token.strip().replace("-", "#minus")
            paint_latex(scene, shown, scene.pixel(right, y), dict(base, align=32))


def _stats_lines(prim: Primitive) -> list[Any]:
    """The lines ``TPaveStats::Paint`` paints: its ``TLatex`` ones."""
    return [line for line in prim.get("fLines") or [] if getattr(line, "classname", "") == "TLatex"]


def stats_box(scene: Scene, prim: Primitive, _option: str) -> None:
    """``TPaveStats::Paint``: the box, the histogram's name over a rule, and its statistics."""
    corners = draw_box(scene, prim)
    lines = _stats_lines(prim)
    if not lines:
        return
    x1, y1, x2, y2 = corners
    row = (y2 - y1) / len(lines)
    size = title = float(lookup(prim, "fTextSize", 0.0) or 0.0)
    if not size:
        size, title = _stats_sizes(scene, prim, lines, row, x2 - x1)
    margin = float(lookup(prim, "fMargin", MARGIN) or 0.0) * (x2 - x1)
    named = int(lookup(prim, "fOptStat", 0) or 0) % 10
    y = y2 + 0.5 * row
    for line in lines:
        y -= row
        text = str(line.get("fTitle", ""))
        base = _stats_attributes(prim, line, size)
        if "=" in text and named == 0:
            _stats_pair(scene, text, (x1 + margin, x2 - margin, y), base)
            continue
        named = 0
        paint_latex(scene, text, scene.pixel(0.5 * (x1 + x2), y), dict(base, align=22, size=title))
        _outline(scene, prim, [(x1, y2 - row), (x2, y2 - row)])


#: How each of these classes draws; a legend's own painter is added by :mod:`.legend`.
PAVES = {
    "TPave": pave,
    "TPaveText": pave_text,
    "TPavesText": pave_text,
    "TPaveLabel": pave_label,
    "TPaveStats": stats_box,
}
