"""Text as ROOT puts it in a picture: a string, its pixel, its alignment, its font.

``TASImage::DrawText`` draws a ``TText``: the string is laid out by
FreeType from its point - rounded to a whole pixel of the canvas - and moved
by its alignment, measured in the string's own glyphs: centred on half the
pen's advance, right-aligned on all of it, bottom-aligned on the baseline,
top-aligned on the highest glyph's top and centred on half that height.
Nothing about the font's own ascent or descent comes into it. The same
alignment turned by the text's angle turns the string round its point.

Everything here is in the canvas's pixels, ``x`` from the left and ``y``
from the top, as ROOT counts them; :func:`glyphs` draws one string there
and is what :mod:`.latex` draws each piece of a formula with.
"""

from __future__ import annotations

import math
from typing import Any

from . import fonts
from .scene import Scene

__all__ = ["SYMBOLS", "glyphs", "pixel_size", "symbol_text"]

#: The pixel ROOT's canvas never draws beyond, as ``pixel_boundary`` clips to.
LIMIT = 30000

#: Symbol's own code points against Unicode's, for what is written in fonts 12x and 15x.
SYMBOLS: dict[str, str] = {
    **dict(zip("abcdefghijklmnopqrstuvwxyz", "αβχδεφγηιϕκλμνοπθρστυϖωξψζ")),
    **dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "ΑΒΧΔΕΦΓΗΙϑΚΛΜΝΟΠΘΡΣΤΥςΩΞΨΖ")),
    '"': "∀", "$": "∃", "'": "∋", "*": "∗", "-": "−", "@": "≅", "\\": "∴", "^": "⊥",
    "`": "‾", "~": "∼",
    **dict(zip(
        (chr(code) for code in range(0xA1, 0xFF)),
        "ϒ′≤⁄∞ƒ♣♦♥♠↔←↑→↓°±″≥×∝∂•÷≠≡≈…⏐⎯↵ℵℑℜ℘⊗⊕∅∩∪⊃⊇⊄⊂⊆∈∉∠∇®©™∏√⋅¬∧∨⇔⇐⇑⇒⇓◊⟨®©™∑⎛⎜⎝⎡⎢⎣⎧⎨⎩⎪ "
        "⟩∫⌠⎮⌡⎞⎟⎠⎤⎥⎦⎫⎬⎭",
    )),
}  # fmt: skip


def symbol_text(text: str, font: int) -> str:
    """``text`` as Unicode: itself, or for Symbol's fonts each character by Symbol's code."""
    if int(font) // 10 not in (12, 15):
        return text
    return "".join(SYMBOLS.get(char, char) for char in text)


def nint(value: float) -> int:
    """``TMath::Nint`` of a pixel, within ``pixel_boundary``'s limits."""
    return int(round(max(-LIMIT, min(LIMIT, value))))


def pixel_size(scene: Scene, size: float, font: int) -> float:
    """A ``fTextSize`` in pixels: of the pad's shorter side, or for precision 3 itself, truncated."""
    if int(font) % 10 == 3:
        return float(int(size))
    return float(size) * min(scene.whole)


def _alignment(extent: fonts.Extent, align: int, angle: float) -> tuple[int, int]:
    """``TASImage::DrawText``'s ``ftal``: how far the string moves for its alignment, turned."""
    across, up = divmod(int(align), 10)
    ax = {2: extent.advance // 2, 3: extent.advance}.get(across, 0)
    ay = {2: extent.top // 2, 3: extent.top}.get(up, 0)
    turn = math.radians(angle)
    cos, sin = math.cos(turn), math.sin(turn)
    return int(cos * ax - sin * ay) >> 6, int(sin * ax + cos * ay) >> 6


def glyphs(
    scene: Scene,
    text: str,
    at: tuple[float, float],
    font: int,
    pixels: float,
    color: Any,
    align: int = 11,
    angle: float = 0.0,
) -> Any:
    """One string drawn as ``TASImage`` draws it at canvas pixel ``at``: its artist, or ``None``.

    ``pixels`` is the size ROOT measures in, of which 0.985 is drawn.
    """
    em = fonts.draw_em(pixels)
    shown = symbol_text(text, font)
    if em <= 0 or not shown.strip():
        return None
    across, up = _alignment(fonts.extent(shown, int(font), em), align, angle)
    x, y = nint(at[0]) - across, nint(at[1]) + up
    return scene.ax.text(
        x,
        y,
        shown,
        transform=scene.display,
        fontproperties=fonts.properties(int(font), shown),
        fontsize=em * 72.0 / scene.figure.dpi,
        color=color,
        ha="left",
        va="baseline",
        rotation=angle,
        rotation_mode="anchor",
        clip_on=False,
        zorder=scene.layer(),
    )
