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

import functools
import math
import warnings
from typing import Any

from . import fonts
from .scene import Scene

__all__ = ["SYMBOLS", "glyphs", "pixel_size", "symbol_text"]

#: The pixel ROOT's canvas never draws beyond, as ``pixel_boundary`` clips to.
LIMIT = 30000

#: Symbol's own code points against Unicode's, for what is written in fonts 12x and 15x.
SYMBOLS: dict[str, str] = {
    **dict(zip("abcdefghijklmnopqrstuvwxyz", "αβχδεφγηιϕκλμνοπθρστυϖωξψζ", strict=False)),
    **dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "ΑΒΧΔΕΦΓΗΙϑΚΛΜΝΟΠΘΡΣΤΥςΩΞΨΖ", strict=False)),
    # The glyphs themselves are the point of this table, look-alikes and all.
    '"': "∀", "$": "∃", "'": "∋", "*": "∗", "-": "−", "@": "≅",  # noqa: RUF001
    "\\": "∴", "^": "⊥", "`": "‾", "~": "∼",  # noqa: RUF001
    **dict(zip(
        (chr(code) for code in range(0xA1, 0xFF)),
        "ϒ′≤⁄∞ƒ♣♦♥♠↔←↑→↓°±″≥×∝∂•÷≠≡≈…⏐⎯↵ℵℑℜ℘⊗⊕∅∩∪⊃⊇⊄⊂⊆∈∉∠∇®©™∏√⋅¬∧∨⇔⇐⇑⇒⇓◊⟨®©™∑⎛⎜⎝⎡⎢⎣⎧⎨⎩⎪ "  # noqa: RUF001
        "⟩∫⌠⎮⌡⎞⎟⎠⎤⎥⎦⎫⎬⎭", strict=False,
    )),
}  # fmt: skip


def symbol_text(text: str, font: int) -> str:
    """``text`` as Unicode: itself, or for Symbol's fonts each character by Symbol's code."""
    if int(font) // 10 not in (12, 15):
        return text
    return "".join(SYMBOLS.get(char, char) for char in text)


def nint(value: float) -> int:
    """``TMath::Nint`` of a pixel, within ``pixel_boundary``'s limits."""
    # int() stays: a numpy float rounds to a numpy float, not to an int.
    return int(round(max(-LIMIT, min(LIMIT, value))))  # noqa: RUF046


def pixel_size(scene: Scene, size: float, font: int) -> float:
    """A ``fTextSize`` in pixels: of the pad's shorter side, or for precision 3 itself, cut."""
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
    drawn = _glyph_text()(
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
    scene.ax.add_artist(drawn)
    return drawn


#: What matplotlib says of a letter a face has no glyph for, which it asks the Symbol font
#: for when measuring a line's height by "lp" - before 3.7 without falling back to another.
MEASURING = r"Glyph (108|112) \((l|p)\) missing"


@functools.cache
def _glyph_text() -> Any:
    """matplotlib's ``Text``, drawn without its warning that the Symbol font has no "lp".

    matplotlib sizes each line of a text by the height of "lp" in the text's
    own face; ROOT's Symbol has Greek where those letters are, and a Symbol
    with no Latin at all leaves matplotlib before 3.7 warning of the letters
    it never draws.
    """
    from matplotlib.text import Text

    class GlyphText(Text):  # type: ignore[misc]
        """A string of ROOT's glyphs, one face and size, placed at a pixel."""

        def draw(self, renderer: Any) -> None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", MEASURING, UserWarning)
                super().draw(renderer)

    return GlyphText
