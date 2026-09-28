"""ROOT's fonts: which face each ``fTextFont`` is, how big, and how wide a string is in it.

ROOT draws text in a picture with FreeType, from fonts it ships: font 4x is
TeX Gyre Heros, a Helvetica, 1x, 2x, 3x and 13x the FreeSerif Times, 8x to
11x FreeMono's Courier, and 12x and 15x Symbol, whose Greek letters sit
where the Latin ones would. None of those is taken from ROOT here; each is
the first installed of a list of faces with the same widths - TeX Gyre
Heros, Nimbus Sans, Helvetica, Arial, Liberation Sans, then matplotlib's
own DejaVu Sans, which is always there and a little wider - and Symbol is
matplotlib's STIX, its Greek written as Unicode. Only TrueType files are
taken (``.ttf`` and ``.ttc``), which every matplotlib writes into a PDF;
an OpenType Heros installed as ``.otf`` is passed over for the next.

A size is ROOT's too. A text size is a fraction of the pad's shorter side
in pixels (or, for a font of precision 3, pixels); FreeType is given
``int(size * 0.93376068 + 0.5)`` pixels to the em - ``TTF::SetTextSize``'s
scale - when a string is measured, and 0.985 of that size when
``TASImage`` draws it, so text is drawn a little smaller than it was laid
out for, as ROOT's is. A string is measured as ``TTF::LayoutGlyphs``
measures it: unhinted advances, kerning, and the box round the glyphs in
whole pixels.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

__all__ = ["Extent", "draw_em", "extent", "face", "measure_em", "properties"]

#: FreeType's scale in ``TTF::SetTextSize``: a size's pixels to an em's.
TTF_SCALE = 0.93376068
#: How much smaller ``TASImage::DrawText`` draws text than it is measured.
DRAW_SCALE = 0.985

#: Faces with Helvetica's widths, the first installed of which is ROOT's 4x.
SANS = (
    "TeX Gyre Heros", "Nimbus Sans", "Nimbus Sans L", "Helvetica", "Arial",
    "Liberation Sans", "FreeSans", "DejaVu Sans",
)  # fmt: skip
#: Faces with Times's widths, for ROOT's FreeSerif.
SERIF = (
    "FreeSerif", "TeX Gyre Termes", "Nimbus Roman", "Nimbus Roman No9 L", "Times New Roman",
    "Times", "Liberation Serif", "DejaVu Serif",
)  # fmt: skip
#: Faces with Courier's widths, for ROOT's FreeMono.
MONO = (
    "FreeMono", "TeX Gyre Cursor", "Nimbus Mono PS", "Nimbus Mono L", "Courier New",
    "Liberation Mono", "Courier", "DejaVu Sans Mono",
)  # fmt: skip
#: The faces ROOT's Symbol font is drawn in: a Symbol of Adobe's widths if one is
#: installed, else matplotlib's own STIX, its Greek a Times's.
SYMBOL = ("Symbol", "Standard Symbols PS", "STIXGeneral")
#: Each ``fTextFont // 10``: the faces it may be, and whether italic and bold.
FONTS: dict[int, tuple[tuple[str, ...], bool, bool]] = {
    0: (SANS, False, True),
    1: (SERIF, True, False),
    2: (SERIF, False, True),
    3: (SERIF, True, True),
    4: (SANS, False, False),
    5: (SANS, True, False),
    6: (SANS, False, True),
    7: (SANS, True, True),
    8: (MONO, False, False),
    9: (MONO, True, False),
    10: (MONO, False, True),
    11: (MONO, True, True),
    12: (SYMBOL, False, False),
    13: (SERIF, False, False),
    14: (SANS, False, False),
    15: (SYMBOL, True, False),
}
#: The files a face may be read from: TrueType, which every matplotlib embeds in a PDF.
TRUETYPE = (".ttf", ".ttc")
#: The weights at and above which a face is bold, as matplotlib numbers them.
BOLD = 600


class Extent(NamedTuple):
    """A string laid out as ``TTF::LayoutGlyphs`` lays it out, in pixels.

    ``width`` is ``TTF::GetTextExtent``'s - the right of the glyphs' box, and
    the width of any blanks after them - and ``ascent`` and ``descent`` the
    box's top above the baseline and bottom below it. ``advance`` and
    ``top`` are ``fgWidth`` and ``fgAscent``, the pen's advance and the
    highest glyph's bearing, in FreeType's 64ths of a pixel, which is what
    ``TASImage`` aligns a string by.
    """

    width: int
    ascent: int
    descent: int
    advance: int
    top: int


def _weight(entry: Any) -> int:
    weight = entry.weight
    if isinstance(weight, str):
        return BOLD if "bold" in weight.lower() or weight.lower() == "heavy" else 400
    return int(weight)


def _matches(entry: Any, family: str, italic: bool, bold: bool) -> bool:
    """Whether a face matplotlib found is ``family`` in the style asked for.

    A collection (``.ttc``) is read by its first face, its regular one, so
    it stands only for that.
    """
    suffix = Path(entry.fname).suffix.lower()
    return (
        entry.name == family
        and (entry.style != "normal") == italic
        and (_weight(entry) >= BOLD) == bold
        and suffix in TRUETYPE
        and (suffix != ".ttc" or not (italic or bold))
    )


@functools.lru_cache(maxsize=None)
def face(code: int) -> str:
    """The file ``fTextFont`` ``code`` is drawn from: the first of its faces installed.

    A face with no file of the style asked for falls to the next; when none
    has one, matplotlib's own nearest DejaVu is taken.
    """
    from matplotlib import font_manager

    faces, italic, bold = FONTS.get(int(code) // 10, FONTS[4])
    entries = font_manager.fontManager.ttflist
    for family in faces:
        found = sorted(entry.fname for entry in entries if _matches(entry, family, italic, bold))
        if found:
            return found[0]
    prop = font_manager.FontProperties(
        family="DejaVu Sans", style="italic" if italic else "normal", weight="bold" if bold else "normal"
    )
    return str(font_manager.findfont(prop))


#: Where a Symbol character its face has not is looked for: matplotlib's STIX, and
#: its face of the pieces of big brackets.
SYMBOL_FALLBACKS = ("STIXGeneral", "STIXSizeOneSym")


@functools.lru_cache(maxsize=None)
def _fallback(family: str) -> str:
    from matplotlib import font_manager

    return str(font_manager.findfont(font_manager.FontProperties(family=family)))


def _has(path: str, text: str) -> bool:
    font = _font(path)
    return all(font.get_char_index(ord(char)) for char in text)


@functools.lru_cache(maxsize=4096)
def face_of(code: int, text: str) -> str:
    """The file ``text`` in ``fTextFont`` ``code`` is drawn from.

    A Symbol character the installed Symbol has no glyph for is drawn from
    the first of matplotlib's STIX faces that has it.
    """
    own = face(code)
    if int(code) // 10 not in (12, 15) or not text or _has(own, text):
        return own
    return next((path for path in map(_fallback, SYMBOL_FALLBACKS) if _has(path, text)), own)


def properties(code: int, text: str = "") -> Any:
    """The ``FontProperties`` matplotlib draws ``text`` in ``fTextFont`` ``code`` with."""
    from matplotlib.font_manager import FontProperties

    return FontProperties(fname=face_of(code, text))


def measure_em(pixels: float) -> int:
    """The em FreeType is given for text ``pixels`` big, as ``TTF::SetTextSize`` rounds it."""
    scaled = np.float32(pixels) * np.float32(TTF_SCALE) + np.float32(0.5)
    return int(scaled)


def draw_em(pixels: float) -> int:
    """The em ``TASImage::DrawText`` draws text ``pixels`` big at: 0.985 of it, measured."""
    return measure_em(float(np.float32(pixels) * np.float32(DRAW_SCALE)))


def _flags() -> tuple[Any, Any]:
    """FreeType's no-hinting load flag and unfitted kerning mode, in this matplotlib's spelling."""
    from matplotlib import ft2font

    if hasattr(ft2font, "LoadFlags"):
        return ft2font.LoadFlags.NO_HINTING, ft2font.Kerning.UNFITTED
    return ft2font.LOAD_NO_HINTING, ft2font.KERNING_UNFITTED  # pragma: no cover - before 3.10


@functools.lru_cache(maxsize=None)
def _font(path: str) -> Any:
    """A FreeType face of its own for measuring, laid out at one pixel to the pixel.

    matplotlib's own faces are stretched eight times across for hinting
    until 3.11, which deprecates the stretch; this one never is.
    """
    import warnings

    from matplotlib.ft2font import FT2Font

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return FT2Font(path, hinting_factor=1)
        except TypeError:  # pragma: no cover - matplotlib 3.13 has no stretch to undo
            return FT2Font(path)


def _floor(value: int) -> int:
    return value // 64


def _ceil(value: int) -> int:
    return -(-value // 64)


@functools.lru_cache(maxsize=65536)
def extent(text: str, code: int, em: int) -> Extent:
    """``text`` in ``fTextFont`` ``code`` at ``em`` pixels, laid out as ``TTF`` lays it out.

    Characters the face has no glyph for are left out, as ROOT leaves them.
    """
    font = _font(face_of(code, text))
    font.set_size(float(em), 72.0)
    no_hinting, unfitted = _flags()
    pen = top = 0
    box = [32000, 32000, -32000, -32000]
    previous = 0
    for char in text:
        index = font.get_char_index(ord(char))
        if not index:
            continue
        if previous:
            pen += int(font.get_kerning(previous, index, unfitted))
        previous = index
        glyph = font.load_glyph(index, no_hinting)
        xmin, ymin, xmax, ymax = glyph.bbox
        box = [
            min(box[0], _floor(pen + xmin)),
            min(box[1], _floor(ymin)),
            max(box[2], _ceil(pen + xmax)),
            max(box[3], _ceil(ymax)),
        ]
        pen += int(glyph.horiAdvance)
        top = max(top, int(glyph.horiBearingY))
    if box[0] > box[2]:
        return Extent(0, 0, 0, pen, top)
    blanks = len(text) - len(text.rstrip(" "))
    trailing = _floor(int(font.load_char(32, no_hinting).horiAdvance)) * blanks if blanks else 0
    width = box[2] + max(-box[0], 0) + trailing
    return Extent(width, box[3], abs(box[1]), pen, top)
