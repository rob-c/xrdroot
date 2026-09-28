"""``TLatex``'s ``#`` commands: symbols, accents, brackets, fractions, roots and settings.

Each is one branch of ``TLatex::Analyse``, sizing its piece and painting it
the way ROOT does: a Greek letter or a symbol is the Symbol font's character
for it; ``#bar`` and the other accents, the big brackets, ``#frac``'s bar and
``#sqrt``'s sign are lines, sized as fractions of the pad's shorter side
(``TLatex::GetHeight``); ``#color``, ``#font``, ``#scale``, ``#bf`` and
``#it`` change the spec of what they hold; ``#kern`` and ``#lower`` move it.
A command ROOT would refuse - ``#color`` with no number - is a
:class:`~.latexscan.LatexError`, and nothing of the formula is drawn.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

from .latexform import Box, Form, Mark, Spec, placed
from .latexscan import ABOVE, GREEK, SPECIAL, Found, LatexError

__all__ = ["command", "draw_shape"]

#: The Symbol font, upright and italic, that Greek letters and symbols are drawn in.
SYMBOL, SYMBOL_ITALIC = 122, 152
#: ``#bf``'s and ``#it``'s tables: the family each family becomes.
BOLD = (3, 13, 1, 6, 7, 4, 5, 10, 11, 8, 9, 12, 2, 14, 15)
ITALIC = (13, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 15, 1, 14, 12)
#: How much bigger ``#sum`` (66) and ``#int`` (79) are drawn than their text.
BIG = {66: 1.8, 79: 2.3}
#: ``sscanf``'s ``%d`` and ``%f``: the number a setting starts with.
INTEGER = re.compile(r"\s*[+-]?\d+")
REAL = re.compile(r"\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")

Handler = Callable[[Any, str, Spec, Spec, Found, bool], Box]


def _line(points: tuple[tuple[float, float], ...], spec: Spec, width: int) -> Mark:
    return Mark("line", points, spec, width=width)


def _text(x: float, y: float, spec: Spec, text: str, align: int = 11, tilt: float = 0.0) -> Mark:
    return Mark("text", ((x, y),), spec, text, align=align, tilt=tilt)


def _poly(layout: Any, points: tuple[tuple[float, float], ...], spec: Spec, scale: float) -> Mark:
    """``TLatex::DrawPolyLine``: a fill for ``scale`` 1 or more, else a line at least that wide."""
    if scale >= 1:
        return Mark("fill", points, spec)
    width = max(round(layout.height * spec.size * scale), layout.line)
    return _line(points, spec, width)


# -- symbols drawn by hand ---------------------------------------------------------------


def _boxed(layout: Any, text: str, spec: Spec, italic: bool) -> Box:
    """``#Box``: a square outline, then what follows."""
    square = layout.square(spec)
    rest = layout.analyse(text[4:], spec, italic)
    adjust = layout.height * spec.size / 20
    x1, x2, y2 = adjust, square - adjust, adjust - square
    corners = ((x1, 0.0), (x2, 0.0), (x2, y2), (x1, y2), (x1, 0.0))
    marks = tuple(_line((corners[i], corners[i + 1]), spec, layout.line) for i in range(4))
    return Box(rest.form.beside(Form(square, square, 0.0)), placed([(rest, square, 0.0)], marks))


def _circle(layout: Any, centre: tuple[float, float], radius: float, spec: Spec) -> Mark:
    """``TLatex::DrawCircle``: forty sides round ``centre``, a pixel at the least."""
    radius = max(radius, 1.0)
    points = tuple(
        (centre[0] + radius * math.cos(i * 2 * math.pi / 40), centre[1] + radius * math.sin(i * 2 * math.pi / 40))
        for i in range(41)
    )
    return _line(points, spec, layout.line)


def _odot(layout: Any, text: str, spec: Spec, italic: bool) -> Box:
    """``#odot``: a circle with a dot in it."""
    square = layout.square(spec)
    rest = layout.analyse(text[5:], spec, italic)
    adjust = layout.height * spec.size / 20
    centre = (0.6 * square, -0.3 * square - adjust)
    marks = (_circle(layout, centre, 0.62 * square, spec), _circle(layout, centre, 0.0062 * square, spec))
    return Box(rest.form.beside(Form(square, square, 0.0)), placed([(rest, 1.3 * square, 0.0)], marks))


def _lettered(layout: Any, text: str, spec: Spec, italic: bool, name: str) -> Box:
    """``#hbar``, ``#minus``, ``#plus``, ``#mp`` and ``#backslash``: a character, and a bar for ``#hbar``."""
    square = layout.square(spec)
    rest = layout.analyse(text[len(name) + 1 :], spec, italic)
    marks: tuple[Mark, ...]
    if name == "hbar":
        marks = (_text(0.0, 0.0, spec._replace(font=12), "h"),
                 _line(((0.0, -0.8 * square), (0.75 * square, -square)), spec, layout.line))  # fmt: skip
    elif name == "mp":
        marks = (_text(square, -1.25 * square, spec._replace(font=SYMBOL), "\xb1", tilt=180.0),)
    elif name == "backslash":
        marks = (_text(0.0, 0.0, spec, "\\"),)
    else:
        marks = (_text(0.0, 0.0, spec._replace(font=SYMBOL), "-" if name == "minus" else "+"),)
    return Box(rest.form.beside(Form(square, square, 0.0)), placed([(rest, square, 0.0)], marks))


def _upright(layout: Any, text: str, spec: Spec, italic: bool, name: str) -> Box:
    """``#perp`` and ``#parallel``: an upside-down T, or two upright bars."""
    square = layout.square(spec, 1.4)
    rest = layout.analyse(text[len(name) + 1 :], spec, italic)
    if name == "perp":
        x0, y1 = 0.5 * square, 0.6 * square
        bars = (((x0 - 0.48 * square, y1), (x0 + 0.48 * square, y1)), ((x0, y1), (x0, y1 - 1.3 * square)))
        form = rest.form
    else:
        y1 = 0.3 * square
        bars = tuple(((x, y1), (x, y1 - 1.3 * square)) for x in (0.15 * square, 0.45 * square))
        form = rest.form.beside(Form(square, square, 0.0))
    marks = tuple(_line(bar, spec, layout.line) for bar in bars)
    return Box(form, placed([(rest, 0.5 * square, 0.0)], marks))


# -- characters of the Symbol font -------------------------------------------------------


def _greek(layout: Any, text: str, spec: Spec, italic: bool, name: str) -> Box:
    """A Greek letter: the Symbol font's, at the letter's place in the alphabet."""
    index = GREEK.index(name)
    letter = {52: "\xa1", 53: "\xce"}.get(index, chr(97 + index - (58 if index > 25 else 0)))
    shown = spec._replace(font=SYMBOL_ITALIC if italic else SYMBOL)
    one = layout.plain(letter, shown)
    rest = layout.analyse(text[len(name) + 1 :], spec, italic)
    return Box(one.form.beside(rest.form), placed([(rest, one.form.width, 0.0), (one, 0.0, 0.0)]))


def _special(layout: Any, text: str, spec: Spec, italic: bool, name: str) -> Box:
    """A symbol: the Symbol font's character for it, ``#sum`` and ``#int`` drawn bigger."""
    index = SPECIAL.index(name)
    shown = spec._replace(font=SYMBOL_ITALIC if italic else SYMBOL)
    letter = {75: "\xc5", 76: "\xe5", 80: '"', 81: "$"}.get(index, chr(0xA3 + index))
    if index in (75, 76):
        shown = spec  # the Angstrom is the text's own
    if index in BIG:
        shown = shown.sized(spec.size * BIG[index])
    one = layout.plain(letter, shown)
    form, drop = one.form, 0.0
    if index in BIG:
        form = Form(form.width, form.over * 0.45, form.over * 0.45)
        drop = form.under / 2
    rest = layout.analyse(text[len(name) + 1 :], spec, italic)
    return Box(form.beside(rest.form), placed([(rest, form.width, 0.0), (one, 0.0, drop)]))


# -- accents -------------------------------------------------------------------------------


def _accent_marks(layout: Any, name: str, form: Form, spec: Spec) -> tuple[Mark, ...]:
    """The lines, dots or tilde an accent draws over text of ``form``, from its baseline's left."""
    sub = layout.height * spec.size / 14
    width, top = form.width, -sub - form.over
    middle = width / 2
    if name in ("dot", "ddot"):
        return _dots(layout, name, middle, top, sub, spec)
    if name == "vec":
        head = layout.height * spec.size / 8
        mid = top - head
        return (
            _poly(layout, ((0.0, mid), (width, mid)), spec, 0.03),
            _poly(layout, ((width - 2 * head, mid - head), (width, mid), (width - 2 * head, mid + head)), spec, 0.03),
        )
    if name == "tilde":
        return (_text(middle, -form.over, spec.sized(0.9 * spec.size), "~", align=22),)
    strokes = {
        "bar": ((0.0, top), (width, top)),
        "hat": ((middle - width / 3, top), (middle, top - 2 * sub), (middle + width / 3, top)),
        "acute": ((middle, top), (middle + 3 * sub, top - 2.5 * sub)),
        "grave": ((middle + sub, top), (middle - sub, top - 2 * sub)),
        "check": ((middle - 2 * sub, top - 2 * sub), (middle, top), (middle + 2 * sub, top - 2 * sub)),
        "slash": ((0.8 * width, -form.over - sub), (0.3 * width, -form.over - sub + form.height + 2 * sub)),
    }
    return (_poly(layout, strokes[name], spec, 0.03),)


def _dots(layout: Any, name: str, middle: float, top: float, sub: float, spec: Spec) -> tuple[Mark, ...]:
    """``#dot``'s dot, or ``#ddot``'s two, as filled squares."""
    size = max(0.5 * layout.line, 0.5 * sub)
    centres = (middle,) if name == "dot" else (middle - 1.5 * sub, middle + 1.5 * sub)
    mid = top - size
    return tuple(
        _poly(layout, ((x - size, mid + size), (x - size, mid - size), (x + size, mid - size),
                       (x + size, mid + size), (x - size, mid + size)), spec, 10.0)
        for x in centres
    )  # fmt: skip


def _accent(layout: Any, text: str, spec: Spec, italic: bool, name: str) -> Box:
    """An accent over what follows it, a third of its size higher (a quarter for ``#vec``)."""
    under = layout.analyse(text[len(name) + 1 :], spec, italic)
    form = under.form
    marks = _accent_marks(layout, name, form, spec)
    raised = layout.height * spec.size / (4 if name == "vec" else 3)
    return Box(Form(form.width, form.over + raised, form.under), placed([(under, 0.0, 0.0)], marks))


# -- big brackets ---------------------------------------------------------------------------


def _bracketed(layout: Any, text: str, spec: Spec, italic: bool, name: str) -> Box:
    """``#[]{}``, ``#||{}`` and ``#{}{}``: brackets as tall as what they hold."""
    inner = layout.analyse(text[3:], spec, italic)
    form = inner.form
    unit = layout.height * spec.size / 4
    if name == "{}":
        return _curly(layout, inner, spec, min(form.height / 8, unit))
    half, top, bottom = unit / 2, -form.over, form.under
    right = half + form.width + 2 * unit
    bars = [((half, top), (half, bottom)), ((right, top), (right, bottom))]
    if name == "[]":
        bars += [((half, top), (half + unit, top)), ((half, bottom), (half + unit, bottom)),
                 ((right, top), (right - unit, top)), ((right, bottom), (right - unit, bottom))]  # fmt: skip
    marks = tuple(_line(bar, spec, layout.line) for bar in bars)
    return Box(Form(form.width + 3 * unit, form.over, form.under), placed([(inner, half + unit, 0.0)], marks))


def _curly(layout: Any, inner: Box, spec: Spec, tip: float) -> Box:
    """``#{}{}``: braces, their points at the middle of what they hold."""
    form, unit = inner.form, tip
    half = layout.height * spec.size / 8
    top, bottom, mid = -form.over, form.under, (form.under - form.over) / 2
    left, right = half + tip, half + tip + form.width + 2 * unit
    bars = (
        ((left, top), (left, mid - tip)), ((left, mid + tip), (left, bottom)),
        ((left, top), (left + unit, top)), ((left, bottom), (left + unit, bottom)),
        ((half, mid), (left, mid - tip)), ((half, mid), (left, mid + tip)),
        ((right, top), (right, mid - tip)), ((right, mid + tip), (right, bottom)),
        ((right - unit, top), (right, top)), ((right - unit, bottom), (right, bottom)),
        ((right, mid - tip), (right + tip, mid)), ((right, mid + tip), (right + tip, mid)),
    )  # fmt: skip
    marks = tuple(_line(bar, spec, layout.line) for bar in bars)
    return Box(Form(form.width + 5 * tip, form.over, form.under), placed([(inner, unit + tip + half, 0.0)], marks))


def _arc(layout: Any, centre: tuple[float, float], radii: tuple[float, float], turn: float, spec: Spec) -> Mark:
    """``TLatex::DrawParenthesis``: seventy degrees of an ellipse round ``turn``."""
    first, rx, ry = math.radians(turn - 35), max(radii[0], 1.0), max(radii[1], 1.0)
    step = math.radians(70) / 40
    points = tuple((centre[0] + rx * math.cos(first + i * step), centre[1] + ry * math.sin(first + i * step))
                   for i in range(41))  # fmt: skip
    return _line(points, spec, layout.line)


def _parenthesis(layout: Any, text: str, spec: Spec, italic: bool) -> Box:
    """``#(){}``: parentheses as tall as what they hold."""
    inner = layout.analyse(text[3:], spec, italic)
    form = inner.form
    unit = layout.height * spec.size / 4
    tall = form.height
    wide = tall * 2 / 3
    bend = wide * (1 - math.cos(math.radians(35)))
    mid = -(form.over - form.under) / 2
    marks = (
        _arc(layout, (unit / 2 + wide, mid), (wide, tall), 180.0, spec),
        _arc(layout, (5 * unit / 2 + 2 * bend + form.width - wide, mid), (wide, tall), 360.0, spec),
    )
    form = Form(form.width + 3 * unit + 2 * bend, form.over, form.under)
    return Box(form, placed([(inner, 3 * unit / 2 + bend, 0.0)], marks))


# -- two lines, and roots ---------------------------------------------------------------------


def _stacked(layout: Any, text: str, spec: Spec, found: Found, italic: bool, name: str) -> Box:
    """``#frac`` and ``#splitline``: one piece over the other, a fraction centred and barred."""
    if found.curly_curly == -1:
        raise LatexError("Missing denominator for #frac" if name == "frac" else "Missing second line for #splitline")
    start = found.command[1] + len(name) + 2
    top = layout.analyse(text[start : found.curly_curly], spec, italic)
    bottom = layout.analyse(text[found.curly_curly + 2 : -1], spec, italic)
    one, two = top.form, bottom.form
    eighth = layout.height * spec.size / 8
    shift_top = shift_bottom = 0.0
    marks: tuple[Mark, ...] = ()
    if name == "frac":
        shift_top, shift_bottom = max(two.width - one.width, 0.0) / 2, max(one.width - two.width, 0.0) / 2
        marks = (_line(((0.0, -2 * eighth), (max(one.width, two.width), -2 * eighth)), spec, layout.line),)
    form = Form(max(one.width, two.width), one.height + 3 * eighth, two.height - eighth)
    parts = [(bottom, shift_bottom, two.over - eighth), (top, shift_top, -one.under - 3 * eighth)]
    return Box(form, placed(parts, marks))


def _root(layout: Any, text: str, spec: Spec, small: Spec, found: Found, italic: bool) -> Box:
    """``#sqrt``: the sign and its bar over what it holds, with an index in brackets."""
    at, unit = found.command[1], layout.height * spec.size
    if found.square_curly > -1:
        return _indexed_root(layout, text, spec, small, found, italic)
    inner = layout.analyse(text[at + 5 :], spec, italic)
    form = inner.form
    x1, x2 = unit * 2 / 5, unit / 2 + form.width
    y1, y2 = -form.over, form.under
    y3 = y1 - unit / 4
    dx = (y2 - y3) / 8
    thick = form.over > 12
    marks = (
        _line(((x1 - 2 * dx, y1), (x1 - dx, y2)), spec, max(2, int(dx / 2)) if thick else 1),
        _line(((x1 - dx, y2), (x1, y3), (x2, y3)), spec, max(1, int(dx / 4)) if thick else 1),
    )
    return Box(Form(form.width + unit / 2, form.over + unit / 4, form.under), placed([(inner, unit / 2, 0.0)], marks))


def _indexed_root(layout: Any, text: str, spec: Spec, small: Spec, found: Found, italic: bool) -> Box:
    """``#sqrt[n]{...}``: the index small, over the sign's first stroke."""
    at, unit = found.command[1], layout.height * spec.size
    index = layout.analyse(text[at + 6 : found.square_curly], small, italic)
    inner = layout.analyse(text[found.square_curly + 1 :], spec, italic)
    one, two = index.form, inner.form
    step = max(unit / 2, one.width)
    beyond = step + unit / 10
    y1, y2 = -two.over, two.under
    y3 = y1 - unit / 4
    strokes = (((0.0, y1), (step, y2)), ((step, y2), (step, y3)), ((step, y3), (beyond + two.width, y3)))
    marks = tuple(_line(stroke, spec, layout.line) for stroke in strokes)
    form = Form(two.width + unit / 10 + step, two.over + one.height + unit / 4, two.under)
    return Box(form, placed([(inner, beyond, 0.0), (index, 0.0, -two.over - one.under)], marks))


# -- settings -----------------------------------------------------------------------------------


def _setting(text: str, found: Found, name: str, pattern: re.Pattern[str]) -> float:
    """The number in a command's brackets, read as ``sscanf`` reads it, or ROOT's refusal."""
    if found.square_curly == -1:
        raise LatexError(f"Missing setting. Syntax is #{name}[nb]{{ ... }}")
    given = text[found.command[1] + len(name) + 2 : found.square_curly]
    number = pattern.match(given)
    if number is None:
        raise LatexError(f"Invalid setting. Syntax is #{name}[nb]{{ ... }}")
    return float(number.group())


def _set(layout: Any, text: str, spec: Spec, found: Found, italic: bool, name: str) -> Box:
    """``#color``, ``#font``, ``#scale``, ``#url``, ``#kern`` and ``#lower`` on what follows."""
    after = text[found.square_curly + 1 :] if found.square_curly > -1 else ""
    if name == "url":
        _setting(text, found, name, re.compile(r".*"))
        return layout.analyse(after, spec, italic)
    value = _setting(text, found, name, INTEGER if name in ("color", "font") else REAL)
    changed = {
        "color": lambda: spec._replace(color=int(value)),
        "font": lambda: spec._replace(font=int(value)),
        "scale": lambda: spec.sized(value * spec.size),
    }
    if name in changed:
        return layout.analyse(after, changed[name](), italic)
    inner = layout.analyse(after, spec, italic)
    form = inner.form
    if name == "kern":
        moved = value * form.width
        return Box(Form(form.width + moved, form.over, form.under), placed([(inner, moved, 0.0)]))
    moved = value * form.height
    return Box(Form(form.width, form.over + moved, form.under + moved), placed([(inner, 0.0, moved)]))


def _face(layout: Any, text: str, spec: Spec, italic: bool, name: str) -> Box:
    """``#bf``, ``#it`` and ``#mbox``: what follows in the bold or italic of its font, or as it is."""
    if name == "mbox":
        return layout.analyse(text[5:], spec, italic)
    table = BOLD if name == "bf" else ITALIC
    family = spec.font // 10
    if 1 <= family <= len(table):
        family = table[family - 1]
    return layout.analyse(text[3:], spec._replace(font=family * 10 + spec.font % 10), italic ^ (name == "it"))


def command(layout: Any, text: str, spec: Spec, small: Spec, found: Found, italic: bool) -> Box:
    """The piece a ``#`` command begins, by the command."""
    name = found.command[0]
    if name in GREEK:
        return _greek(layout, text, spec, italic, name)
    if name in ABOVE:
        return _accent(layout, text, spec, italic, name)
    if name in SPECIAL:
        return _special(layout, text, spec, italic, name)
    return BY_NAME[name](layout, text, spec, small, found, italic, name)


BY_NAME: dict[str, Callable[..., Box]] = {
    "Box": lambda layout, text, spec, small, found, italic, name: _boxed(layout, text, spec, italic),
    "odot": lambda layout, text, spec, small, found, italic, name: _odot(layout, text, spec, italic),
    **{
        name: lambda layout, text, spec, small, found, italic, name: _lettered(layout, text, spec, italic, name)
        for name in ("hbar", "minus", "plus", "mp", "backslash")
    },
    **{
        name: lambda layout, text, spec, small, found, italic, name: _upright(layout, text, spec, italic, name)
        for name in ("perp", "parallel")
    },
    **{
        name: lambda layout, text, spec, small, found, italic, name: _bracketed(layout, text, spec, italic, name)
        for name in ("[]", "||", "{}")
    },
    "()": lambda layout, text, spec, small, found, italic, name: _parenthesis(layout, text, spec, italic),
    **{
        name: lambda layout, text, spec, small, found, italic, name: _stacked(layout, text, spec, found, italic, name)
        for name in ("frac", "splitline")
    },
    "sqrt": lambda layout, text, spec, small, found, italic, name: _root(layout, text, spec, small, found, italic),
    **{
        name: lambda layout, text, spec, small, found, italic, name: _set(layout, text, spec, found, italic, name)
        for name in ("color", "font", "scale", "url", "kern", "lower")
    },
    **{
        name: lambda layout, text, spec, small, found, italic, name: _face(layout, text, spec, italic, name)
        for name in ("bf", "it", "mbox")
    },
}


# -- drawing -----------------------------------------------------------------------------------


def draw_shape(scene: Any, mark: Mark, points: list[tuple[int, int]], color: Any) -> None:
    """A formula's line or filled shape, at whole pixels of the canvas."""
    from matplotlib.patches import Polygon

    from .raster import add_line

    if mark.kind == "fill":
        scene.ax.add_artist(Polygon(points, closed=True, transform=scene.display, clip_on=False,
                                    zorder=scene.layer(), facecolor=color, edgecolor=color, linewidth=0.0))
        return
    add_line(scene, points, color, mark.width)
