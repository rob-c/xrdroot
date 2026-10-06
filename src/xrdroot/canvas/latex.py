"""ROOT's ``TLatex``, laid out as ROOT lays it out and drawn string by string.

ROOT spells mathematics with ``#`` where TeX spells it with a backslash -
``#mu^{+}#mu^{-}``, ``#sqrt{s} = 13 TeV``, ``p_{T} [GeV]`` - and draws it
itself: ``TLatex::Analyse`` splits a formula at its first operator, sizes
each piece in pixels from the glyphs of its font, and paints each run of
plain text as a string of its own, at the pixel the layout puts it. This is
that layout, piece for piece: scripts two thirds the size and raised or
lowered by ROOT's fractions, Greek and symbols from the Symbol font, bars
and fractions and roots drawn as lines, ``#color``, ``#font``, ``#bf`` and
``#it`` as they change a piece's spec. What is not an operator - ``_``
without a brace after it, ``x^2``, a word after ``#`` that ROOT does not know
- is text, as it is in ROOT. A formula ROOT refuses, an operator left open,
draws nothing, as in ROOT.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from . import fonts, latexmarks
from .latexform import EMPTY, Box, Form, Mark, Spec, placed
from .latexscan import Found, LatexError, check, scan
from .scene import Scene
from .text import glyphs, nint, symbol_text

__all__ = ["Layout", "LatexError", "paint_latex", "plain_text"]

#: ``TLatex``'s ``fFactorSize``: how much smaller a script is than what it is on.
FACTOR_SIZE = 1.5
#: ``fFactorPos``: how far up or down a script moves, as a fraction of what it is on.
FACTOR_POS = 0.6
#: ``fLimitFactorSize``: how many times a script may shrink before it stops shrinking.
LIMIT_FACTOR = 3
#: An ``@`` escaping a brace or bracket, which a piece of text does not show.
ESCAPE = re.compile(r"@(?=[{}\[\]])")


def plain_text(text: str) -> str:
    """A run of text as it is drawn: its escaped braces and brackets without their ``@``."""
    return ESCAPE.sub("", text)


def _stripped(text: str) -> str:
    """``text`` without the braces round all of it, as many pairs as there are."""
    while len(text) >= 2 and text[0] == "{" and text[-1] == "}" and _one_group(text):
        text = text[1:-1]
    return text


def _one_group(text: str) -> bool:
    """Whether the first brace of ``text`` closes only at its end."""
    level = 0
    for at, char in enumerate(text):
        if char == "{" and not (at > 0 and text[at - 1] == "@"):
            level += 1
        if char == "}" and text[at - 1] != "@":
            level -= 1
        if level == 0 and at < len(text) - 2:
            return False
    return True


class Layout:
    """One formula being laid out, in a pad of ``basis`` pixels on its shorter side.

    ``height`` is ``TLatex::GetHeight``, the same side unrounded, which the
    operators' spacing is a fraction of; ``origin`` the size the formula
    started at, below which a script's script stops shrinking; ``line`` the
    ``TLatex``'s line width, which its bars and roots are drawn with.
    """

    def __init__(self, basis: float, height: float, origin: float, line: int = 2) -> None:
        """A layout with nothing analysed yet; each piece is analysed once, then remembered."""
        self.basis = basis
        self.height = height
        self.origin = origin
        self.line = line
        self._boxes: dict[tuple[str, Spec, bool], Box] = {}

    def analyse(self, text: str, spec: Spec, italic: bool = False) -> Box:
        """``TLatex::Analyse`` of ``text``: its form and how it paints."""
        key = (text, spec, italic)
        if key not in self._boxes:
            self._boxes[key] = self._analysed(_stripped(text), spec, italic)
        return self._boxes[key]

    def _analysed(self, text: str, spec: Spec, italic: bool) -> Box:
        """One piece split at its first operator, by the first of :data:`OPERATORS` it has."""
        if not text:
            return EMPTY
        found = scan(text)
        small = spec.sized(self.indice(spec.size))
        for applies, operator in OPERATORS:
            if applies(found, text):
                return operator(self, text, spec, small, found, italic)
        return self.plain(text, spec)

    def indice(self, size: float) -> float:
        """The size of a script on text ``size`` big: smaller, down to its limit."""
        smaller = size / FACTOR_SIZE
        if smaller < self.origin / math.exp(LIMIT_FACTOR * math.log(FACTOR_SIZE)) - 0.001:
            return size
        return smaller

    def em(self, spec: Spec) -> float:
        """A spec's size in the pad's pixels."""
        return spec.size * self.basis

    def measure(self, text: str, spec: Spec) -> Form:
        """A run of plain text's form: ``GetTextExtent``'s width, and its glyphs' box."""
        shown = symbol_text(plain_text(text), spec.font)
        found = fonts.extent(shown, spec.font, fonts.measure_em(self.em(spec)))
        return Form(float(found.width), float(found.ascent), float(found.descent))

    def plain(self, text: str, spec: Spec) -> Box:
        """A run of text with no operator in it, painted at its baseline's left."""
        shown = plain_text(text)

        def paint(x: float, y: float, marks: list[Mark]) -> None:
            """The run as one string at ``(x, y)``."""
            marks.append(Mark("text", ((x, y),), spec, shown))

        return Box(self.measure(text, spec), paint)

    def square(self, spec: Spec, divide: float = 2.0) -> float:
        """``GetHeight() * spec.fSize / divide``: the unit a drawn symbol is sized in."""
        return self.height * spec.size / divide


# -- the operators that split a piece ------------------------------------------------


def _split(layout: Layout, text: str, spec: Spec, _small: Spec, found: Found, italic: bool) -> Box:
    """A ``}`` with more after it: the piece up to it, and the rest beside it."""
    first = layout.analyse(text[: found.close_curly + 1], spec, italic)
    rest = layout.analyse(text[found.close_curly + 1 :], spec, italic)
    parts = [(rest, first.form.width, 0.0), (first, 0.0, 0.0)]
    return Box(first.form.beside(rest.form), placed(parts))


def _base(layout: Layout, text: str, at: int, spec: Spec, italic: bool) -> tuple[Form, Box]:
    """What a script is on: its form as measured, and the box painted.

    ``{}`` right before a script puts it before what follows, as a chemical
    element's numbers are: ROOT measures the ``{}`` as an ``I`` and paints it
    as a blank.
    """
    if at >= 2 and text[at - 2 : at] == "{}":
        measured = layout.analyse(text[: at - 2] + "I", spec, italic)
        return measured.form, layout.analyse(text[: at - 2] + " ", spec, italic)
    base = layout.analyse(text[:at], spec, italic)
    return base.form, base


#: How far over and under an ``#int`` (1) or a ``#sum`` (2) its limits go: under, over.
LIMITS = {1: (0.8, 1.75), 2: (0.9, 1.75)}


def _both(layout: Layout, text: str, spec: Spec, small: Spec, found: Found, italic: bool) -> Box:
    """A superscript and a subscript on the same text, the first found first."""
    low, high = sorted((found.power, found.under))
    one, base = _base(layout, text, low, spec, italic)
    first = layout.analyse(text[low + 1 : high], small, italic)
    second = layout.analyse(text[high + 1 :], small, italic)
    power_first = found.power < found.under
    if found.above_place:
        return _both_limits(one, base, first, second, power_first, found.above_place)
    two, three = first.form, second.form
    if power_first:
        up, down = -one.over * FACTOR_POS - two.under, one.under + three.over * FACTOR_POS
        form = Form(
            one.width + max(two.width, three.width),
            one.over * FACTOR_POS + two.height,
            one.under + three.height - three.over * (1 - FACTOR_POS),
        )
    else:
        up, down = one.under + two.over * FACTOR_POS, -one.over * FACTOR_POS - three.under
        form = Form(
            one.width + max(two.width, three.width),
            one.over * FACTOR_POS + three.height,
            one.under + two.height - two.over * (1 - FACTOR_POS),
        )
    return Box(form, placed([(second, one.width, down), (first, one.width, up), (base, 0.0, 0.0)]))


def _both_limits(
    one: Form, base: Box, first: Box, second: Box, power_first: bool, place: int
) -> Box:
    """Limits over and under an ``#int`` or ``#sum``, each centred on it."""
    under, over = LIMITS[place]
    two, three = first.form, second.form
    widest = max(one.width, two.width, three.width)
    if power_first:
        up, down = -one.over * over - two.under, one.under * under + three.over
        form = Form(widest, one.over * over + two.height, one.under * under + three.height)
    else:
        up, down = one.under * under + two.over, -one.over * over - three.under
        form = Form(widest, one.over * over + three.height, one.under * under + two.height)
    return Box(form, placed([
        (second, (widest - three.width) / 2, down),
        (first, (widest - two.width) / 2, up),
        (base, (widest - one.width) / 2, 0.0),
    ]))  # fmt: skip


def _centred(one: Form, script: Form) -> tuple[int, int]:
    """How far a limit and what it is on move to be centred on each other, in whole pixels."""
    if script.width > one.width:
        return int((script.width - one.width) / 2), 0
    return 0, int((one.width - script.width) / 2)


def _power(layout: Layout, text: str, spec: Spec, small: Spec, found: Found, italic: bool) -> Box:
    """A superscript: raised by ``fFactorPos`` of what it is on, or over an ``#int``."""
    one, base = _base(layout, text, found.power, spec, italic)
    script = layout.analyse(text[found.power + 1 :], small, italic)
    two = script.form
    if found.above_place:
        pos, pos2 = _centred(one, two)
        form = Form(max(one.width, two.width), one.over * 1.75 + two.height, one.under)
        return Box(form, placed([(script, pos2, -one.over * 1.75 - two.under), (base, pos, 0.0)]))
    over = one.over if one.over > 0 else 1.5 * two.over
    form = Form(one.width + two.width, one.over * FACTOR_POS + two.over, one.under)
    return Box(
        form, placed([(script, one.width, -over * FACTOR_POS - two.under), (base, 0.0, 0.0)])
    )


def _under(layout: Layout, text: str, spec: Spec, small: Spec, found: Found, italic: bool) -> Box:
    """A subscript: lowered by ``fFactorPos`` of itself, or under a ``#sum``."""
    one, base = _base(layout, text, found.under, spec, italic)
    script = layout.analyse(text[found.under + 1 :], small, italic)
    two = script.form
    if found.above_place:
        pos, pos2 = _centred(one, two)
        form = Form(max(one.width, two.width), one.over, one.under * 0.9 + two.height)
        return Box(form, placed([(script, pos2, one.under * 0.9 + two.over), (base, pos, 0.0)]))
    form = Form(one.width + two.width, one.over, one.under + two.under + two.over * FACTOR_POS)
    parts = [(script, one.width, one.under + two.over * FACTOR_POS), (base, 0.0, 0.0)]
    return Box(form, placed(parts))


OPERATORS: tuple[tuple[Callable[[Found, str], bool], Callable[..., Box]], ...] = (
    (lambda found, text: -1 < found.close_curly < len(text) - 1, _split),
    (lambda found, text: found.power > -1 and found.under > -1, _both),
    (lambda found, text: found.power > -1, _power),
    (lambda found, text: found.under > -1, _under),
    (lambda found, text: found.command[1] > -1, latexmarks.command),
)


# -- painting a formula -----------------------------------------------------------------


def _turned(
    point: tuple[float, float], origin: tuple[float, float], angle: float
) -> tuple[float, float]:
    """``TLatex::Rotate``: a point of the unturned layout, turned round the text's origin."""
    turn = math.radians(angle)
    cos, sin = math.cos(turn), math.sin(turn)
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    return cos * dx + sin * dy + origin[0], -sin * dx + cos * dy + origin[1]


def draw_marks(scene: Scene, marks: list[Mark], origin: tuple[float, float], basis: float) -> None:
    """Every mark of a formula, turned by its piece's angle round the formula's origin."""
    for mark in marks:
        points = [_turned(point, origin, mark.spec.angle) for point in mark.points]
        color = scene.colors.rgb(mark.spec.color)
        if mark.kind == "text":
            glyphs(scene, mark.text, points[0], mark.spec.font, mark.spec.size * basis, color,
                   mark.align, mark.spec.angle + mark.tilt)  # fmt: skip
        else:
            latexmarks.draw_shape(scene, mark, [(nint(x), nint(y)) for x, y in points], color)


def paint_latex(
    scene: Scene, text: str, at: tuple[float, float], attributes: dict[str, Any]
) -> None:
    """``TLatex::PaintLatex``: ``text`` at canvas pixel ``at``, aligned as ``fTextAlign`` says.

    ``attributes`` are the ``TLatex``'s: ``font``, ``size`` (of the pad, or
    pixels for a font of precision 3), ``color``, ``align``, ``angle``, and
    the ``line`` width its bars are drawn with.
    """
    size, font = float(attributes["size"]), int(attributes["font"])
    if size <= 0 or not text:
        return
    if font % 10 < 2:
        color = scene.colors.rgb(attributes["color"])
        glyphs(scene, text, at, font, size * min(scene.whole), color,
               int(attributes["align"]), float(attributes["angle"]))  # fmt: skip
        return
    if font % 10 > 2:
        size, font = size / min(scene.whole), 10 * (font // 10) + 2
    try:
        checked = check(text.replace("#hbox", "#mbox").replace("\\", "#"))
        _laid_out(scene, checked, at, size, font, attributes)
    except LatexError:
        return  # ROOT says what is wrong on its standard error, and draws nothing


def _laid_out(
    scene: Scene,
    text: str,
    at: tuple[float, float],
    size: float,
    font: int,
    attributes: dict[str, Any],
) -> None:
    """A formula ``check`` passed, laid out at ``size`` in ``font``, aligned on ``at`` and drawn."""
    basis = float(min(scene.whole))
    layout = Layout(basis, scene.height, size, int(attributes.get("line", 2)))
    spec = Spec(size, font, int(attributes["color"]), float(attributes["angle"]))
    form = layout.analyse(text, spec).form
    across, up = divmod(int(attributes["align"]), 10)
    x, y = at
    y += {0: -form.under, 2: form.height * 0.5 - form.under + 1.0, 3: form.over}.get(up, 0.0)
    x -= {2: form.width / 2, 3: form.width}.get(across, 0.0)
    marks: list[Mark] = []
    layout.analyse(text, spec).paint(x, y, marks)
    draw_marks(scene, marks, at, basis)


def formula_form(text: str, size: float, font: int, whole: tuple[int, int], height: float) -> Form:
    """How big a formula is laid out, in pixels, as ``TLatex::GetBoundingBox`` measures it.

    ``whole`` is the pad's width and height in whole pixels and ``height``
    its shorter side unrounded, as :class:`~.scene.Scene` has them.
    """
    basis = float(min(whole))
    if font % 10 > 2:
        size, font = size / basis, 10 * (font // 10) + 2
    try:
        return Layout(basis, height, size).analyse(check(text), Spec(size, font, 1, 0.0)).form
    except LatexError:  # what ROOT refuses measures nothing, as GetXsize and GetYsize give
        return Form(0.0, 0.0, 0.0)
