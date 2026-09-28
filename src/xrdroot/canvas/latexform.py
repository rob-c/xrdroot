"""What ``TLatex`` lays a formula out as: sizes, and the marks each piece paints.

``TLatex::Analyse`` works in the canvas's pixels, ``y`` down, with the
baseline at ``y``: every piece has a :class:`Form` - how wide it is, how far
it reaches over the baseline and under it, ROOT's ``TLatexFormSize`` - and,
once placed, paints :class:`Mark` s: strings, lines and filled shapes at
pixels. A :class:`Box` is the two together, a piece's form and how to paint
it wherever it is put, which is what ROOT's two passes over the formula -
one to size it, one to paint it - come to.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

__all__ = ["Box", "Form", "Mark", "Spec", "EMPTY"]


class Spec(NamedTuple):
    """``TextSpec_t``: the size (a fraction of the pad), font, colour and angle of a piece."""

    size: float
    font: int
    color: int
    angle: float

    def sized(self, size: float) -> Spec:
        return self._replace(size=size)


class Form(NamedTuple):
    """``TLatexFormSize``: a piece's width, and how far it reaches over and under its baseline."""

    width: float
    over: float
    under: float

    @property
    def height(self) -> float:
        return self.over + self.under

    def beside(self, other: Form) -> Form:
        """Two pieces side by side, as ``TLatexFormSize::operator+`` puts them."""
        return Form(self.width + other.width, max(self.over, other.over), max(self.under, other.under))


class Mark(NamedTuple):
    """One thing a formula paints, at the canvas's pixels before the text's turn.

    ``kind`` is ``"text"`` (``text`` at ``points[0]``, its baseline's left),
    ``"line"`` (through ``points``, ``width`` pixels wide) or ``"fill"`` (the
    shape ``points`` bound).
    """

    kind: str
    points: tuple[tuple[float, float], ...]
    spec: Spec
    text: str = ""
    width: int = 1
    #: How a string is aligned on its point, and how much more it is turned than its piece.
    align: int = 11
    tilt: float = 0.0


#: How a placed piece paints: given its baseline's left, the marks it adds.
Painter = Callable[[float, float, list[Mark]], None]


class Box(NamedTuple):
    """A piece of a formula: its form, and how it paints where it is put."""

    form: Form
    paint: Painter


def _nothing(_x: float, _y: float, _marks: list[Mark]) -> None:
    """What an empty piece paints."""


#: The empty piece: nothing wide, nothing painted.
EMPTY = Box(Form(0.0, 0.0, 0.0), _nothing)


def placed(parts: list[tuple[Box, float, float]], marks: tuple[Mark, ...] = ()) -> Painter:
    """A painter putting each box at its offset from the piece's point, then ``marks`` moved there."""

    def paint(x: float, y: float, out: list[Mark]) -> None:
        for box, dx, dy in parts:
            box.paint(x + dx, y + dy, out)
        for mark in marks:
            out.append(mark._replace(points=tuple((x + px, y + py) for px, py in mark.points)))

    return paint
