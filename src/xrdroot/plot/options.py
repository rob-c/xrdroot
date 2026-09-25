"""ROOT's drawing options, read the way ``THistPainter`` and ``TGraphPainter`` read them.

``h->Draw("E1 SAME")``, ``g->Draw("AP")``, ``h2->Draw("COLZ")``: an option is
a run of short words with nothing between them, in any case, and ROOT finds
each by looking for it in the string. This splits the string into those
words - the longest that fits at each place, so ``COLZ`` is ``COL`` and ``Z``
and ``E1`` is not ``E`` then ``1`` - and says what each asks for.

Every word is checked against what it can mean for the thing being drawn.
A word this does not draw is refused by name with the reason, and so is a
word that means nothing for this object - ``COLZ`` on a one-dimensional
histogram, which ROOT would quietly ignore - because an option that does
nothing looks, in the picture, like an option that worked.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import NamedTuple

__all__ = ["KINDS", "Chosen", "choose"]

#: Every spelling ROOT takes that is drawn here, against the word it means.
SPELLINGS = {
    "SAME": "SAME", "SAMES": "SAME", "HIST": "HIST", "FUNC": "FUNC", "NORM": "NORM",
    "AXIS": "AXIS", "E": "E", "E0": "E0", "E1": "E1", "E2": "E2", "E3": "E3", "E4": "E4",
    "P": "P", "P0": "P", "L": "L", "C": "C", "B": "B", "BAR": "B", "X0": "X0", "*": "*",
    "][": "][", "COL": "COL", "Z": "Z", "BOX": "BOX", "BOX1": "BOX", "CONT": "CONT",
    "CONT0": "CONT", "CONT4": "CONT", "CONT1": "CONTL", "CONT2": "CONTL", "CONT3": "CONTL",
    "LEGO": "LEGO", "LEGO1": "LEGO", "LEGO2": "LEGO", "LEGO3": "LEGO", "LEGO4": "LEGO",
    "SURF": "SURF", "SURF1": "SURF", "SURF2": "SURF", "SURF3": "SURF", "SURF4": "SURF",
    "SURF5": "SURF", "SURF6": "SURF", "SURF7": "SURF", "ISO": "ISO", "NOSTACK": "NOSTACK",
    "NOSTACKB": "NOSTACKB", "A": "A", "F": "F", "2": "2", "3": "3", "4": "4", "X": "X",
    "PLC": "PLC", "PMC": "PMC", "PFC": "PFC", "TEXT": "TEXT",
}  # fmt: skip

#: Spellings ROOT takes that are not drawn here, and why not.
REFUSED = {
    "SCAT": "ROOT itself has retired the scatter plot of a histogram; COL shades the same bins",
    "ARR": "arrows along a gradient are not a picture any backend here draws",
    "PIE": "a pie chart of bins is not drawn here",
    "POL": "polar coordinates are not drawn here",
    "CYL": "cylindrical coordinates are not drawn here",
    "SPH": "spherical coordinates are not drawn here",
    "PSR": "pseudo-rapidity coordinates are not drawn here",
    "HBAR": "horizontal bars are not drawn here; B draws them upright",
    "CANDLE": "candle plots are not drawn here",
    "VIOLIN": "violin plots are not drawn here",
    "TRI": "Delaunay triangles are not drawn here",
    "SPEC": "the TSpectrum2Painter is not drawn here",
    "E5": "E5 and E6 ignore ROOT's own error bars; E3 is the band they draw",
    "E6": "E5 and E6 ignore ROOT's own error bars; E3 is the band they draw",
    "GL": "OpenGL pictures need ROOT's canvas; LEGO and SURF draw in three dimensions here",
    "PADS": "a stack drawn one histogram to a pad is a grid of plots, one plot() each",
    "LOGX": "ROOT sets a log scale on the pad, not in the option: pass logx=True",
    "LOGY": "ROOT sets a log scale on the pad, not in the option: pass logy=True",
    "LOGZ": "ROOT sets a log scale on the pad, not in the option: pass logz=True",
    "[]": "the brackets of TGraphAsymmErrors are not drawn here",
    "|>": "arrow heads on error bars are not drawn here",
    ">": "arrow heads on error bars are not drawn here",
}

#: ``TEXT`` with the angle ROOT writes the numbers at, ``TEXT45`` for 45 degrees.
_TEXT = r"TEXT(?P<angle>\d{1,2})?"

_WORDS = re.compile(
    "|".join(
        [_TEXT]
        + [re.escape(word) for word in sorted({*SPELLINGS, *REFUSED}, key=len, reverse=True)]
    )
)

_HIST1 = {"SAME", "HIST", "FUNC", "NORM", "AXIS", "E", "E0", "E1", "E2", "E3", "E4", "P", "L",
          "C", "B", "X0", "*", "][", "TEXT", "PLC", "PMC", "PFC"}  # fmt: skip
_HIST2 = {"SAME", "FUNC", "NORM", "AXIS", "COL", "Z", "BOX", "CONT", "CONTL", "LEGO", "SURF",
          "TEXT"}  # fmt: skip
_GRAPH = {"SAME", "A", "P", "L", "C", "*", "B", "F", "2", "3", "4", "X", "Z", "PLC", "PMC",
          "PFC"}  # fmt: skip

#: What each kind of thing can be asked to be drawn as.
KINDS = {
    "histogram": frozenset(_HIST1),
    "two-dimensional histogram": frozenset(_HIST2),
    "three-dimensional histogram": frozenset({"SAME", "BOX", "ISO", "NORM"}),
    "graph": frozenset(_GRAPH),
    "function": frozenset({"SAME", "L", "C", "P", "*", "F", "PLC", "PMC", "PFC"}),
    "stack": frozenset(_HIST1 | {"NOSTACK", "NOSTACKB"}),
}


class Chosen(NamedTuple):
    """The words an option string was made of, and the ``TEXT`` angle if it had one."""

    words: frozenset[str]
    angle: float = 0.0

    def has(self, *words: str) -> bool:
        """Does the option ask for any of ``words``?"""
        return any(word in self.words for word in words)

    def without(self, *words: str) -> Chosen:
        """The same option with ``words`` taken out, as ROOT blanks what it has read."""
        return self._replace(words=self.words - set(words))

    @property
    def drawing(self) -> frozenset[str]:
        """The words that say how to draw, rather than where or how much."""
        return self.words - {"SAME", "NORM", "PLC", "PMC", "PFC", "A", "AXIS", "FUNC", "Z"}


def _split(option: str) -> tuple[list[tuple[str, str]], float]:
    """The option cut into ``(spelling, word)`` pairs, the longest word at each place."""
    text = re.sub(r"[\s,]+", "", str(option).upper())
    pairs: list[tuple[str, str]] = []
    angle = 0.0
    at = 0
    while at < len(text):
        found = _WORDS.match(text, at)
        if found is None:
            raise ValueError(
                f"option {option!r} has {text[at:]!r}, which begins with no drawing option "
                f"ROOT has; the words drawn here are {', '.join(sorted(set(SPELLINGS)))}"
            )
        spelling = found.group(0)
        if found.group("angle"):
            angle = float(found.group("angle"))
        word = "TEXT" if spelling.startswith("TEXT") else SPELLINGS.get(spelling, spelling)
        pairs.append((spelling, word))
        at = found.end()
    return pairs, angle


def _refuse(pairs: Iterable[tuple[str, str]], kind: str) -> None:
    allowed = KINDS[kind]
    for spelling, word in pairs:
        if spelling in REFUSED:
            raise ValueError(f"option {spelling!r} is not drawn here: {REFUSED[spelling]}")
        if word not in allowed:
            raise ValueError(
                f"option {spelling!r} does not draw a {kind}; what does is "
                f"{', '.join(sorted(allowed))}"
            )


def choose(option: str, kind: str) -> Chosen:
    """The option read for a ``kind`` of thing, refusing what does not draw it.

    >>> sorted(choose("colz same", "two-dimensional histogram").words)
    ['COL', 'SAME', 'Z']
    """
    pairs, angle = _split(option)
    _refuse(pairs, kind)
    return Chosen(frozenset(word for _, word in pairs), angle)
