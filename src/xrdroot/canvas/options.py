"""ROOT's draw options, read the way ``THistPainter`` and ``TGraphPainter`` read them.

An option is letters run together with no separator - ``"hist same"``,
``"e1p"``, ``"colz"``, ``"alp"`` - and ROOT picks it apart by looking for
the words it knows, the long ones first so that ``COLZ`` is not read as
``COL`` and a stray ``Z``. The histogram's words and the graph's letters
are different languages, so each has a reader of its own.
"""

from __future__ import annotations

__all__ = ["graph_option", "histogram_option", "strip_same"]

#: The words of a histogram's option, the longer of any two that overlap first.
HISTOGRAM_WORDS = (
    "SAMES", "SAME", "HIST", "TEXT", "COLZ", "COL", "BOX", "SCAT", "CONTZ", "CONT",
    "LEGO", "SURF", "ARR", "POL", "CYL", "SPH", "PSR", "FUNC", "AXIS", "NOSTACK",
    "PFC", "PLC", "PMC", "BAR", "MIN0", "E0", "E1", "E2", "E3", "E4", "E5", "E6",
    "E", "P0", "P", "L", "C", "B", "*H", "][", "Z", "9",
)  # fmt: skip
#: The error-bar options of a histogram.
ERRORS = frozenset({"E", "E0", "E1", "E2", "E3", "E4", "E5", "E6"})
#: What a histogram can be drawn as, besides its outline.
SHAPES = frozenset({"P", "P0", "L", "C", "B", "BAR", "*H", "TEXT"}) | ERRORS
#: The pictures of a two-dimensional histogram this draws: each is itself.
PICTURES_2D = frozenset({"COL", "COLZ", "BOX", "TEXT", "CONT", "CONTZ"})


def strip_same(option: str) -> str:
    """``option`` without ``same``, which says where to draw rather than what."""
    upper = option.upper()
    for word in ("SAMES", "SAME"):
        upper = upper.replace(word, "")
    return upper


def histogram_option(option: str) -> frozenset[str]:
    """The words of a histogram's draw option: ``"e1 same"`` is ``{"E1", "SAME"}``."""
    rest = option.upper().replace(" ", "")
    found = set()
    for word in HISTOGRAM_WORDS:
        if word in rest:
            found.add(word)
            rest = rest.replace(word, " ")
    return frozenset(found)


def graph_option(option: str) -> frozenset[str]:
    """The letters of a graph's draw option, with ``A`` for axes: ``"ap"`` is ``{"A", "P"}``.

    A graph given no way of drawing its points - no line, curve, markers,
    bars or fill - is drawn as a line through them, as ROOT draws one.
    """
    rest = strip_same(option).replace(" ", "")
    found = {letter for letter in rest if letter in "ALCP*BFXZ12345"}
    if "[]" in rest:
        found.add("[]")
    if not found & set("LCP*BF"):
        found.add("L")
    return frozenset(found)
