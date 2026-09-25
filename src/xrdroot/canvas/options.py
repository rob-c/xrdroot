"""The little of ROOT's draw options a pad needs to know before anything is drawn.

What an option draws is :mod:`xrdroot.plot`'s to say. What a pad needs
beforehand is less: whether something is drawn ``SAME``, over the frame
another drew, and whether a histogram is drawn with its errors, which its
frame then makes room for. An option is words run together with nothing
between them - ``"hist same"``, ``"e1p"`` - found the long ones first, so
that ``E1`` is not read as ``E`` and a stray ``1``.
"""

from __future__ import annotations

__all__ = ["ERRORS", "histogram_option", "strip_same"]

#: The words of a histogram's option, the longer of any two that overlap first.
HISTOGRAM_WORDS = (
    "SAMES", "SAME", "HIST", "TEXT", "COLZ", "COL", "BOX", "SCAT", "CONTZ", "CONT",
    "LEGO", "SURF", "ARR", "POL", "CYL", "SPH", "PSR", "FUNC", "AXIS", "NOSTACK",
    "PFC", "PLC", "PMC", "BAR", "MIN0", "E0", "E1", "E2", "E3", "E4", "E5", "E6",
    "E", "P0", "P", "L", "C", "B", "*H", "][", "Z", "9",
)  # fmt: skip
#: The error-bar options of a histogram.
ERRORS = frozenset({"E", "E0", "E1", "E2", "E3", "E4", "E5", "E6"})


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
