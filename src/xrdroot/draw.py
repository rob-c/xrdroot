"""Pictures without a canvas or a drawing library: characters.

ROOT draws through a ``TCanvas``, which this library does not carry. The
drawing libraries Python has are reached through :mod:`xrdroot.plot`; what
is here is the plainest picture of all, text, which needs nothing installed
and goes anywhere a string goes.
"""

from __future__ import annotations

import math

from .errors import UnsupportedFeatureError

__all__ = ["bar", "missing_picture", "shade"]

#: A bar grown one eighth of a character at a time, then a full block.
EIGHTHS = "▏▎▍▌▋▊▉█"
#: A grid cell, darker the fuller it is.
SHADES = " ░▒▓█"


def bar(fraction: float, width: int) -> str:
    """A horizontal bar filling ``fraction`` of ``width`` characters.

        >>> bar(0.5, 4)
        '██'
    """
    eighths = round(max(0.0, min(1.0, fraction)) * width * 8)
    text = "█" * (eighths // 8)
    if eighths % 8:
        text += EIGHTHS[eighths % 8 - 1]
    return text


def shade(fraction: float) -> str:
    """One cell of a grid, in five steps from empty to full.

        >>> shade(0.9)
        '█'
    """
    if fraction <= 0:
        return SHADES[0]
    return SHADES[max(1, min(4, math.ceil(fraction * 4)))]


def missing_picture(what: str, dimensions: int) -> UnsupportedFeatureError:
    """The refusal for a histogram with more axes than a picture has."""
    return UnsupportedFeatureError(
        f"a {dimensions}-dimensional {what} has no honest flat picture; take "
        f".values() and slice it down to the two dimensions you want to see"
    )
