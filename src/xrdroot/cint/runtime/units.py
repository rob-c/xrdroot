"""User-defined literals the library defines: ROOT 7's pad lengths and ``std::chrono``'s.

``0.1_normal`` calls ``ROOT::Experimental::operator""_normal(0.1)``, which
makes an ``RPadLength::Normal``; ``20_px`` and ``80_user`` make the pixel
and user-coordinate lengths. ROOT's class is looked up only when the
literal is evaluated, the way every ROOT name a translation uses is, so
a translation imports without it. ``100us`` is ``std::chrono``'s: a
duration, which is a :class:`datetime.timedelta` here - what
``std::this_thread::sleep_for`` and the clocks of :mod:`.threads` take.
"""

from __future__ import annotations

import datetime
from typing import Any

from .root import ROOT

__all__ = ["user_literal", "DURATIONS"]

#: ``std::chrono``'s suffixes, as the :class:`datetime.timedelta` keyword each counts in.
DURATIONS = {
    "ns": ("microseconds", 1e-3),
    "us": ("microseconds", 1.0),
    "ms": ("milliseconds", 1.0),
    "s": ("seconds", 1.0),
    "min": ("minutes", 1.0),
    "h": ("hours", 1.0),
}

#: ROOT 7's suffixes, as the ``RPadLength`` member class each makes.
PAD_LENGTHS = {"_normal": "Normal", "_px": "Pixel", "_user": "User"}


def user_literal(suffix: str, value: Any) -> Any:
    """What the library's ``operator""`` for ``suffix`` makes of the number ``value``."""
    duration = DURATIONS.get(suffix)
    if duration is not None:
        unit, scale = duration
        return datetime.timedelta(**{unit: value * scale})
    return getattr(ROOT.RPadLength, PAD_LENGTHS[suffix])(value)
