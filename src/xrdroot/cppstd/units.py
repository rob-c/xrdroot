"""Standard chrono literals keep the existing timedelta units and rounding."""

from __future__ import annotations

import datetime
from typing import Any

__all__ = ["duration_literal", "DURATIONS"]

#: ``std::chrono``'s suffixes, as the :class:`datetime.timedelta` keyword each counts in.
DURATIONS = {
    "ns": ("microseconds", 1e-3),
    "us": ("microseconds", 1.0),
    "ms": ("milliseconds", 1.0),
    "s": ("seconds", 1.0),
    "min": ("minutes", 1.0),
    "h": ("hours", 1.0),
}


def duration_literal(suffix: str, value: Any) -> datetime.timedelta | None:
    """Return a standard duration, leaving unknown literal suffixes to the caller."""
    duration = DURATIONS.get(suffix)
    if duration is not None:
        unit, scale = duration
        return datetime.timedelta(**{unit: value * scale})
    return None
