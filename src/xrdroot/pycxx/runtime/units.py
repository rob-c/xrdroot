"""User-defined literals the standard library defines: ``std::chrono``'s.

``100us`` is ``std::chrono``'s: a duration, which is a
:class:`datetime.timedelta` here - what ``std::this_thread::sleep_for`` and
the clocks of :mod:`.threads` take. A dialect whose library defines literals
of its own exports its own ``user_literal``, which a translation in that
dialect calls instead.
"""

from __future__ import annotations

from typing import Any

from ...cppstd.units import DURATIONS as DURATIONS
from ...cppstd.units import duration_literal
from ..errors import Unsupported

__all__ = ["user_literal", "DURATIONS"]


def user_literal(suffix: str, value: Any) -> Any:
    """What the standard library's ``operator""`` for ``suffix`` makes of the number ``value``."""
    duration = duration_literal(suffix, value)
    if duration is None:
        raise Unsupported(f'operator""{suffix} is a literal the standard library does not define')
    return duration
