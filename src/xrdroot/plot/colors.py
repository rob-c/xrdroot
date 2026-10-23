"""Compatibility names for ROOT's shared colour tables and arithmetic.

These are the same objects used by geometry and canvas rendering; optional
plotting libraries are not needed to resolve a ROOT colour.
"""

from __future__ import annotations

from ..colors import (
    _GRADIENTS as _GRADIENTS,
)
from ..colors import (
    _NAMED as _NAMED,
)
from ..colors import (
    _PALETTE_NUMBERS as _PALETTE_NUMBERS,
)
from ..colors import (
    _STOPS as _STOPS,
)
from ..colors import (
    COLORS as COLORS,
)
from ..colors import (
    ECOLOR as ECOLOR,
)
from ..colors import (
    PALETTES as PALETTES,
)
from ..colors import (
    PETROFF as PETROFF,
)
from ..colors import (
    UNKNOWN as UNKNOWN,
)
from ..colors import (
    _between as _between,
)
from ..colors import (
    _gradient as _gradient,
)
from ..colors import (
    _hex as _hex,
)
from ..colors import (
    _named as _named,
)
from ..colors import (
    color as color,
)
from ..colors import (
    palette as palette,
)
from ..colors import (
    palette_color as palette_color,
)

__all__ = [
    "COLORS", "ECOLOR", "PALETTES", "PETROFF", "UNKNOWN", "color", "palette", "palette_color",
]
