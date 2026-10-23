"""Compatibility names for the shared geometry projection arithmetic.

Canvas and geometry keep one implementation so both paths use the same
rounding and the same view objects.
"""

from __future__ import annotations

from ..geom.projection import (
    AXIS_ENDS as AXIS_ENDS,
)
from ..geom.projection import (
    CORNER_ORDER as CORNER_ORDER,
)
from ..geom.projection import (
    RAD as RAD,
)
from ..geom.projection import (
    Angles as Angles,
)
from ..geom.projection import (
    Corners as Corners,
)
from ..geom.projection import (
    View3D as View3D,
)
from ..geom.projection import (
    _direction as _direction,
)
from ..geom.projection import (
    _rotation as _rotation,
)
from ..geom.projection import (
    _translation as _translation,
)

__all__ = ["View3D"]
