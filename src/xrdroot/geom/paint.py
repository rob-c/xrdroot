"""Compatibility access to pad painters, with scene-only points available without graphics.

The painters belong to canvas; looking up one here returns that exact object.
Only an explicit request for a graphics symbol imports the rendering component.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .scene import extent as extent
from .scene import points_of as points_of
from .view import make_view as make_view

__all__ = ["GEOMETRY", "paint_geometry", "paint_polyline3d", "paint_polymarker3d", "points_of"]

if TYPE_CHECKING:
    GEOMETRY: dict[str, Callable[[Any, Any, str], None]]
    paint_geometry: Callable[[Any, Any, str], None]
    paint_polyline3d: Callable[[Any, Any, str], None]
    paint_polymarker3d: Callable[[Any, Any, str], None]

#: Legacy painter names, including the helpers callers could previously import.
_GRAPHICS = (
    "GEOMETRY", "paint_geometry", "paint_polyline3d", "paint_polymarker3d", "_view", "_pixels",
)


def __getattr__(name: str) -> Any:
    if name not in _GRAPHICS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from ..canvas import geompaint

    return getattr(geompaint, name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_GRAPHICS})
