"""A gradient colour on a pad: the shade laid over what is filled with it, clipped to it.

A canvas saved with a ``TLinearGradient`` or ``TRadialGradient`` among its
colours carries the stops the colour runs through and where it runs - a
start and an end, or a centre and a radius, in the unit square of the thing
filled (``kObjectBoundingMode``) or of the pad. A patch filled with such a
colour is drawn hollow, and an image of the shade is laid under it over its
box and clipped to its outline, which every one of matplotlib's backends -
PNG, PDF and SVG - writes as an image with a clip path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Gradient", "gradient_of", "shade"]

#: ``TColorGradient::ECoordinateMode``: the unit square is the pad's, or the object's box.
PAD_MODE, OBJECT_MODE = 0, 1
#: How many pixels across the shade is sampled: smooth at any size a pad is saved at.
RESOLUTION = 160
#: The gradient classes a canvas may hold among its colours.
CLASSES = ("TLinearGradient", "TRadialGradient")


@dataclass(frozen=True)
class Gradient:
    """The stops a gradient runs through, and where it runs in its unit square."""

    kind: str
    positions: tuple[float, ...]
    stops: tuple[tuple[float, float, float, float], ...]
    mode: int
    start: tuple[float, float]
    end: tuple[float, float]
    radius: float = 0.0

    def colours(self, t: Any) -> Any:
        """The colour at each ``t`` of 0 to 1, each channel laid straight between the stops."""
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        return np.stack(
            [np.interp(t, self.positions, [stop[k] for stop in self.stops]) for k in range(4)],
            axis=-1,
        )

    def parameter(self, u: Any, v: Any) -> Any:
        """Where in the gradient each point of the unit square is: along the line, or out."""
        if self.kind == "linear":
            dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
            along = (u - self.start[0]) * dx + (v - self.start[1]) * dy
            return along / ((dx * dx + dy * dy) or 1.0)
        return np.hypot(u - self.end[0], v - self.end[1]) / (self.radius or 1.0)


def _point(given: Any) -> tuple[float, float]:
    """``fStart`` or ``fEnd`` as read: a ``Point``'s members, or nothing."""
    if given is None:
        return (0.0, 0.0)
    return float(given.get("fX", 0.0)), float(given.get("fY", 0.0))


def gradient_of(prim: Any) -> Gradient:
    """A gradient from the members of a ``TLinearGradient`` or ``TRadialGradient`` saved."""
    flat = [float(c) for c in prim.get("fColors", ())]
    return Gradient(
        kind="linear" if prim.classname == "TLinearGradient" else "radial",
        positions=tuple(float(p) for p in prim.get("fColorPositions", ())),
        stops=tuple((flat[at], flat[at + 1], flat[at + 2], flat[at + 3])
                    for at in range(0, len(flat) - 3, 4)),
        mode=int(prim.get("fCoordinateMode", OBJECT_MODE)),
        start=_point(prim.get("fStart")),
        end=_point(prim.get("fEnd")),
        radius=float(prim.get("fR2", 0.0)),
    )  # fmt: skip


def shade(scene: Any, patch: Any, index: Any) -> bool:
    """Lay the gradient of colour ``index`` under ``patch``, which is on the axes already and
    is left hollow; ``False``, and nothing done, when ``index`` is no gradient."""
    from matplotlib.transforms import IdentityTransform

    found = scene.colors.gradients.get(int(index))
    if found is None:
        return False
    square = patch.get_extents() if found.mode == OBJECT_MODE else _pad_square(scene)
    x0, y0, x1, y1 = square.x0, square.y0, square.x1, square.y1
    if x1 <= x0 or y1 <= y0:
        return False
    u, v = np.meshgrid(np.linspace(0, 1, RESOLUTION), np.linspace(0, 1, RESOLUTION))
    image = scene.ax.imshow(
        found.colours(found.parameter(u, v)), extent=(x0, x1, y0, y1), origin="lower",
        transform=IdentityTransform(), interpolation="bilinear", zorder=patch.get_zorder() - 0.001,
        alpha=patch.get_alpha(), aspect="auto",
    )  # fmt: skip
    image.set_clip_path(patch)
    patch.set_facecolor("none")
    return True


def _pad_square(scene: Any) -> Any:
    """The pad's box in the figure's pixels: the unit square of a gradient in pad mode."""
    from matplotlib.transforms import Bbox

    corners = scene.ndc.transform([(0.0, 0.0), (1.0, 1.0)])
    return Bbox(corners)
