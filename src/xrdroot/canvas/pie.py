"""A ``TPie`` on a pad, as ``TPie::Paint`` paints one: slices, then their labels.

Each slice is a wedge of the circle, from ``fAngularOffset`` anticlockwise
in order, as wide as its value's share of the whole, moved out along its
middle by its radius offset (a fraction of the radius) and filled and
outlined as its own attributes say - a gradient fill shaded. Its label,
made by ``fLabelFormat``, sits ``fLabelsOffset`` of the radius beyond the
rim along that middle: along the radius with option ``R``, along the
tangent with ``T``, upright otherwise, in the slice's colour with ``SC``
and left out with ``NOL``. The pseudo-3D of option ``3D`` is painted flat.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .gradient import shade
from .model import Primitive, lookup
from .scene import Scene
from .shapes import canvas_point, patch_style, write

__all__ = ["PIE"]

#: How many points along a slice's arc per degree of it, and the fewest for any slice.
ARC_STEP, ARC_LEAST = 2.0, 4


def _label(pie: Primitive, piece: Any, value: float, total: float) -> str:
    """``fLabelFormat`` with the slice's title, value, fraction and percentage put in."""
    text = str(lookup(pie, "fLabelFormat", "%txt"))
    share = value / total if total else 0.0
    made = (
        ("%txt", str(lookup(piece, "fTitle", ""))),
        ("%val", str(lookup(pie, "fValueFormat", "%4.2f")) % value),
        ("%frac", str(lookup(pie, "fFractionFormat", "%3.2f")) % share),
        ("%perc", str(lookup(pie, "fPercentFormat", "%3.1f")) % (100 * share) + " %"),
    )
    for key, replacement in made:
        text = text.replace(key, replacement)
    return text


def _text_style(option: str, mid: float, piece: Any) -> dict[str, Any]:
    """How a label at angle ``mid`` is written: turned and aligned by the option, so that it
    reads outward on the right and is turned round on the left."""
    upper = option.upper()
    left = 90 < mid % 360 < 270
    found: dict[str, Any] = {}
    if "R" in upper:
        found["angle"], found["align"] = (mid + 180, 32) if left else (mid, 12)
    elif "T" in upper:
        found["angle"], found["align"] = (mid + 90 if left else mid - 90), 22
    else:
        found["align"] = 32 if left else 12
    if "SC" in upper:
        found["color"] = int(lookup(piece, "fFillColor", 1))
    return found


def _slice(
    scene: Scene, piece: Any, circle: tuple[float, float, float], a0: float, a1: float
) -> None:
    """One wedge, centre to rim along the arc and back, filled and outlined as the slice is."""
    from matplotlib.patches import Polygon

    cx, cy, r = circle
    turn = np.radians(np.linspace(a0, a1, max(ARC_LEAST, int((a1 - a0) * ARC_STEP) + 1)))
    xs = np.concatenate([[cx], cx + r * np.cos(turn)])
    ys = np.concatenate([[cy], cy + r * np.sin(turn)])
    patch = Polygon(np.column_stack([xs, ys]), closed=True, transform=scene.ax.transData,
                    clip_on=False, zorder=scene.layer(), **patch_style(scene, piece))  # fmt: skip
    scene.ax.add_artist(patch)
    shade(scene, patch, lookup(piece, "fFillColor", 0))


def pie(scene: Scene, prim: Primitive, option: str) -> None:
    """A ``TPie``: its slices in order from ``fAngularOffset``, and their labels."""
    slices = list(prim.get("fPieSlices") or [])
    values = [max(float(lookup(piece, "fValue", 0.0)), 0.0) for piece in slices]
    total = sum(values)
    x, y, r = (float(prim.get(k, d)) for k, d in (("fX", 0.5), ("fY", 0.5), ("fRadius", 0.4)))
    angle = float(prim.get("fAngularOffset", 0.0))
    reach = 1.0 + float(prim.get("fLabelsOffset", 0.1))
    for piece, value in zip(slices, values, strict=True):
        span = 360.0 * value / total if total else 0.0
        mid = angle + span / 2
        out = r * float(lookup(piece, "fRadiusOffset", 0.0))
        along = (math.cos(math.radians(mid)), math.sin(math.radians(mid)))
        _slice(scene, piece, (x + out * along[0], y + out * along[1], r), angle, angle + span)
        if "NOL" not in option.upper():
            at = canvas_point(scene, x + (out + r * reach) * along[0],
                              y + (out + r * reach) * along[1], False)  # fmt: skip
            write(scene, _label(prim, piece, value, total), at,
                  scene.attributes(prim, **_text_style(option, mid, piece)), True)  # fmt: skip
        angle += span


#: How a pie chart draws.
PIE = {"TPie": pie}
