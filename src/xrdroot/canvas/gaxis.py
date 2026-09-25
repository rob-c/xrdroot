"""A ``TGaxis``: an axis drawn anywhere in a pad, with a scale of its own.

A ``TGaxis`` is a line from ``(fX1, fY1)`` to ``(fX2, fY2)`` in the units
of the pad's axes, graduated from ``fWmin`` to ``fWmax`` - which need have
nothing to do with the pad's own range, which is the point of one: a second
scale on the right of a frame, an axis on a picture with none. ROOT's
``fChopt`` says how: ``G`` a logarithmic scale, ``+`` and ``-`` which side
the ticks stand on (both, given both), ``=`` the labels on the ticks' side
rather than the other, ``U`` no labels at all.

The ticks are where matplotlib's locators put them over the scale, as many
as the units of ``fNdiv`` ask for, since ROOT's own ``THLimitsFinder``
finds the same round numbers for any range a reader would draw. The side a
tick stands on is the one to the left walking from the first end to the
second - above a horizontal axis, left of a vertical one - which is ROOT's
"positive" side.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .latex import translate
from .model import Primitive, lookup
from .scene import Scene
from .shapes import draw_text

__all__ = ["GAXIS", "gaxis", "graduations"]

#: What ``TGaxis`` gives an axis never styled: tick length (of the axis's own
#: length), label size and offset, as fractions of the pad.
TICK_SIZE, LABEL_SIZE, LABEL_OFFSET = 0.03, 0.04, 0.005


def graduations(low: float, high: float, divisions: int, log: bool) -> np.ndarray[Any, Any]:
    """The values along a scale from ``low`` to ``high`` that are ticked and labelled."""
    from matplotlib.ticker import LogLocator, MaxNLocator

    lo, hi = min(low, high), max(low, high)
    if log and lo > 0:
        found = LogLocator(base=10.0).tick_values(lo, hi)
    else:
        found = MaxNLocator(nbins=max(divisions % 100, 1), steps=[1, 2, 2.5, 5, 10]).tick_values(
            lo, hi
        )
    found = np.asarray(found, dtype=float)
    span = hi - lo
    return found[(found >= lo - 1e-9 * span) & (found <= hi + 1e-9 * span)]


def _fraction(values: np.ndarray[Any, Any], low: float, high: float, log: bool) -> Any:
    """How far along the axis each value is, from its first end to its second."""
    if log and low > 0 and high > 0:
        return (np.log10(values) - math.log10(low)) / (math.log10(high) - math.log10(low))
    return (values - low) / ((high - low) or 1.0)


def _label(value: float) -> str:
    """A graduation as ROOT labels one: a plain number, with no trailing zeros."""
    return f"{value:.6g}"


def _pixels(scene: Scene, prim: Primitive) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """The axis's two ends in the pad's pixels, from its units or its fractions."""
    ends = [(float(prim.get(f"fX{n}", 0.0)), float(prim.get(f"fY{n}", 0.0))) for n in (1, 2)]
    if not prim.ndc:
        ends = [scene.to_ndc(x, y) for x, y in ends]
    size = np.asarray(scene.pixels, dtype=float)
    return np.asarray(ends[0]) * size, np.asarray(ends[1]) * size


def _sides(chopt: str) -> tuple[list[float], float]:
    """Which ways the ticks stand, and which way the labels go, as signs of the normal."""
    ticks = [side for mark, side in (("+", 1.0), ("-", -1.0)) if mark in chopt] or [1.0]
    labels = ticks[0] if "=" in chopt else -ticks[0]
    return ticks, labels


def gaxis(scene: Scene, prim: Primitive, _option: str) -> None:
    """The line, its ticks, and their labels, where and as ``fChopt`` says."""
    start, end = _pixels(scene, prim)
    length = float(np.hypot(*(end - start))) or 1.0
    along = (end - start) / length
    normal = np.array([-along[1], along[0]])
    chopt = str(prim.get("fChopt", ""))
    low, high = float(prim.get("fWmin", 0.0)), float(prim.get("fWmax", 1.0))
    log = "G" in chopt
    values = graduations(low, high, int(prim.get("fNdiv", 510) or 510), log)
    places = [start + (end - start) * f for f in _fraction(values, low, high, log)]
    tick = float(prim.get("fTickSize", TICK_SIZE) or TICK_SIZE) * length
    ticks, labels = _sides(chopt)
    segments = [(start, end)] + [(at, at + side * tick * normal) for at in places for side in ticks]
    _segments(scene, prim, segments)
    if "U" not in chopt:
        _labels(scene, prim, values, places, labels * normal)
    _title(scene, prim, end, labels * normal)


def _segments(scene: Scene, prim: Primitive, segments: list[Any]) -> None:
    """The axis line and its ticks, each from one point in pixels to another."""
    from matplotlib.lines import Line2D

    size = np.asarray(scene.pixels, dtype=float)
    style = scene.line(prim)
    for a, b in segments:
        scene.ax.add_artist(
            Line2D(
                [a[0] / size[0], b[0] / size[0]],
                [a[1] / size[1], b[1] / size[1]],
                transform=scene.ndc,
                clip_on=False,
                zorder=scene.layer(),
                **style,
            )
        )


def _text_style(scene: Scene, prim: Primitive, size_name: str, direction: Any) -> dict[str, Any]:
    """Text on the side ``direction`` points to, sized by ``size_name``."""
    size = float(lookup(prim, size_name, LABEL_SIZE) or LABEL_SIZE)
    style = scene.text(None, None, scene.text_points(size, lookup(prim, "fLabelFont", 42) or 42))
    style["color"] = scene.colors.rgb(lookup(prim, "fLabelColor", 1))
    style["ha"] = "center" if abs(direction[0]) < 0.5 else ("left" if direction[0] > 0 else "right")
    style["va"] = "center" if abs(direction[1]) < 0.5 else ("bottom" if direction[1] > 0 else "top")
    return style


def _labels(scene: Scene, prim: Primitive, values: Any, places: list[Any], direction: Any) -> None:
    """Each graduation's number, just past the axis on the labels' side."""
    style = _text_style(scene, prim, "fLabelSize", direction)
    offset = (float(prim.get("fLabelOffset", LABEL_OFFSET) or LABEL_OFFSET)) * scene.shorter
    size = np.asarray(scene.pixels, dtype=float)
    for value, at in zip(values, places):
        x, y = (at + offset * direction) / size
        draw_text(scene, _label(float(value)), float(x), float(y), style, ndc=True)


def _title(scene: Scene, prim: Primitive, end: Any, direction: Any) -> None:
    """The axis's title, at its far end, beyond its labels."""
    title = str(prim.get("fTitle", "") or "")
    if not title:
        return
    style = _text_style(scene, prim, "fTitleSize", direction)
    size = np.asarray(scene.pixels, dtype=float)
    x, y = (end + 3 * LABEL_SIZE * scene.shorter * direction) / size
    draw_text(scene, translate(title), float(x), float(y), style, ndc=True)


#: How this class draws.
GAXIS = {"TGaxis": gaxis}
