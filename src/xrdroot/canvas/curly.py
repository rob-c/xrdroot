"""``TCurlyLine`` and ``TCurlyArc``: loops or a sine wave, built in the pad's pixels.

``TCurlyLine::Build`` lays the wave out along a straight line of the line's
length in pixels - a stretch, then forty steps a period, each a phase of
2 pi / 40 further on, the loops' ``x`` swinging back by the amplitude's sine
where a wavy line's goes straight on - and turns it to the line's angle. A
``TCurlyArc`` bends the same straight wave round its circle. The wave length
and amplitude are fractions of the pad's larger side in pixels.
"""

from __future__ import annotations

import math
from typing import Any

from .model import Primitive, lookup
from .scene import Scene
from .shapes import _line_of, canvas_point

__all__ = ["CURLY"]

#: ``TCurlyLine::Build``'s steps a period.
STEPS = 40


def _straight(
    length: float, wave: float, amplitude: float, curly: bool
) -> list[tuple[float, float]]:
    """The wave along a line from ``(0, 0)`` to ``(length, 0)``, in pixels."""
    periods = int((length - wave) / wave) if wave > 0 else 0
    rest = 0.5 * (length - periods * wave - wave)
    steps = max(int(STEPS * periods + STEPS / 2 + 4), 2)
    points = [(0.0, 0.0), (rest, 0.0)]
    phase, x0 = 1.5 * math.pi, amplitude + rest
    for _ in range(2, steps - 1):
        x = x0 + amplitude * math.sin(phase) if curly else x0
        points.append((x, amplitude * math.cos(phase)))
        phase += 2 * math.pi / STEPS
        x0 += wave / STEPS
    points.append((length, 0.0))
    return points[:steps]


def _sizes(scene: Scene, prim: Primitive) -> tuple[float, float, bool]:
    """The wave length and amplitude in pixels, and whether the wave curls."""
    side = max(scene.pixels)
    wave = side * float(lookup(prim, "fWaveLength", 0.02))
    amplitude = side * float(lookup(prim, "fAmplitude", 0.01))
    return wave, amplitude, bool(lookup(prim, "fIsCurly", True))


def curly_line(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TCurlyLine``: the wave from ``(fX1, fY1)`` to ``(fX2, fY2)``."""
    x1, y1 = canvas_point(scene, float(prim.get("fX1", 0.0)), float(prim.get("fY1", 0.0)), False)
    x2, y2 = canvas_point(scene, float(prim.get("fX2", 0.0)), float(prim.get("fY2", 0.0)), False)
    angle = math.atan2(y2 - y1, x2 - x1)
    cos, sin = math.cos(angle), math.sin(angle)
    straight = _straight(math.hypot(x2 - x1, y2 - y1), *_sizes(scene, prim))
    _line_of(scene, prim, [(x1 + u * cos - v * sin, y1 + u * sin + v * cos) for u, v in straight])


def curly_arc(scene: Scene, prim: Primitive, _option: str) -> None:
    """A ``TCurlyArc``: the wave bent round the circle, from ``fPhimin`` to ``fPhimax``."""
    cx, cy = canvas_point(scene, float(prim.get("fX1", 0.0)), float(prim.get("fY1", 0.0)), False)
    edge, _ = canvas_point(scene, float(prim.get("fX1", 0.0)) + float(prim.get("fR1", 0.0)),
                           float(prim.get("fY1", 0.0)), False)  # fmt: skip
    radius = abs(edge - cx)
    low, high = float(prim.get("fPhimin", 0.0)), float(prim.get("fPhimax", 360.0))
    span = (high - low) % 360.0 or 360.0
    if radius <= 0:
        return
    straight = _straight(math.pi * radius * span / 180.0, *_sizes(scene, prim))
    start = math.radians(low)
    points = []
    for u, v in straight:
        turn, reach = start + u / radius, radius + v
        points.append((cx + reach * math.cos(turn), cy - reach * math.sin(turn)))  # y counts down
    _line_of(scene, prim, points)


#: How each of these classes draws.
CURLY: dict[str, Any] = {"TCurlyLine": curly_line, "TCurlyArc": curly_arc}
