"""The layers of a one-dimensional histogram, from its edges, values and errors.

``THistPainter`` draws a ``TH1`` as whatever its option asks, in a fixed
order: bars, the outline, the error boxes or band, the error bars and
markers, a line through the bin centres, and the numbers on top. This makes
the same layers in the same order out of plain arrays, so that a histogram,
a profile and a histogram inside a stack - its values sitting on the ones
below - all go through one piece of code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from .model import Band, Bars, Boxes, Curve, Labels, Look, Points, Steps
from .request import Request

__all__ = ["Binned", "binned_layers", "number"]

#: The words that draw error bars, and the ones that draw markers with no bars.
BARRED = ("E", "E0", "E1")
MARKED = ("P", "*", "E2")


class Binned(NamedTuple):
    """One histogram's worth of numbers: edges, a value and an error per bin."""

    edges: Any
    values: Any
    errors: Any
    look: Look
    #: What the values sit on, for a histogram in a stack; zero otherwise.
    baseline: Any = None


def number(value: float) -> str:
    """A bin's value as ROOT's ``TEXT`` writes it, ``%g``."""
    return f"{value:g}"


def _centres(data: Binned) -> Any:
    return (data.edges[:-1] + data.edges[1:]) / 2


def _bottom(data: Binned) -> Any:
    return np.zeros_like(data.values) if data.baseline is None else data.baseline


def _steps(data: Binned, request: Request) -> list[Any]:
    return [Steps(data.edges, data.values, data.look, data.baseline)]


def _bars(data: Binned, request: Request) -> list[Any]:
    widths = np.diff(data.edges)
    left = data.edges[:-1] + 0.1 * widths
    return [Bars(left, left + 0.8 * widths, data.values, data.look)]


def _shown(data: Binned, request: Request) -> Any:
    """The bins ROOT draws a marker or bar for: all of them with ``E0``, the non-empty ones else."""
    if request.chosen.has("E0"):
        return np.ones(len(data.values), dtype=bool)
    return (data.values != _bottom(data)) | (data.errors != 0)


def _points(data: Binned, request: Request) -> list[Any]:
    chosen = request.chosen
    barred = chosen.has(*BARRED)
    if not barred and not chosen.has(*MARKED):
        return []
    keep = _shown(data, request)
    half = np.diff(data.edges)[keep] / 2
    across = np.zeros_like(half) if chosen.has("X0") or not barred else half
    bars = data.errors[keep] if barred else np.zeros(int(keep.sum()))
    return [
        Points(
            _centres(data)[keep],
            data.values[keep],
            across,
            across,
            bars,
            bars,
            data.look,
            caps=chosen.has("E1"),
        )
    ]


def _boxes(data: Binned, request: Request) -> list[Any]:
    if not request.chosen.has("E2"):
        return []
    fill = data.look.fill or data.look.color
    look = data.look._replace(fill=fill, alpha=data.look.alpha if data.look.fill else 0.35)
    return [
        Boxes(
            data.edges[:-1],
            data.edges[1:],
            data.values - data.errors,
            data.values + data.errors,
            look,
        )
    ]


def _band(data: Binned, request: Request) -> list[Any]:
    chosen = request.chosen
    if not chosen.has("E3", "E4"):
        return []
    fill = data.look.fill or data.look.color
    look = data.look._replace(fill=fill, alpha=data.look.alpha if data.look.fill else 0.35)
    return [
        Band(
            _centres(data),
            data.values - data.errors,
            data.values + data.errors,
            look,
            smooth=chosen.has("E4"),
        )
    ]


def _line(data: Binned, request: Request) -> list[Any]:
    chosen = request.chosen
    if not chosen.has("L", "C"):
        return []
    return [Curve(_centres(data), data.values, data.look, smooth=chosen.has("C"))]


def _labels(data: Binned, request: Request) -> list[Any]:
    if not request.chosen.has("TEXT"):
        return []
    texts = tuple(number(float(value)) for value in data.values - _bottom(data))
    return [
        Labels(_centres(data), data.values, texts, request.chosen.angle, data.look.marker_color)
    ]


def _when(word: str, make: Callable[[Binned, Request], list[Any]]) -> Any:
    def made(data: Binned, request: Request) -> list[Any]:
        return make(data, request) if request.chosen.has(word) else []

    return made


#: ``THistPainter``'s order: what is drawn first is underneath.
ORDER = (_when("B", _bars), _when("HIST", _steps), _boxes, _band, _points, _line, _labels)


def binned_layers(data: Binned, request: Request) -> list[Any]:
    """Every layer the option asks for, bottom first."""
    return [layer for make in ORDER for layer in make(data, request)]
