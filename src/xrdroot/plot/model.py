"""What a picture is made of, before any drawing library has been asked.

ROOT's painters go straight from an object to pixels. Here there is a step
between: a histogram, graph or function becomes a :class:`Picture` - a
:class:`Frame` and a few layers of plain arrays, each layer one kind of mark
(steps, points with bars, a band, a curve, a shaded grid) with the
:class:`Look` it is drawn in. Every backend draws the same picture its own
way, which is what makes ``backend="plotly"`` give the same plot as
matplotlib rather than a cousin of it, and what lets a test look at what is
to be drawn without drawing anything.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Union

__all__ = [
    "Band",
    "Bars",
    "Boxes",
    "Cloud",
    "Contour",
    "Curve",
    "Frame",
    "Labels",
    "Layer",
    "Look",
    "Mesh",
    "Picture",
    "Points",
    "Steps",
    "Surface",
]

Array = Any


class Look(NamedTuple):
    """How one layer is drawn: ROOT's line, fill and marker attributes, resolved.

    Colours are ones every backend understands - ``"#rrggbb"`` from ROOT's
    table, or whatever name the caller gave. A ``fill`` of ``None`` is
    hollow, and a ``marker`` of ``None`` draws no markers.
    """

    color: str | None = "#000000"
    width: float = 1.0
    dash: str = "solid"
    fill: str | None = None
    alpha: float = 1.0
    hatch: str | None = None
    marker: str | None = None
    hollow: bool = False
    marker_color: str | None = "#000000"
    marker_size: float = 1.0
    label: str | None = None


class Steps(NamedTuple):
    """A histogram's outline: one level per bin, filled down to ``baseline`` if it has a fill."""

    edges: Array
    values: Array
    look: Look
    baseline: Array = None


class Bars(NamedTuple):
    """ROOT's ``B``: a bar from ``left`` to ``right`` up to each value."""

    left: Array
    right: Array
    values: Array
    look: Look


class Points(NamedTuple):
    """Markers at ``(x, y)`` with bars reaching ``low`` below and ``high`` above them.

    Each bar is a distance, not a position, and zero is no bar. ``caps``
    puts ROOT's ``E1`` ticks on the ends of the bars.
    """

    x: Array
    y: Array
    xlow: Array
    xhigh: Array
    ylow: Array
    yhigh: Array
    look: Look
    caps: bool = False


class Boxes(NamedTuple):
    """Rectangles, one per ``(x0, x1, y0, y1)``: ``E2``'s error boxes and ``BOX``'s cells."""

    x0: Array
    x1: Array
    y0: Array
    y1: Array
    look: Look


class Band(NamedTuple):
    """A filled band from ``low`` to ``high`` through ``x``: ``E3``, and ``E4`` smoothed."""

    x: Array
    low: Array
    high: Array
    look: Look
    smooth: bool = False


class Curve(NamedTuple):
    """A line through ``(x, y)``: ``L``, ``C`` smoothed, and every function."""

    x: Array
    y: Array
    look: Look
    smooth: bool = False


class Labels(NamedTuple):
    """ROOT's ``TEXT``: each value written where it is, at ``angle`` degrees."""

    x: Array
    y: Array
    texts: tuple[str, ...]
    angle: float = 0.0
    color: str | None = "#000000"


class Mesh(NamedTuple):
    """A grid of cells shaded by value: ``COL``, and ``COLZ`` with its scale.

    ``values`` is shaped ``(x, y)``, as a histogram's are, and a cell that is
    NaN is left unpainted - ROOT does not paint an empty bin.
    """

    xedges: Array
    yedges: Array
    values: Array
    palette: Any
    scale: bool = True


class Contour(NamedTuple):
    """Lines, or filled bands, of equal value over a grid of ``(x, y)`` centres."""

    x: Array
    y: Array
    values: Array
    palette: Any
    scale: bool = False
    levels: int = 20
    filled: bool = True


class Surface(NamedTuple):
    """A grid as heights: ``SURF`` a smooth surface, ``LEGO`` a block per bin."""

    xedges: Array
    yedges: Array
    values: Array
    palette: Any
    scale: bool = False
    lego: bool = False


class Cloud(NamedTuple):
    """A three-dimensional histogram: a marker per filled bin, or ``ISO`` its surfaces."""

    x: Array
    y: Array
    z: Array
    values: Array
    palette: Any
    iso: bool = False


Layer = Union[Steps, Bars, Points, Boxes, Band, Curve, Labels, Mesh, Contour, Surface, Cloud]

#: The layers that need axes with depth to be drawn on.
DEEP = (Surface, Cloud)


class Frame(NamedTuple):
    """What the axes say: the title, what each axis is, and how it is scaled."""

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""
    logx: bool = False
    logy: bool = False
    logz: bool = False
    xlim: Any = None
    ylim: Any = None
    legend: bool | None = None
    grid: bool = False


class Picture(NamedTuple):
    """A whole plot: its layers in the order they are drawn, and its frame.

    ``same`` is ROOT's ``SAME``: draw over what was drawn last. ``native``
    is whatever keywords the caller gave that this library does not know,
    handed as they are to the drawing library's call for the first layer.
    """

    layers: tuple[Any, ...]
    frame: Frame
    same: bool = False
    native: dict[str, Any] = {}  # noqa: RUF012 - a NamedTuple default, never mutated

    @property
    def deep(self) -> bool:
        """Does any layer need three-dimensional axes?"""
        return any(isinstance(layer, DEEP) for layer in self.layers)
