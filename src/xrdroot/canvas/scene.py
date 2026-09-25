"""One pad being drawn: where it is, and how its units become the figure's.

ROOT places everything a pad draws in one of two ways - in the units of its
axes, as a histogram's bins are, or in NDC, as fractions of the pad from its
bottom left corner, as a legend or a label usually is. A :class:`Scene`
carries both: the axes the pad's frame is, whose data transform is the
first, and a transform of its own for the second. It also turns ROOT's
attribute members into the keywords matplotlib draws with, since what
they mean - a text size a fraction of the pad, a line width in pixels -
depends on the pad they are drawn in.
"""

from __future__ import annotations

from typing import Any

from . import styles
from .colors import Colors
from .model import Pad, lookup

__all__ = ["Scene"]

#: The text size ROOT's own kit gives a ``TAttText`` that was never set.
TEXT_SIZE = 0.05
#: Where the drawing classes start stacking, above the data, and how far apart.
LAYERS, LAYER_STEP = 3.0, 0.001


class Scene:
    """A pad on a figure, with its axes and the ways into them."""

    __slots__ = (
        "figure",
        "ax",
        "pad",
        "colors",
        "box",
        "pixels",
        "skipped",
        "owner",
        "ndc",
        "stats",
        "depth",
    )

    def __init__(
        self,
        figure: Any,
        pad: Pad,
        box: tuple[float, float, float, float],
        colors: Colors,
        canvas_pixels: tuple[float, float],
        skipped: list[str],
    ) -> None:
        from matplotlib.transforms import Affine2D

        self.figure = figure
        self.pad = pad
        self.colors = colors
        #: Where the pad is on the figure, as fractions: left, bottom, width, height.
        self.box = box
        #: How big the pad is, in ROOT's pixels.
        self.pixels = (box[2] * canvas_pixels[0], box[3] * canvas_pixels[1])
        #: The classes met that this does not draw, for the warning at the end.
        self.skipped = skipped
        #: What drew the frame and its axes - the first histogram, or a graph drawn "A".
        self.owner: tuple[Any, str] | None = None
        self.ax: Any = None
        #: Pad NDC into the figure's display, for what is placed by fractions.
        self.ndc = Affine2D().scale(box[2], box[3]).translate(box[0], box[1]) + figure.transFigure
        #: How many stats boxes the pad has drawn, which offsets each new one.
        self.stats = 0
        self.depth = 0

    def layer(self) -> float:
        """The next height to draw at, over everything drawn in the pad before.

        ROOT paints a pad's primitives in order, each over the last, so each
        drawing here stands a little above the one before it - and above the
        data, which matplotlib draws low down.
        """
        self.depth += 1
        return LAYERS + self.depth * LAYER_STEP

    # -- sizes ---------------------------------------------------------------

    @property
    def shorter(self) -> float:
        """The pad's shorter side in pixels, which ROOT sizes text against."""
        return min(self.pixels)

    def text_points(self, size: Any, font: Any = 42) -> float:
        """A ``fTextSize`` in points: pixels for a font of precision 3, else of the pad."""
        pixel_sized = styles.font(font)[3]
        size = float(size)
        return styles.points(size if pixel_sized else size * self.shorter)

    def to_ndc(self, x: float, y: float) -> tuple[float, float]:
        """A point in the pad's axes' units, as a fraction of the pad."""
        display = self.ax.transData.transform((x, y))
        u, v = self.ndc.inverted().transform(display)
        return float(u), float(v)

    def where(self, ndc: bool) -> Any:
        """The transform for something placed in NDC, or in the axes' units."""
        return self.ndc if ndc else self.ax.transData

    # -- attributes ----------------------------------------------------------

    def line(self, obj: Any) -> dict[str, Any]:
        """``TAttLine`` as matplotlib's colour, width and dashes."""
        width = float(lookup(obj, "fLineWidth", 1))
        return {
            "color": self.colors.rgb(lookup(obj, "fLineColor", 1)),
            "linewidth": styles.points(width),
            "linestyle": styles.dashes(lookup(obj, "fLineStyle", 1), width),
        }

    def fill(self, obj: Any) -> dict[str, Any] | None:
        """``TAttFill`` as a face colour and hatch, or ``None`` for a hollow one."""
        fills, hatch, alpha = styles.fill(lookup(obj, "fFillStyle", 0))
        if not fills:
            return None
        color = self.colors.rgba(lookup(obj, "fFillColor", 0), alpha)
        if hatch is None:
            return {"facecolor": color, "hatch": None, "hatchcolor": None}
        return {"facecolor": "none", "hatch": hatch, "hatchcolor": color}

    def marker(self, obj: Any) -> dict[str, Any]:
        """``TAttMarker`` as matplotlib's marker, size and colours."""
        style = lookup(obj, "fMarkerStyle", 1)
        shape, filled = styles.marker(style)
        color = self.colors.rgb(lookup(obj, "fMarkerColor", 1))
        return {
            "marker": shape,
            "markersize": styles.marker_size(style, lookup(obj, "fMarkerSize", 1)),
            "markerfacecolor": color if filled else "none",
            "markeredgecolor": color,
        }

    def text(self, obj: Any, inherited: Any = None, size: float | None = None) -> dict[str, Any]:
        """``TAttText`` as a font, size, colour and alignment.

        A member left at zero - as a line of a pave or an entry of a legend
        leaves it - takes the value of ``inherited``, the pave or legend it
        is in, as ROOT's painters do.
        """
        font = _attribute(obj, inherited, "fTextFont", 42)
        family, style, weight, _pixels = styles.font(font)
        across, up = styles.align(_attribute(obj, inherited, "fTextAlign", 11))
        points = (
            size
            if size is not None
            else self.text_points(_attribute(obj, inherited, "fTextSize", TEXT_SIZE), font)
        )
        return {
            "fontsize": points,
            "color": self.colors.rgb(_attribute(obj, inherited, "fTextColor", 1)),
            "family": family,
            "style": style,
            "weight": weight,
            "math_fontfamily": styles.MATH[family],
            "ha": across,
            "va": up,
            "rotation": float(lookup(obj, "fTextAngle", 0.0)),
        }


def _attribute(obj: Any, inherited: Any, name: str, default: Any) -> Any:
    """``name`` of ``obj``, or of what it inherits where it is left at zero.

    A zero size, font or alignment is unset and takes ``default``; a zero
    colour that nothing overrides is white, which is what colour 0 is.
    """
    value = lookup(obj, name, None)
    if not value and inherited is not None:
        value = lookup(inherited, name, None)
    if value is None or (not value and name != "fTextColor"):
        return default
    return value
