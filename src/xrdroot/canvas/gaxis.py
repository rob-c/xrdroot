"""A ``TGaxis``: an axis drawn anywhere in a pad, with a scale of its own.

A ``TGaxis`` is a line from ``(fX1, fY1)`` to ``(fX2, fY2)`` in the units
of the pad's axes, graduated from ``fWmin`` to ``fWmax`` - which need have
nothing to do with the pad's own range, which is the point of one: a second
scale on the right of a frame, an axis on a picture with none. ROOT's
``fChopt`` says how: ``G`` a logarithmic scale, ``+`` and ``-`` which side
the ticks stand on (both, given both), ``=`` the labels on the ticks' side
rather than the other, ``U`` no labels at all.

It is painted by ``TGaxis::PaintAxis`` as a frame's axes are, from its own
attributes - ``TGaxis``'s defaults are a bold label font (62) and labels
and title 0.04 of the pad - by :func:`~.axis.paint_axis`.
"""

from __future__ import annotations

from .axis import Axis, paint_axis
from .dressing import draw_painted
from .model import Primitive
from .scene import Scene
from .text import nint

__all__ = ["GAXIS", "gaxis"]

#: What ``TGaxis`` gives an axis never styled: tick length (of the axis's own
#: length), label size and offset, as fractions of the pad, and its label font.
TICK_SIZE, LABEL_SIZE, LABEL_OFFSET, LABEL_FONT = 0.03, 0.04, 0.005, 62


def _ends(scene: Scene, prim: Primitive) -> tuple[float, float, float, float]:
    """The axis's two ends in the pad's NDC, from its units or its fractions."""
    ends = [(float(prim.get(f"fX{n}", 0.0)), float(prim.get(f"fY{n}", 0.0))) for n in (1, 2)]
    if not prim.ndc:
        ends = [scene.to_ndc(x, y) for x, y in ends]
    return ends[0][0], ends[0][1], ends[1][0], ends[1][1]


def _axis(scene: Scene, prim: Primitive) -> Axis:
    """The :class:`~.axis.Axis` a saved ``TGaxis`` is, with ``TGaxis``'s own defaults.

    A ``TGaxis`` keeps its title's font and colour as its ``TAttText``, and its
    line's colour as its ``TAttLine``, where a ``TAxis`` has members of its own.
    """
    x0, y0, x1, y1 = _ends(scene, prim)
    label_size = float(prim.get("fLabelSize", LABEL_SIZE))
    return Axis(
        x0, y0, x1, y1,
        wmin=float(prim.get("fWmin", 0.0)), wmax=float(prim.get("fWmax", 1.0)),
        ndiv=int(prim.get("fNdiv", 510)), chopt=str(prim.get("fChopt", "")),
        grid_length=float(prim.get("fGridLength", 0.0) or 0.0),
        label_font=int(prim.get("fLabelFont", LABEL_FONT) or LABEL_FONT), label_size=label_size,
        label_color=int(prim.get("fLabelColor", 1)),
        label_offset=float(prim.get("fLabelOffset", LABEL_OFFSET)),
        tick_size=float(prim.get("fTickSize", TICK_SIZE)),
        title=str(prim.get("fTitle", "") or ""), title_offset=float(prim.get("fTitleOffset", 1.0)),
        title_size=float(prim.get("fTitleSize", label_size)),
        title_font=int(prim.get("fTextFont", 62) or 62), title_color=int(prim.get("fTextColor", 1)),
        line_color=int(prim.get("fLineColor", 1)), pad=scene.pixels,
        changed=tuple(prim.get("_changed_labels", ())),
    )  # fmt: skip


def gaxis(scene: Scene, prim: Primitive, _option: str) -> None:
    """The line, its ticks, their labels and its title, where and as ``fChopt`` says."""

    def pixel(u: float, v: float) -> tuple[int, int]:
        """A point of NDC in the whole pixels ``PaintAxis`` measures its title's angle in."""
        px, py = scene.pixel(u, v)
        return nint(px), nint(py)

    axis = _axis(scene, prim)
    draw_painted(scene, paint_axis(axis, pixel), axis)


#: How this class draws.
GAXIS = {"TGaxis": gaxis}
