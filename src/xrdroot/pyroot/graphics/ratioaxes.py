"""A ``TRatioPlot``'s axes: ``TGaxis`` drawn over both pads, so they line up as one plot.

The pads' own axes are not drawn (``A`` and ``IA``); instead a top pad over
both holds a ``TGaxis`` for each side of each pad, placed in that pad's NDC
at its frame's edges and graduated over its frame's range - the upper x
axis unlabelled, the lower y axis's ticks lengthened by the ratio of the
two frames' heights so they look alike - each taking its attributes from
the axis it stands for, as ``TRatioPlot::ImportAxisAttributes`` does. Where
the pads meet too closely, a label of one would overwrite the other's: one
is hidden, as ``fHideLabelMode`` says. Mirrors of each are drawn on the top
and right when the parent pad has ticks there, or a frame with no fill.
They are placed afresh whenever the canvas is drawn, as ROOT does in
``TRatioPlot::Paint``, so a range or title set after ``Draw`` is shown.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .text import TGaxis

__all__ = [
    "f32",
    "HIDE_LOW",
    "HIDE_UP",
    "FORCE_HIDE_LOW",
    "FORCE_HIDE_UP",
    "NO_HIDE",
    "create",
    "update",
]

#: ``TRatioPlot::HideLabelMode``.
HIDE_UP, HIDE_LOW, NO_HIDE, FORCE_HIDE_UP, FORCE_HIDE_LOW = 1, 2, 3, 4, 5
#: The narrowest gap between the pads that leaves room for both their labels.
CROWDED = 0.025
#: Each axis, by its member, with the chopt it is made with.
AXES = {"upper_x": "+U", "upper_y": "S", "lower_x": "+S", "lower_y": "-S"}


def create(rp: Any) -> None:
    """``CreateVisualAxes``: the four axes, and the mirrors the parent pad asks for."""
    parent = rp._parent_pad
    mirrored = parent.GetFrameFillStyle() == 0
    axis_top = parent.GetTickx() == 1 or mirrored
    axis_right = parent.GetTicky() == 1 or mirrored
    for name, chopt in AXES.items():
        if rp._axes.get(name) is None:
            made = TGaxis(0, 0, 1, 1, 0, 1, 510, chopt)
            rp._top_pad.add(made, "")
            rp._axes[name] = made
    wanted = {"upper_x": axis_top, "lower_x": axis_top, "upper_y": axis_right,
              "lower_y": axis_right}  # fmt: skip
    for name, want in wanted.items():
        if want and rp._mirrors.get(name) is None:
            made = rp._axes[name].Clone()
            rp._top_pad.add(made, "")
            rp._mirrors[name] = made
    update(rp)


def import_attributes(gaxis: Any, axis: Any) -> None:
    """``ImportAxisAttributes``: the ``TAxis``'s colours, fonts, sizes, offsets and title."""
    gaxis.SetLineColor(axis.GetAxisColor())
    gaxis.SetTextColor(axis.GetTitleColor())
    gaxis.SetTextFont(axis.GetTitleFont())
    gaxis.SetLabelColor(axis.GetLabelColor())
    gaxis.SetLabelFont(axis.GetLabelFont())
    gaxis.SetLabelSize(axis.GetLabelSize())
    gaxis.SetLabelOffset(axis.GetLabelOffset())
    gaxis.SetTickSize(axis.GetTickLength())
    gaxis.SetTitle(axis.GetTitle())
    gaxis.SetTitleOffset(axis.GetTitleOffset())
    gaxis.SetTitleSize(axis.GetTitleSize())
    gaxis.SetTimeFormat(axis.GetTimeFormat())


def _place(gaxis: Any, ends: tuple[float, float, float, float], scale: tuple[float, float]) -> None:
    gaxis.SetX1(ends[0])
    gaxis.SetY1(ends[1])
    gaxis.SetX2(ends[2])
    gaxis.SetY2(ends[3])
    gaxis.SetWmin(scale[0])
    gaxis.SetWmax(scale[1])


def _powered(low: float, high: float, log: bool) -> tuple[float, float]:
    """A frame's y range, powers of ten turned back into its ends on a logarithmic pad.

    ROOT checks the ends are above zero, which a power of ten always is.
    """
    return (math.pow(10, low), math.pow(10, high)) if log else (low, high)


def ranges(rp: Any) -> dict[str, Any]:
    """What the axes are placed and graduated by: the shared x range, each pad's y range."""
    upper, lower = rp._upper_pad, rp._lower_pad
    shared = rp._shared_x
    first = shared.GetBinLowEdge(shared.GetFirst())
    last = shared.GetBinUpEdge(shared.GetLast())
    logx = bool(upper.GetLogx() or lower.GetLogx())
    if logx and (first <= 0 or last <= 0):
        rp.Error("UpdateVisualAxes", "Cannot set X axis to log scale")
    return {
        "x": (first, last),
        "up": _powered(upper.GetUymin(), upper.GetUymax(), bool(upper.GetLogy())),
        "low": _powered(lower.GetUymin(), lower.GetUymax(), bool(lower.GetLogy())),
        "xopt": "G" if logx else "",
        "upopt": "G" if upper.GetLogy() else "",
        "lowopt": "G" if lower.GetLogy() else "",
    }


def _edges(rp: Any) -> dict[str, float]:
    """Where the frames' edges are in the top pad: the upper pad above ``sf``, the lower below."""
    upper, lower, sf = rp._upper_pad, rp._lower_pad, rp._split_fraction
    return {
        "upleft": upper.GetLeftMargin(), "upright": 1 - upper.GetRightMargin(),
        "upbottom": upper.GetBottomMargin() * f32(1 - sf) + sf,
        "uptop": (1 - upper.GetTopMargin()) * f32(1 - sf) + sf,
        "lowleft": lower.GetLeftMargin(), "lowright": 1 - lower.GetRightMargin(),
        "lowbottom": lower.GetBottomMargin() * sf, "lowtop": (1 - lower.GetTopMargin()) * sf,
    }  # fmt: skip


def _tick_ratio(rp: Any) -> float:
    """How much taller the upper frame is than the lower: the lower y axis's ticks grow by it."""
    upper, lower, sf = rp._upper_pad, rp._lower_pad, rp._split_fraction
    up = (upper.GetBottomMargin() - (1 - upper.GetTopMargin())) * f32(1 - sf)
    low = (lower.GetBottomMargin() - (1 - lower.GetTopMargin())) * sf
    return float(up / low)


def f32(value: float) -> float:
    """``value`` as a ``Float_t`` holds it: ROOT keeps margins and fractions in single precision."""
    return float(np.float32(value))


def update(rp: Any) -> None:
    """``UpdateVisualAxes``: every axis placed at its frame's edge and graduated over its range."""
    r, e = ranges(rp), _edges(rp)
    axes = rp._axes
    references = {"upper_x": rp.GetUpperRefXaxis(), "upper_y": rp.GetUpperRefYaxis(),
                  "lower_x": rp.GetLowerRefXaxis(), "lower_y": rp.GetLowerRefYaxis()}  # fmt: skip
    for name, axis in axes.items():
        import_attributes(axis, references[name])
    axes["lower_x"].SetTitle(axes["upper_x"].GetTitle())
    axes["upper_x"].SetTitle("")
    _place(axes["upper_x"], (e["upleft"], e["upbottom"], e["upright"], e["upbottom"]), r["x"])
    _place(axes["upper_y"], (e["upleft"], e["upbottom"], e["upleft"], e["uptop"]), r["up"])
    _place(axes["lower_x"], (e["lowleft"], e["lowbottom"], e["lowright"], e["lowbottom"]), r["x"])
    _place(axes["lower_y"], (e["lowleft"], e["lowbottom"], e["lowleft"], e["lowtop"]), r["low"])
    shared_ndiv = rp._shared_x.GetNdivisions()
    up_ndiv, low_ndiv = rp._up_y.GetNdivisions(), rp._low_y.GetNdivisions()
    for name, ndiv in (("upper_x", shared_ndiv), ("upper_y", up_ndiv),
                       ("lower_x", shared_ndiv), ("lower_y", low_ndiv)):  # fmt: skip
        axes[name].SetNdivisions(ndiv)
    for name, opt in (("upper_x", "+U" + r["xopt"]), ("upper_y", "S" + r["upopt"]),
                      ("lower_x", "+S" + r["xopt"]), ("lower_y", "-S" + r["lowopt"])):  # fmt: skip
        axes[name].SetOption(opt)
    axes["upper_x"].SetLabelSize(0.0)
    ticksize = f32(axes["upper_y"].GetTickSize() * _tick_ratio(rp))
    axes["lower_y"].SetTickSize(ticksize)
    _hide(rp)
    _mirror(rp, r, e, references, ticksize)


def _hide(rp: Any) -> None:
    """A label hidden where the pads meet too closely for both - or where told to be."""
    mode, upper_y, lower_y = rp._hide_label_mode, rp._axes["upper_y"], rp._axes["lower_y"]
    if mode == FORCE_HIDE_UP:
        upper_y.ChangeLabel(1, -1, 0)
    elif mode == FORCE_HIDE_LOW:
        lower_y.ChangeLabel(-1, -1, 0)
    elif rp.GetSeparationMargin() < CROWDED:
        if mode == HIDE_UP:
            upper_y.ChangeLabel(1, -1, 0)
        elif mode == HIDE_LOW:
            lower_y.ChangeLabel(-1, -1, 0)
    elif mode == HIDE_UP:
        upper_y.ChangeLabel(0)
    elif mode == HIDE_LOW:
        lower_y.ChangeLabel(0)


def _mirror(
    rp: Any, r: dict[str, Any], e: dict[str, float], references: dict[str, Any], ticksize: float
) -> None:
    """The unlabelled twins on the far sides, ticked towards the frame."""
    places = {
        "upper_x": ((e["upleft"], e["uptop"], e["upright"], e["uptop"]), r["x"], "-S" + r["xopt"]),
        "upper_y": ((e["upright"], e["upbottom"], e["upright"], e["uptop"]), r["up"],
                    "+S" + r["upopt"]),
        "lower_x": ((e["lowleft"], e["lowtop"], e["lowright"], e["lowtop"]), r["x"],
                    "-S" + r["xopt"]),
        "lower_y": ((e["lowright"], e["lowbottom"], e["lowright"], e["lowtop"]), r["low"],
                    "+S" + r["lowopt"]),
    }  # fmt: skip
    divisions = {"upper_x": rp._shared_x, "lower_x": rp._shared_x, "upper_y": rp._up_y,
                 "lower_y": rp._low_y}  # fmt: skip
    for name, mirror in rp._mirrors.items():
        ends, scale, chopt = places[name]
        import_attributes(mirror, references[name])
        mirror.SetTitle("")
        _place(mirror, ends, scale)
        mirror.SetOption(chopt)
        if name == "lower_y":
            mirror.SetTickSize(ticksize)
        mirror.SetNdivisions(divisions[name].GetNdivisions())
        mirror.SetLabelSize(0.0)
