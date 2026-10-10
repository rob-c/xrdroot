"""A gradient colour painted: the shade laid under what is filled with it, clipped to it.

Whatever a gradient fills - a pad, a frame, a stats box, a histogram's
bars, a box or an ellipse - is drawn hollow with an image of the shade
under it, clipped to its outline, in object or pad coordinates.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.canvas import Canvas, Primitive
from xrdroot.canvas.gradient import Gradient, gradient_of, shade

pytest.importorskip("matplotlib")

STOPS = ((1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.5))


def _linear(number, mode=1, start=(0.0, 0.5), end=(1.0, 0.5)):
    return Primitive("TLinearGradient", {
        "fNumber": number, "fRed": 1.0, "fGreen": 0.0, "fBlue": 0.0, "fColorPositions": [0.0, 1.0],
        "fColors": [c for stop in STOPS for c in stop], "fCoordinateMode": mode,
        "fStart": {"fX": start[0], "fY": start[1]}, "fEnd": {"fX": end[0], "fY": end[1]},
    })  # fmt: skip


def _radial(number):
    return Primitive("TRadialGradient", {
        "fNumber": number, "fRed": 1.0, "fGreen": 0.0, "fBlue": 0.0, "fColorPositions": [0.0, 1.0],
        "fColors": [c for stop in STOPS for c in stop], "fCoordinateMode": 1,
        "fStart": {"fX": 0.5, "fY": 0.5}, "fR1": 0.0, "fEnd": {"fX": 0.5, "fY": 0.5}, "fR2": 0.5,
    })  # fmt: skip


def test_a_gradient_runs_its_stops_along_its_line_or_out_from_its_centre():
    linear = gradient_of(_linear(300))
    assert linear.colours([0.0, 0.5, 1.0, 2.0]).tolist() == [
        [1.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.5, 0.75], [0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 1.0, 0.5]]
    assert linear.parameter(np.array([0.25, 1.0]), np.array([0.9, 0.0])).tolist() == [0.25, 1.0]
    radial = gradient_of(_radial(301))
    assert radial.kind == "radial" and radial.radius == 0.5
    assert radial.parameter(np.array([0.5, 1.0]), np.array([0.5, 0.5])).tolist() == [0.0, 1.0]
    flat = Gradient("linear", (0.0, 1.0), STOPS, 1, (0.2, 0.2), (0.2, 0.2))  # no line to run
    assert flat.parameter(0.7, 0.7) == 0.0
    assert gradient_of(Primitive("TLinearGradient", {})).start == (0.0, 0.0)


def _canvas(primitives, options, colours, fill=0):
    members = {
        "fX1": 0.0, "fY1": 0.0, "fX2": 1.0, "fY2": 1.0, "fFillColor": fill, "fFillStyle": 1001,
        "fLeftMargin": 0.1, "fRightMargin": 0.1, "fBottomMargin": 0.1, "fTopMargin": 0.1,
        "fPrimitives": _listed([*primitives, colours], [*options, ""]),
    }  # fmt: skip
    return Canvas("TCanvas", {"TPad": members, "fCw": 200, "fCh": 200})


def _listed(items, options):
    from xrdroot.buffer import Listed

    return Listed(items, options)


def _images(figure):
    from matplotlib.image import AxesImage

    return [a for ax in figure.axes for a in ax.get_children() if isinstance(a, AxesImage)]


@pytest.mark.parametrize("mode", [0, 1])
def test_shapes_filled_with_a_gradient_are_shaded_under_their_outline(mode):
    box = Primitive("TBox", {"fX1": 0.1, "fY1": 0.1, "fX2": 0.6, "fY2": 0.5, "fFillColor": 300,
                             "fFillStyle": 1001, "fLineWidth": 1})  # fmt: skip
    disc = Primitive("TEllipse", {"fX1": 0.7, "fY1": 0.7, "fR1": 0.2, "fR2": 0.2, "fPhimin": 0.0,
                                  "fPhimax": 360.0, "fTheta": 0.0, "fFillColor": 301,
                                  "fFillStyle": 1001, "fLineWidth": 1})  # fmt: skip
    canvas = _canvas([box, disc], ["", ""], [_linear(300, mode), _radial(301)])
    figure = canvas.plot()
    images = _images(figure)
    assert len(images) == 2 and images[1].get_clip_path() is not None  # the disc's outline
    if mode == 1:  # over the box itself, which, a rectangle, clips as a box
        x0, x1, y0, y1 = images[0].get_extent()
        assert images[0].clipbox.extents == pytest.approx((x0, y0, x1, y1)) and x0 < x1
    flat = Primitive("TBox", {"fX1": 0.1, "fY1": 0.3, "fX2": 0.6, "fY2": 0.3, "fFillColor": 300,
                              "fFillStyle": 1001})  # fmt: skip
    assert not _images(_canvas([flat], [""], [_linear(300)]).plot())  # no box to shade


def test_the_pad_the_frame_the_stats_box_and_the_bars_take_a_gradient_too():
    from xrdroot import Histogram

    h = Histogram.book("h", (4, 0.0, 1.0))
    h.fill(np.array([0.2, 0.4, 0.4, 0.8]))
    h._core["TAttFill"]["fFillColor"], h._core["TAttFill"]["fFillStyle"] = 300, 1001
    stats = Primitive("TPaveStats", {"fX1NDC": 0.6, "fY1NDC": 0.6, "fX2NDC": 0.9, "fY2NDC": 0.9,
                                     "fFillColor": 301, "fFillStyle": 1001, "fLines": [],
                                     "fOptStat": 1111, "fOption": "brNDC"})  # fmt: skip
    frame = Primitive("TFrame", {"fX1": 0.0, "fY1": 0.0, "fX2": 1.0, "fY2": 1.0,
                                 "fFillColor": 300, "fFillStyle": 1001})  # fmt: skip
    canvas = _canvas([h, frame, stats], ["", "", ""], [_linear(300, 0), _radial(301)], fill=301)
    assert len(_images(canvas.plot())) >= 4  # the pad, the frame, the bars and the box


def test_a_colour_that_is_no_gradient_leaves_the_patch_as_it_is():
    from matplotlib.patches import Rectangle

    canvas = _canvas([], [], [_linear(300)])
    figure = canvas.plot()
    scene = type("S", (), {"colors": type("C", (), {"gradients": {}})()})()
    assert shade(scene, Rectangle((0, 0), 1, 1), 2) is False and figure is not None
