"""A scatter plot's frame, colour scale, marker sizes and the layers it is drawn as."""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.plot import picture
from xrdroot.plot.model import Cloud, Dots
from xrdroot.scatterplot import ScatterPlot


def _plot(**kwargs):
    x, y = np.array([0.0, 10.0, 20.0, 30.0]), np.array([1.0, 3.0, 2.0, 4.0])
    return ScatterPlot.new("s", x, y, title="Points;across;up;shade", **kwargs)


def test_the_frame_is_a_histogram_over_the_points_a_tenth_wider_titled_from_the_title():
    plot = _plot(colors=[1.0, 2.0, 3.0, 4.0], sizes=[10.0, 20.0, 30.0, 40.0])
    frame = plot.frame()
    assert frame is plot.frame() and frame.name == "s_h" and frame.title == "Points"
    assert (frame.axes[0].low, frame.axes[0].high) == (-3.0, 33.0)
    assert (frame.axes[1].low, frame.axes[1].high) == pytest.approx((0.7, 4.3))
    assert (frame.axes[0].title, frame.axes[1].title) == ("across", "up")
    assert frame._core["fZaxis"]["TNamed"]["fTitle"] == "shade"
    assert (plot.classname, len(plot), plot.z, plot.name) == ("TScatter", 4, None, "s")
    assert ScatterPlot.new("one", [2.0], [2.0]).frame().axes[0].low == pytest.approx(1.9)


def test_colours_run_between_the_values_ends_or_the_z_range_set():
    plot = _plot(colors=[1.0, 2.0, 3.0, 5.0], sizes=[10.0, 20.0, 30.0, 50.0])
    assert plot.colour_scale() == (1.0, 5.0)
    assert plot.colour_fractions().tolist() == [0.0, 0.25, 0.5, 1.0]
    plot.frame()._core["fMinimum"], plot.frame()._core["fMaximum"] = 2.0, 4.0
    assert plot.colour_scale() == (2.0, 4.0)
    assert plot.colour_fractions().tolist() == [0.0, 0.0, 0.5, 1.0]
    plot.members["fLogC"] = True
    assert plot.colour_fractions()[1] == 0.0 and 0 < plot.colour_fractions()[2] < 1.0
    bare = _plot()
    assert bare.colour_fractions() is None and bare.colour_scale() == (0.0, 1.0)
    same = _plot(colors=[2.0] * 4)
    assert same.colour_fractions().tolist() == [0.0] * 4


def test_sizes_run_between_the_marker_sizes_as_the_values_lie_between_theirs():
    plot = _plot(colors=[1.0, 2.0, 3.0, 5.0], sizes=[10.0, 20.0, 30.0, 50.0])
    assert plot.marker_sizes().tolist() == [1.0, 2.0, 3.0, 5.0]
    plot.members["fMinMarkerSize"], plot.members["fMaxMarkerSize"] = 0.5, 2.5
    assert plot.marker_sizes().tolist() == [0.5, 1.0, 1.5, 2.5]
    plot.members["fLogS"] = True
    assert plot.marker_sizes()[1] == pytest.approx(0.5 + 2.0 * np.log10(2) / np.log10(5))
    assert _plot().marker_sizes().tolist() == [1.0] * 4  # no sizes: the marker's own
    assert _plot(sizes=[3.0] * 4).marker_sizes().tolist() == [1.0] * 4  # all alike: the least


def test_the_picture_is_a_dot_per_point_in_the_palettes_colour_or_the_markers_own():
    shaded = picture(_plot(colors=[1.0, 2.0, 3.0, 5.0], sizes=[1.0, 2.0, 3.0, 4.0]), "A")
    layer = shaded.layers[0]
    assert isinstance(layer, Dots) and layer.scale == (1.0, 5.0) and len(layer.colors) == 4
    assert layer.colors[0] != layer.colors[3]
    assert layer.sizes.tolist() == pytest.approx([1.0, 7 / 3, 11 / 3, 5.0])
    assert (shaded.frame.xlabel, shaded.frame.ylabel, shaded.frame.zlabel) == (
        "across", "up", "shade")
    plain = picture(_plot(), "A").layers[0]
    assert plain.scale is None and set(plain.colors) == {"#000000"}
    mapped = picture(_plot(colors=[0.0, 1.0, 2.0, 3.0]), "A", {"palette": lambda t: (t, 0, 0, 1)})
    assert mapped.layers[0].colors[-1] == "#ff0000"


def test_a_scatter_plot_in_space_is_a_cloud_of_its_points_coloured_by_their_values():
    plot = ScatterPlot.new("s", [0.0, 1.0], [0.0, 1.0], [5.0, 6.0], colors=[1.0, 2.0])
    frame = plot.frame3d()
    assert plot.classname == "TScatter2D" and plot.z.tolist() == [5.0, 6.0]
    assert (frame._core["fMinimum"], frame._core["fMaximum"]) == pytest.approx((4.9, 6.1))
    layer = picture(plot, "").layers[0]
    assert isinstance(layer, Cloud) and layer.values.tolist() == [0.0, 1.0]
    assert picture(ScatterPlot.new("b", [0.0], [0.0], [0.0]), "").layers[0].values.tolist() == [0.0]


def test_matplotlib_draws_the_dots_with_a_colour_bar_and_a_canvas_titles_the_frame_by_it():
    pytest.importorskip("matplotlib")
    from xrdroot.canvas import Canvas
    from xrdroot.canvas.scatterplot import colormap
    from xrdroot.plot import plot
    from xrdroot.pyroot.core.wrapping import wrap

    shaded = _plot(colors=[1.0, 2.0, 3.0, 5.0], sizes=[1.0, 2.0, 3.0, 4.0])
    ax = plot(shaded, option="A")
    assert len(ax.figure.axes) == 2  # the frame and the colour bar
    assert plot(_plot(), option="A").collections  # no colours: no bar, the dots still
    cmap = colormap("kBird")
    assert colormap(cmap) is cmap
    members = {
        "fX1": 0.0, "fY1": 0.0, "fX2": 1.0, "fY2": 1.0, "fFillColor": 0, "fFillStyle": 1001,
        "fLeftMargin": 0.1, "fRightMargin": 0.1, "fBottomMargin": 0.1, "fTopMargin": 0.1,
        "fPrimitives": _listed([shaded], ["A"]),
    }  # fmt: skip
    canvas = Canvas("TCanvas", {"TPad": members, "fCw": 200, "fCh": 200})
    assert canvas.plot() is not None
    from xrdroot.canvas.frame import _painted_title

    assert _painted_title(shaded) == "Points"  # the frame's title, from the scatter's own
    assert wrap(shaded).GetN() == 4 and wrap(shaded).ClassName() == "TScatter"


def _listed(items, options):
    from xrdroot.buffer import Listed

    return Listed(items, options)
