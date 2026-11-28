"""``TScatter``: points with a colour and a size each, and the frame they are drawn in.

A scatter plot is made here rather than read: ``ScatterPlot.new`` books the graph and keeps the
columns a file would hold, so the engine's own arithmetic - the frame's range, the colour scale, a
marker's size - is testable without a drawing back end or a file. That is what keeps
``xrdroot.scatterplot`` inside the ``hist`` component's isolation scope.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.hist import Histogram
from xrdroot.scatterplot import FRAME_BINS, NO_STATS, ScatterPlot

#: Five points either side of zero, so a range widened by the margin is easy to read.
X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
Y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])


def test_a_scatter_plot_in_a_plane_books_a_frame_over_its_points():
    """``GetHistogram`` is a ``TH2F`` of the range widened by ``fMargin`` each way, its titles
    split from ``title;x;y;z``, and ``TH1::kNoStats`` set so no statistics box is drawn."""
    plot = ScatterPlot.new("sp", X, Y, title="points;px;py;pz")
    assert (plot.name, plot.title, len(plot)) == ("sp", "points;px;py;pz", 5)
    frame = plot.frame()
    assert isinstance(frame, Histogram)
    xaxis, yaxis = frame.axes[0], frame.axes[1]
    assert (len(xaxis), len(yaxis)) == (FRAME_BINS, FRAME_BINS)  # ROOT books 100 bins each way
    assert (xaxis.low, xaxis.high) == pytest.approx((-0.4, 4.4))  # 0..4, a tenth of it each end
    assert (yaxis.low, yaxis.high) == pytest.approx((6.0, 54.0))  # 10..50, likewise
    assert (xaxis.title, yaxis.title) == ("px", "py")
    assert frame._core["TNamed"]["fBits"] & NO_STATS  # kNoStats
    assert plot.frame() is frame  # booked once, and kept


def test_a_scatter_plot_in_space_narrows_the_frame_to_the_heights():
    """``TScatter2D``'s frame carries the heights' own range in ``fMinimum``/``fMaximum`` - what
    ``SetRangeUser`` on a ``TH2``'s z axis sets - and its z axis is titled from the fourth part."""
    plot = ScatterPlot.new("sp2", X, Y, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), title="in space;;;w")
    frame = plot.frame3d()
    assert (frame._core["fMinimum"], frame._core["fMaximum"]) == pytest.approx((0.6, 5.4))
    assert frame._core["fZaxis"]["TNamed"]["fTitle"] == "w"


def test_a_points_colour_lies_along_the_scale_the_colours_set():
    """``colour_fractions`` is where each colour lies between the smallest and the largest, 0 to
    1 - and the logarithm's way along for ``logc``."""
    colors = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    plot = ScatterPlot.new("sp", X, Y, colors=colors)
    assert plot.colour_fractions().tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert plot.colour_scale() == pytest.approx((0.0, 100.0))
    # a logarithm's way along needs positive values: 1 and 100 map to the ends, 10 to halfway
    logged = ScatterPlot.new("log", X, Y, colors=np.array([1.0, 10.0, 100.0, 10.0, 1.0]))
    logged.members["fLogC"] = True
    fractions = logged.colour_fractions()
    assert (fractions[0], fractions[2]) == pytest.approx((0.0, 1.0))
    assert fractions[1] == pytest.approx(0.5)  # log10(10) is halfway from log10(1) to log10(100)
    plain = ScatterPlot.new("none", X, Y)
    assert plain.colour_fractions() is None  # points coloured all alike
    assert plain.colour_scale() == (0.0, 1.0)  # no colours, so the unit scale
    nothing = ScatterPlot.new("no colours", X, Y, colors=np.array([]))
    assert nothing.colour_scale() == (0.0, 1.0)  # and an empty column likewise


def test_a_points_marker_size_runs_from_the_least_to_the_most():
    """``marker_sizes`` places each size between ``fMinMarkerSize`` and ``fMaxMarkerSize``; a plot
    given no sizes keeps the marker's own size for every point."""
    sizes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    plot = ScatterPlot.new("sp", X, Y, sizes=sizes)
    # a 1..5 column between the ends 1 and 5 is itself, and between 2 and 10 runs 2 to 10
    assert plot.marker_sizes().tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])
    plot.members["fMinMarkerSize"], plot.members["fMaxMarkerSize"] = 2.0, 10.0
    assert plot.marker_sizes().tolist() == pytest.approx([2.0, 4.0, 6.0, 8.0, 10.0])
    plot.members["fLogS"] = True
    assert plot.marker_sizes()[0] == pytest.approx(2.0)  # the smallest value is the least size
    unsized = ScatterPlot.new("plain", X, Y)
    assert np.all(unsized.marker_sizes() == float(unsized.members["TAttMarker"]["fMarkerSize"]))


def test_a_colour_scale_that_was_set_on_the_frame_wins_over_the_colours():
    """In space the frame's z range is the heights' and not the colours', so a range set on the
    frame is the scale - the colours are then read against the heights, as ROOT draws them."""
    colors = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    plot = ScatterPlot.new("sp2", X, Y, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), colors=colors)
    frame = plot.frame3d()
    frame._core["fMinimum"], frame._core["fMaximum"] = 0.0, 4.0
    assert plot.colour_scale() == pytest.approx((0.0, 4.0))
    assert plot.colour_fractions().tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_a_scatter_plot_needs_points_to_have_a_range():
    """A frame of no points is the unit range, which ``_span`` answers rather than a crash."""
    plot = ScatterPlot.new("empty", np.array([]), np.array([]))
    frame = plot.frame()
    assert (frame.axes[0].low, frame.axes[0].high) == pytest.approx((-0.1, 1.1))
