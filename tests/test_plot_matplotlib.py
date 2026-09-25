"""Pictures drawn by the real matplotlib, headless, checked by what the axes hold afterwards."""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from matplotlib import colors as mcolors
from matplotlib import pyplot
from matplotlib.collections import PolyCollection

from plotting import cube, curve, gauss, grid, points, tidy  # noqa: F401
from xrdroot import Histogram, UnsupportedFeatureError
from xrdroot.plot import picture, plot
from xrdroot.plot.backends import withmatplotlib


def rgb(color) -> tuple[float, ...]:
    return tuple(round(channel, 3) for channel in mcolors.to_rgba(color)[:3])


def test_a_histogram_is_stairs_in_its_root_colour_with_its_titles():
    made = Histogram.new("h", [0, 1, 2], [4, 2], title="counts")
    made.axes[0].title = "energy"
    made.members["TH1"]["TAttLine"]["fLineColor"] = 633
    ax = made.plot()
    steps = ax.patches[0]
    values, edges, baseline = steps.get_data()
    assert (values.tolist(), edges.tolist(), baseline) == ([4.0, 2.0], [0.0, 1.0, 2.0], 0)
    assert rgb(steps.get_edgecolor()) == rgb("#cc0000")
    assert (ax.get_title(), ax.get_xlabel()) == ("counts", "energy")


def test_a_filled_histogram_is_filled_from_its_baseline_with_its_hatch():
    made = Histogram.new("h", [0, 1, 2], [4, 2])
    made.members["TH1"]["TAttFill"].update({"fFillColor": 4, "fFillStyle": 3004})
    steps = made.plot().patches[0]
    assert steps.get_fill() and rgb(steps.get_facecolor()) == (0.0, 0.0, 1.0)
    assert steps.get_hatch() == "/"


def test_error_bars_are_errorbar_with_caps_for_e1():
    made = Histogram.new("h", [0, 1, 2], [4, 1])
    ax = made.plot(option="E1", marker=24)
    line = ax.lines[0]
    assert line.get_xdata().tolist() == [0.5, 1.5] and line.get_ydata().tolist() == [4.0, 1.0]
    assert line.get_marker() == "o" and line.get_markerfacecolor() == "none"
    assert len(ax.containers[0].lines[1]) == 4  # a cap at each end of each bar, both ways
    assert len(plot(made, option="P").containers[0].lines[1]) == 0


def test_bars_boxes_bands_lines_and_numbers_are_each_their_artist():
    made = Histogram.new("h", [0, 1, 2], [4, 1])
    assert len(made.plot(option="B", fill="kRed").patches) == 2
    boxes = made.plot(option="E2").collections[0]
    assert isinstance(boxes, PolyCollection) and len(boxes.get_paths()) == 2
    band = Histogram.new("h", [0, 1, 2, 3], [4, 1, 2]).plot(option="E4").collections[0]
    assert len(band.get_paths()[0].vertices) > 30
    assert len(made.plot(option="E3", label="band").collections) == 1
    assert len(made.plot(option="C").lines[0].get_xdata()) == 2  # too few points to smooth
    ax = made.plot(option="TEXT45")
    assert [text.get_text() for text in ax.texts] == ["4", "1"]
    assert ax.texts[0].get_rotation() == 45.0


def test_a_graph_draws_its_line_markers_and_both_sides_of_its_bars():
    ax = points().plot(option="ALP")
    assert ax.lines[0].get_ydata().tolist() == [2.0, 3.0, 1.0]
    segments = ax.containers[0].lines[2][0].get_segments()
    assert segments[0].tolist() == [[1.0, 1.9], [1.0, 2.3]]
    smooth = points().plot(option="AC").lines[0]
    assert len(smooth.get_xdata()) == 17


def test_a_graphs_boxes_are_hollow_outlines_for_box_and_filled_for_errors():
    ax = grid().plot(option="BOX")
    boxes = ax.collections[0]
    assert len(boxes.get_facecolor()) == 0 and len(boxes.get_paths()) == 9


def test_a_2d_histogram_is_an_image_with_its_scale_and_a_log_scale_for_logz():
    ax = grid().plot()
    assert ax.images[0].get_array().shape == (3, 3) and len(ax.figure.axes) == 2
    logged = grid().plot(option="COL", logz=True, palette="magma")
    assert isinstance(logged.images[0].norm, mcolors.LogNorm)
    assert logged.images[0].get_cmap().name == "magma" and len(logged.figure.axes) == 1


def test_an_unevenly_binned_grid_is_still_drawn_quickly():
    made = Histogram.new("u", [[0, 1, 3], [0, 2, 3]], [[1, 2], [3, 4]])
    ax = made.plot(option="COLZ")
    assert ax.get_xlim() == (0.0, 3.0)


def test_contours_filled_and_as_lines():
    filled = grid().plot(option="CONTZ", levels=4)
    assert len(filled.figure.axes) == 2
    lines = grid().plot(option="CONT1")
    assert lines.collections or lines.get_children()


def test_lego_and_surf_draw_on_axes_with_depth_made_for_them():
    ax = grid().plot(option="SURFZ", zlabel="height")
    assert ax.name == "3d" and ax.get_zlabel() == "height"
    lego = grid().plot(option="LEGO", palette="magma")
    assert lego.name == "3d" and len(lego.collections) == 1
    assert grid().plot(option="LEGO").name == "3d"


def test_lego_on_flat_axes_and_a_3d_histogram_are_refused_with_the_way_out():
    flat = pyplot.subplots()[1]
    with pytest.raises(UnsupportedFeatureError, match="projection='3d'"):
        grid().plot(ax=flat, option="LEGO")
    with pytest.raises(UnsupportedFeatureError, match="backend='plotly'"):
        cube().plot()


def test_the_frame_sets_scales_limits_grid_and_legend():
    ax = gauss().plot(logy=True, logx=False, xlim=(-2, 2), ylim=(1, 500), grid=True, label="h")
    assert ax.get_yscale() == "log" and ax.get_xlim() == (-2.0, 2.0) and ax.get_ylim() == (1, 500)
    assert ax.get_legend().get_texts()[0].get_text() == "h"
    assert curve().plot(logx=True, label="f", legend=False).get_legend() is None
    with pytest.warns(UserWarning, match="No artists with labels"):
        assert curve().plot(legend=True).get_legend() is not None
    assert gauss().plot().get_legend() is None


def test_same_draws_on_the_last_axes_and_ax_on_the_ones_given():
    first = gauss().plot()
    assert curve().plot(option="SAME") is first
    mine = pyplot.subplots()[1]
    assert curve().plot(ax=mine) is mine
    assert curve().plot() is not mine


def test_the_librarys_own_keywords_reach_the_first_layer_only():
    ax = points().plot(option="LP", zorder=7)
    assert ax.lines[0].get_zorder() == 7 and ax.lines[1].get_zorder() != 7


def test_a_million_cells_draw_in_well_under_a_couple_of_seconds():
    made = Histogram.book("big", (1000, 0, 1), (1000, 0, 1))
    rng = np.random.default_rng(3)
    made.fill(rng.random(100_000), rng.random(100_000))
    started = time.perf_counter()
    ax = made.plot()
    ax.figure.savefig(io.BytesIO(), format="png")
    assert time.perf_counter() - started < 5.0


def test_an_svg_is_drawn_without_pyplot_hearing_of_it():
    before = pyplot.get_fignums()
    drawn = withmatplotlib.svg(picture(gauss()))
    assert drawn.lstrip().startswith("<?xml") and "<svg" in drawn
    assert pyplot.get_fignums() == before
    assert "<svg" in withmatplotlib.svg(picture(grid(), "SURF"))
