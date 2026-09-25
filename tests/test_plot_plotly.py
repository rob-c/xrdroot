"""Pictures drawn as plotly traces, checked by what the figure holds."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from plotting import cube, curve, gauss, grid, points, tidy  # noqa: F401
from xrdroot import Histogram
from xrdroot.plot import picture, set_backend
from xrdroot.plot.backends import withplotly


def test_a_histogram_is_a_step_line_through_its_edges_in_its_root_colour():
    made = Histogram.new("h", [0, 1, 2], [4, 2], title="counts")
    made.axes[0].title = "energy"
    made.members["TH1"]["TAttLine"]["fLineColor"] = 4
    figure = made.plot(backend="plotly")
    trace = figure.data[0]
    assert isinstance(figure, go.Figure) and trace.line.shape == "hv"
    assert (list(trace.x), list(trace.y)) == ([0.0, 1.0, 2.0], [4.0, 2.0, 2.0])
    assert trace.line.color == "#0000ff" and not trace.showlegend
    assert figure.layout.title.text == "counts" and figure.layout.xaxis.title.text == "energy"


def test_a_filled_histogram_fills_to_zero_and_a_stacked_one_to_the_one_below():
    low, high = Histogram.new("a", [0, 1, 2], [1, 2]), Histogram.new("b", [0, 1, 2], [3, 1])
    from xrdroot.plot import stack

    figure = stack([low, high], ["a", "b"], backend="plotly")
    fills = [trace.fill for trace in figure.data]
    assert fills == [None, "tonexty", None, "tonexty"]
    assert list(figure.data[3].y) == [4.0, 3.0, 3.0] and figure.data[3].name == "b"
    alone = Histogram.new("h", [0, 1, 2], [1, 2]).plot(backend="plotly", fill="kRed")
    assert alone.data[0].fill == "tozeroy" and alone.data[0].fillcolor == "#ff0000"


def test_error_bars_are_asymmetric_errors_on_markers_with_roots_symbol():
    figure = points().plot(backend="plotly", option="AP", marker=24, label="scan")
    trace = figure.data[0]
    assert trace.mode == "markers" and trace.marker.symbol == "circle-open"
    assert list(trace.error_y.array) == [0.3, 0.2, 0.1]
    assert list(trace.error_y.arrayminus) == [0.1, 0.2, 0.1]
    assert trace.error_y.width == 4 and not trace.error_x.visible
    assert trace.name == "scan" and trace.showlegend
    assert points().plot(backend="plotly", option="P", marker=3).data[0].marker.symbol == (
        "asterisk-open"
    )
    assert gauss().plot(backend="plotly", option="E").data[0].marker.size == 3


def test_boxes_bands_bars_curves_and_numbers_are_each_their_trace():
    made = Histogram.new("h", [0, 1, 2], [4, 1])
    boxes = made.plot(backend="plotly", option="E2").data[0]
    assert boxes.fill == "toself" and np.isnan(boxes.x[5]) and boxes.opacity == 0.35
    hollow = grid().plot(backend="plotly", option="BOX").data[0]
    assert hollow.fill is None and hollow.line.width == 1
    below, above = made.plot(backend="plotly", option="E4").data
    assert above.fill == "tonexty" and above.line.shape == "spline" and below.line.width == 0
    bar = made.plot(backend="plotly", option="B").data[0]
    assert isinstance(bar, go.Bar) and list(bar.width) == pytest.approx([0.8, 0.8])
    assert made.plot(backend="plotly", option="C").data[0].line.shape == "spline"
    assert list(made.plot(backend="plotly", option="TEXT").data[0].text) == ["4", "1"]


def test_a_2d_histogram_is_a_heatmap_over_its_edges_in_roots_palette():
    figure = grid().plot(backend="plotly")
    heat = figure.data[0]
    assert isinstance(heat, go.Heatmap) and list(heat.x) == [0.0, 1.0, 2.0, 3.0]
    assert np.asarray(heat.z).tolist() == grid().values().T.tolist() and heat.showscale
    assert heat.colorscale[0][1] == "#352a86"
    logged = grid().plot(backend="plotly", logz=True, zlabel="n", palette="magma").data[0]
    assert np.asarray(logged.z)[0, 0] == pytest.approx(np.log10(501))
    assert logged.colorbar.title.text == "log10 n"


def test_contours_surfaces_lego_and_3d_histograms():
    contour = grid().plot(backend="plotly", option="CONT1").data[0]
    assert isinstance(contour, go.Contour) and contour.contours.coloring == "lines"
    assert grid().plot(backend="plotly", option="CONT", logz=True).data[0].contours.coloring == (
        "fill"
    )
    surface = grid().plot(backend="plotly", option="SURF", zlabel="n")
    assert isinstance(surface.data[0], go.Surface) and surface.layout.scene.zaxis.title.text == "n"
    lego = grid().plot(backend="plotly", option="LEGO").data[0]
    assert list(lego.x) == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0] and np.asarray(lego.z).shape == (6, 6)
    cloud = cube().plot(backend="plotly").data[0]
    assert isinstance(cloud, go.Scatter3d) and list(cloud.x) == [0.5]
    assert isinstance(cube().plot(backend="plotly", option="ISO").data[0], go.Isosurface)


def test_a_cloud_of_nothing_still_draws():
    empty = Histogram.book("e", (2, 0, 1), (2, 0, 1), (2, 0, 1))
    assert len(empty.plot(backend="plotly").data[0].x) == 0


def test_the_frame_sets_log_axes_limits_in_decades_and_the_legend():
    figure = gauss().plot(backend="plotly", logy=True, ylim=(1, 1000), xlim=(-2, 2), legend=False)
    assert figure.layout.yaxis.type == "log" and list(figure.layout.yaxis.range) == [0.0, 3.0]
    assert list(figure.layout.xaxis.range) == [-2.0, 2.0] and figure.layout.showlegend is False
    assert curve().plot(backend="plotly", logx=True, grid=True).layout.xaxis.showgrid


def test_same_adds_to_the_last_figure_and_a_row_of_a_subplot_can_be_the_target():
    first = gauss().plot(backend="plotly")
    assert curve().plot(backend="plotly", option="SAME") is first and len(first.data) == 2
    from plotly.subplots import make_subplots

    rows = make_subplots(rows=2, cols=1)
    assert curve().plot(ax=(rows, 2), backend="plotly") is rows
    assert rows.data[0].yaxis == "y2"


def test_the_librarys_own_keywords_reach_the_first_trace():
    figure = points().plot(backend="plotly", option="LP", opacity=0.5)
    assert figure.data[0].opacity == 0.5 and figure.data[1].opacity is None


def test_the_backend_set_is_the_one_drawn_with():
    set_backend("plotly")
    assert isinstance(gauss().plot(), go.Figure)


def test_html_is_a_div_with_plotlys_script_from_its_cdn():
    made = withplotly.html(picture(gauss()))
    assert made.startswith("<div") and "cdn.plot.ly" in made
