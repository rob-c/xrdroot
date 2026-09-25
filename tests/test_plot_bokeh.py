"""Pictures drawn as bokeh glyphs, checked by what the figure's renderers hold."""

from __future__ import annotations

import pytest
from bokeh.models import ColorBar, GlyphRenderer, LinearColorMapper, LogColorMapper, LogScale
from bokeh.plotting import figure

from plotting import cube, curve, gauss, grid, points, tidy  # noqa: F401
from xrdroot import Histogram, UnsupportedFeatureError


def glyphs(fig) -> list[str]:
    return [type(r.glyph).__name__ for r in fig.renderers if isinstance(r, GlyphRenderer)]


def data(fig, index: int = 0) -> dict:
    return dict(fig.renderers[index].data_source.data)


def test_a_histogram_is_a_step_through_its_edges_in_its_root_colour():
    made = Histogram.new("h", [0, 1, 2], [4, 2], title="counts")
    made.axes[0].title = "energy"
    made.members["TH1"]["TAttLine"]["fLineColor"] = 2
    fig = made.plot(backend="bokeh")
    assert glyphs(fig) == ["Step"] and fig.renderers[0].glyph.line_color == "#ff0000"
    assert list(data(fig)["x"]) == [0.0, 1.0, 2.0] and list(data(fig)["y"]) == [4.0, 2.0, 2.0]
    assert fig.title.text == "counts" and fig.xaxis[0].axis_label == "energy"


def test_a_filled_histogram_is_a_stepped_area_under_its_outline():
    made = Histogram.new("h", [0, 1, 2], [4, 2])
    fig = made.plot(backend="bokeh", fill="kRed", hatch="/", label="h")
    assert glyphs(fig) == ["VAreaStep", "Step"]
    assert fig.renderers[0].glyph.fill_color == "#ff0000"
    assert fig.renderers[0].glyph.hatch_pattern == "/"
    from xrdroot.plot import stack

    piled = stack([made, made], ["a", "b"], backend="bokeh")
    assert list(data(piled, 2)["y1"]) == [4.0, 2.0, 2.0]


def test_points_are_markers_with_segments_for_their_bars():
    fig = points().plot(backend="bokeh", option="AP", marker=24)
    assert glyphs(fig) == ["Segment", "Scatter"]
    assert list(data(fig)["y1"]) == pytest.approx([2.3, 3.2, 1.1])
    scatter = fig.renderers[1].glyph
    assert scatter.marker == "circle" and scatter.fill_alpha == 0.0
    across = Histogram.new("h", [0, 1, 2], [4, 1]).plot(backend="bokeh", option="E")
    assert glyphs(across) == ["Segment", "Segment", "Scatter"]


def test_bars_boxes_bands_lines_and_numbers_are_each_their_glyph():
    made = Histogram.new("h", [0, 1, 2, 3], [4, 1, 2])
    assert glyphs(made.plot(backend="bokeh", option="B")) == ["Quad"]
    boxes = made.plot(backend="bokeh", option="E2").renderers[0].glyph
    assert boxes.fill_alpha == 0.35 and boxes.line_color is None
    hollow = grid().plot(backend="bokeh", option="BOX").renderers[0].glyph
    assert hollow.fill_alpha == 0.0 and hollow.line_color is not None
    band = made.plot(backend="bokeh", option="E4")
    assert glyphs(band) == ["VArea"] and len(data(band)["x"]) == 17
    assert glyphs(made.plot(backend="bokeh", option="E3")) == ["VArea"]
    assert len(data(made.plot(backend="bokeh", option="C"))["x"]) == 17
    assert glyphs(made.plot(backend="bokeh", option="L")) == ["Line"]
    text = made.plot(backend="bokeh", option="TEXT90")
    assert list(data(text)["text"]) == ["4", "1", "2"] and text.renderers[0].glyph.angle == 90


def test_a_2d_histogram_is_an_image_when_even_and_quads_when_not():
    fig = grid().plot(backend="bokeh")
    assert glyphs(fig) == ["Image"] and isinstance(fig.right[0], ColorBar)
    mapper = fig.renderers[0].glyph.color_mapper
    assert isinstance(mapper, LinearColorMapper) and mapper.palette[0] == "#352a86"
    uneven = Histogram.new("u", [[0, 1, 3], [0, 2, 3]], [[1, 2], [3, 4]])
    cells = uneven.plot(backend="bokeh", option="COL", logz=True, palette="Viridis256")
    assert glyphs(cells) == ["Quad"] and len(data(cells)["value"]) == 4
    assert isinstance(cells.renderers[0].glyph.fill_color.transform, LogColorMapper)
    empty = Histogram.new("e", [[0, 1], [0, 1]], [[0]]).plot(backend="bokeh", option="COL")
    assert empty.renderers[0].glyph.color_mapper.high == 1.0


def test_a_palette_bokeh_does_not_have_is_refused_by_name():
    with pytest.raises(ValueError, match=r"palette='nonesuch'.*Viridis256"):
        grid().plot(backend="bokeh", palette="nonesuch")


def test_contours_filled_with_a_scale_and_as_lines():
    filled = grid().plot(backend="bokeh", option="CONTZ", levels=4)
    assert type(filled.renderers[0]).__name__ == "ContourRenderer"
    assert type(filled.right[0]).__name__ == "ContourColorBar"
    lines = grid().plot(backend="bokeh", option="CONT1")
    assert not len(lines.renderers[0].fill_renderer.data_source.data.get("xs", []))


def test_depth_is_refused_with_plotly_named():
    for thing, option in ((grid(), "LEGO"), (grid(), "SURF"), (cube(), "")):
        with pytest.raises(UnsupportedFeatureError, match="backend='plotly'"):
            thing.plot(backend="bokeh", option=option)


def test_log_axes_are_made_with_the_figure_and_refused_on_one_made_linear():
    fig = gauss().plot(backend="bokeh", logy=True, xlim=(-2, 2), ylim=(1, 500), grid=True)
    assert isinstance(fig.y_scale, LogScale) and (fig.x_range.start, fig.x_range.end) == (-2, 2)
    assert fig.xgrid[0].visible
    with pytest.raises(ValueError, match="y_axis_type='log'"):
        gauss().plot(ax=figure(), backend="bokeh", logy=True)


def test_same_and_a_figure_given_are_drawn_on_and_legend_false_hides_it():
    first = gauss().plot(backend="bokeh", label="h")
    assert curve().plot(backend="bokeh", option="SAME", label="f", legend=False) is first
    assert not first.legend[0].visible
    mine = figure()
    assert curve().plot(ax=mine, backend="bokeh", line_alpha=0.5) is mine
    assert mine.renderers[0].glyph.line_alpha == 0.5
