"""Drawing ROOT's canvases with matplotlib.

The canvases here are built in memory, a pad's members and the objects it
draws - histograms and graphs as this library makes them, the drawing
classes as :class:`~xrdroot.canvas.Primitive` - which is what reading one
gives back, and the tests check what lands on the figure: an axes per pad
where the pad's margins put its frame, the data by its draw option, text,
lines, paves and legends where ROOT would put them, in ROOT's colours. One
test draws ROOT's own ``tcanvas.root``; none compares pictures.

What ROOT draws as lines of whole pixels is drawn here the same way, as a
:class:`~xrdroot.canvas.raster.PixelLine` holding its polylines in the
canvas's pixels (``y`` down), and its text as matplotlib text placed in
those pixels too; the tests read both back.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

from xrdroot import Canvas, Function, Graph, Histogram, UnsupportedFeatureError, open_root
from xrdroot.buffer import Listed
from xrdroot.canvas import Pad, Primitive, render
from xrdroot.canvas.marks import MARKER_GID
from xrdroot.canvas.paint import CanvasWarning
from xrdroot.canvas.raster import PixelLine
from xrdroot.profile import Profile
from xrdroot.stacks import MultiGraph, Stack

DATA = __file__.rsplit("/", 1)[0] + "/data"

#: What a drawing class's attributes are when a test does not say.
ATTRIBUTES = {
    "fLineColor": 1,
    "fLineStyle": 1,
    "fLineWidth": 1,
    "fFillColor": 0,
    "fFillStyle": 1001,
    "fMarkerColor": 1,
    "fMarkerStyle": 1,
    "fMarkerSize": 1.0,
    "fTextAlign": 11,
    "fTextColor": 1,
    "fTextFont": 42,
    "fTextSize": 0.05,
    "fBits": 0x03000000,
}
NDC = 1 << 14


@pytest.fixture(autouse=True)
def _nothing_warns_unless_a_test_says():
    """A canvas drawn here raises any warning it gives, matplotlib's included."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def prim(classname: str, **members):
    return Primitive(classname, {**ATTRIBUTES, **members})


def listed(primitives):
    return Listed([obj for obj, _ in primitives], [option for _, option in primitives])


def pad_members(name, primitives, **members):
    return {
        "fName": name,
        "fTitle": name,
        "fX2": 1.0,
        "fY2": 1.0,
        "fWNDC": 1.0,
        "fHNDC": 1.0,
        "fFillColor": 0,
        "fFillStyle": 1001,
        "fPrimitives": listed(primitives),
        **members,
    }


def sub(name, primitives, **members):
    return Pad("TPad", pad_members(name, primitives, **members))


def make(primitives, width=700, height=500, **members):
    return Canvas(
        "TCanvas", {"TPad": pad_members("c", primitives, **members), "fCw": width, "fCh": height}
    )


def filled(bins=(10, 0.0, 10.0), *values, name="h", title=""):
    h = Histogram.book(name, bins, title=title)
    h.fill(np.asarray(values or [1, 2, 2, 3, 3, 3, 4, 4, 5, 7.5], dtype=float))
    return h


def only(fig, label):
    (ax,) = [ax for ax in fig.axes if ax.get_label() == label]
    return ax


def lines(ax, color=None, clipped=None):
    """The pixel lines on ``ax``, those of one colour, or those clipped to a frame or not."""
    found = [a for a in ax.get_children() if isinstance(a, PixelLine)]
    if color is not None:
        found = [a for a in found if to_rgb(a.color) == color]
    if clipped is not None:
        found = [a for a in found if (a.pixel_clip is not None) == clipped]
    return found


def polylines(ax, color=None, clipped=None):
    """Every polyline those lines hold, as lists of pixels."""
    return [line.tolist() for a in lines(ax, color, clipped) for line in a.lines]


def words(ax):
    """What is written on ``ax``, a piece at a time, without the spaces round each."""
    return [t.get_text().strip() for t in ax.texts]


def data_words(ax):
    """What is written on ``ax`` in its data's units: a bin's content, drawn ``TEXT``."""
    return [t.get_text() for t in ax.texts if t.get_transform() is ax.transData]


def written(ax):
    """The pieces of text on ``ax`` by what they say."""
    return {t.get_text().strip(): t for t in ax.texts}


def marks(ax):
    """The markers on ``ax``, in one collection per drawing, placed at pixels' centres."""
    return [c for c in ax.collections if c.get_gid() == MARKER_GID]


def fills(ax):
    """The filled polygons on ``ax`` in the canvas's pixels: bands, boxes, bars and areas."""
    return [p for p in ax.get_children() if isinstance(p, Polygon)]


# -- ROOT's own canvas -----------------------------------------------------------


def test_roots_canvas_draws_one_axes_where_its_margins_put_the_frame():
    with open_root(f"{DATA}/tcanvas.root") as f:
        c = f["c1"]
    fig = c.plot()
    (ax,) = fig.axes
    assert ax.get_label() == "c1"
    assert ax.get_position().bounds == pytest.approx((0.1, 0.1, 0.8, 0.8))
    assert tuple(fig.get_size_inches() * fig.dpi) == pytest.approx((296, 372))
    # "alp": a line through the points, in the frame's pixels, and a marker on each
    (graph, _fit) = lines(ax, clipped=True)
    assert graph.lines[0].tolist() == [[30, 335], [83, 200], [137, 64], [191, 267], [245, 132]]
    assert len(marks(ax)[0].get_offsets()) == 5
    # a tenth of their spread each side, but not below zero where none of them are
    assert ax.get_xlim() == pytest.approx((0.0, 4.4))
    assert ax.get_ylim() == pytest.approx((0.0, 4.4))


def test_a_canvas_saves_as_png_pdf_and_svg(tmp_path):
    with open_root(f"{DATA}/tcanvas.root") as f:
        c = f["c1"]
    for suffix in ("png", "pdf", "svg"):
        out = tmp_path / f"c1.{suffix}"
        assert c.save(out) == out
        assert out.stat().st_size > 1000
    assert c.save_as(tmp_path / "again.png").read_bytes()[:4] == b"\x89PNG"
    assert render(c, tmp_path / "r.png").stat().st_size > 1000


def test_render_takes_a_canvas_or_pad_as_members_and_refuses_anything_else(tmp_path):
    canvas_members = {"TPad": pad_members("c", []), "fCw": 300, "fCh": 200}
    assert render(canvas_members, tmp_path / "c.png").stat().st_size > 100
    assert render(pad_members("p", []), tmp_path / "p.png").stat().st_size > 100
    with pytest.raises(TypeError, match="only a canvas or a pad"):
        render(Histogram.book("h", (2, 0.0, 1.0)), tmp_path / "h.png")


def test_a_canvas_draws_onto_a_figure_it_is_given():
    fig = Figure()
    assert make([]).plot(figure=fig) is fig
    assert len(fig.axes) == 1


def test_drawing_a_canvas_without_matplotlib_says_how_to_get_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    with pytest.raises(UnsupportedFeatureError, match="pip install matplotlib"):
        make([]).plot()


# -- pads ------------------------------------------------------------------------


def test_every_pad_is_an_axes_at_its_place_framed_by_its_margins():
    h = filled()
    left = sub("c_1", [(h, "hist")], fXlowNDC=0.0, fYlowNDC=0.0, fWNDC=0.5, fHNDC=1.0)
    right = sub(
        "c_2",
        [(filled(name="g"), "")],
        fXlowNDC=0.5,
        fWNDC=0.5,
        fHNDC=0.5,
        fYlowNDC=0.5,
        fLeftMargin=0.2,
        fRightMargin=0.0,
        fBottomMargin=0.3,
        fTopMargin=0.1,
    )
    fig = make([(left, ""), (right, "")]).plot()
    assert [ax.get_label() for ax in fig.axes] == ["c", "c_1", "c_2"]
    assert not only(fig, "c").axison  # a pad with no frame has no axis lines
    assert only(fig, "c_1").get_position().bounds == pytest.approx((0.05, 0.1, 0.4, 0.8))
    assert only(fig, "c_2").get_position().bounds == pytest.approx((0.6, 0.65, 0.4, 0.3))


def test_a_pad_draws_alone_filling_a_figure():
    p = sub("p", [(filled(), "")], fXlowNDC=0.5, fWNDC=0.5)
    fig = p.plot()
    assert only(fig, "p").get_position().bounds == pytest.approx((0.1, 0.1, 0.8, 0.8))


def test_a_pad_is_scaled_gridded_and_ticked_as_it_says():
    h = filled()
    fig = make([(h, "")], fLogy=1, fLogx=1, fGridx=True, fGridy=True, fTickx=1, fTicky=1).plot()
    (ax,) = fig.axes
    assert ax.get_xscale() == ax.get_yscale() == "log"
    grid = polylines(ax) and [line for a in lines(ax) if a.dashes == (1, 2) for line in a.lines]
    upright = [line for line in grid if line[0, 0] == line[1, 0]]
    across = [line for line in grid if line[0, 1] == line[1, 1]]
    assert upright and across  # dotted across the frame at each tick, both ways
    assert all(sorted(line[:, 1]) == [50, 450] for line in upright)
    ticked = [line for line in polylines(ax, (0.0, 0.0, 0.0), clipped=False) if len(line) == 2]
    assert any(min(y for _, y in line) == 50 < max(y for _, y in line) < 70 for line in ticked)
    assert any(max(x for x, _ in line) == 630 > min(x for x, _ in line) > 610 for line in ticked)
    low, high = ax.get_ylim()
    assert low == pytest.approx(0.5) and high == pytest.approx(
        6.0
    )  # half the lowest bin, twice the highest


def test_a_pad_drawn_before_it_was_saved_keeps_the_frame_it_was_drawn_with():
    frame = prim("TFrame", fFillColor=5, fFillStyle=1001, fLineColor=2, fLineWidth=3)
    fig = make(
        [(filled(), ""), (frame, "")], fUxmin=-1.0, fUxmax=11.0, fUymin=0.0, fUymax=2.0, fLogy=1
    ).plot()
    (ax,) = fig.axes
    assert ax.get_xlim() == (-1.0, 11.0)
    assert ax.get_ylim() == pytest.approx((1.0, 100.0))
    backs = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_zorder() == -50]
    assert backs
    assert backs[0].get_facecolor()[:3] == (1.0, 1.0, 0.0)
    (edge,) = lines(ax, (1.0, 0.0, 0.0))
    assert edge.thick == 3
    assert edge.lines[0].tolist() == [[70, 50], [630, 50], [630, 450], [70, 450], [70, 50]]


def test_a_pad_with_no_frame_is_ranged_by_its_own_coordinates():
    line = prim("TLine", fX1=10.0, fY1=10.0, fX2=20.0, fY2=20.0)
    fig = make([(line, "")], fX1=0.0, fY1=0.0, fX2=40.0, fY2=40.0).plot()
    (ax,) = fig.axes
    assert ax.get_xlim() == (0.0, 40.0)
    assert not ax.axison
    assert polylines(ax) == [[[175, 375], [350, 250]]]  # a quarter and a half of 700 by 500


@pytest.mark.parametrize(("mode", "top"), [(1, "light"), (-1, "dark")])
def test_a_pad_with_a_border_is_raised_or_sunken(mode, top):
    fig = make([], fBorderMode=mode, fBorderSize=3, fFillColor=16).plot()
    upper, lower = [p for p in fig.axes[0].patches if isinstance(p, Polygon)]
    brighter = sum(upper.get_facecolor()[:3]) > sum(lower.get_facecolor()[:3])
    assert brighter == (top == "light")


def test_a_hollow_pad_draws_no_background():
    fig = make([], fFillStyle=0).plot()
    assert not [p for p in fig.axes[0].patches if p.get_zorder() == -100]


def test_the_canvas_background_is_the_canvas_colour():
    fig = make([], fFillColor=2).plot()
    assert fig.get_facecolor()[:3] == (1.0, 0.0, 0.0)


def test_a_frame_without_a_tframe_takes_its_pads_frame_colours():
    fig = make([(filled(), "")], fFrameFillColor=3, fFrameFillStyle=1001, fFrameLineColor=4).plot()
    (ax,) = fig.axes
    backs = [p for p in ax.patches if p.get_zorder() == -50]
    assert backs[0].get_facecolor()[:3] == (0.0, 1.0, 0.0)
    assert len(lines(ax, (0.0, 0.0, 1.0))) == 1  # the frame's edge


def test_a_hollow_frame_is_not_filled():
    fig = make([(filled(), "")], fFrameFillStyle=0).plot()
    assert not [p for p in fig.axes[0].patches if p.get_zorder() == -50]


# -- histograms by option ---------------------------------------------------------


def _drawn(h, option, **pad):
    fig = make([(h, option)], **pad).plot()
    return fig, only(fig, "c")


#: The frame of a 700 by 500 canvas with ROOT's margins, in its pixels: left, top, right, bottom.
FRAME = (70, 50, 630, 450)
#: The pixel rows of 0, 1, 2 and 3 in a frame reaching 3.15, and the middles of its bins.
ROWS = {0: 450, 1: 323, 2: 196, 3: 69}
MIDDLES = [98, 154, 210, 266, 322, 378, 434, 490, 546, 602]


def _levels(outline):
    """The rows a histogram's outline runs along, a bin at a time."""
    points = outline.lines[0]
    return [int(points[i, 1]) for i in range(1, len(points), 2)]


def test_a_histogram_drawn_hist_is_its_outline_with_its_axis_titles_and_title():
    h = filled(title="p_{T} spectrum")
    h.axes[0].title = "p_{T} [GeV]"
    h.members["TH1"]["fXaxis"]["TNamed"]["fTitle"] = "p_{T} [GeV]"
    h.members["TH1"]["TAttLine"]["fLineColor"] = 4
    _fig, ax = _drawn(h, "hist")
    (outline,) = lines(ax, (0.0, 0.0, 1.0))
    assert outline.pixel_clip == FRAME
    assert _levels(outline) == [ROWS[int(v)] for v in h.values()]
    said = words(ax)
    assert said.count("p") == said.count("T") == 2  # in the title, and under the axis
    assert "[GeV]" in said and "spectrum" in said
    assert ax.get_xlabel() == ax.get_title() == ""  # both are drawn as ROOT draws them
    assert ax.get_xlim() == (0.0, 10.0)
    assert ax.get_ylim() == pytest.approx((0.0, 3.15))  # five percent over the highest bin


def test_a_histogram_with_no_title_bit_or_title_draws_none():
    h = filled(title="shown")
    named = h.members["TH1"]["TNamed"]
    named["fBits"] = named.get("fBits", 0) | 1 << 17
    _fig, ax = _drawn(h, "hist")
    assert "shown" not in words(ax)


def test_a_histogram_is_drawn_filled_and_hatched_as_its_fill_says():
    h = filled()
    h.members["TH1"]["TAttFill"].update(fFillColor=2, fFillStyle=1001)
    _fig, ax = _drawn(h, "")
    area = fills(ax)[0]
    assert area.get_facecolor()[:3] == (1.0, 0.0, 0.0)
    h.members["TH1"]["TAttFill"].update(fFillStyle=3004)
    _fig, ax = _drawn(h, "")
    assert [p.get_hatch() for p in fills(ax)][:1] == ["//"]


def _bars(ax):
    """Each error bar's upright arms, as the column they stand in and their ends' rows."""
    arms = [line for line in polylines(ax, clipped=True) if len(line) == 2]
    upright = [a for a in arms if a[0][0] == a[1][0] and abs(a[0][1] - a[1][1]) > 4]
    return sorted({(line[0][0], line[0][1]) for line in upright})


def test_a_histogram_drawn_e1_has_bars_with_ends_and_skips_empty_bins():
    h = filled()
    _fig, ax = _drawn(h, "e1")
    (centres,) = marks(ax)
    columns = [154, 210, 266, 322, 378, 490]
    assert [x for x, _ in centres.get_offsets()] == [m + 0.5 for m in columns]
    assert [column for column, _ in _bars(ax)] == [154, 210, 266, 322, 378, 490]
    ends = [line for line in polylines(ax, clipped=True) if line[0][1] == line[1][1]]
    assert [[152, 289], [156, 289]] in ends  # a cap of two pixels either side of the arm
    assert ax.get_ylim()[1] == pytest.approx((3 + np.sqrt(3)) * 1.05)


def test_a_histogram_drawn_e0_keeps_its_empty_bins():
    _fig, ax = _drawn(filled(), "e0")
    columns = {line[0][0] for line in polylines(ax, clipped=True) if len(line) == 2}
    assert set(MIDDLES) <= columns


def test_a_weighted_histogram_and_a_profile_draw_error_bars_unasked():
    h = filled()
    h.sumw2()
    _fig, ax = _drawn(h, "")
    assert _bars(ax) and marks(ax)
    p = Profile.book("p", (4, 0.0, 4.0))
    p.fill(np.array([0.5, 1.5]), np.array([2.0, 3.0]))
    _fig, ax = _drawn(p, "")
    # one entry a bin has no spread: a point and its bin's width, and no outline
    assert marks(ax)[0].get_offsets().tolist() == [[140.5, 196.5], [280.5, 69.5]]
    assert polylines(ax, clipped=True)[:2] == [[[140, 196], [70, 196]], [[140, 196], [210, 196]]]


def _mapped(ax):
    """The one thing on ``ax`` drawn in the colours of a scale."""
    (mapped,) = [a for a in (*ax.images, *ax.collections) if getattr(a, "norm", None) is not None]
    return mapped


def test_a_histogram_drawn_e2_is_a_box_round_each_bin():
    _fig, ax = _drawn(filled(), "e2")
    boxes = fills(ax)[:10]
    heights = [np.ptp(np.asarray(box.get_xy())[:, 1]) for box in boxes]
    assert heights[3] == round(2 * np.sqrt(3) / ax.get_ylim()[1] * 400)  # 3, give or take its root
    assert heights[0] == 0  # an empty bin's box is flat


def test_a_histogram_drawn_e3_is_a_band_through_its_bins():
    _fig, ax = _drawn(filled(), "e3")
    (band,) = fills(ax)[:1]
    assert [int(x) for x, _ in band.get_xy()][:6] == MIDDLES[:6]


def test_a_histogram_drawn_p_or_l_marks_its_bins():
    _fig, ax = _drawn(filled(), "p")
    (centres,) = marks(ax)
    assert [x - 0.5 for x, _ in centres.get_offsets()] == [154, 210, 266, 322, 378, 490]
    _fig, ax = _drawn(filled(), "l")
    (line,) = lines(ax, clipped=True)
    assert [x for x, _ in line.lines[0]] == MIDDLES
    assert not marks(ax)


def test_an_option_the_picture_refuses_draws_as_without_it_and_says_so():
    with pytest.warns(CanvasWarning, match=r"drawn without its option '\*h'"):
        _fig, ax = _drawn(filled(), "*h")
    (outline,) = lines(ax, clipped=True)  # HIST, as without it
    assert len(outline.lines[0]) == 20


def test_a_histogram_drawn_bar_is_a_bar_per_bin():
    _fig, ax = _drawn(filled(), "bar")
    bars = [line for line in polylines(ax, clipped=True) if len(line) == 5]
    assert len(bars) == 10
    assert bars[3] == [[244, 450], [288, 450], [288, 69], [244, 69], [244, 450]]  # a tenth in


def test_a_histogram_drawn_text_writes_each_bin_that_is_not_empty():
    _fig, ax = _drawn(filled(), "text")
    shown = sorted(t for t in data_words(ax) if t in ("1", "2", "3"))
    assert shown == sorted(["1", "2", "3", "2", "1", "1"])


def test_a_two_dimensional_histogram_drawn_colz_has_its_colour_scale_beside_it():
    h = Histogram.book("h2", (4, 0.0, 4.0), (2, 0.0, 2.0))
    h.fill(np.array([0.5, 1.5, 1.5]), np.array([0.5, 0.5, 0.5]))
    fig, ax = _drawn(h, "colz")
    assert _mapped(ax).get_clim() == (1.0, 2.0)  # an empty bin is not painted
    assert only(fig, "c palette").get_position().x0 == pytest.approx(0.905)
    assert (ax.get_xlim(), ax.get_ylim()) == ((0.0, 4.0), (0.0, 2.0))


def test_a_colour_scale_goes_where_its_saved_palette_axis_was():
    h = Histogram.book("h2", (2, 0.0, 2.0), (2, 0.0, 2.0))
    h.fill(np.array([0.5]), np.array([0.5]))
    h.functions.append(prim("TPaletteAxis", fX1NDC=0.91, fY1NDC=0.2, fX2NDC=0.95, fY2NDC=0.8))
    fig, _ax = _drawn(h, "colz")
    assert only(fig, "c palette").get_position().bounds == pytest.approx((0.91, 0.2, 0.04, 0.6))


def test_a_colour_plot_on_a_pad_drawn_logz_is_scaled_logarithmically():
    from matplotlib.colors import LogNorm

    h = Histogram.book("h2", (2, 0.0, 2.0), (2, 0.0, 2.0))
    h.fill(np.array([0.5, 0.5, 1.5]), np.array([0.5, 0.5, 1.5]))
    _fig, ax = _drawn(h, "col", fLogz=1)
    assert isinstance(_mapped(ax).norm, LogNorm)


def test_a_two_dimensional_histogram_with_no_option_is_shaded():
    h = Histogram.book("h2", (2, 0.0, 2.0), (1, 0.0, 1.0))
    h.fill(np.array([0.5, 1.5]), np.array([0.5, 0.5]), weight=np.array([-2.0, 3.0]))
    _fig, ax = _drawn(h, "")
    assert _mapped(ax).get_clim() == (-2.0, 3.0)


def test_a_two_dimensional_histogram_drawn_box_cont_and_text():
    h = Histogram.book("h2", (2, 0.0, 2.0), (2, 0.0, 2.0))
    h.fill(np.array([0.5, 0.5, 1.5]), np.array([0.5, 0.5, 1.5]))
    _fig, ax = _drawn(h, "box")
    boxes = [line for line in polylines(ax, clipped=True) if len(line) == 5]
    assert boxes == [  # the fullest bin fills its cell, the other a box of half its area
        [[70, 450], [350, 450], [350, 250], [70, 250], [70, 450]],
        [[391, 221], [589, 221], [589, 79], [391, 79], [391, 221]],
    ]
    _fig, ax = _drawn(h, "cont")
    assert ax.collections
    _fig, ax = _drawn(h, "text")
    assert sorted(t for t in data_words(ax) if t in ("1", "2")) == ["1", "2"]


@pytest.mark.parametrize("option", ["lego", "surf"])
def test_a_two_dimensional_histogram_drawn_lego_or_surf_stands_in_a_box_with_no_frame(option):
    h = Histogram.book("h2", (2, 0.0, 2.0), (2, 0.0, 2.0))
    h.fill(np.array([0.5]), np.array([0.5]))
    _fig, ax = _drawn(h, option)
    assert not ax.axison
    front = [a for a in lines(ax) if a.dashes == () and len(a.lines) == 2]
    assert [len(line) for line in front[-1].lines] == [4, 4]  # the box's two front faces
    assert [a for a in lines(ax) if a.dashes == (1, 2)]  # its back walls lined at z's divisions
    assert {"0", "1", "2"} <= set(words(ax))  # and its axes, labelled


def test_a_three_dimensional_histogram_is_left_out_with_a_warning():
    h = Histogram.book("h3", (2, 0.0, 2.0), (2, 0.0, 2.0), (2, 0.0, 2.0))
    with pytest.warns(CanvasWarning, match="three dimensions"):
        make([(h, "")]).plot()


# -- stats boxes ------------------------------------------------------------------


def test_a_histogram_saved_without_a_stats_box_is_drawn_with_gstyles():
    _fig, ax = _drawn(filled(), "")
    texts = words(ax)
    for expected in ("h", "Entries", "10", "Mean", "3.45", "Std Dev", "1.739"):
        assert expected in texts


def test_a_histogram_told_kno_stats_or_drawn_same_has_no_stats_box():
    h = filled()
    named = h.members["TH1"]["TNamed"]
    named["fBits"] = named.get("fBits", 0) | 1 << 9
    _fig, ax = _drawn(h, "")
    assert "Entries" not in [t.get_text() for t in ax.texts]
    other = filled(name="first")
    fig = make(
        [(other, "hist"), (filled(name="second"), "same"), (filled(name="third"), "sames")]
    ).plot()
    names = [t.get_text() for t in fig.axes[0].texts]
    assert "first" in names
    assert "second" not in names
    assert "third" in names


def test_a_saved_stats_box_is_drawn_with_the_lines_it_was_saved_with():
    h = filled()
    # only TLatex lines are painted, as TPaveStats::Paint paints them
    saved = Listed([prim("TLatex", fTitle="h"), prim("TLatex", fTitle="Entries = 10"), "TUnknown"])
    h.functions.append(
        prim(
            "TPaveStats",
            fX1NDC=0.7,
            fY1NDC=0.7,
            fX2NDC=0.9,
            fY2NDC=0.9,
            fOption="brNDC",
            fLines=saved,
            fTextSize=0.0,
            fBorderSize=1,
            fOptStat=11,
        )
    )
    _fig, ax = _drawn(h, "")
    texts = words(ax)
    assert texts.count("Entries") == 1
    assert "10" in texts
    assert "Mean" not in texts


def test_a_stats_box_lists_what_each_digit_of_fOptStat_asks_for():
    from xrdroot.canvas.statbox import stats_rows

    h = filled()
    names = [name for name, _ in stats_rows(h, 111111111)]
    assert names == [
        "h",
        "Entries",
        "Mean",
        "Std Dev",
        "Underflow",
        "Overflow",
        "Integral",
        "Skewness",
        "Kurtosis",
    ]
    (mean,) = [value for name, value in stats_rows(h, 200) if name == "Mean"]
    assert "#pm" in mean
    h2 = Histogram.book("h2", (2, 0.0, 2.0), (2, 0.0, 2.0))
    h2.fill(np.array([0.5]), np.array([0.5]))
    assert [name for name, _ in stats_rows(h2, 1001110)] == [
        "Entries",
        "Mean x",
        "Mean y",
        "Std Dev x",
        "Std Dev y",
        "Integral",
    ]


def test_a_stats_box_describes_the_fit_hung_on_the_histogram_as_fOptFit_asks():
    from xrdroot.canvas.statbox import fit_rows

    h = filled()
    assert fit_rows(h, 111) == []
    f = Function("f", "pol1", range=(0.0, 10.0), parameters=[1.0, 2.0])
    h.attach(f)
    assert fit_rows(h, 0) == []
    assert [name for name, _ in fit_rows(h, 111)] == ["p0", "p1"]  # never fitted: no chi-square
    f.fit_result = {"chi2": 2.0, "ndf": 3, "npfits": 5}
    rows = fit_rows(h, 111)
    assert rows[0] == ("#chi^{2} / ndf", "2 / 3")
    assert rows[1][0] == "Prob"
    assert [name for name, _ in fit_rows(h, 1)] == ["p0", "p1"]


def test_a_second_stats_box_made_in_a_pad_goes_below_the_first():
    fig = make([(filled(name="a"), ""), (filled(name="b"), "sames")]).plot()
    tops = [t.get_position()[1] for t in fig.axes[0].texts if t.get_text() in ("a", "b")]
    assert tops[0] < tops[1]  # rows of pixels count down the canvas


# -- functions --------------------------------------------------------------------


def test_a_function_is_drawn_over_its_range_and_a_fit_with_its_histogram():
    f = Function("f", "pol1", range=(0.0, 10.0), parameters=[1.0, 2.0])
    fig = make([(f, "")]).plot()
    (ax,) = fig.axes
    (line,) = lines(ax, clipped=True)
    assert line.lines[0][0].tolist() == [70, 432]  # 1 at the left edge of a frame up to 22.05
    assert line.lines[0][-1].tolist() == [630, round(450 - 21 / 22.05 * 400)]
    assert (line.color, line.thick) == ("#ff0000", 2)  # gStyle's function colour and width
    h = filled()
    h.attach(Function("fit", "pol0", range=(0.0, 10.0), parameters=[2.0]))
    hidden = Function("hidden", "pol0", range=(0.0, 10.0), parameters=[9.0])
    hidden.members["TNamed"]["fBits"] = hidden.members["TNamed"].get("fBits", 0) | 1 << 9
    h.attach(hidden)
    _fig, ax = _drawn(h, "")
    (fit,) = lines(ax, (1.0, 0.0, 0.0), clipped=True)
    assert {y for _, y in fit.lines[0]} == {ROWS[2]}  # the fit, not the hidden function
    _fig, ax = _drawn(h, "hist")
    assert not lines(ax, (1.0, 0.0, 0.0))  # HIST draws the histogram alone


def test_a_function_of_two_variables_is_its_contours_and_one_that_will_not_evaluate_is_left_out():
    two = Function("f2", "x*y", range=[(0.0, 1.0), (0.0, 1.0)])
    assert make([(two, "")]).plot().axes[0].collections

    def refuses(x, p):
        raise UnsupportedFeatureError("this model is compiled code with nothing saved")

    broken = Function.from_callable("broken", refuses, 0, range=(0.0, 1.0))
    with pytest.warns(CanvasWarning, match="compiled code"):
        fig = make([(broken, "")]).plot()
    assert fig.axes[0].get_ylim() == (0.0, 1.0)


# -- graphs -----------------------------------------------------------------------


def _graph(errors=True):
    if errors:
        return Graph.new(
            "g", [1.0, 2.0, 3.0], [2.0, 4.0, 3.0], yerr=[0.5, 0.5, 0.5], xerr=[0.1] * 3
        )
    return Graph.new("g", [1.0, 2.0, 3.0], [2.0, 4.0, 3.0], title="the graph")


def test_a_graph_drawn_ap_is_markers_and_bars_on_axes_it_makes():
    fig = make([(_graph(), "ap")]).plot()
    (ax,) = fig.axes
    (centres,) = marks(ax)
    assert centres.get_offsets().tolist() == [[138.5, 361.5], [350.5, 139.5], [562.5, 250.5]]
    assert [[136, 306], [140, 306]] in polylines(ax)  # the top of the first bar
    assert ax.get_xlim() == pytest.approx((0.9 - 0.22, 3.1 + 0.22))
    assert ax.get_ylim() == pytest.approx((1.5 - 0.3, 4.5 + 0.3))


def test_a_graph_without_errors_draws_its_title_and_a_line_by_default():
    fig = make([(_graph(errors=False), "a")]).plot()
    (ax,) = fig.axes
    (line,) = lines(ax, clipped=True)
    assert line.lines[0].tolist() == [[117, 417], [350, 83], [583, 250]]
    assert "the graph" in words(ax)


def test_a_graph_drawn_with_a_star_x_or_z_marks_its_points_so():
    from xrdroot.canvas.marks import marker_path

    fig = make([(_graph(), "a*")]).plot()
    (stars,) = marks(fig.axes[0])
    star = stars.get_paths()[0]  # ROOT's asterisk, kStar: four strokes through the point
    assert star.codes.tolist() == [1, 2] * 4
    assert len(star.vertices) == len(marker_path(3, 1.0)[0].vertices)
    fig = make([(_graph(), "apx")]).plot()
    assert not polylines(fig.axes[0], clipped=True) and marks(fig.axes[0])
    fig = make([(_graph(), "apz")]).plot()
    arms = polylines(fig.axes[0], clipped=True)
    assert len(arms) == 12 and [[136, 306], [140, 306]] not in arms  # bars without their ends


def test_a_graph_drawn_2_or_3_draws_its_errors_as_boxes_or_a_band():
    fig = make([(_graph(), "a2")]).plot()
    boxes = fills(fig.axes[0])
    assert np.asarray(boxes[0].get_xy())[:4].tolist() == [
        [117.0, 417.0], [159.0, 417.0], [159.0, 306.0], [117.0, 306.0],
    ]  # fmt: skip
    assert len(boxes) == 3
    fig = make([(_graph(), "a3")]).plot()
    (band,) = fills(fig.axes[0])
    assert len(band.get_xy()) >= 6  # up the tops of the bars and back down their bottoms


def test_a_graph_drawn_f_and_b_is_filled_and_barred():
    fig = make([(_graph(), "af")]).plot()
    (area,) = fills(fig.axes[0])
    corners = np.asarray(area.get_xy())[:3].tolist()
    assert corners == [[138.0, 361.0], [350.0, 139.0], [562.0, 250.0]]
    fig = make([(_graph(), "ab")]).plot()
    bars = [line for line in polylines(fig.axes[0], clipped=True) if len(line) == 5]
    assert len(bars) == 3


def test_a_graph_on_log_axes_is_ranged_from_its_positive_points():
    fig = make([(_graph(errors=False), "ap")], fLogy=1, fLogx=1).plot()
    assert fig.axes[0].get_ylim() == pytest.approx((1.0, 8.0))


def test_a_graph_drawn_same_after_a_histogram_takes_its_axes():
    fig = make([(filled(), "hist"), (_graph(errors=False), "p")]).plot()
    (ax,) = fig.axes
    assert ax.get_xlim() == (0.0, 10.0)
    (points,) = marks(ax)  # the point above the frame is left out
    assert points.get_offsets().tolist() == [[126.5, 196.5], [238.5, 69.5]]


def test_a_multigraph_draws_each_graph_by_its_own_option_or_its_own():
    one, two = _graph(), _graph(errors=False)
    mg = MultiGraph(
        "TMultiGraph",
        {"TNamed": {"fName": "mg", "fTitle": ""}, "fGraphs": Listed([one, two], ["", "l"])},
    )
    fig = make([(mg, "ap")]).plot()
    (ax,) = fig.axes
    assert len(marks(ax)) == 1  # the first graph's points and bars
    assert polylines(ax, clipped=True)[-1] == [[138, 361], [350, 139], [562, 250]]  # the second
    assert ax.get_ylim()[1] == pytest.approx(4.5 + 0.3)


def test_the_fit_made_to_a_whole_multigraph_is_drawn_over_its_graphs():
    mg = MultiGraph(
        "TMultiGraph",
        {"TNamed": {"fName": "mg", "fTitle": ""}, "fGraphs": Listed([_graph()], [""])},
    )
    mg.functions.append(Function("f", "pol0", range=(1.0, 3.0), parameters=[3.0]))
    mg.functions.append(prim("TPaveStats"))  # a box, not a fit: drawn with nothing here
    (fit,) = lines(make([(mg, "ap")]).plot().axes[0], (1.0, 0.0, 0.0))
    assert {y for _, y in fit.lines[0]} == {250}  # 3, in a frame from 1.2 to 4.8


def test_the_command_line_prints_roots_own_canvas_to_a_picture(tmp_path, capsys):
    from xrdroot.cli import main

    out = tmp_path / "c1.png"
    main(["print", f"{DATA}/tcanvas.root:c1", "-o", str(out)])
    assert out.read_bytes()[:4] == b"\x89PNG"
    assert out.stat().st_size > 1000


def test_a_multigraph_or_stack_of_nothing_draws_nothing():
    mg = MultiGraph("TMultiGraph", {"TNamed": {"fName": "mg", "fTitle": ""}, "fGraphs": []})
    stack = Stack("THStack", {"TNamed": {"fName": "s", "fTitle": ""}, "fHists": []})
    for empty, option in ((mg, "a"), (stack, "")):
        fig = make([(empty, option)]).plot()
        assert fig.axes[0].get_xlim() == (0.0, 1.0)


# -- stacks -----------------------------------------------------------------------


def _stack():
    a, b = filled(name="a"), filled(name="b")
    a.members["TH1"]["TAttFill"]["fFillColor"] = 2
    b.members["TH1"]["TAttFill"]["fFillColor"] = 4
    return Stack(
        "THStack", {"TNamed": {"fName": "s", "fTitle": ""}, "fHists": Listed([a, b], ["", ""])}
    )


def test_a_stack_is_drawn_each_histogram_on_the_ones_added_before_it():
    fig = make([(_stack(), "")]).plot()
    (ax,) = fig.axes
    outlines = lines(ax, clipped=True)
    tops = [int(min(y for _, y in outline.lines[0])) for outline in outlines]
    assert tops == [round(450 - 3 / 6.3 * 400), round(450 - 6 / 6.3 * 400)]  # the second on top
    first, second = fills(ax)
    assert (first.get_facecolor()[:3], second.get_facecolor()[:3]) == ((1, 0, 0), (0, 0, 1))
    bottom = np.asarray(second.get_xy())[len(second.get_xy()) // 2 :, 1]
    assert min(bottom) == tops[0]  # filled down to the first's top, which still shows
    assert ax.get_ylim()[1] == pytest.approx(6.3)


def test_a_stack_drawn_nostack_draws_each_histogram_by_itself():
    fig = make([(_stack(), "nostack")]).plot()
    outlines = lines(fig.axes[0], clipped=True)
    assert len({int(min(y for _, y in outline.lines[0])) for outline in outlines}) == 1


# -- text, lines and shapes -------------------------------------------------------


def _texts():
    latex = prim(
        "TLatex", fTitle="#sqrt{s} = 13 TeV", fX=0.2, fY=0.8, fBits=0x03000000 | NDC,
        fTextColor=2, fTextAlign=22, fTextAngle=30.0,
    )  # fmt: skip
    text = prim("TText", fTitle="cost $5", fX=5.0, fY=1.0, fTextFont=43, fTextSize=20)
    fig = make([(filled(), "hist"), (latex, ""), (text, "")]).plot()
    return fig, written(fig.axes[0])


def test_latex_is_laid_out_as_tlatex_lays_it_out_at_its_place_in_ndc():
    fig, texts = _texts()
    root, rest = texts["s"], texts["= 13 TeV"]  # the root's argument, then the rest beside it
    for piece in (root, rest):
        assert (piece.get_color(), piece.get_rotation()) == ((1.0, 0.0, 0.0), 30.0)
        assert (piece.get_ha(), piece.get_va()) == ("left", "baseline")
    assert root.get_position() == (102, 134) and rest.get_position() == (112, 128)  # up the slope
    sign = lines(fig.axes[0], (1.0, 0.0, 0.0))  # the root sign, drawn: its tick and its top
    tick, top = [a.lines[0].tolist() for a in sign]
    assert (tick, top) == ([[89, 126], [98, 137]], [[98, 137], [90, 118], [102, 111]])
    assert fig.axes[0].texts[0].get_transform() is not fig.axes[0].transData  # in pixels


def test_text_is_placed_by_the_axes_units_and_sized_in_pixels_for_precision_3():
    _fig, texts = _texts()
    plain = texts["cost $5"]  # a TText is not TLatex: its dollar sign is a dollar sign
    assert plain.get_position() == (350, 323)  # 5 and 1 in a frame of 0 to 10 and 0 to 3.15
    assert plain.get_fontsize() == pytest.approx(18 * 0.72)  # 20 pixels as FreeType draws them


def test_a_text_size_is_a_fraction_of_the_shorter_side_of_its_pad():
    latex = prim("TLatex", fTitle="x", fX=0.5, fY=0.5, fTextSize=0.1, fTextFont=132)
    fig = make([(latex, "")], width=800, height=400).plot()
    (text,) = fig.axes[0].texts
    # 40 pixels, as TTF sizes it: int(40 * 0.93376068 + 0.5) = 37, in points at 72 a 100 pixels
    assert text.get_fontsize() == pytest.approx(37 * 0.72)
    assert text.get_position() == (400, 200)
    assert text.get_fontproperties().get_style() == "normal"


def _shapes():
    """A pad of one of each shape, drawn."""
    shapes = [
        prim("TLine", fX1=0.1, fY1=0.1, fX2=0.9, fY2=0.9, fLineColor=2, fLineStyle=2, fLineWidth=2),
        prim("TLine", fX1=0.1, fY1=0.9, fX2=0.9, fY2=0.1, fBits=0x03000000 | NDC),
        prim("TArrow", fX1=0.1, fY1=0.5, fX2=0.9, fY2=0.5, fOption="<|>", fArrowSize=0.05),
        prim("TArrow", fX1=0.1, fY1=0.4, fX2=0.9, fY2=0.4, fOption="->-", fFillStyle=0),
        prim("TBox", fX1=0.6, fY1=0.6, fX2=0.2, fY2=0.8, fFillColor=5, fFillStyle=3005),
        prim("TEllipse", fX1=0.5, fY1=0.5, fR1=0.2, fR2=0.1, fPhimax=360.0, fTheta=45.0),
        prim("TEllipse", fX1=0.5, fY1=0.5, fR1=0.2, fR2=0.2, fPhimin=0.0, fPhimax=90.0),
        prim("TMarker", fX=0.5, fY=0.5, fMarkerStyle=24, fMarkerColor=4, fMarkerSize=2.0),
    ]
    return make([(shape, "") for shape in shapes]).plot().axes[0]


def test_lines_and_markers_are_drawn_in_their_style_and_place():
    ax = _shapes()
    dashed, plain = lines(ax)[:2]
    assert (dashed.color, dashed.thick, dashed.dashes) == ((1.0, 0.0, 0.0), 2, (3, 3))
    assert dashed.lines[0].tolist() == [[70, 450], [630, 50]]
    assert plain.lines[0].tolist() == [[70, 50], [630, 450]]  # the same place, in NDC
    (marker,) = ax.lines
    assert (marker.get_marker(), marker.get_markerfacecolor()) == ("o", "none")
    assert marker.get_markersize() == pytest.approx(16 * 0.72)


def test_arrows_and_boxes_are_drawn_in_their_style_and_place():
    ax = _shapes()
    drawn = polylines(ax)[2:7]
    assert drawn[0] == [[94, 250], [606, 250]]  # the shaft stops where a closed head begins
    assert drawn[1] == [[606, 264], [630, 250], [606, 236], [606, 264]]
    assert drawn[2] == [[94, 236], [70, 250], [94, 264], [94, 236]]
    assert drawn[3:] == [[[70, 300], [630, 300]], [[606, 314], [630, 300], [606, 286]]]
    (box,) = [p for p in ax.patches if p.get_hatch()]
    assert (box.get_x(), box.get_y(), box.get_width()) == pytest.approx((0.2, 0.6, 0.4))


def test_an_ellipse_is_drawn_whole_or_as_the_slice_it_is_limited_to():
    ax = _shapes()
    whole, slice_ = [p for p in ax.patches if isinstance(p, Polygon)][-2:]
    assert len(whole.get_xy()) < len(slice_.get_xy()) + 2
    assert slice_.get_xy()[-2] == pytest.approx((0.5, 0.5))  # a slice closes on its centre


# -- paves and legends -------------------------------------------------------------


def test_a_pave_text_stacks_its_lines_in_its_box_with_its_shadow():
    held = [
        prim("TText", fTitle="first", fTextSize=0.0, fTextAlign=0),
        prim("TLatex", fTitle="#alpha", fTextSize=0.0, fTextAlign=0),
        prim("TLine", fX1=0.0),
        prim("TText", fTitle="placed", fX=0.5, fY=0.25, fTextSize=0.03, fTextAlign=22),
    ]
    pave = prim(
        "TPaveText",
        fX1NDC=0.1,
        fY1NDC=0.6,
        fX2NDC=0.5,
        fY2NDC=0.9,
        fOption="tlNDC",
        fLines=held,
        fTextSize=0.0,
        fTextAlign=0,
        fBorderSize=3,
        fShadowColor=1,
        fFillColor=0,
    )
    fig = make([(pave, "")]).plot()
    ax = fig.axes[0]
    box, shadow = fills(ax)
    assert np.max(np.asarray(shadow.get_xy())[:, 1]) > np.max(np.asarray(box.get_xy())[:, 1])
    box_edge, rule = polylines(ax)
    assert box_edge == [[70, 200], [70, 50], [350, 50], [350, 200], [70, 200]]
    assert rule == [[70, 106], [350, 106]]  # a line at x 0 is ruled right across the box
    texts = written(ax)
    alpha = texts["\u03b1"]  # the Greek letter, from the Symbol font
    assert texts["first"].get_position()[1] < alpha.get_position()[1]  # down the box
    assert texts["placed"].get_position() == (190, 167)  # centred where it was put, in the box


def test_a_pave_placed_in_the_axes_units_is_converted_to_the_pads():
    pave = prim(
        "TPaveText", fX1=0.0, fY1=0.0, fX2=5.0, fY2=1.5, fOption="br", fLines=[], fBorderSize=0
    )
    fig = make([(filled(), "hist"), (pave, "")]).plot()
    (box,) = [p for p in fills(fig.axes[0]) if p.get_xy()[0][0] == pytest.approx(0.1)]
    top = 0.1 + 1.5 / 3.15 * 0.8
    corners = [[0.1, 0.1], [0.1, top], [0.5, top], [0.5, 0.1], [0.1, 0.1]]
    np.testing.assert_allclose(box.get_xy(), corners)
    assert len(lines(fig.axes[0], clipped=False)) > 1  # with no border, only the stats box's


def test_a_pave_a_pave_label_and_a_title_pave_are_drawn():
    label = prim(
        "TPaveLabel",
        fX1NDC=0.1,
        fY1NDC=0.1,
        fX2NDC=0.4,
        fY2NDC=0.2,
        fOption="NDC",
        fLabel="label",
        fTextSize=0.0,
    )
    sized = prim(
        "TPaveLabel",
        fX1NDC=0.5,
        fY1NDC=0.1,
        fX2NDC=0.9,
        fY2NDC=0.2,
        fOption="NDC",
        fLabel="sized",
        fTextSize=0.5,
    )
    plain = prim(
        "TPave", fX1NDC=0.1, fY1NDC=0.3, fX2NDC=0.4, fY2NDC=0.4, fOption="NDC", fBorderSize=1
    )
    fig = make([(label, ""), (sized, ""), (plain, "")]).plot()
    texts = written(fig.axes[0])
    assert texts["label"].get_position() == (74, 449)  # in from the box's corner, by fTextAlign
    # half the box's 50 pixels, as FreeType sizes them: int(25 * 0.93376068 + 0.5) = 23
    assert texts["sized"].get_fontsize() == pytest.approx(23 * 0.72)
    assert len(fills(fig.axes[0])) == 3
    assert polylines(fig.axes[0]) == [[[70, 350], [70, 300], [280, 300], [280, 350], [70, 350]]]


def test_a_painted_pads_title_pave_stands_in_for_the_histograms_title():
    title = prim(
        "TPaveText",
        fX1NDC=0.3,
        fY1NDC=0.9,
        fX2NDC=0.7,
        fY2NDC=1.0,
        fOption="blNDC",
        fName="title",
        fLines=[prim("TText", fTitle="saved title", fTextSize=0.0)],
        fBorderSize=0,
    )
    frame = prim("TFrame")
    painted = {"fUxmax": 10.0, "fUymax": 5.0}
    fig = make([(filled(title="own title"), ""), (frame, ""), (title, "")], **painted).plot()
    texts = words(fig.axes[0])
    assert "saved title" in texts and "own title" not in texts
    assert "Entries" not in texts  # nor was a stats box saved with it


def test_a_legend_draws_each_entry_s_symbol_and_label_in_rows_and_columns():
    h = filled()
    h.members["TH1"]["TAttLine"]["fLineColor"] = 2
    entries = Listed(
        [
            prim("TLegendEntry", fLabel="Header", fOption="h", fTextSize=0.0, fTextAlign=0),
            prim("TLegendEntry", fLabel="data", fOption="lep", fObject=h, fTextSize=0.0),
            prim("TLegendEntry", fLabel="fill", fOption="fl", fFillColor=3, fTextSize=0.0),
            prim("TLegendEntry", fLabel="#mu", fOption="p", fTextSize=0.0),
        ]
    )
    legend = prim(
        "TLegend",
        fX1NDC=0.5,
        fY1NDC=0.5,
        fX2NDC=0.9,
        fY2NDC=0.9,
        fOption="brNDC",
        fPrimitives=entries,
        fNColumns=2,
        fMargin=0.25,
        fTextSize=0.0,
        fBorderSize=1,
    )
    fig = make([(legend, "")]).plot()
    ax = fig.axes[0]
    texts = {text: piece.get_position() for text, piece in written(ax).items()}
    assert set(texts) == {"Header", "data", "fill", "μ"}
    # the header a row of its own, then two columns, rows downwards
    assert texts["Header"][1] < texts["data"][1] == texts["fill"][1] < texts["μ"][1]
    assert texts["data"][0] == texts["μ"][0] < texts["fill"][0]
    red = polylines(ax, (1.0, 0.0, 0.0))
    assert red == [[[355, 150], [380, 150]], [[368, 150], [368, 130]], [[368, 150], [368, 170]]]
    (green,) = [p for p in fills(ax) if p.get_facecolor()[:3] == (0.0, 1.0, 0.0)]
    assert np.asarray(green.get_xy())[0].tolist() == pytest.approx([0.762, 0.653], abs=1e-3)


def test_an_empty_legend_is_its_box():
    legend = prim(
        "TLegend",
        fX1NDC=0.5,
        fY1NDC=0.5,
        fX2NDC=0.9,
        fY2NDC=0.9,
        fOption="NDC",
        fPrimitives=[],
        fBorderSize=1,
    )
    fig = make([(legend, "")]).plot()
    assert not fig.axes[0].texts
    assert polylines(fig.axes[0]) == [[[350, 250], [350, 50], [630, 50], [630, 250], [350, 250]]]


# -- colours ----------------------------------------------------------------------


def test_the_colours_a_canvas_saved_are_the_ones_it_draws_with():
    colors = [prim("TColor", fNumber=2, fRed=0.0, fGreen=0.5, fBlue=0.0)]
    palette = [prim("TColor", fNumber=2, fRed=0.0, fGreen=0.5, fBlue=0.0)] * 2
    line = prim("TLine", fX1=0.0, fY1=0.0, fX2=1.0, fY2=1.0, fLineColor=2)
    h2 = Histogram.book("h2", (2, 0.0, 2.0), (1, 0.0, 1.0))
    h2.fill(np.array([0.5]), np.array([0.5]))
    h = filled()
    h.members["TH1"]["TAttLine"]["fLineColor"] = 2
    h.members["TH1"]["TAttFill"].update(fFillColor=2, fFillStyle=1001)
    shaded, outlined = sub("c_1", [(h2, "col")]), sub("c_2", [(h, "hist")])
    fig = make([(colors, ""), (palette, ""), (line, ""), (shaded, ""), (outlined, "")]).plot()
    assert [a.color for a in lines(fig.axes[0])] == [(0.0, 0.5, 0.0)]
    assert _mapped(only(fig, "c_1")).cmap.name == "saved"
    (area,) = fills(only(fig, "c_2"))[:1]
    assert area.get_facecolor()[:3] == pytest.approx((0.0, 0.5, 0.0), abs=0.01)
    (outline,) = lines(only(fig, "c_2"), clipped=True)
    assert to_rgb(outline.color) == pytest.approx((0.0, 0.5, 0.0), abs=0.01)


@pytest.mark.parametrize(
    ("index", "rgb"),
    [
        (0, (1.0, 1.0, 1.0)),
        (1, (0.0, 0.0, 0.0)),
        (632, (1.0, 0.0, 0.0)),  # kRed
        (632 - 9, (1.0, 0.6, 0.6)),  # kRed-9, a lighter ring
        (600 + 2, (0.0, 0.0, 0.6)),  # kBlue+2, a darker one
        (920, (0.8, 0.8, 0.8)),  # kGray
        (800, (1.0, 0.8, 0.0)),  # kOrange
        (800 - 9, (1.0, 0.8, 0.6)),  # kOrange-9
        (800 + 10, (1.0, 0.2, 0.0)),  # kOrange+10
        (123456, (0.0, 0.0, 0.0)),  # nobody's
    ],
)
def test_roots_colour_table_has_its_names_circles_rectangles_and_greys(index, rgb):
    from xrdroot.canvas.colors import Colors

    assert Colors().rgb(index) == pytest.approx(rgb)


def test_roots_colour_table_has_its_spectrum_and_takes_what_a_canvas_saved():
    from xrdroot.canvas.colors import Colors

    table = Colors()
    assert table.rgb(51)[2] > table.rgb(51)[0]  # the spectrum starts at violet
    assert table.rgba(2, 0.5) == (1.0, 0.0, 0.0, 0.5)
    table.adopt([prim("TColor", fNumber=-1), prim("TColor", fNumber=7, fRed=0.25)])
    assert (table.rgb(7), table.colormap().name) == ((0.25, 0.0, 0.0), "kBird")


@pytest.mark.parametrize(
    ("meaning", "expected"),
    [
        (("dashes", 1), "solid"),
        (("dashes", 99), "solid"),
        (("dashes", 2, 2.0), (0, (1.5, 1.5))),  # "12 12" a quarter as long, in widths of 2
        (("marker", 20), ("o", True)),
        (("marker", 999), ("o", True)),
        (("marker_size", 1, 5.0), pytest.approx(0.72)),
        (("fill", 0), (False, None, 0.0)),
        (("fill", 4000), (False, None, 0.0)),
        (("fill", 4050), (True, None, 0.5)),
        (("fill", 3004), (True, "//", 1.0)),
        (("fill", 3999), (True, "//", 1.0)),
        (("fill", 1001), (True, None, 1.0)),
        (("font", 43), ("sans-serif", "normal", "normal", True)),
        (("font", 999), ("sans-serif", "normal", "normal", False)),
        (("align", 33), ("right", "top")),
        (("align", 0), ("left", "bottom")),
    ],
)
def test_roots_attribute_numbers_mean_what_tattline_tattmarker_and_tattfill_say(meaning, expected):
    from xrdroot.canvas import styles

    name, *arguments = meaning
    assert getattr(styles, name)(*arguments) == expected


# -- what is left out --------------------------------------------------------------


def test_what_a_canvas_does_not_draw_is_named_in_a_warning():
    axis = prim("TButton", fName="axis")
    quiet = [prim("TFrame"), prim("TPaletteAxis"), prim("TLegendEntry")]
    with pytest.warns(CanvasWarning) as caught:
        make([("TUnknown", ""), (axis, ""), (object(), ""), *((q, "") for q in quiet)]).plot()
    (warning,) = caught
    message = str(warning.message)
    assert "3 things" in message
    assert "TUnknown (a class this file does not describe)" in message
    assert "TButton 'axis'" in message
    assert "object" in message


# -- the corners ------------------------------------------------------------------


def test_a_graph_drawn_without_axes_before_a_histogram_leaves_the_frame_to_it():
    fig = make([(_graph(errors=False), "p"), (filled(), "hist")]).plot()
    assert fig.axes[0].get_xlim() == (0.0, 10.0)


def test_a_logarithmic_frame_a_pad_was_drawn_with_is_in_powers_of_ten():
    frame = prim("TFrame")
    fig = make([(filled(), ""), (frame, "")], fLogx=1, fUxmin=0.0, fUxmax=1.0, fUymax=1.0).plot()
    assert fig.axes[0].get_xlim() == pytest.approx((1.0, 10.0))


def test_a_hollow_box_and_an_empty_label_draw_their_outline():
    box = prim("TBox", fX1=0.1, fY1=0.1, fX2=0.2, fY2=0.2, fFillStyle=0)
    label = prim(
        "TPaveLabel", fX1NDC=0.1, fY1NDC=0.1, fX2NDC=0.4, fY2NDC=0.2, fOption="NDC", fLabel=""
    )
    fig = make([(box, ""), (label, "")]).plot()
    shapes = [p for p in fig.axes[0].patches if isinstance(p, Rectangle) and p.get_zorder() > 3]
    (hollow,) = [p for p in shapes if p.get_facecolor()[3] == 0.0]
    assert hollow.get_edgecolor()[:3] == (0.0, 0.0, 0.0)


def test_a_stats_box_of_a_two_dimensional_histogram_leaves_out_what_only_one_has():
    from xrdroot.canvas.statbox import fit_rows, stats_rows

    h2 = Histogram.book("h2", (2, 0.0, 2.0), (2, 0.0, 2.0))
    assert stats_rows(h2, 110000) == []  # no underflow or overflow line for a grid
    h = filled()
    h.attach(Function("f", "pol0", range=(0.0, 10.0), parameters=[1.0]))
    h.functions[0].fit_result = {"chi2": 1.0, "ndf": 1, "npfits": 2}
    assert [name for name, _ in fit_rows(h, 110)] == ["#chi^{2} / ndf", "Prob"]


def test_a_hatch_takes_its_colour_where_this_matplotlib_keeps_one(monkeypatch):
    import builtins

    from xrdroot.canvas import shapes

    real = builtins.__import__

    def missing(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    assert shapes._hatch_colour() == "hatchcolor"


# -- polylines, crowns, arcs and axes of their own ------------------------------------


def test_a_polyline_is_its_points_joined_or_filled_and_a_polymarker_marks_them():
    xs, ys = [0.1, 0.5, 0.9], [0.1, 0.9, 0.1]
    shapes = [
        prim("TPolyLine", fN=3, fX=xs, fY=ys, fOption="", fLineColor=2),
        prim("TPolyLine", fN=3, fX=xs, fY=ys, fOption="f", fFillColor=3),
        prim("TPolyMarker", fN=2, fX=xs, fY=ys, fMarkerStyle=20),
    ]
    ax = make([(shape, "") for shape in shapes]).plot().axes[0]
    line, markers = ax.lines
    assert list(line.get_xdata()) == xs and line.get_color() == (1.0, 0.0, 0.0)
    assert list(markers.get_xdata()) == xs[:2] and markers.get_linestyle() == "None"
    (area,) = [p for p in ax.patches if isinstance(p, Polygon)]
    assert tuple(area.get_facecolor()[:3]) == (0.0, 1.0, 0.0)


def test_a_crown_is_the_ring_between_its_radii_and_an_arc_an_ellipse_of_one_radius():
    shapes = [
        prim("TCrown", fX1=0.5, fY1=0.5, fR1=0.1, fR2=0.2, fPhimin=0.0, fPhimax=360.0),
        prim("TArc", fX1=0.5, fY1=0.5, fR1=0.3, fR2=0.3, fPhimin=0.0, fPhimax=360.0),
    ]
    ax = make([(shape, "") for shape in shapes]).plot().axes[0]
    ring, arc = [p for p in ax.patches if isinstance(p, Polygon)]
    radii = np.hypot(*(ring.get_xy() - 0.5).T)
    assert radii.min() == pytest.approx(0.1) and radii.max() == pytest.approx(0.2)
    assert np.hypot(*(arc.get_xy() - 0.5).T).max() == pytest.approx(0.3)


def _axis(**members):
    base = {
        "fX1": 0.1, "fY1": 0.2, "fX2": 0.9, "fY2": 0.2, "fWmin": 0.0, "fWmax": 10.0,
        "fNdiv": 510, "fChopt": "", "fTitle": "", "fLabelSize": 0.04, "fTickSize": 0.03,
    }  # fmt: skip
    return prim("TGaxis", **{**base, **members})


def test_an_axis_of_its_own_is_graduated_over_its_scale_and_labelled_on_the_other_side():
    ax = make([(_axis(fTitle="x [cm]"), "")]).plot().axes[0]
    labels = words(ax)
    assert labels == [str(n) for n in range(11)] + ["x [cm]"]
    drawn = polylines(ax)
    assert drawn[0] == [[70, 400], [630, 400]]  # the axis, a fifth of the way up
    assert drawn[1] == [[70, 388], [70, 400]]  # ticks stand up from it: 0.03 of its length
    assert drawn[2] == [[81, 394], [81, 400]]  # and the secondary ones half as long
    assert all(t.get_position()[1] > 400 for t in ax.texts)  # labels hang below


def test_an_axis_says_by_its_chopt_which_side_its_ticks_and_labels_go_and_whether_logarithmic():
    vertical = _axis(fX1=0.9, fY1=0.1, fX2=0.9, fY2=0.9, fChopt="-=", fBits=0x03000000 | NDC)
    log = _axis(fWmin=1.0, fWmax=1000.0, fChopt="G+-")
    bare = _axis(fChopt="U")
    ax = make([(vertical, ""), (log, ""), (bare, "")]).plot().axes[0]
    labels = words(ax)
    assert labels[:11] == [str(n) for n in range(11)]
    assert labels[11:] == ["1", "10", "2", "10", "3", "10"]  # 1, 10 and 10 to the 2 and 3
    drawn = polylines(ax)
    assert drawn[1] == [[647, 450], [630, 450]]  # "-": the vertical axis's ticks to its right
    assert [[70, 388], [70, 412]] in drawn  # "+-": the logarithmic one's either side of it
    assert [[70, 400], [630, 400]] in drawn  # and the bare axis a line alone


@pytest.mark.parametrize(
    ("members", "labels"),
    [
        ({"fNdiv": 505}, ["0", "2", "4", "6", "8", "10"]),
        ({"fNdiv": 2}, ["0", "5", "10"]),
        ({"fWmin": 0.0, "fWmax": 1.0, "fNdiv": 505}, ["0", "0.2", "0.4", "0.6", "0.8", "1"]),
    ],
)
def test_graduations_keep_to_the_scale_and_divide_as_asked(members, labels):
    ax = make([(_axis(**members), "")]).plot().axes[0]
    assert words(ax) == labels


def test_a_histogram_with_a_range_is_framed_by_its_range():
    h = filled()
    h._core["fXaxis"]["fFirst"], h._core["fXaxis"]["fLast"] = 3, 5
    _fig, ax = _drawn(h, "hist")
    assert ax.get_xlim() == pytest.approx((2.0, 5.0))
    assert ax.get_ylim()[1] == pytest.approx(3 * 1.05)
    h._core["fXaxis"]["fLast"] = 99  # a range off the axis is all of it
    assert _drawn(h, "hist")[1].get_xlim() == pytest.approx((0.0, 10.0))


def test_a_two_dimensional_histogram_with_a_range_is_framed_by_both():
    h2 = Histogram.book("h2", (4, 0.0, 4.0), (4, 0.0, 4.0))
    h2.fill([0.5], [1.5])
    h2._core["fYaxis"]["fFirst"], h2._core["fYaxis"]["fLast"] = 2, 3
    _fig, ax = _drawn(h2, "col")
    assert (ax.get_xlim(), ax.get_ylim()) == ((0.0, 4.0), (1.0, 3.0))


def test_an_efficiency_frames_its_pad_and_is_drawn_as_points_or_a_grid():
    from xrdroot.efficiency import Efficiency

    e = Efficiency.book("e", (4, 0.0, 4.0))
    e.fill([True, False, True, True], [0.5, 1.5, 2.5, 3.5])
    _fig, ax = _drawn(e, "")
    assert ax.get_xlim() == (0.0, 4.0) and ax.get_ylim()[0] == pytest.approx(0.0)
    e2 = Efficiency.book("e2", (2, 0.0, 2.0), (2, 0.0, 4.0))
    e2.fill([True], [0.5], [1.0])
    assert _drawn(e2, "colz")[1].get_ylim() == (0.0, 4.0)


def test_tick_labels_are_plain_numbers_and_divisions_follow_fndivisions():
    h = filled()
    h._core["fXaxis"]["TAttAxis"]["fNdivisions"] = 505
    _fig, ax = _drawn(h, "hist")
    below = [t for t in ax.texts if t.get_position()[1] > 460]
    assert [t.get_text() for t in below] == ["0", "2", "4", "6", "8", "10"]
