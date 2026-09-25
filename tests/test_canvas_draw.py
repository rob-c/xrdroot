"""Drawing ROOT's canvases with matplotlib.

The canvases here are built in memory, a pad's members and the objects it
draws - histograms and graphs as this library makes them, the drawing
classes as :class:`~xrdroot.canvas.Primitive` - which is what reading one
gives back, and the tests check what lands on the figure: an axes per pad
where the pad's margins put its frame, the data by its draw option, text,
lines, paves and legends where ROOT would put them, in ROOT's colours. One
test draws ROOT's own ``tcanvas.root``; none compares pixels.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, StepPatch

from xrdroot import Canvas, Function, Graph, Histogram, UnsupportedFeatureError, open_root
from xrdroot.buffer import Listed
from xrdroot.canvas import Pad, Primitive, render
from xrdroot.canvas.paint import CanvasWarning
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


# -- ROOT's own canvas -----------------------------------------------------------


def test_roots_canvas_draws_one_axes_where_its_margins_put_the_frame():
    with open_root(f"{DATA}/tcanvas.root") as f:
        c = f["c1"]
    fig = c.plot()
    (ax,) = fig.axes
    assert ax.get_label() == "c1"
    assert ax.get_position().bounds == pytest.approx((0.1, 0.1, 0.8, 0.8))
    assert tuple(fig.get_size_inches() * fig.dpi) == pytest.approx((296, 372))
    _xs, ys = ax.lines[0].get_data()
    assert list(ys) == [0.0, 2.0, 4.0, 1.0, 3.0]  # "alp": a line through the points
    assert ax.get_xlim() == pytest.approx((-0.4, 4.4))  # a tenth of their spread each side
    assert ax.get_ylim() == pytest.approx((0.0, 4.4))  # not below zero, as none are


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
    assert ax.xaxis._major_tick_kw["gridOn"]
    assert ax.yaxis._major_tick_kw["gridOn"]
    assert ax.xaxis._major_tick_kw["tick2On"]
    assert ax.yaxis._major_tick_kw["tick2On"]
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
    assert ax.spines["bottom"].get_edgecolor()[:3] == (1.0, 0.0, 0.0)


def test_a_pad_with_no_frame_is_ranged_by_its_own_coordinates():
    line = prim("TLine", fX1=10.0, fY1=10.0, fX2=20.0, fY2=20.0)
    fig = make([(line, "")], fX1=0.0, fY1=0.0, fX2=40.0, fY2=40.0).plot()
    (ax,) = fig.axes
    assert ax.get_xlim() == (0.0, 40.0)
    assert not ax.axison
    assert list(ax.lines[0].get_xdata()) == [10.0, 20.0]


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
    assert ax.spines["left"].get_edgecolor()[:3] == (0.0, 0.0, 1.0)


def test_a_hollow_frame_is_not_filled():
    fig = make([(filled(), "")], fFrameFillStyle=0).plot()
    assert not [p for p in fig.axes[0].patches if p.get_zorder() == -50]


# -- histograms by option ---------------------------------------------------------


def _drawn(h, option, **pad):
    fig = make([(h, option)], **pad).plot()
    return fig, only(fig, "c")


def test_a_histogram_drawn_hist_is_its_outline_with_its_axis_titles_and_title():
    h = filled(title="p_{T} spectrum")
    h.axes[0].title = "p_{T} [GeV]"
    h.members["TH1"]["fXaxis"]["TNamed"]["fTitle"] = "p_{T} [GeV]"
    h.members["TH1"]["TAttLine"]["fLineColor"] = 4
    _fig, ax = _drawn(h, "hist")
    (steps,) = [p for p in ax.patches if isinstance(p, StepPatch)]
    np.testing.assert_array_equal(steps.get_data().values, h.values())
    assert steps.get_edgecolor()[:3] == (0.0, 0.0, 1.0)
    assert ax.get_xlabel() == r"$\mathrm{p}_{\mathrm{T}}\mathrm{\ [GeV]}$"
    assert ax.get_title() == ""  # the title is the pad's, drawn in NDC
    assert r"$\mathrm{p}_{\mathrm{T}}\mathrm{\ spectrum}$" in [t.get_text() for t in ax.texts]
    assert ax.get_xlim() == (0.0, 10.0)
    assert ax.get_ylim() == pytest.approx((0.0, 3.15))  # five percent over the highest bin


def test_a_histogram_with_no_title_bit_or_title_draws_none():
    h = filled(title="shown")
    named = h.members["TH1"]["TNamed"]
    named["fBits"] = named.get("fBits", 0) | 1 << 17
    _fig, ax = _drawn(h, "hist")
    assert "shown" not in [t.get_text() for t in ax.texts]


def test_a_histogram_is_drawn_filled_and_hatched_as_its_fill_says():
    h = filled()
    h.members["TH1"]["TAttFill"].update(fFillColor=2, fFillStyle=1001)
    _fig, ax = _drawn(h, "")
    (steps,) = [p for p in ax.patches if isinstance(p, StepPatch)]
    assert steps.get_fill()
    assert steps.get_facecolor()[:3] == (1.0, 0.0, 0.0)
    h.members["TH1"]["TAttFill"].update(fFillStyle=3004)
    _fig, ax = _drawn(h, "")
    (steps,) = [p for p in ax.patches if isinstance(p, StepPatch)]
    assert steps.get_hatch() == "/"


def test_a_histogram_drawn_e1_has_bars_with_ends_and_skips_empty_bins():
    h = filled()
    _fig, ax = _drawn(h, "e1")
    (bars,) = ax.containers
    xs, ys = bars.lines[0].get_data()
    assert list(xs) == [1.5, 2.5, 3.5, 4.5, 5.5, 7.5]
    assert list(ys) == [1, 2, 3, 2, 1, 1]
    assert bars.lines[1]  # the ends
    assert ax.get_ylim()[1] == pytest.approx((3 + np.sqrt(3)) * 1.05)


def test_a_histogram_drawn_e0_keeps_its_empty_bins():
    _fig, ax = _drawn(filled(), "e0")
    xs, _ys = ax.containers[0].lines[0].get_data()
    assert len(xs) == 10


def test_a_weighted_histogram_and_a_profile_draw_error_bars_unasked():
    h = filled()
    h.sumw2()
    _fig, ax = _drawn(h, "")
    assert ax.containers
    p = Profile.book("p", (4, 0.0, 4.0))
    p.fill(np.array([0.5, 1.5]), np.array([2.0, 3.0]))
    _fig, ax = _drawn(p, "")
    assert ax.containers
    assert not [x for x in ax.patches if isinstance(x, StepPatch)]


def _mapped(ax):
    """The one thing on ``ax`` drawn in the colours of a scale."""
    (mapped,) = [a for a in (*ax.images, *ax.collections) if getattr(a, "norm", None) is not None]
    return mapped


def test_a_histogram_drawn_e2_is_a_box_round_each_bin():
    _fig, ax = _drawn(filled(), "e2")
    (boxes,) = [c for c in ax.collections if isinstance(c, PolyCollection)]
    heights = [np.ptp(path.vertices[:, 1]) for path in boxes.get_paths()]
    assert len(heights) == 10
    assert heights[3] == pytest.approx(2 * np.sqrt(3))


def test_a_histogram_drawn_e3_is_a_band_through_its_bins():
    _fig, ax = _drawn(filled(), "e3")
    assert any(isinstance(c, PolyCollection) for c in ax.collections)


def test_a_histogram_drawn_p_or_l_marks_its_bins():
    _fig, ax = _drawn(filled(), "p")
    xs, _ys = ax.containers[0].lines[0].get_data()
    assert list(xs) == [1.5, 2.5, 3.5, 4.5, 5.5, 7.5]
    _fig, ax = _drawn(filled(), "l")
    assert (len(ax.lines[0].get_xdata()), ax.lines[0].get_marker()) == (10, "None")


def test_an_option_the_picture_refuses_draws_as_without_it_and_says_so():
    with pytest.warns(CanvasWarning, match=r"drawn without its option '\*h'"):
        _fig, ax = _drawn(filled(), "*h")
    assert [p for p in ax.patches if isinstance(p, StepPatch)]  # HIST, as without it


def test_a_histogram_drawn_bar_is_a_bar_per_bin():
    _fig, ax = _drawn(filled(), "bar")
    bars = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_zorder() == 1]
    assert len(bars) == 10


def test_a_histogram_drawn_text_writes_each_bin_that_is_not_empty():
    _fig, ax = _drawn(filled(), "text")
    written = sorted(t.get_text() for t in ax.texts if t.get_text() in ("1", "2", "3"))
    assert written == sorted(["1", "2", "3", "2", "1", "1"])


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
    assert [c for c in ax.collections if isinstance(c, PolyCollection)]
    _fig, ax = _drawn(h, "cont")
    assert ax.collections
    _fig, ax = _drawn(h, "text")
    assert sorted(t.get_text() for t in ax.texts if t.get_text() in ("1", "2")) == ["1", "2"]


def test_a_two_dimensional_histogram_drawn_lego_is_shaded_on_the_flat_pad_and_says_so():
    h = Histogram.book("h2", (2, 0.0, 2.0), (2, 0.0, 2.0))
    h.fill(np.array([0.5]), np.array([0.5]))
    with pytest.warns(CanvasWarning, match="three dimensions"):
        _fig, ax = _drawn(h, "lego")
    assert _mapped(ax)


def test_a_three_dimensional_histogram_is_left_out_with_a_warning():
    h = Histogram.book("h3", (2, 0.0, 2.0), (2, 0.0, 2.0), (2, 0.0, 2.0))
    with pytest.warns(CanvasWarning, match="three dimensions"):
        make([(h, "")]).plot()


# -- stats boxes ------------------------------------------------------------------


def test_a_histogram_saved_without_a_stats_box_is_drawn_with_gstyles():
    _fig, ax = _drawn(filled(), "")
    texts = [t.get_text() for t in ax.texts]
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
    lines = Listed([prim("TText", fTitle="h"), prim("TText", fTitle="Entries = 10"), "TUnknown"])
    h.functions.append(
        prim(
            "TPaveStats",
            fX1NDC=0.7,
            fY1NDC=0.7,
            fX2NDC=0.9,
            fY2NDC=0.9,
            fOption="brNDC",
            fLines=lines,
            fTextSize=0.0,
            fBorderSize=1,
            fOptStat=11,
        )
    )
    _fig, ax = _drawn(h, "")
    texts = [t.get_text() for t in ax.texts]
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
    assert tops[0] > tops[1]


# -- functions --------------------------------------------------------------------


def test_a_function_is_drawn_over_its_range_and_a_fit_with_its_histogram():
    f = Function("f", "pol1", range=(0.0, 10.0), parameters=[1.0, 2.0])
    fig = make([(f, "")]).plot()
    (ax,) = fig.axes
    (line,) = ax.lines
    assert line.get_xdata()[0] == 0.0
    assert line.get_ydata()[-1] == pytest.approx(21.0)
    h = filled()
    h.attach(Function("fit", "pol0", range=(0.0, 10.0), parameters=[2.0]))
    hidden = Function("hidden", "pol0", range=(0.0, 10.0), parameters=[9.0])
    hidden.members["TNamed"]["fBits"] = hidden.members["TNamed"].get("fBits", 0) | 1 << 9
    h.attach(hidden)
    _fig, ax = _drawn(h, "")
    assert [list(line.get_ydata()[:1]) for line in ax.lines] == [[2.0]]
    _fig, ax = _drawn(h, "hist")
    assert not ax.lines  # HIST draws the histogram alone


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
    (bars,) = ax.containers
    assert list(bars.lines[0].get_ydata()) == [2.0, 4.0, 3.0]
    assert ax.get_xlim() == pytest.approx((0.9 - 0.22, 3.1 + 0.22))
    assert ax.get_ylim() == pytest.approx((1.5 - 0.3, 4.5 + 0.3))


def test_a_graph_without_errors_draws_its_title_and_a_line_by_default():
    fig = make([(_graph(errors=False), "a")]).plot()
    (ax,) = fig.axes
    assert list(ax.lines[0].get_ydata()) == [2.0, 4.0, 3.0]
    assert r"$\mathrm{the\ graph}$" not in [t.get_text() for t in ax.texts]
    assert "the graph" in [t.get_text() for t in ax.texts]


def test_a_graph_drawn_with_a_star_x_or_z_marks_its_points_so():
    fig = make([(_graph(), "a*")]).plot()
    assert fig.axes[0].containers[0].lines[0].get_marker() == (6, 2, 0)  # ROOT's asterisk
    fig = make([(_graph(), "apx")]).plot()
    (bars,) = fig.axes[0].containers
    assert not bars.has_yerr
    fig = make([(_graph(), "apz")]).plot()
    assert len(fig.axes[0].containers) == 1


def test_a_graph_drawn_2_or_3_draws_its_errors_as_boxes_or_a_band():
    fig = make([(_graph(), "a2")]).plot()
    (boxes,) = [c for c in fig.axes[0].collections if isinstance(c, PolyCollection)]
    assert len(boxes.get_paths()) == 3
    fig = make([(_graph(), "a3")]).plot()
    assert any(isinstance(c, PolyCollection) for c in fig.axes[0].collections)


def test_a_graph_drawn_f_and_b_is_filled_and_barred():
    fig = make([(_graph(), "af")]).plot()
    assert fig.axes[0].collections or fig.axes[0].patches
    fig = make([(_graph(), "ab")]).plot()
    bars = [p for p in fig.axes[0].patches if isinstance(p, Rectangle) and p.get_zorder() == 1]
    assert len(bars) == 3


def test_a_graph_on_log_axes_is_ranged_from_its_positive_points():
    fig = make([(_graph(errors=False), "ap")], fLogy=1, fLogx=1).plot()
    assert fig.axes[0].get_ylim() == pytest.approx((1.0, 8.0))


def test_a_graph_drawn_same_after_a_histogram_takes_its_axes():
    fig = make([(filled(), "hist"), (_graph(errors=False), "p")]).plot()
    (ax,) = fig.axes
    assert ax.get_xlim() == (0.0, 10.0)
    assert len(ax.lines) == 1


def test_a_multigraph_draws_each_graph_by_its_own_option_or_its_own():
    one, two = _graph(), _graph(errors=False)
    mg = MultiGraph(
        "TMultiGraph",
        {"TNamed": {"fName": "mg", "fTitle": ""}, "fGraphs": Listed([one, two], ["", "l"])},
    )
    fig = make([(mg, "ap")]).plot()
    (ax,) = fig.axes
    assert len(ax.containers) == 1  # the first graph's points and bars
    assert list(ax.lines[-1].get_ydata()) == [2.0, 4.0, 3.0]  # the second, drawn "l"
    assert ax.get_ylim()[1] == pytest.approx(4.5 + 0.3)


def test_the_fit_made_to_a_whole_multigraph_is_drawn_over_its_graphs():
    mg = MultiGraph(
        "TMultiGraph",
        {"TNamed": {"fName": "mg", "fTitle": ""}, "fGraphs": Listed([_graph()], [""])},
    )
    mg.functions.append(Function("f", "pol0", range=(1.0, 3.0), parameters=[3.0]))
    mg.functions.append(prim("TPaveStats"))  # a box, not a fit: drawn with nothing here
    (line,) = make([(mg, "ap")]).plot().axes[0].lines[-1:]
    assert list(line.get_ydata()[:2]) == [3.0, 3.0]


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


def test_a_stack_is_drawn_stacked_the_top_first():
    fig = make([(_stack(), "")]).plot()
    (ax,) = fig.axes
    steps = [p for p in ax.patches if isinstance(p, StepPatch)]
    tops = sorted(float(np.max(s.get_data().values)) for s in steps)
    assert tops == [3.0, 6.0]  # the second stands on the first
    assert ax.get_ylim()[1] == pytest.approx(6.3)


def test_a_stack_drawn_nostack_draws_each_histogram_by_itself():
    fig = make([(_stack(), "nostack")]).plot()
    steps = [p for p in fig.axes[0].patches if isinstance(p, StepPatch)]
    assert [s.get_data().values.max() for s in steps] == [3.0, 3.0]


# -- text, lines and shapes -------------------------------------------------------


def _texts():
    latex = prim(
        "TLatex", fTitle="#sqrt{s} = 13 TeV", fX=0.2, fY=0.8, fBits=0x03000000 | NDC,
        fTextColor=2, fTextAlign=22, fTextAngle=30.0,
    )  # fmt: skip
    text = prim("TText", fTitle="cost $5", fX=5.0, fY=1.0, fTextFont=43, fTextSize=20)
    fig = make([(filled(), "hist"), (latex, ""), (text, "")]).plot()
    return fig, {t.get_text(): t for t in fig.axes[0].texts}


def test_latex_is_drawn_in_mathtext_at_its_place_in_ndc():
    fig, texts = _texts()
    drawn = texts[r"$\sqrt{\mathrm{s}}\mathrm{\ =\ 13\ TeV}$"]
    assert (drawn.get_color(), drawn.get_rotation()) == ((1.0, 0.0, 0.0), 30.0)
    assert (drawn.get_ha(), drawn.get_va()) == ("center", "center")
    at = fig.transFigure.inverted().transform(drawn.get_transform().transform(drawn.get_position()))
    assert tuple(at) == pytest.approx((0.2, 0.8))


def test_text_is_drawn_as_it_is_in_the_axes_units_sized_in_pixels_for_precision_3():
    fig, texts = _texts()
    plain = texts[r"cost \$5"]
    assert plain.get_fontsize() == pytest.approx(14.4)
    assert plain.get_transform() is fig.axes[0].transData


def test_a_text_size_is_a_fraction_of_the_shorter_side_of_its_pad():
    latex = prim("TLatex", fTitle="x", fX=0.5, fY=0.5, fTextSize=0.1, fTextFont=132)
    fig = make([(latex, "")], width=800, height=400).plot()
    (text,) = fig.axes[0].texts
    assert text.get_fontsize() == pytest.approx(0.1 * 400 * 0.72)
    assert text.get_fontfamily() == ["serif"]
    assert text.get_fontstyle() == "normal"


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
    first, second, marker = ax.lines
    assert (first.get_color(), first.get_linestyle()) == ((1.0, 0.0, 0.0), "--")
    assert (first.get_transform() is ax.transData, second.get_transform() is ax.transData) == (
        True, False,
    )  # fmt: skip
    assert (marker.get_marker(), marker.get_markerfacecolor()) == ("o", "none")
    assert marker.get_markersize() == pytest.approx(16 * 0.72)


def test_arrows_and_boxes_are_drawn_in_their_style_and_place():
    ax = _shapes()
    assert len([p for p in ax.patches if isinstance(p, FancyArrowPatch)]) == 2
    (box,) = [p for p in ax.patches if p.get_hatch()]
    assert (box.get_x(), box.get_y(), box.get_width()) == pytest.approx((0.2, 0.6, 0.4))


def test_an_ellipse_is_drawn_whole_or_as_the_slice_it_is_limited_to():
    ax = _shapes()
    whole, slice_ = [p for p in ax.patches if isinstance(p, Polygon)][-2:]
    assert len(whole.get_xy()) < len(slice_.get_xy()) + 2
    assert slice_.get_xy()[-2] == pytest.approx((0.5, 0.5))  # a slice closes on its centre


def test_arrow_options_name_the_ends_their_heads_are_on():
    from xrdroot.canvas.shapes import _arrowstyle

    assert [_arrowstyle(o) for o in (">", "|>", "<", "<|", "<>", "<|>", "->-", "-<|-", "")] == [
        "->",
        "-|>",
        "<-",
        "<|-",
        "<->",
        "<|-|>",
        "->",
        "<|-",
        "-",
    ]


# -- paves and legends -------------------------------------------------------------


def test_a_pave_text_stacks_its_lines_in_its_box_with_its_shadow():
    lines = [
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
        fLines=lines,
        fTextSize=0.0,
        fTextAlign=0,
        fBorderSize=3,
        fShadowColor=1,
        fFillColor=0,
    )
    fig = make([(pave, "")]).plot()
    ax = fig.axes[0]
    shadow, box = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_zorder() > 3]
    assert shadow.get_x() < box.get_x()
    assert shadow.get_y() > box.get_y()  # top left
    texts = {t.get_text(): t for t in ax.texts}
    assert texts["first"].get_position()[1] > texts[r"${\alpha}$"].get_position()[1]
    assert texts["first"].get_ha() == "left"
    assert texts["placed"].get_position() == pytest.approx((0.3, 0.675))


def test_a_pave_placed_in_the_axes_units_is_converted_to_the_pads():
    pave = prim(
        "TPaveText", fX1=0.0, fY1=0.0, fX2=5.0, fY2=1.5, fOption="br", fLines=[], fBorderSize=0
    )
    fig = make([(filled(), "hist"), (pave, "")]).plot()
    boxes = [p for p in fig.axes[0].patches if isinstance(p, Rectangle) and p.get_zorder() > 3]
    (box,) = [p for p in boxes if p.get_x() == pytest.approx(0.1)]  # the other is the stats box
    assert box.get_width() == pytest.approx(0.4)
    assert box.get_linewidth() == 0.0


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
    texts = {t.get_text(): t for t in fig.axes[0].texts}
    assert texts["label"].get_position() == pytest.approx((0.25, 0.15))
    assert texts["sized"].get_fontsize() == pytest.approx(0.5 * 500 * 0.72)
    assert (
        len([p for p in fig.axes[0].patches if isinstance(p, Rectangle) and p.get_zorder() > 3])
        == 3
    )


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
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert (
        "saved title" in texts
        and r"$\mathrm{own\ title}$" not in texts
        and "own title" not in texts
    )
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
    texts = {t.get_text(): t.get_position()[1] for t in ax.texts}
    assert set(texts) == {"Header", "data", "fill", r"${\mu}$"}
    assert texts["Header"] == texts["data"] > texts["fill"]  # two columns, rows downwards
    red = [line for line in ax.lines if line.get_color() == (1.0, 0.0, 0.0)]
    assert len(red) == 2  # data's line and bar, in the histogram's colour
    green = [p for p in ax.patches if p.get_facecolor()[:3] == (0.0, 1.0, 0.0)]
    assert green[0].get_linewidth() > 0


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
    assert fig.axes[0].patches


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
    assert fig.axes[0].lines[0].get_color() == (0.0, 0.5, 0.0)
    assert _mapped(only(fig, "c_1")).cmap.name == "saved"
    (steps,) = [p for p in only(fig, "c_2").patches if isinstance(p, StepPatch)]
    assert steps.get_facecolor()[:3] == pytest.approx((0.0, 0.5, 0.0), abs=0.01)
    assert steps.get_edgecolor()[:3] == pytest.approx((0.0, 0.5, 0.0), abs=0.01)


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
        (("dashes", 2, 2.0), (0, (3.0, 3.0))),
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
    axis = prim("TGaxis", fName="axis")
    quiet = [prim("TFrame"), prim("TPaletteAxis"), prim("TLegendEntry")]
    with pytest.warns(CanvasWarning) as caught:
        make([("TUnknown", ""), (axis, ""), (object(), ""), *((q, "") for q in quiet)]).plot()
    (warning,) = caught
    message = str(warning.message)
    assert "3 things" in message
    assert "TUnknown (a class this file does not describe)" in message
    assert "TGaxis 'axis'" in message
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
