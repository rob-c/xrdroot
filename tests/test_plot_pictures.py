"""What each object is drawn as, before any backend draws it: ROOT's defaults and options."""

from __future__ import annotations

import numpy as np
import pytest

from plotting import DATA, cube, curve, gauss, grid, points, surface, tidy  # noqa: F401
from xrdroot import Efficiency, Function, Graph, Histogram, UnsupportedFeatureError, open_root
from xrdroot.plot import picture
from xrdroot.plot.drawers import NOT_DRAW
from xrdroot.plot.model import (
    Band,
    Bars,
    Boxes,
    Cloud,
    Contour,
    Curve,
    Labels,
    Mesh,
    Points,
    Steps,
    Surface,
)
from xrdroot.stacks import MultiGraph, Stack


def kinds(made) -> list[type]:
    return [type(layer) for layer in made.layers]


# -- one-dimensional histograms ---------------------------------------------------------------


def test_a_histogram_is_drawn_as_hist_until_it_keeps_its_squares_of_weights():
    plain = Histogram.new("h", [0, 1, 2], [4, 2], title="counts")
    assert kinds(picture(plain)) == [Steps]
    plain.sumw2()
    assert kinds(picture(plain)) == [Points]


def test_a_histograms_frame_is_its_title_and_its_axes_titles():
    made = gauss()
    made.members["TH1"]["fYaxis"]["TNamed"]["fTitle"] = "entries"
    frame = picture(made, "", {"logy": True}).frame
    assert (frame.title, frame.xlabel, frame.ylabel, frame.logy) == (
        "h",
        "x [GeV]",
        "entries",
        True,
    )


def test_error_bars_leave_out_empty_bins_unless_e0_and_are_half_a_bin_wide_unless_x0():
    made = Histogram.new("h", [0, 1, 2, 3], [4, 0, 2])
    bars = picture(made, "E").layers[0]
    assert bars.x.tolist() == [0.5, 2.5] and bars.xlow.tolist() == [0.5, 0.5]
    assert bars.ylow.tolist() == [2.0, np.sqrt(2)] and not bars.caps
    assert len(picture(made, "E0").layers[0].x) == 3
    assert picture(made, "E1X0").layers[0].xlow.tolist() == [0.0, 0.0]
    assert picture(made, "E1").layers[0].caps


def test_markers_alone_have_no_bars_and_a_star_is_roots_asterisk():
    made = Histogram.new("h", [0, 1, 2], [4, 2])
    marked = picture(made, "P").layers[0]
    assert marked.yhigh.tolist() == [0.0, 0.0] and marked.look.marker == "point"
    assert picture(made, "*").layers[0].look.marker == "asterisk"


def test_e2_draws_boxes_and_markers_and_e3_e4_a_band():
    made = Histogram.new("h", [0, 1, 2], [4, 1])
    boxes, marks = picture(made, "E2").layers
    assert isinstance(boxes, Boxes) and isinstance(marks, Points)
    assert (boxes.y0.tolist(), boxes.y1.tolist()) == ([2.0, 0.0], [6.0, 2.0])
    assert boxes.look.alpha == 0.35 and boxes.look.fill == "#000000"
    filled = picture(made, "E2", {"fill": "kRed"}).layers[0]
    assert (filled.look.fill, filled.look.alpha) == ("#ff0000", 1.0)
    band = picture(made, "E4", {"fill": "kRed"}).layers[0]
    assert isinstance(band, Band) and band.smooth and band.look.alpha == 1.0
    assert not picture(made, "E3").layers[0].smooth


def test_lines_bars_and_numbers_are_drawn_in_roots_order():
    made = Histogram.new("h", [0, 2, 4], [4, 1])
    assert kinds(picture(made, "TEXT L B HIST")) == [Bars, Steps, Curve, Labels]
    bars = picture(made, "B").layers[0]
    assert bars.left.tolist() + bars.right.tolist() == pytest.approx([0.2, 2.2, 1.8, 3.8])
    assert picture(made, "C").layers[0].smooth
    labels = picture(made, "TEXT45").layers[0]
    assert labels.texts == ("4", "1") and labels.angle == 45.0


def test_norm_scales_to_a_sum_of_one_and_is_refused_for_a_profile():
    made = Histogram.new("h", [0, 1, 2], [3, 1])
    assert picture(made, "HIST NORM").layers[0].values.tolist() == [0.75, 0.25]
    empty = Histogram.new("h", [0, 1, 2], [0, 0])
    assert picture(empty, "HIST NORM").layers[0].values.tolist() == [0.0, 0.0]
    with open_root(f"{DATA}/tprofile.root") as root:
        profile = root["p1d"]
    assert kinds(picture(profile)) == [Points]
    with pytest.raises(ValueError, match="profile, whose bins are means"):
        picture(profile, "NORM")


# -- the fits a histogram carries -----------------------------------------------------------------


def test_a_fit_is_drawn_over_the_histogram_unless_hist_says_not():
    made = gauss()
    made.fit("gaus", "Q")
    assert kinds(picture(made)) == [Steps, Curve]  # ROOT's outline, and its fit
    fitted = picture(made).layers[1]
    assert len(fitted.x) == 101 and fitted.look.color == "#ff0000" and fitted.look.width == 2
    assert kinds(picture(made, "HIST")) == [Steps]
    assert kinds(picture(made, "FUNC")) == [Curve]


def test_a_fit_made_with_option_zero_is_kept_but_not_drawn():
    made = gauss()
    made.fit("gaus", "Q0")
    assert made.functions[0].members["TNamed"]["fBits"] & NOT_DRAW
    assert kinds(picture(made)) == [Steps]


def test_a_function_that_cannot_be_worked_out_is_left_off_the_histogram():
    def refuses(x, params):
        raise UnsupportedFeatureError("no")

    made = Histogram.new("h", [0, 1, 2], [4, 2])
    made.attach(Function.from_callable("f", refuses, 0, range=(0, 2)))
    made.attach({"a ROOT 5 function": "kept as it was read"})
    made.attach(surface())
    assert kinds(picture(made)) == [Steps]


def test_functions_draw_as_their_curve_or_as_points():
    made = picture(curve(), "", {"label": "fit"})
    line = made.layers[0]
    assert isinstance(line, Curve) and line.x[0] == -3 and line.x[-1] == 3 and len(line.x) == 101
    assert line.look.label == "fit" and made.frame.title == "gaus"
    assert isinstance(picture(curve(), "P").layers[0], Points)
    assert picture(curve(), "C").layers[0].smooth


def test_a_tf2_is_its_contour_lines_and_a_tf3_has_no_picture():
    made = picture(surface()).layers[0]
    assert isinstance(made, Contour) and not made.filled and made.values.shape == (30, 30)
    assert made.values[15, 15] == pytest.approx(np.exp(-2 * (1 / 30) ** 2))
    assert isinstance(picture(surface(), "COLZ").layers[0], Mesh)
    cubed = Function("f3", "x+y+z", range=[(0, 1), (0, 1), (0, 1)])
    with pytest.raises(UnsupportedFeatureError, match="function of 3 variables"):
        picture(cubed)


# -- two and three dimensions ---------------------------------------------------------------------


def test_a_2d_histogram_is_colz_with_its_empty_cells_unpainted():
    made = grid()
    made.members["TH2"]["TH1"]["fZaxis"]["TNamed"]["fTitle"] = "count"
    shaded = picture(made)
    mesh = shaded.layers[0]
    assert isinstance(mesh, Mesh) and mesh.scale and mesh.palette == "bird"
    assert mesh.values.tolist() == made.values().tolist()
    assert shaded.frame.zlabel == "count"
    empty = Histogram.new("e", [[0, 1, 2], [0, 1]], [[0], [3]])
    assert np.isnan(picture(empty, "COL").layers[0].values[0, 0])
    assert not picture(empty, "COL", {"palette": "viridis"}).layers[0].scale


def test_each_2d_option_makes_its_layer_in_order():
    made = picture(grid(), "COLZ BOX CONT CONT1 LEGO SURF TEXT", {"levels": 5})
    assert kinds(made) == [Mesh, Boxes, Contour, Contour, Surface, Surface, Labels]
    assert made.layers[2].filled and not made.layers[3].filled and made.layers[2].levels == 5
    assert made.layers[4].lego and not made.layers[5].lego and made.deep
    biggest = np.argmax(grid().values())
    boxes = made.layers[1]
    assert boxes.x1[biggest] - boxes.x0[biggest] == pytest.approx(1.0)


def test_boxes_of_an_empty_grid_are_none():
    empty = Histogram.new("e", [[0, 1], [0, 1]], [[0]])
    assert len(picture(empty, "BOX").layers[0].x0) == 0


def test_a_3d_histogram_is_a_cloud_of_its_filled_bins_or_iso_of_them_all():
    made = picture(cube())
    cloud = made.layers[0]
    assert isinstance(cloud, Cloud) and made.deep
    assert (cloud.x.tolist(), cloud.y.tolist(), cloud.z.tolist()) == ([0.5], [1.5], [0.5])
    assert len(picture(cube(), "ISO").layers[0].values) == 8
    assert picture(cube(), "BOX NORM").layers[0].values.tolist() == [1.0]


# -- graphs ---------------------------------------------------------------------------------------


def test_a_graph_is_drawn_alp_or_as_its_saved_option():
    made = points()
    line, marks = picture(made).layers
    assert isinstance(line, Curve) and isinstance(marks, Points) and marks.caps
    assert (marks.ylow.tolist(), marks.yhigh.tolist()) == ([0.1, 0.2, 0.1], [0.3, 0.2, 0.1])
    made.members["fOption"] = "P"
    assert kinds(picture(made)) == [Points]


def test_a_graphs_axis_titles_are_on_the_histogram_it_draws_its_frame_with():
    made = Graph.new("g", [1, 2], [3, 4], title="scan")
    frame = Histogram.new("frame", [0, 1], [0])
    frame.axes[0].title = "time"
    frame.members["TH1"]["fYaxis"]["TNamed"]["fTitle"] = "height"
    made.members["fHistogram"] = frame
    shown = picture(made).frame
    assert (shown.title, shown.xlabel, shown.ylabel) == ("scan", "time", "height")


def test_x_drops_the_error_bars_z_their_ticks_and_2_3_4_draw_boxes_and_bands():
    made = points()
    assert picture(made, "PX").layers[0].yhigh.tolist() == [0.0] * 3
    assert not picture(made, "PZ").layers[0].caps
    assert kinds(picture(made, "A2P")) == [Boxes, Points]
    band = picture(made, "A4").layers[0]
    assert isinstance(band, Band) and band.smooth
    assert band.high.tolist() == pytest.approx([2.3, 3.2, 1.1])


def test_bars_of_a_graph_are_most_of_the_way_to_the_nearest_point():
    bars = picture(points(), "AB").layers[0]
    assert (bars.left.tolist(), bars.right.tolist()) == ([0.6, 1.6, 2.6], [1.4, 2.4, 3.4])
    lonely = picture(Graph.new("g", [1], [2]), "B").layers[0]
    assert (lonely.left.tolist(), lonely.right.tolist()) == ([0.6], [1.4])


def test_a_graph_with_layers_of_errors_draws_a_set_of_bars_per_layer():
    with open_root(f"{DATA}/tgme.root") as root:
        made = root["gme"]
    first, second = picture(made, "AP", {"label": "both"}).layers
    assert first.look.label == "both" and second.look.label is None
    assert second.xlow.tolist() == [0.0] * 5
    assert kinds(picture(Graph.new("g", [1], [2]), "P")) == [Points]


def test_a_graphs_fits_are_drawn_over_it():
    made = Graph.new("g", [0, 1, 2], [0, 1, 2])
    made.fit("pol1", "Q")
    assert kinds(picture(made, "AP")) == [Points, Curve]


def test_palette_colours_follow_plc_and_pmc():
    marks = picture(points(), "P PLC PMC PFC").layers[0]
    assert marks.look.color == marks.look.marker_color == marks.look.fill == "#352a86"


# -- efficiencies, stacks and multigraphs ---------------------------------------------------------


def test_an_efficiency_is_its_tried_bins_with_their_intervals_or_a_shaded_grid():
    with open_root(f"{DATA}/tefficiency.root") as root:
        flat, flat2, cubic = root["eff1"], root["eff2"], root["eff3"]
    marks = picture(flat).layers[0]
    tried = flat.total.values() != 0
    assert marks.y.tolist() == flat.values()[tried].tolist()
    assert marks.ylow.tolist() == flat.errors()[0][tried].tolist()
    assert kinds(picture(flat, "A3")) == [Band]
    assert isinstance(picture(flat2).layers[0], Mesh)
    assert isinstance(picture(flat2, "CONT").layers[0], Contour)
    with pytest.raises(UnsupportedFeatureError, match="efficiency of 3 axes"):
        picture(cubic)


def test_a_stack_piles_each_histogram_on_the_ones_before_it():
    low, high = Histogram.new("a", [0, 1, 2], [1, 2]), Histogram.new("b", [0, 1, 2], [3, 1])
    held = Stack("THStack", {"TNamed": {"fName": "s", "fTitle": "pile"}, "fHists": [low, high]})
    first, second = picture(held, "", {"labels": ["a", "b"]}).layers
    assert second.values.tolist() == [4.0, 3.0] and second.baseline.tolist() == [1.0, 2.0]
    assert (first.look.label, second.look.label) == ("a", "b")
    marks = picture(held, "E").layers[1]
    assert marks.y.tolist() == [4.0, 3.0] and marks.ylow.tolist() == [np.sqrt(3), 1.0]
    assert picture(held, "NOSTACK").layers[1].values.tolist() == [3.0, 1.0]
    bars = picture(held, "NOSTACKB PFC").layers
    assert bars[1].left.tolist() == [0.5, 1.5] and bars[1].look.fill == "#f9f90e"
    assert picture(held).frame.title == "pile"


def test_an_empty_stack_is_an_empty_picture_and_a_stack_of_grids_is_refused():
    nothing = Stack("THStack", {"TNamed": {"fName": "s", "fTitle": "t"}, "fHists": []})
    assert picture(nothing).layers == ()
    grids = Stack("THStack", {"TNamed": {"fName": "s", "fTitle": ""}, "fHists": [grid()]})
    with pytest.raises(ValueError, match=r"histograms of one axis.*LEGO"):
        picture(grids)


def test_a_multigraph_draws_each_graph_with_its_option_and_its_title():
    with open_root(f"{DATA}/tgme.root") as root:
        held = root["mg"]
    made = picture(held, "AP PLC")
    # The three graphs, then the pol1 ROOT fitted to them all together.
    assert kinds(made) == [Points] * 3 + [Curve] and made.frame.title == held.title
    assert made.layers[0].look.color != made.layers[2].look.color
    assert kinds(picture(held)) == [Curve, Points] * 3 + [Curve]
    assert isinstance(held, MultiGraph)


def test_a_fit_made_to_a_multigraph_is_drawn_over_its_range():
    with open_root(f"{DATA}/tgme.root") as root:
        held = root["mg"]
    (fit,) = held.functions
    line = picture(held).layers[-1]
    assert (line.x[0], line.x[-1]) == pytest.approx(fit.range)
    assert line.y.tolist() == pytest.approx(fit(line.x).tolist())


# -- the picture as a whole -----------------------------------------------------------------------


def test_one_label_is_kept_once_however_many_layers_carry_it():
    made = picture(points(), "", {"label": "scan"})
    assert [layer.look.label for layer in made.layers] == ["scan", None]


def test_same_and_the_librarys_own_keywords_are_carried_to_the_backend():
    made = picture(points(), "P SAME", {"zorder": 3})
    assert made.same and made.native == {"zorder": 3}


def test_what_is_not_drawn_here_is_refused_by_what_it_is():
    with pytest.raises(TypeError, match="a dict is not something plot"):
        picture({})


def test_an_efficiency_is_an_efficiency():
    assert isinstance(Efficiency.book("e", (2, 0, 1)), Efficiency)
