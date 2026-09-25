"""Ratio plots, comparisons and stacks on every backend, and the styles they are drawn in."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from matplotlib import pyplot, rcParams

import xrdroot
from plotting import curve, gauss, grid, points, tidy  # noqa: F401
from xrdroot import Histogram, UnsupportedFeatureError
from xrdroot.plot import PETROFF, compare, label, ratio, stack, use_style
from xrdroot.plot.composite import WORKED

# -- ratio plots ----------------------------------------------------------------------------------


def test_divsym_is_th1_divide_with_the_errors_of_both():
    n, d = np.array([4.0, 2.0, 1.0]), np.array([2.0, 2.0, 0.0])
    value, low, high = WORKED["divsym"](n, np.sqrt(n), d, np.sqrt(d))
    assert value[:2].tolist() == [2.0, 1.0] and np.isnan(value[2])
    assert low[0] == pytest.approx(np.sqrt((2 / 2) ** 2 + (4 * np.sqrt(2) / 4) ** 2))
    assert low.tolist()[:2] == high.tolist()[:2]


def test_diff_and_diffsig_are_the_difference_and_it_over_its_error():
    n, d = np.array([4.0, 1.0]), np.array([1.0, 1.0])
    value, low, _ = WORKED["diff"](n, np.array([3.0, 0.0]), d, np.array([4.0, 0.0]))
    assert value.tolist() == [3.0, 0.0] and low.tolist() == [5.0, 0.0]
    pull, bars, _ = WORKED["diffsig"](n, np.array([3.0, 0.0]), d, np.array([4.0, 0.0]))
    assert pull[0] == 0.6 and np.isnan(pull[1]) and bars.tolist() == [0.0, 0.0]


def test_pois_is_the_binomial_interval_on_a_ratio_of_counts():
    from xrdroot.efficiency import beta_quantile

    value, low, high = WORKED["pois"](
        np.array([4.0, 0.0, 3.0]), None, np.array([2.0, 5.0, 0.0]), None
    )
    upper = beta_quantile(1 - (1 - 0.682689492137086) / 2, 5, 2)
    assert value[0] == 2.0 and high[0] == pytest.approx(upper / (1 - upper) - 2)
    assert low[1] == 0.0 and np.isnan(value[2]) and np.isnan(low[2])


def test_a_ratio_on_matplotlib_is_two_axes_sharing_x_with_the_ratio_about_one():
    data, model = gauss("data", 1), gauss("model", 2)
    upper, lower = ratio(data, model, labels=["data", "model"], logy=True)
    assert upper.get_yscale() == "log" and upper.get_xlabel() == ""
    assert lower.get_xlabel() == "x [GeV]" and lower.get_ylabel() == "ratio"
    reference, points_ = lower.lines[0], lower.lines[1]
    assert reference.get_ydata().tolist() == [1.0, 1.0]
    expected = data.values() / model.values()
    assert points_.get_ydata().tolist() == pytest.approx(expected[model.values() != 0].tolist())
    assert sorted(t.get_text() for t in upper.get_legend().get_texts()) == ["data", "model"]


def test_a_ratio_against_a_fit_is_its_pulls():
    data = gauss()
    data.fit("gaus", "Q")
    _, lower = ratio(data, data.functions[0], "diffsig")
    assert lower.get_ylabel() == "pull" and lower.lines[0].get_ydata().tolist() == [0.0, 0.0]


def test_a_ratio_is_refused_for_what_is_not_a_ratio_of_two_1d_histograms():
    with pytest.raises(ValueError, match="numerator of a ratio plot is a histogram of one axis"):
        ratio(grid(), gauss())
    with pytest.raises(ValueError, match="binned differently"):
        ratio(gauss(), Histogram.new("h", [0, 1], [1]))
    with pytest.raises(ValueError, match="'pois' is a ratio of two counts"):
        ratio(gauss(), curve(), "pois")
    with pytest.raises(ValueError, match="option='quotient' is not one of TRatioPlot's"):
        ratio(gauss(), gauss(), "quotient")


def test_a_ratio_on_plotly_is_one_figure_of_two_rows():
    figure = ratio(gauss("a", 1), gauss("b", 2), "diff", backend="plotly", style="ROOT")
    assert {trace.yaxis for trace in figure.data} == {"y", "y2"}
    assert figure.layout.yaxis2.title.text == "difference"
    assert figure.layout.template.layout.plot_bgcolor == "white"


def test_a_ratio_on_bokeh_is_a_column_of_two_figures_sharing_x():
    column = ratio(gauss("a", 1), gauss("b", 2), "pois", backend="bokeh", style="ROOT")
    upper, lower = column.children
    assert lower.x_range is upper.x_range and lower.yaxis[0].axis_label == "ratio"
    assert upper.xaxis[0].major_tick_in == 8


def test_a_ratio_in_characters_is_the_two_pictures_one_after_the_other():
    written = ratio(gauss("a", 1), gauss("b", 2), backend="text")
    assert written.count("ratio") == 1 and "*" in written and "█" in written


# -- comparisons and stacks -----------------------------------------------------------------------


def test_compare_draws_each_in_a_colour_of_its_own_with_a_legend():
    ax = compare([gauss("a", 1), gauss("b", 2)], option="HIST", norm=True)
    colours = [patch.get_edgecolor() for patch in ax.patches]
    assert colours[0] != colours[1]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]
    assert sum(ax.patches[0].get_data()[0]) == pytest.approx(1.0)
    figure = compare([points(), curve()], ["g", "f"], backend="plotly", colors=["red", "blue"])
    assert figure.data[0].line.color == "red" and figure.data[-1].name == "f"


def test_stack_fills_each_in_the_petroff_colours_first_at_the_bottom():
    low, high = Histogram.new("a", [0, 1, 2], [1, 2]), Histogram.new("b", [0, 1, 2], [3, 1])
    ax = stack([low, high], title="pile")
    top = ax.patches[1]
    assert top.get_data()[0].tolist() == [4.0, 3.0] and top.get_data()[2].tolist() == [1.0, 2.0]
    assert top.get_facecolor()[:3] == pytest.approx(pyplot.matplotlib.colors.to_rgb(PETROFF[1]))
    assert ax.get_title() == "pile"
    assert len(stack([low, high], option="NOSTACKB", colors=["red", "blue"]).patches) == 4


def test_a_stack_read_from_a_file_draws_itself():
    low, high = Histogram.new("a", [0, 1, 2], [1, 2]), Histogram.new("b", [0, 1, 2], [3, 1])
    held = xrdroot.Stack(
        "THStack", {"TNamed": {"fName": "s", "fTitle": "t"}, "fHists": [low, high]}
    )
    assert held.plot(option="NOSTACK").patches[1].get_data()[0].tolist() == [3.0, 1.0]


# -- styles and labels ----------------------------------------------------------------------------


def test_the_root_style_is_in_force_while_drawing_and_not_after():
    ax = gauss().plot(style="ROOT")
    assert ax.xaxis.get_ticks_position() in ("both", "default")
    assert ax.xaxis.get_label().get_horizontalalignment() == "right"
    assert rcParams["xtick.direction"] == "out"


def test_mplheps_styles_are_used_and_one_it_does_not_have_is_refused():
    assert gauss().plot(style="CMS") is not None
    with pytest.raises(ValueError, match="style='Nonesuch' is not one of mplhep's"):
        gauss().plot(style="Nonesuch")


def test_mplhep_not_installed_is_refused_with_the_way_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "mplhep", None)
    with pytest.raises(UnsupportedFeatureError, match=r"pip install mplhep.*style='ROOT'"):
        gauss().plot(style="ATLAS")


def test_a_style_for_another_backend_is_refused_as_not_theirs():
    with pytest.raises(ValueError, match="plotly draws style='ROOT' only"):
        gauss().plot(backend="plotly", style="CMS")
    with pytest.raises(ValueError, match="nothing to style in characters"):
        gauss().plot(backend="text", style="ROOT")


def test_plotly_and_bokeh_draw_in_roots_style():
    figure = gauss().plot(backend="plotly", style="ROOT")
    assert figure.layout.xaxis.ticks == "inside" and figure.layout.yaxis.mirror == "allticks"
    fig = gauss().plot(backend="bokeh", style="ROOT")
    assert fig.yaxis[0].minor_tick_in == 4 and not fig.xgrid[0].visible


def test_use_style_sets_matplotlibs_settings_from_now_on():
    saved = dict(rcParams)
    try:
        use_style("ROOT")
        assert rcParams["xtick.direction"] == "in"
    finally:
        rcParams.update(saved)


def test_an_experiments_label_is_written_above_the_plot_on_every_backend():
    ax = label(gauss().plot(), "CMS", "Preliminary", lumi=138)
    written = [text.get_text() for text in ax.texts]
    assert written == [r"$\mathbf{CMS}$ $\mathit{Preliminary}$", "138 fb$^{-1}$ (13 TeV)"]
    pair = label(ratio(gauss("a", 1), gauss("b", 2)), "LHCb", "", energy=None)
    assert pair[0].texts[0].get_text() == r"$\mathbf{LHCb}$" and pair[0].texts[1].get_text() == ""
    figure = label(gauss().plot(backend="plotly"), lumi=59.7)
    assert figure.layout.annotations[1].text == "59.7 fb⁻¹ (13 TeV)"
    fig = label(gauss().plot(backend="bokeh"), "ATLAS", "Internal")
    assert fig.above[-1].text == "ATLAS Internal"
    assert label("picture", "ALICE", energy=5.02).startswith("ALICE Preliminary   (5.02 TeV)\n")
