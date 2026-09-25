"""Pictures in characters, and choosing the backend a picture is drawn by."""

from __future__ import annotations

import sys

import pytest

from plotting import cube, curve, gauss, grid, points, tidy  # noqa: F401
from xrdroot import Histogram, UnsupportedFeatureError
from xrdroot.plot import BACKENDS, get_backend, set_backend
from xrdroot.plot.backends import astext


def test_a_histogram_in_characters_is_a_line_per_bin_under_its_titles():
    made = Histogram.new("h", [0, 1, 2], [4, 2], title="counts")
    made.axes[0].title = "energy"
    lines = made.plot(backend="text").splitlines()
    assert lines[:3] == ["counts", "energy", ""]
    assert lines[3].endswith(" 4") and "█" * 50 in lines[3] and "[1, 2)" in lines[4]


def test_bars_points_curves_bands_and_boxes_are_drawn_as_what_they_are():
    made = Histogram.new("h", [0, 1, 2], [4, 2])
    assert "[0.1, 0.9)" in made.plot(backend="text", option="B")
    for option in ("E", "L", "E3", "E2", "TEXT"):
        assert "*" in made.plot(backend="text", option=option) or option == "TEXT"
    assert "*" in curve().plot(backend="text")
    stars = points().plot(backend="text", option="P").splitlines()
    assert stars[2].startswith(f"{3:>10} |") and stars[-1].endswith("3")


def test_a_grid_is_shaded_with_y_upward():
    shaded = grid().plot(backend="text", option="COLZ").splitlines()[2:]
    assert len(shaded) == 3 and set("".join(shaded)) <= set(" ░▒▓█")
    for option in ("CONT", "SURF"):
        assert grid().plot(backend="text", option=option).count("\n") == 4
    assert astext.shades(Histogram.new("e", [[0, 1], [0, 1]], [[0]]).values()) == " "


def test_what_has_no_flat_picture_and_nothing_to_draw_say_so():
    with pytest.raises(UnsupportedFeatureError, match="no honest flat picture"):
        cube().plot(backend="text")
    assert astext.stars([], []) == "(no points)"
    assert "*" in astext.stars([1, 1], [2, 2])


def test_same_writes_after_the_last_picture_and_a_string_given_is_written_after():
    first = gauss().plot(backend="text")
    both = curve().plot(backend="text", option="SAME")
    assert both.startswith(first) and both.count("*") > 10
    assert curve().plot(backend="text", ax="before").startswith("before\n\n")


def test_the_backend_is_set_by_name_and_refused_when_it_is_not_one():
    assert get_backend() == "matplotlib" and list(BACKENDS) == [
        "matplotlib", "plotly", "bokeh", "text",
    ]  # fmt: skip
    set_backend("TEXT")
    assert get_backend() == "text" and isinstance(gauss().plot(), str)
    with pytest.raises(ValueError, match="backend='gnuplot' is not one this draws with"):
        set_backend("gnuplot")


def test_a_backend_not_installed_is_refused_with_the_way_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly", None)
    with pytest.raises(UnsupportedFeatureError, match=r"pip install plotly.*backend='text'"):
        gauss().plot(backend="plotly")
