"""Points in space: their ranges, the histogram they draw with, and ``TGraph2D::Fit``.

The histogram is ROOT's ``GetHistogram``: ``fNpx`` by ``fNpy`` bins over the
points' range - their bars' reach, and ``fMargin`` more - each filled at its
centre with the surface's height, its z range the points'. A fit weighs
each height by its z error, or by one when there are none, and with errors
in x and y widens each by the slope there, as the effective variance does.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.fit.data import COORD_ERROR, NO_ERROR, VALUE_ERROR, DataOptions, from_graph2d
from xrdroot.function import Function
from xrdroot.graph2d import UNSET, Graph2D

pytest.importorskip("iminuit")


def _grid():
    gx, gy = np.meshgrid(np.linspace(-1, 1, 6), np.linspace(0, 2, 5), indexing="ij")
    return gx.ravel(), gy.ravel()


def test_the_histogram_is_the_surface_at_each_bins_centre_over_the_points_range():
    x, y = _grid()
    g = Graph2D.new("g", x, y, 3 * x + y, title="plane")
    h = g.histogram()
    assert (h.name, h.title, h.axes[0].low, h.axes[0].high) == ("g", "plane", -1.0, 1.0)
    assert h.values()[0, 0] == pytest.approx(3 * (-1 + 1 / 40) + 2 / 80)
    assert (h._core["fMinimum"], h._core["fMaximum"]) == (-3.0, 5.0)
    assert g.histogram() is h and g.histogram(empty=True) is not h
    assert g.histogram(empty=True).entries == 0
    g.members.update(fMargin=0.5, fMinimum=-10.0, fMaximum=UNSET)
    g.changed()
    wide = g.histogram()
    assert (wide.axes[1].low, wide.axes[1].high, wide._core["fMinimum"]) == (-1.0, 3.0, -10.0)
    assert wide.values()[0, 0] == 0.0  # outside the hull


def test_ranges_reach_the_bars_of_a_graph_that_has_them():
    g = Graph2D.new("e", [0, 1, 2], [0, 1, 0], [5, 6, 7], errors=([0.5] * 3, [1] * 3, [2] * 3))
    assert g.extent(0) == (0.0, 2.0) and g.extent(0, bars=True) == (-0.5, 2.5)
    assert g.extent(2, bars=True) == (3.0, 9.0) and len(g) == 3
    assert Graph2D.new("none", [], [], []).extent(1) == (0.0, 0.0)
    flat = Graph2D.new("flat", [1, 1, 1], [0, 1, 2], [0, 0, 0])
    assert flat.histogram().axes[0].low == 0.0  # a range of no width, widened by one
    with pytest.raises(ValueError, match="laid over three or more"):
        Graph2D.new("two", [0, 1], [0, 1], [0, 1]).interpolate(0.5, 0.5)


def _plane():
    return Function("plane", "[0]*x + [1]*y + [2]", range=[(-1, 1), (0, 2)])


def test_a_graph_without_errors_is_fitted_with_errors_of_one(capsys):
    x, y = _grid()
    g = Graph2D.new("g", x, y, 2 * x - y + 0.5 + 0.01 * np.sin(7 * x * y))
    model = _plane()
    model.set_parameters(1, 1, 1)
    found = g.fit(model, "Q")
    assert found.parameters == pytest.approx([2, -1, 0.5], abs=0.01)
    assert found.ndf == len(x) - 3 and capsys.readouterr().out == ""
    assert g.functions[0].range == ((-1.0, 1.0), (0.0, 2.0))


def test_the_kind_of_points_comes_from_the_errors_the_graph_has():
    x, y = _grid()
    n = len(x)
    plain = Graph2D.new("p", x, y, x)
    zonly = Graph2D.new("z", x, y, x, errors=(np.zeros(n), np.zeros(n), np.r_[0, np.ones(n - 1)]))
    every = Graph2D.new("c", x, y, x, errors=(np.full(n, 0.1), np.zeros(n), np.zeros(n)))
    assert from_graph2d(plain, DataOptions(), [None, None]).kind == NO_ERROR
    kept = from_graph2d(zonly, DataOptions(), [(-0.5, 1), None])
    assert (kept.kind, kept.size) == (VALUE_ERROR, int(np.sum(x >= -0.5)) - 0)
    coord = from_graph2d(every, DataOptions(), [None, (0.0, 1.0)])
    assert coord.kind == COORD_ERROR and coord.xerr.shape == (coord.size, 2)
    assert from_graph2d(every, DataOptions(errors1=True), [None, None]).kind == NO_ERROR
    assert from_graph2d(every, DataOptions(coord_errors=False), [None, None]).kind == VALUE_ERROR


def test_errors_in_x_and_y_widen_each_heights_error_by_the_slope(capsys):
    x, y = _grid()
    n = len(x)
    z = 2 * x - y + 0.5
    g = Graph2D.new("c", x, y, z, errors=(np.full(n, 0.1), np.full(n, 0.2), np.full(n, 0.3)))
    model = _plane()
    model.set_parameters(1, -2, 0)
    found = g.fit(model, "Q")
    assert found.parameters == pytest.approx([2, -1, 0.5], abs=1e-4)
    assert found.chi2 == pytest.approx(0.0, abs=1e-6)
