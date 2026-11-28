"""The parts of ``TH1`` a reader meets first: an axis, the sums, the density and the text
picture. ``Axis`` is what a caller indexes, iterates and prints; ``sum``/``density`` are
what a plot's normalisation asks for; ``text`` is ROOT's character drawing, the thing a log
file has instead of a canvas. Each is reachable from a histogram booked here, so no file is
opened - which keeps this test inside the ``hist`` component's own scope. Where a number
matters it is ROOT's own definition and the arithmetic is shown: a density is the content
divided by the bin's width and by the total, so it integrates to one."""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot import UnsupportedFeatureError
from xrdroot.hist import Histogram


def _three_bins() -> Histogram:
    """Three bins of 0..3, each filled twice: the total is six and every bin holds two."""
    made = Histogram.book("h", (3, 0.0, 3.0))
    for value in (0.5, 0.5, 1.5, 1.5, 2.5, 2.5):
        made.fill(value)
    return made


def test_an_axis_yields_its_bins_edges_and_indexes_them_from_zero():
    """``axis[0]`` is the first bin's (low, high), a negative index counts from the end, and one
    past the end names the axis and the count rather than reading whatever is there."""
    axis = _three_bins().axes[0]
    assert axis[0] == (0.0, 1.0)
    assert axis[-1] == (2.0, 3.0)
    assert list(axis) == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]  # iteration is the pairs
    with pytest.raises(IndexError, match="bin 3 of an axis of 3"):
        axis[3]


def test_two_axes_are_equal_when_their_edges_are_and_print_their_range():
    """An axis compares by its edges - ROOT's own identity for a binned axis - and its repr names
    the axis, the count and the range."""
    one = Histogram.book("a", (3, 0.0, 3.0)).axes[0]
    assert one == Histogram.book("b", (3, 0.0, 3.0)).axes[0]
    assert one != Histogram.book("c", (3, 0.0, 4.0)).axes[0]
    assert one.__eq__(4.0) is NotImplemented  # anything that is not an axis
    assert repr(one) == "<Axis 'xaxis' of 3 bins from 0 to 3>"


def test_the_widths_are_what_a_density_divides_by():
    """``widths`` is the difference of the edges: one for an even axis, and the uneven bins' own for
    a variable one."""
    assert _three_bins().axes[0].widths().tolist() == [1.0, 1.0, 1.0]
    uneven = Histogram.book("u", [0.0, 1.0, 3.0, 6.0])  # type: ignore[arg-type]
    assert uneven.axes[0].widths().tolist() == [1.0, 2.0, 3.0]


def test_the_sum_of_the_bins_is_their_total_and_the_flow_is_its_own():
    """``sum`` adds the bins - the flow bins only when asked - which is the total weight a
    normalisation divides by."""
    found = _three_bins()
    found.fill(-1.0)  # into the underflow
    assert found.sum() == 6.0
    assert found.sum(flow=True) == 7.0


def test_the_density_integrates_to_one_over_the_bins_widths():
    """A density is the content over the bin's width and the total, so the density times the widths
    adds up to one - ROOT's ``TH1::GetBinContent / (GetBinWidth * GetSumOfWeights)``."""
    found = _three_bins()
    density = found.density()
    assert np.sum(density * found.axes[0].widths()) == pytest.approx(1.0)


def test_a_density_of_the_flow_bins_is_refused_because_they_have_no_width():
    """``density(flow=True)`` has nothing to divide the flow bins by, and says so in a sentence."""
    with pytest.raises(ValueError, match="flow bins is not a thing"):
        _three_bins().density(flow=True)


def test_a_density_of_a_plane_divides_by_both_widths():
    """For two axes the volume of a cell is the product of the two widths, so the density times the
    cells' volumes adds to one."""
    plane = Histogram.book("p", (2, 0.0, 2.0), (2, 0.0, 4.0))
    plane.fill(0.5, 1.0)
    plane.fill(1.5, 3.0)
    plane.fill(1.5, 3.0)
    density = plane.density()
    volumes = np.multiply.outer(plane.axes[0].widths(), plane.axes[1].widths())
    assert np.sum(density * volumes) == pytest.approx(1.0)


def test_the_text_picture_is_one_line_a_bin_for_a_line_and_a_grid_for_a_plane():
    """``text`` is ROOT's character drawing: a row per bin with its edges, a bar and the value for
    one axis, a shaded grid for two - and a refusal for three, which have no flat picture."""
    picture = _three_bins().text()
    assert picture.count("\n") == 2  # three bins, no trailing newline
    assert "2" in picture  # the bin contents are in the drawing
    plane = Histogram.book("p", (2, 0.0, 2.0), (2, 0.0, 2.0))
    plane.fill(0.5, 0.5)
    assert isinstance(plane.text(), str)
    solid = Histogram.book("s", (2, 0.0, 2.0), (2, 0.0, 2.0), (2, 0.0, 2.0))
    with pytest.raises(UnsupportedFeatureError, match="picture"):
        solid.text()


def test_a_histogram_prints_its_class_its_shape_and_its_entries():
    """``repr`` is what a terminal shows: the class, the name, how many bins on each axis and how
    many entries went in."""
    assert repr(_three_bins()) == "<TH1D 'h' of 3 bins, 6 entries>"


def test_turning_a_histogram_into_a_hist_hist_keeps_the_values_and_the_weights():
    """``to_hist`` hands the bins to the ``hist`` package with the flow, and a weighted histogram's
    variances go across with them - the plotting protocol's own conversion."""
    pytest.importorskip("hist")
    made = _three_bins().to_hist()
    assert made.values(flow=True).tolist()[:1] == [0.0]  # the underflow, empty here
    weighted = Histogram.book("w", (2, 0.0, 2.0))
    weighted.fill(0.5, weight=2.0)
    carried = weighted.to_hist()
    assert carried.values().tolist() == pytest.approx([2.0, 0.0])
    assert carried.variances().tolist() == pytest.approx([4.0, 0.0])  # the square of the weight
