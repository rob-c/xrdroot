"""``Histogram.of``: any histogram Python has, read as one this library writes.

ROOT's own histograms arrive as ROOT files, but a notebook's do not: what
``numpy.histogram``, ``numpy.histogram2d`` and ``numpy.histogramdd`` give back,
and anything speaking the plotting protocol (``hist``, ``boost_histogram``,
``mplhep``), is a pair of arrays and a set of axis objects. That is plain data, so
the conversion is exercised here without a file, which keeps this test inside the
``hist`` component's own isolation scope.

The protocol is checked against a **stand-in** rather than only against the real
package, because the stand-in is what pins the three things the conversion has to
survive: an object that refuses ``flow=True``, one whose flowed array is not the
shape the axes imply, and a ``kind`` that is an enum member rather than a string.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xrdroot import UnsupportedFeatureError
from xrdroot.hist import Histogram


class _Traits:
    """What ``hist``'s axis traits look like from the outside: two flags."""

    def __init__(self, *, discrete: bool = False, circular: bool = False) -> None:
        self.discrete = discrete
        self.circular = circular


class _Axis:
    """One axis of a plotting-protocol histogram: its bin edges and its label."""

    def __init__(self, pairs: list[tuple[float, float]], label: str = "") -> None:
        self._pairs = pairs
        self.label = label
        self.traits = _Traits()

    def __iter__(self) -> Any:
        return iter(self._pairs)


class _Kind:
    """``hist``'s ``Kind`` enum: the conversion reads ``.value`` off it."""

    def __init__(self, value: str) -> None:
        self.value = value


class _Plottable:
    """The plotting protocol as small as it can be: a kind, two arrays, some axes.

    ``refuses_flow`` is a histogram whose ``values``/``variances`` take no
    ``flow`` argument at all - the protocol allows that - and ``flow_shape``,
    when given, is what ``flow=True`` answers even though it is not two bins
    wider than the axes, which is the case a caller who lies about its flow
    produces.
    """

    def __init__(
        self,
        values: Any,
        variances: Any,
        axes: list[_Axis],
        *,
        kind: Any = "COUNT",
        refuses_flow: bool = False,
        flow_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.kind = kind
        self.axes = axes
        self._values = np.asarray(values, dtype=np.float64)
        self._variances = np.asarray(variances, dtype=np.float64)
        self._refuses = refuses_flow
        self._flow_shape = flow_shape

    def _read(self, flow: bool) -> tuple[Any, Any]:
        if self._refuses:
            if flow:
                raise TypeError("values() takes no keyword argument 'flow'")
            return self._values, self._variances
        if flow and self._flow_shape is not None:
            return (
                np.resize(self._values, self._flow_shape),
                np.resize(self._variances, self._flow_shape),
            )
        if flow:
            return np.pad(self._values, 1), np.pad(self._variances, 1)
        return self._values, self._variances

    def values(self, flow: bool = False) -> Any:
        return self._read(flow)[0]

    def variances(self, flow: bool = False) -> Any:
        return self._read(flow)[1]


def _three_bins() -> list[_Axis]:
    """Three bins of 0..3, which is what every stand-in below is binned on."""
    return [_Axis([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], "the x")]


def test_a_histogram_handed_to_of_is_the_same_object():
    """``of`` takes a ``Histogram`` as it is: nothing is copied, so a fill of the answer
    is a fill of the argument."""
    mine = Histogram.book("h", (3, 0.0, 3.0))
    assert Histogram.of(mine) is mine


def test_what_numpy_histogram_gives_back_becomes_a_one_dimensional_histogram():
    """``numpy.histogram`` gives ``(values, edges)``, and that is exactly the bins an
    evenly or unevenly binned ``TH1D`` is booked with."""
    values, edges = np.histogram([0.1, 0.4, 1.2, 1.3, 1.3], bins=3, range=(0.0, 3.0))
    made = Histogram.of((values, edges), name="from numpy")
    assert made.classname == "TH1D"
    assert made.name == "from numpy"
    assert made.shape == (3,)
    assert made.values().tolist() == [2.0, 3.0, 0.0]
    assert made.edges().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert made.entries == 5.0


def test_histogram2d_s_three_parts_become_a_plane():
    """``numpy.histogram2d`` gives ``(values, xedges, yedges)``, one part per axis."""
    values, xedges, yedges = np.histogram2d(
        [0.1, 0.5, 1.5], [0.2, 1.2, 0.7], bins=2, range=[[0.0, 2.0], [0.0, 2.0]]
    )
    made = Histogram.of((values, xedges, yedges), name="plane")
    assert made.classname == "TH2D"
    assert made.shape == (2, 2)
    assert made.values().tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert made.entries == 3.0


def test_histogramdd_s_list_of_edges_becomes_a_plane():
    """``numpy.histogramdd`` gives its edges as one list, which is the same histogram as
    ``histogram2d``'s three parts - and the two readings agree bin for bin."""
    values, edges = np.histogramdd(
        [[0.1, 0.5, 1.5], [0.2, 1.2, 0.7]], bins=2, range=[[0.0, 2.0], [0.0, 2.0]]
    )
    listed = Histogram.of((values, edges), name="listed")
    spread = Histogram.of((values, *edges), name="spread")
    assert listed.shape == (2, 2)
    assert listed.values().tolist() == spread.values().tolist()
    assert [axis.low for axis in listed.axes] == [0.0, 0.0]
    assert [axis.high for axis in listed.axes] == [2.0, 2.0]


def test_a_plottable_histogram_is_read_with_its_labels_and_its_variances():
    """A protocol object's axes give their bins as ``(low, high)`` pairs, and the label on
    each becomes the ROOT axis title - so a plotting name survives the conversion."""
    stand_in = _Plottable([1.0, 2.0, 3.0], [1.0, 4.0, 9.0], _three_bins(), kind=_Kind("COUNT"))
    stand_in.name = "stand-in"
    stand_in.label = "a histogram"
    made = Histogram.of(stand_in)
    assert made.classname == "TH1D"
    assert made.name == "stand-in"
    assert made.title == "a histogram"
    assert made.values().tolist() == [1.0, 2.0, 3.0]
    assert made.variances().tolist() == [1.0, 4.0, 9.0]
    assert made.axes[0].title == "the x"
    assert made.axes[0].name == "xaxis"  # ROOT's own name, since the protocol has none


def test_a_plottable_that_speaks_no_flow_is_read_without_it():
    """The protocol lets ``values()`` take no ``flow`` argument; asking anyway raises
    ``TypeError`` and the conversion then reads the inner bins alone."""
    stand_in = _Plottable([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], _three_bins(), refuses_flow=True)
    made = Histogram.of(stand_in, name="noflow")
    assert made.values(flow=True).tolist() == [0.0, 1.0, 2.0, 3.0, 0.0]
    assert made.name == "noflow"
    assert made.title == ""  # a protocol object that carries no label carries nothing


def test_a_plottable_whose_flow_is_not_the_shape_its_axes_imply_is_read_without_it():
    """A flowed array two bins wider is taken as it is; one that is the wrong width is
    thrown away and the inner bins are read instead, which is the only reading that can
    be trusted."""
    flowed = _Plottable([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], _three_bins())
    assert Histogram.of(flowed).values(flow=True).tolist() == [0.0, 1.0, 2.0, 3.0, 0.0]
    wrong = _Plottable([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], _three_bins(), flow_shape=(7,))
    assert Histogram.of(wrong).values().tolist() == [1.0, 2.0, 3.0]


def test_a_plottable_that_keeps_means_is_refused_as_a_profile():
    """A histogram of ``MEAN`` storage is a ``TProfile``, which this writer does not make;
    the refusal says so and says what to do instead."""
    stand_in = _Plottable([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], _three_bins(), kind="MEAN")
    with pytest.raises(UnsupportedFeatureError, match="keeps means rather than counts"):
        Histogram.of(stand_in)


def test_a_categorical_axis_is_refused_before_any_bin_is_read():
    """An axis of categories, or one that wraps round, has no edges to write, so it is
    refused in a sentence rather than given a made-up range."""
    axis = _Axis([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], "colour")
    axis.traits = _Traits(discrete=True)
    with pytest.raises(UnsupportedFeatureError, match="no ROOT spelling"):
        Histogram.of(_Plottable([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [axis]))
    axis.traits = _Traits(circular=True)
    with pytest.raises(UnsupportedFeatureError, match="one that wraps round"):
        Histogram.of(_Plottable([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [axis]))


def test_something_that_is_no_histogram_is_refused_by_what_it_is():
    """Neither a plotting-protocol object nor numpy's parts: the refusal names the type it
    was handed, so a caller knows what it passed."""
    with pytest.raises(TypeError, match="a str is not a histogram"):
        Histogram.of("not a histogram")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="a int is not a histogram"):
        Histogram.of(3)  # type: ignore[arg-type]


def test_a_tuple_numpy_did_not_make_is_not_one_of_numpy_s_histograms():
    """Only a ``(values, edges)``-shaped tuple whose values are an array of the axes'
    dimension counts; every other tuple falls through to the protocol check and is
    refused, rather than read as a histogram of nothing."""
    refused = (
        (np.zeros(3),),  # one part, no edges
        ("not an array", np.arange(3)),  # the values are not numpy's
        (np.zeros((2, 2)), np.arange(3)),  # two dimensions of values, one set of edges
        (np.zeros((2, 2)), np.zeros((3, 2)), np.arange(3)),  # an edge with two dimensions
    )
    for parts in refused:
        with pytest.raises(TypeError, match="is not a histogram"):
            Histogram.of(parts)  # type: ignore[arg-type]


def test_recognises_answers_without_building_anything():
    """``recognises`` is the question ``of`` answers by making a histogram; it takes what
    ``of`` would take and nothing else."""
    values, edges = np.histogram([0.1], bins=3, range=(0.0, 3.0))
    assert Histogram.recognises(Histogram.book("h", (3, 0.0, 3.0)))
    assert Histogram.recognises((values, edges))
    assert Histogram.recognises(_Plottable([1.0], [1.0], [_Axis([(0.0, 1.0)])]))
    assert not Histogram.recognises((values,))
    assert not Histogram.recognises("not a histogram")
    assert not Histogram.recognises(3)
