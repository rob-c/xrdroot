"""``Fill`` of one entry at a time: the same doubles as an array of them, without the arrays.

A loop of single fills - every tutorial's way - takes the scalar path; a
histogram filled that way holds bin for bin and moment for moment what the
same entries filled as one array leave, flows, weights and uneven axes
included, and integer bins, which saturate, still go the careful way.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from xrdroot import Histogram

ENTRIES = [-5.0, -0.25, 0.0, 0.3, 0.999, 1.0, float("nan"), 0.5]
WEIGHTS = [1.0, 1.0, 2.5, 1.0, 0.5, 1.0, 3.0, -1.0]


def _moments(histogram):
    return {name: home[name] for name, home in histogram._moment_homes().items()}


def _pair(*args, **kwargs):
    made = Histogram.book("h", *args, **kwargs)
    return made, copy.deepcopy(made)


@pytest.mark.parametrize(
    "booked",
    [
        (((10, 0.0, 1.0),), {}),
        (([0.0, 0.1, 0.5, 0.75, 1.0],), {}),
        (((10, 0.0, 1.0),), {"kind": "F"}),
    ],
)
def test_one_entry_at_a_time_leaves_what_an_array_of_them_leaves(booked):
    args, kwargs = booked
    one, many = _pair(*args, **kwargs)
    for x, w in zip(ENTRIES, WEIGHTS, strict=True):
        assert one.fill_one([x], w) == int(one.find_bin(x))
    many.fill(np.array(ENTRIES), weight=np.array(WEIGHTS))
    assert one.values(flow=True).tolist() == many.values(flow=True).tolist()
    assert one.variances(flow=True).tolist() == many.variances(flow=True).tolist()
    assert one.entries == many.entries
    assert _moments(one) == _moments(many)


def test_a_two_dimensional_entry_counts_the_cross_moment_as_rows_do():
    one, many = _pair((4, 0.0, 1.0), (4, 0.0, 1.0))
    for x, y in [(0.1, 0.2), (0.6, 0.9), (2.0, 0.5)]:
        one.fill_one([x, y])
        many.fill(x, y)
    assert one._moment_homes()["fTsumwxy"]["fTsumwxy"] == pytest.approx(0.1 * 0.2 + 0.6 * 0.9)
    assert one.values(flow=True).tolist() == many.values(flow=True).tolist()


def test_integer_bins_and_a_wrong_count_of_coordinates_go_the_array_way():
    one, many = _pair((10, 0.0, 1.0), kind="I")
    assert one.fill_one([0.25], 2.0) == 3
    many.fill(0.25, weight=2.0)
    assert one.values(flow=True).tolist() == many.values(flow=True).tolist()
    with pytest.raises(ValueError, match="takes one coordinate per axis"):
        one.fill_one([0.1, 0.2])
