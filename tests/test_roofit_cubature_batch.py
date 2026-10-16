"""Independent cubature must keep each group's point stream and floating-point sums."""

import math

import numpy as np
import pytest

from xrdroot.random import libm
from xrdroot.roofit.cubature import adaptive_integral
from xrdroot.roofit.cubature_batch import adaptive_integrals


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize(
    "options",
    [
        {"max_pts": 3000, "size": 3000},
        {"eps_abs": 1e-3, "min_pts": 500},
        {"eps_rel": 1e-3},
        {"max_pts": 10, "min_pts": 40},
    ],
)
def test_independent_integrals_keep_the_same_bits_and_evaluation_counts(ndim, options):
    def function(groups, points):
        delta = points - groups[:, None] * 0.1
        return libm.exp(-np.sum(delta * delta, axis=-1))

    lows, highs = [-2.0] * ndim, [3.0] * ndim
    actual = adaptive_integrals(function, lows, highs, 4, **options)
    expected = [
        adaptive_integral(
            lambda points, i=i: function(np.full(len(points), i), points),
            lows,
            highs,
            vectorized=True,
            **options,
        )
        for i in range(4)
    ]
    for a, e in zip(actual, expected, strict=True):
        assert (a.status, a.neval) == (e.status, e.neval)
        assert np.array_equal(np.asarray(a[:2]).view(np.uint64), np.asarray(e[:2]).view(np.uint64))


def test_groups_that_finish_early_do_not_receive_more_points():
    visits = [[], [], []]

    def value(group, points):
        if group == 0:
            return np.zeros(len(points))
        if group == 1:
            return 1 + points[:, 0] * points[:, 1]
        return 1 / (0.0001 + (points[:, 0] - 0.3) ** 2 + (points[:, 1] + 0.2) ** 2)

    def function(groups, points):
        found = np.empty(len(points))
        for i in range(3):
            chosen = groups == i
            visits[i].extend(points[chosen].tolist())
            found[chosen] = value(i, points[chosen])
        return found

    actual = adaptive_integrals(function, [-1.0, -1.0], [1.0, 1.0], 3, max_pts=2000)
    for i in range(3):
        expected_visits = []

        def scalar(points, group=i, visits=expected_visits):
            visits.extend(points.tolist())
            return value(group, points)

        expected = adaptive_integral(
            scalar, [-1.0, -1.0], [1.0, 1.0], max_pts=2000, vectorized=True
        )
        assert actual[i] == expected
        assert visits[i] == expected_visits
    assert actual[0].neval == actual[1].neval == 17
    assert actual[2].status == 1


def test_nan_integrals_keep_their_status_and_full_evaluation_count():
    found = adaptive_integrals(
        lambda groups, points: np.full(len(points), math.nan),
        [0.0, 0.0],
        [1.0, 1.0],
        2,
        max_pts=200,
    )
    for result in found:
        assert math.isnan(result.value) and math.isnan(result.relerr)
        assert (result.status, result.neval) == (1, 187)


def test_no_integrals_does_not_call_the_function():
    assert (
        adaptive_integrals(
            lambda groups, points: pytest.fail("unexpected evaluation"), [0.0, 0.0], [1.0, 1.0], 0
        )
        == []
    )


def test_a_negative_group_count_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        adaptive_integrals(None, [0.0, 0.0], [1.0, 1.0], -1)


@pytest.mark.parametrize("count", [1, 2])
def test_a_wrong_number_of_returned_values_is_rejected(count):
    with pytest.raises(ValueError, match="one value per requested point"):
        adaptive_integrals(lambda groups, points: [1.0], [0.0, 0.0], [1.0, 1.0], count)
