"""The smoothers' corners - windows above the points, points of no weight, ties - as ROOT's.

Each expected list is what ROOT 6.40's ``TGraphSmooth`` gave for the same
call; only the normal kernel's ``exp`` is the C library's and so held to
1e-13 of ROOT's off its machine (see ``test_pyroot_graphsmooth``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import xrdroot.pyroot as ROOT
from refmachine import roots
from xrdroot import graphsmooth, supsmu

X = [4, 7, 7, 8, 9, 10, 10, 10, 11, 12, 13, 14]
Y = [2, 4, 22, 16, 10, 18, 26, 34, 17, 28, 34, 26]
#: x with a run of one value through the middle half, so the quartiles' spread is nothing.
TIED = [1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5]


def graph(x: Sequence[float], y: Sequence[float]) -> object:
    return ROOT.TGraph(len(x), np.array(x, dtype=float), np.array(y, dtype=float))


def ys(made: object) -> list[float]:
    return [float(v) for v in made.GetY()]  # type: ignore[attr-defined]


def test_a_kernel_window_above_the_first_points_starts_past_them_and_one_far_off_is_zero() -> None:
    x, y = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.5], [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.0]
    smoother = ROOT.TGraphSmooth("s")
    assert ys(smoother.SmoothKern(graph(x, y), "box", 0.3, 2, np.array([0.9, 1.4]))) == [4.0, 3.5]
    far = smoother.SmoothKern(graph(x, y), "normal", 0.3, 2, np.array([0.1, 100.0]))
    assert ys(far) == [roots(1.4121820613647387, rel=1e-13), 0.0]


def test_the_super_smoother_lets_points_of_no_weight_in_and_out_of_its_windows() -> None:
    smoother, weights = ROOT.TGraphSmooth("s"), np.array([0.0] * 3 + [1.0] * 6 + [0.0] * 3)
    assert ys(smoother.SmoothSuper(graph(X, Y), "", 0, 0, False, weights)) == [
        1.8415977961432262, 0.0, 0.0, 13.354910253099172, 17.303129695717505, 20.79233367947946,
        20.79233367947946, 20.79233367947946, 17.000000000000007, 12.550260416666665,
        8.100520833333329, 3.6507812499999925,
    ]  # fmt: skip
    assert ys(smoother.SmoothSuper(graph(X, Y), "", 0, 0.3, False, weights)) == [
        40.0, 0.0, 0.0, 13.666666666666666, 16.545454545454547, 23.875000000000004,
        23.875000000000004, 23.875000000000004, 16.99999999999997, -1.5987211554602254e-13,
        -17.00000000000027, -34.00000000000037,
    ]  # fmt: skip


def test_the_super_smoother_widens_a_spread_of_nothing_and_of_one_x_fits_the_mean() -> None:
    smoother = ROOT.TGraphSmooth("s")
    assert ys(smoother.SmoothSuper(graph(TIED, Y))) == [2.0] + [19.13571699220783] * 8 + [
        24.033690241597576, 27.7283410780911, 31.42299191458462,
    ]  # fmt: skip
    assert ys(smoother.SmoothSuper(graph([1.0] * 5, [0, 1, 2, 3, 4]))) == [2.0] * 5
    assert list(supsmu.supsmu([1.0] * 3, [1.0] * 3, [0.0] * 3, 1, 0.0, 0.0)) == [0.0] * 3


def test_lowess_fits_ties_once_and_two_points_as_they_are() -> None:
    smoother = ROOT.TGraphSmooth("s")
    assert ys(smoother.SmoothLowess(graph(TIED, Y), "", 0.5, 2, 0.5)) == [2.0] + [
        17.889670828908727] * 8 + [28.0, 29.818244881620434, 27.600875188626787]  # fmt: skip
    assert ys(smoother.SmoothLowess(graph([1.0, 2.0], [3.0, 5.0]))) == [3.0, 5.0]
    assert list(graphsmooth.lowess([1.0], [4.0], 0.67, 3, 0.0)) == [4.0]


def test_a_window_the_only_weighted_point_leaves_holds_its_last_means() -> None:
    weights = np.zeros(12)
    weights[0] = 1.0
    found = ROOT.TGraphSmooth("s").SmoothSuper(graph(X, Y), "", 0, 0.05, False, weights)
    assert ys(found) == [2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
