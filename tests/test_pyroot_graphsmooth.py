"""``TGraphSmooth``'s interpolation and smoothers give ROOT 6.40's numbers.

Each expected list is what ROOT 6.40 printed for the same call. The kernel
smoother, lowess and the super smoother are additions, multiplications and a
square root in the order ROOT's code does them, so they are ROOT's to the
bit on any machine - but for the normal kernel's ``exp`` and the bass tone's
``pow``, which another machine's C library may round otherwise in the last
place: there the numbers are held to 1e-13 of themselves, a hundred ulps of
margin and still a billion times finer than any mistake in the formula.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from refmachine import roots

X = [4, 7, 7, 8, 9, 10, 10, 10, 11, 12, 13, 14]
Y = [2, 4, 22, 16, 10, 18, 26, 34, 17, 28, 34, 26]


def graph(x: Sequence[float] = X, y: Sequence[float] = Y) -> Any:
    return ROOT.TGraph(len(x), np.array(x, dtype=float), np.array(y, dtype=float))


def ys(made: object) -> list[float]:
    return [float(v) for v in made.GetY()]  # type: ignore[attr-defined]


def test_approx_merges_the_y_of_points_of_one_x_as_ties_says() -> None:
    x, y = [3.0, 1.0, 2.0, 2.0, 2.0, 4.0], [5.0, 1.0, 7.0, 2.0, 9.0, 3.0]
    smoother = ROOT.TGraphSmooth("s")
    found = {ties: ys(smoother.Approx(graph(x, y), "linear", 7, 0, 0, 0, 0, 0, ties))
             for ties in ("ordered", "MEAN", "min", "max")}  # fmt: skip
    assert found["ordered"] == found["MEAN"] == [1.0, 3.5, 6.0, 5.5, 5.0, 4.0, 3.0]
    assert found["min"] == [1.0, 1.5, 2.0, 3.5, 5.0, 4.0, 3.0]
    assert found["max"] == [1.0, 5.0, 9.0, 7.0, 5.0, 4.0, 3.0]
    assert smoother.GetName() == "s" and smoother.GetTitle() == ""


def test_approx_steps_by_f_and_holds_the_ends_as_rule_says() -> None:
    x, y = [3.0, 1.0, 2.0, 2.0, 2.0, 4.0], [5.0, 1.0, 7.0, 2.0, 9.0, 3.0]
    smoother, xout = ROOT.TGraphSmooth("s"), np.array([0.0, 1.5, 2.0, 5.0])
    stepped = smoother.Approx(graph(x, y), "constant", 7, 0, -1, 9, 0, 0.25)
    assert ys(stepped) == [1.0, 2.25, 6.0, 5.75, 5.0, 4.5, 3.0]
    assert ys(smoother.Approx(graph(x, y), "linear", 4, xout, -1, 9, 2)) == [1.0, 3.5, 6.0, 3.0]
    assert ys(smoother.Approx(graph(x, y), "linear", 4, xout, -1, 9, 0)) == [-1.0, 3.5, 6.0, 9.0]
    assert ys(smoother.Approx(graph(x, y), "", 2, np.array([1.5, 2.5]), 0, 0, 0, 1.0)) == [6.0, 5.0]


def test_approx_and_the_super_smoother_say_what_is_out_of_range_and_return_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    smoother = ROOT.TGraphSmooth("s")
    assert smoother.Approx(graph(), "linear", 5, 0, 0, 0, 0, 2.0) is None
    assert smoother.Approx(graph(), "linear", 5, 0, 0, 0, 0, 0, "foo") is None
    assert smoother.SmoothSuper(graph(), "", 0, 2.0) is None
    assert smoother.SmoothSuper(graph(), "", 0, 0, True) is None
    assert capsys.readouterr().out.splitlines() == [
        "Error: Invalid f value",
        "Error: Method not known: foo",
        "Error: Span must be between 0 and 1",
        "Error: x must be between 0 and 1 for periodic smooth",
    ]


def test_the_kernel_smoother_is_roots_with_a_box_and_with_a_normal() -> None:
    smoother = ROOT.TGraphSmooth("s")
    box = smoother.SmoothKern(graph(), "box", 3.0, 6)
    assert ys(box) == [2.0, 2.0, 9.333333333333334, 11.0, 10.8, 16.5, 21.0, 20.428571428571427,
                       21.285714285714285, 23.857142857142858, 27.8, 26.25]  # fmt: skip
    assert box.GetX()[6] == 9.454545454545455 and box.GetX()[10] == 13.090909090909092
    normal = ys(smoother.SmoothKern(graph(), "normal", 2.0, 6))
    assert normal == roots([2.0, 2.809522176773975, 12.192106563242278, 13.30695240962718,
                            13.94991410620909, 15.436374293266692, 21.32344589046164,
                            23.50885188402698, 23.06794924068702, 27.998378466739872,
                            30.637574670673775, 28.2885698637035], rel=1e-13)  # fmt: skip


def test_the_kernel_smoother_at_given_points_takes_them_in_order() -> None:
    x, y = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.5], [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.0]
    made = ROOT.TGraphSmooth("s").SmoothKern(graph(x, y), "box", 0.3, 3, np.array([0.5, 0.1, 0.9]))
    assert list(made.GetX()) == [0.1, 0.5, 0.9] and ys(made) == [1.5, 1.875, 2.75]


def test_lowess_is_roots_and_goes_on_reweighting_when_the_fit_is_exact() -> None:
    smoother = ROOT.TGraphSmooth("s")
    assert ys(smoother.SmoothLowess(graph(), "", 0.67)) == [
        2.3931154566450132, 13.915777183267894, 13.915777183267894, 15.119386726993826,
        17.215703551789943, 20.924533981159747, 20.924533981159747, 20.924533981159747,
        24.65841238205824, 26.901595174821537, 28.19563811118551, 29.854761139570986,
    ]  # fmt: skip
    # Most points are fitted exactly by the third pass, and ROOT's port reweights on: the last
    # point, then fitted to no point of its own, takes its neighbour's 34.
    found = ys(smoother.SmoothLowess(graph(), "", 0.3, 4, 1.5))
    assert found == [2.0, 4.0, 4.0, 16.0, 10.0, 26.0, 26.0, 26.0, 17.0, 28.0, 34.0, 34.0]


def test_the_super_smoother_is_roots_with_and_without_a_span_a_tone_and_weights() -> None:
    smoother = ROOT.TGraphSmooth("s")
    assert ys(smoother.SmoothSuper(graph())) == [
        2.2422492401215823, 11.9968085106383, 11.9968085106383, 15.023012629253833,
        18.392883217116637, 21.892689108833107, 21.892689108833107, 21.892689108833107,
        24.69252769840242, 26.462046039672554, 27.846863793645806, 29.231681547619058,
    ]  # fmt: skip
    assert ys(smoother.SmoothSuper(graph(), "", 4)) == roots([
        2.2422492401215823, 11.9968085106383, 11.9968085106383, 14.978205643959715,
        18.24950086417546, 21.52916426435822, 21.52916426435822, 21.52916426435822,
        24.215910023571453, 26.061514621769028, 27.62148821645699, 29.181461811144953,
    ], rel=1e-13)  # fmt: skip
    assert ys(smoother.SmoothSuper(graph(), "", 0, 0.4)) == [
        4.371428571428572, 10.8, 10.8, 13.823529411764707, 18.470588235294116,
        23.458333333333332, 23.458333333333332, 23.458333333333332, 27.558823529411764, 27.8,
        27.900000000000002, 28.0,
    ]  # fmt: skip
    weights = np.linspace(1.0, 2.0, 12)
    assert ys(smoother.SmoothSuper(graph(), "", 0, 0, False, weights)) == [
        2.168702617876466, 12.159066023367586, 12.159066023367586, 15.196892149489706,
        18.596793349658633, 22.11557918894905, 22.11557918894905, 22.11557918894905,
        24.834079887774863, 26.549147014447502, 27.862684655440418, 29.176222296433338,
    ]  # fmt: skip


def test_the_super_smoother_of_periodic_x_wraps_the_window_round() -> None:
    periodic = ROOT.TGraphSmooth("s").SmoothSuper(graph([i / 11 for i in range(12)]), "", 0, 0, 1)
    assert ys(periodic) == [
        18.264797593832984, 16.43299262094593, 16.40728652468364, 16.557995056846266,
        17.625142857142855, 19.705714285714283, 22.33314285714285, 24.093857142857136,
        24.056272367770635, 23.247921810139683, 21.684475567065554, 18.597716145869164,
    ]  # fmt: skip


def test_a_smoother_puts_its_input_graphs_points_in_order_of_x() -> None:
    given = graph([3.0, 1.0, 2.0], [30.0, 10.0, 20.0])
    ROOT.TGraphSmooth("s").SmoothLowess(given)
    assert list(given.GetX()) == [1.0, 2.0, 3.0] and ys(given) == [10.0, 20.0, 30.0]
