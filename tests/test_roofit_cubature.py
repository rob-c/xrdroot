"""ROOT's Genz-Malik cubature, bit for bit: ``AdaptiveIntegratorMultiDim`` as RooFit drives it.

Every expected number here came from ROOT 6.40.04. Most came from a
``ROOT::Math::AdaptiveIntegratorMultiDim`` over a ``Functor`` of the same
integrand written in C++. The RooFit ones came from
``pdf.createIntegral(...).getVal()`` on a ``RooGenericPdf``, where RooFit
logged that it chose ``RooAdaptiveIntegratorND``. The values, the relative
errors, the statuses and the evaluation counts are all compared exactly,
because the port makes the same cuts in the same order and adds up the
same terms in the same order as ROOT.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from xrdroot.roofit import cubature
from xrdroot.roofit.cubature import Cubature, adaptive_integral, integrate_nd


def gauss2(x: Sequence[float]) -> float:
    return math.exp(-x[0] * x[0] - x[1] * x[1])


def gauss2_many(points: Any) -> Any:
    return np.exp(-points[:, 0] * points[:, 0] - points[:, 1] * points[:, 1])


def peak2(x: Sequence[float]) -> float:
    dx, dy = x[0] - 0.3, x[1] + 0.2
    return 1.0 / (1e-4 + dx * dx + dy * dy)


def peak2_many(points: Any) -> Any:
    dx, dy = points[:, 0] - 0.3, points[:, 1] + 0.2
    return 1.0 / (1e-4 + dx * dx + dy * dy)


def skewed3(x: Sequence[float]) -> float:
    return math.exp(-(x[0] * x[0] + 2 * x[1] * x[1] + 3 * x[2] * x[2])) * (1 + x[0] * x[1] * x[2])


def polynomial2(x: Sequence[float]) -> float:
    return 1 + x[0] + x[1] * x[1] * x[0]


def upper_corner(x: Sequence[float]) -> float:
    return 1.0 if (x[0] > 0.9 and x[1] > 0.9) else 0.0


def lower_corner(x: Sequence[float]) -> float:
    return 1.0 if (x[0] < 0.1 and x[1] < 0.1) else 0.0


def wave4(x: Sequence[float]) -> float:
    return math.cos(x[0] + x[1] * x[2] - x[3])


def roofit_peak(x: Sequence[float]) -> float:
    return 1.0 / (1e-12 + (x[0] - 0.3) * (x[0] - 0.3) + (x[1] - 0.6) * (x[1] - 0.6))


def test_a_smooth_two_dimensional_integral_matches_root() -> None:
    result = adaptive_integral(gauss2, [-1, -1], [1, 1])
    assert result == (2.2309851414231288, 9.004896973777259e-08, 0, 1785)


def test_a_peaked_two_dimensional_integral_matches_root() -> None:
    result = adaptive_integral(peak2, [-1, -1], [1, 1])
    assert result == (29.279685040587374, 9.98950999385934e-08, 0, 39559)


def test_a_three_dimensional_integral_matches_root() -> None:
    result = adaptive_integral(skewed3, [-1, -2, -1.5], [2, 1, 1.5])
    assert result == (2.041070799900186, 9.995693990188707e-08, 0, 318747)


def test_a_four_dimensional_integral_matches_root() -> None:
    result = adaptive_integral(wave4, [0, 0, 0, 0], [1, 2, 1, 2])
    assert result == (2.929072991235859, 9.986214265040135e-08, 0, 83961)


def test_a_polynomial_is_exact_after_one_rule() -> None:
    result = adaptive_integral(polynomial2, [0, 0], [1, 2])
    assert result == (4.333333333333332, 4.099285014000579e-16, 0, 17)


def test_a_minimum_number_of_points_keeps_dividing_past_the_tolerance() -> None:
    result = adaptive_integral(polynomial2, [0, 0], [1, 2], eps_abs=1e-3, min_pts=5000)
    assert result == (4.333333333333327, 4.628709274841968e-19, 0, 5015)


def test_running_out_of_evaluations_reports_status_one() -> None:
    assert adaptive_integral(upper_corner, [0, 0], [1, 1], max_pts=200) == (
        0.009327131915866484,
        0.9532746074301035,
        1,
        187,
    )
    assert adaptive_integral(peak2, [-1, -1], [1, 1], max_pts=2000) == (
        29.288144855397924,
        0.0007469892456400908,
        1,
        1989,
    )


def test_a_last_box_of_zeros_zeroes_the_result_as_root_does() -> None:
    result = adaptive_integral(lower_corner, [0, 0], [1, 1], max_pts=187)
    assert result == (0.0, 0.9532746074301035, 0, 187)
    assert adaptive_integral(lambda x: 0.0, [0, 0], [1, 1], max_pts=1) == (0.0, 0.0, 0, 17)


def test_reversed_limits_are_integrated_as_root_integrates_them() -> None:
    result = adaptive_integral(gauss2, [1, 1], [-1, -1])
    assert result == (1.64796972072264, 0.35377799321845477, 1, 99977)


def test_a_maximum_below_the_minimum_becomes_ten_times_the_minimum() -> None:
    result = adaptive_integral(gauss2, [-1, -1], [1, 1], max_pts=10, min_pts=40)
    assert result == (2.230985180356792, 1.6246258803536053e-06, 1, 391)


def test_a_nan_integrand_cuts_the_first_axis_until_the_evaluations_run_out() -> None:
    value, relerr, status, neval = adaptive_integral(
        lambda x: math.nan, [0, 0], [1, 1], max_pts=200
    )
    assert math.isnan(value) and math.isnan(relerr)
    assert (status, neval) == (1, 187)


def test_a_vectorised_integrand_gives_the_same_bits() -> None:
    assert adaptive_integral(gauss2_many, [-1, -1], [1, 1], vectorized=True) == (
        2.2309851414231288,
        9.004896973777259e-08,
        0,
        1785,
    )
    assert adaptive_integral(peak2_many, [-1, -1], [1, 1], vectorized=True) == (
        29.279685040587374,
        9.98950999385934e-08,
        0,
        39559,
    )


def test_the_rule_evaluates_its_points_in_the_order_root_does() -> None:
    points = cubature.rule_points([0.0, 0.0], [1.0, 1.0])
    assert points.shape == (17, 2)
    assert points[0].tolist() == [0.0, 0.0]
    assert points[1].tolist() == [-cubature.XL2, 0.0]
    assert points[9].tolist() == [-cubature.XL4, -cubature.XL4]
    corners = points[13:].tolist()
    assert corners == [
        [-cubature.XL5, -cubature.XL5],
        [cubature.XL5, -cubature.XL5],
        [-cubature.XL5, cubature.XL5],
        [cubature.XL5, cubature.XL5],
    ]


def test_roofit_normalises_a_two_dimensional_pdf_as_root_does() -> None:
    def pdf(x: Sequence[float]) -> float:
        return math.exp(-x[0] * x[0] - 2 * x[1] * x[1]) * (1.5 + x[0] * x[1])

    assert integrate_nd(pdf, [-1, -2], [2, 1]).value == 2.9866319853722936


def test_roofit_normalises_a_three_dimensional_pdf_as_root_does() -> None:
    result = integrate_nd(skewed3, [-1, -2, 0], [2, 1, 1])
    assert result.value == 1.0052415826342942
    assert (result.status, result.neval) == (0, 110715)


def test_roofit_counts_a_peak_it_cannot_resolve_and_says_so_when_done() -> None:
    integrator = Cubature(roofit_peak, [0, 0], [1, 1], name="pk")
    assert integrator.integral() == 82.58507280026306
    assert integrator.last is not None and integrator.last.status == 1
    assert integrator.messages == []
    assert integrator.close() == (
        "RooAdaptiveIntegratorND::dtor(pk) WARNING: Number of suppressed warningings about"
        " integral evaluations where target precision was not reached is 1"
    )
    assert len(integrator.messages) == 1


def test_an_allowance_of_warnings_prints_them_then_says_the_rest_are_suppressed() -> None:
    integrator = Cubature(upper_corner, [0, 0], [1, 1], name="c", max_warn=1, vectorized=False)
    integrator.max_eval = 200
    integrator.integral()
    integrator.integral()
    assert integrator.messages == [
        "RooAdaptiveIntegratorND::integral(c) WARNING: target rel. precision not reached due to"
        " nEval limit of 200, estimated rel. precision is 9.5e-01",
        "RooAdaptiveIntegratorND::integral(c) Further warnings on target precision are"
        " suppressed conform specification in integrator specification",
    ]
    assert integrator.close() is not None and integrator.n_error == 2


def test_an_integral_that_converges_leaves_nothing_to_report() -> None:
    integrator = Cubature(gauss2_many, [-1, -1], [1, 1], vectorized=True)
    assert integrator.integral() == 2.2309851414231288
    assert integrator.close() is None and integrator.messages == []


def test_roofit_picks_its_evaluation_limit_by_dimension() -> None:
    assert [cubature.max_evaluations(n) for n in (2, 3, 4, 9)] == [
        100000,
        1000000,
        10000000,
        10000000,
    ]
    with pytest.raises(ValueError, match="at least 2"):
        cubature.max_evaluations(1)


def test_unset_options_take_roots_defaults() -> None:
    limits, eps_abs, eps_rel = cubature._start(2, (-1.0, -1.0, 0, 0, 0))
    assert (eps_abs, eps_rel) == (0.0, 1e-9)
    assert (limits.minpts, limits.maxpts, limits.iwk) == (17, 100000, 100000)


def test_limits_must_pair_up_in_two_to_fifteen_dimensions() -> None:
    with pytest.raises(ValueError, match="2 lower limits but 3 upper"):
        adaptive_integral(gauss2, [0, 0], [1, 1, 1])
    with pytest.raises(ValueError, match="not 1"):
        adaptive_integral(gauss2, [0], [1])
    with pytest.raises(ValueError, match="not 16"):
        adaptive_integral(gauss2, [0] * 16, [1] * 16)


def test_a_full_work_space_reports_status_two() -> None:
    limits = cubature._Limits(7, 17, 17, 100000, 20)
    heap = cubature._Heap(7, [0.0] * 30, 14, 14)
    run = cubature._Run(limits, 0.0, 1e-7, lambda points: [], heap, result=1.0, abserr=1.0)
    assert run.status(run.relerr()) == 2


def test_the_warning_texts_are_roots() -> None:
    assert cubature.not_converged_warning("f", 100000, 1.23456e-5).endswith(
        "nEval limit of 100000, estimated rel. precision is 1.2e-05"
    )
    assert cubature.TOPIC == "NumericIntegration"
