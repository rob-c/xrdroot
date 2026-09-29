"""GSL's QAGS and Brent solver, and MathCore's Brent minimizer, as ROOT's statistics use them."""

from __future__ import annotations

import math

import pytest

from xrdroot.numerics.kronrod import rescale_error
from xrdroot.numerics.minimize1d import BrentMinimizer1D, minim_brent, minim_step
from xrdroot.numerics.qags import qags
from xrdroot.numerics.roots import brent_root


def _inverse_root(x: float) -> float:
    return 1 / math.sqrt(x) if x > 0 else 0.0


def test_qags_integrates_a_smooth_function_to_quadpacks_digits() -> None:
    """The Gaussian over [-3, 2]: QUADPACK's dqagse - scipy's quad - gives the same numbers."""
    found = qags(lambda x: math.exp(-x * x), -3.0, 2.0, 1e-9, 1e-9, 1000)
    assert found == (1.768288739021943, 3.738608272789434e-14, 0)


def test_qags_extrapolates_through_an_integrable_singularity() -> None:
    assert qags(_inverse_root, 0.0, 1.0, 1e-10, 1e-10, 1000) == (
        1.9999999999999984, 5.773159728050814e-15, 0)  # fmt: skip
    assert qags(lambda x: math.log(x) if x > 0 else 0.0, 0.0, 1.0, 1e-12, 1e-12, 1000)[0] == (
        -0.9999999999999999)  # fmt: skip


def test_qags_handles_oscillation_a_cusp_a_peak_and_a_reversed_range() -> None:
    assert qags(lambda x: math.sin(50 * x) ** 2, 0.0, math.pi, 1e-10, 1e-10, 1000)[0] == (
        1.5707963267948966)  # fmt: skip
    assert qags(lambda x: abs(x - 0.3) ** 0.2, 0.0, 1.0, 0.0, 1e-13, 50)[0] == 0.7396715552044779
    assert qags(lambda x: 1 / (1e-8 + x * x), -1.0, 1.0, 1e-12, 1e-12, 100)[0] == (
        31413.926535904597)  # fmt: skip
    assert qags(lambda x: x * x, 2.0, 0.0, 1e-9, 1e-9, 1000)[0] == -2.666666666666667


def test_qags_says_when_its_subdivisions_run_out() -> None:
    """GSL's status 11, ``GSL_EMAXITER``, with the best estimate so far."""
    value, error, status = qags(_inverse_root, 0.0, 1.0, 1e-14, 1e-14, 3)
    assert status == 11 and abs(value - 2.0) < error


def test_qags_refuses_tolerances_it_cannot_meet() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        qags(math.exp, 0.0, 1.0, 0.0, 1e-30, 100)


def test_the_kronrod_error_estimate_is_rescaled_as_quadpack_rescales_it() -> None:
    assert rescale_error(0.0, 1.0, 0.0) == 50 * 2.220446049250313e-16  # the floor of it
    assert rescale_error(1e-3, 1.0, 1.0) == pytest.approx(0.2**1.5)
    assert rescale_error(1.0, 1.0, 1.0) == 1.0  # the scale past one: the whole deviation


def test_brent_root_finds_a_root_and_says_when_it_cannot() -> None:
    ok, root, iterations = brent_root(lambda x: x * x - 2.0, 0.0, 2.0, 100, 1e-12, 1e-12)
    assert ok and root == pytest.approx(math.sqrt(2.0), abs=1e-11) and iterations > 1
    assert brent_root(lambda x: x * x + 1.0, 0.0, 2.0) == (False, 1.0, 0)
    assert brent_root(lambda x: x - 1.0, 0.0, 1.0)[:2] == (True, 1.0)
    assert brent_root(lambda x: x**3 - 2 * x - 5, 2.0, 3.0, 2, 1e-14, 1e-14)[0] is False
    assert brent_root(lambda x: math.inf if x > 1 else x - 0.5, 0.0, 1.0)[0] is True


def test_brent_root_stops_where_the_function_is_not_finite() -> None:
    def odd(x: float) -> float:
        return math.nan if 0.2 < x < 0.9 else x - 0.5

    assert brent_root(odd, 0.0, 1.0)[0] is False


def test_the_brent_minimizer_brackets_on_a_grid_then_converges() -> None:
    minimizer = BrentMinimizer1D()
    minimizer.SetFunction(lambda x: (x - 1.3) ** 2 + 0.5, 3.0, -1.0)
    minimizer.SetNpx(50)
    assert minimizer.Minimize(100, 1e-8, 1e-10)
    assert minimizer.XMinimum() == pytest.approx(1.3, abs=1e-6)
    assert minimizer.FValMinimum() == pytest.approx(0.5)
    assert minimizer.Status() == 0


def test_the_brent_minimizer_says_when_it_does_not_converge() -> None:
    minimizer = BrentMinimizer1D()
    minimizer.SetNpx(3)
    minimizer.SetFunction(lambda x: abs(x - 0.1234567), 0.0, 10.0)
    assert not minimizer.Minimize(1, 1e-15, 1e-15)
    assert minimizer.Status() == -2


def test_the_grid_step_of_one_point_is_the_whole_range() -> None:
    assert minim_step(lambda x: x, 0.0, 4.0, 1) == (2.0, 0.0, 4.0)
    x, converged, _, _ = minim_brent(lambda x: abs(x - 0.25), 0.0, 1.0, 0.5, 1e-10, 1e-10, 200)
    assert converged and x == pytest.approx(0.25, abs=1e-8)


def test_qags_says_when_roundoff_stops_it_at_the_start() -> None:
    """Exact at once, but not within a tolerance tighter than the sign changes allow: 18."""
    assert qags(lambda x: x, -1.0, 2.0, 0.0, 50 * 2.220446049250313e-16, 10)[2] == 18


def test_qags_with_one_subdivision_gives_up_at_once() -> None:
    assert qags(_inverse_root, 0.0, 1.0, 1e-14, 1e-14, 1)[2] == 11


@pytest.mark.parametrize(
    ("f", "limit", "status"),
    [
        (lambda x: 1 / x if x else 0.0, 1000, 11),
        (lambda x: math.sin(1 / x) if x else 0.0, 1000, 11),
        (lambda x: 1 / (x - 0.5) if x != 0.5 else 0.0, 1000, 0),
        (lambda x: x ** -0.9 if x else 0.0, 12, 0),
        (lambda x: math.log(x) ** 2 / math.sqrt(x) if x else 0.0, 1000, 0),
    ],
)
def test_qags_on_integrands_that_defeat_it_says_how(f: object, limit: int, status: int) -> None:
    """Divergent, oscillating, cancelling and steep integrands: GSL's status for each."""
    assert qags(f, 0.0, 1.0, 1e-13, 1e-13, limit)[2] == status  # type: ignore[arg-type]
