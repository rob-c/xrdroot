"""The corners of the ported numerical routines that the everyday integrals, roots and minima
miss: an integrand that is zero everywhere, a Brent minimisation that keeps its oldest point, a
root bracket squeezed to neighbouring doubles, an interpolation Brent's root finder turns down,
Wynn's table filled to its last row - and ``TMath::ChisquareQuantile``, AS 91 as ROOT has it,
checked against ROOT's own digits."""

from __future__ import annotations

import pytest

from xrdroot.numerics.kronrod import qk21
from xrdroot.numerics.minimize1d import minim_brent
from xrdroot.numerics.qelg import DBL_MAX, Table
from xrdroot.numerics.roots import brent_root
from xrdroot.stats import chisquare_quantile


def test_the_kronrod_rule_of_a_zero_function_has_no_error_floor() -> None:
    """With nothing to integrate the error is not raised to the rounding floor: it stays zero."""
    assert qk21(lambda _x: 0.0, 0.0, 1.0) == (0.0, 0.0, 0.0, 0.0)


def test_brents_minimizer_on_a_cusp_finds_its_point() -> None:
    """A cusp turns down steps that are no better than any point kept, so it keeps the old."""
    x, converged, low, high = minim_brent(lambda x: abs(x - 0.31), 0.0, 1.0, 0.77, 1e-12, 1e-12,
                                          200)  # fmt: skip
    assert converged and x == pytest.approx(0.31, abs=1e-11)
    assert (low, high) == (0.0, 1.0)


def test_a_step_function_root_squeezes_the_bracket_to_neighbouring_doubles() -> None:
    """With no tolerance the bracket closes onto the jump, and then stays there to the end."""
    found, root, iterations = brent_root(lambda x: -1.0 if x < 0.3 else 1.0, 0.0, 1.0, 200,
                                         0.0, 0.0)  # fmt: skip
    assert (found, root, iterations) == (False, 0.3, 200)


def test_a_cubic_root_is_found_with_bisection_where_interpolation_is_unsafe() -> None:
    """A triple root makes the interpolated step too long, so Brent bisects - and converges."""
    found, root, iterations = brent_root(lambda x: (x - 0.3) ** 3, 0.0, 1.0)
    assert found and iterations == 70
    assert root == pytest.approx(0.3, abs=1e-8)


def test_wynns_table_filled_to_its_last_row_is_cut_back_by_one() -> None:
    """Fifty estimates that never settle: the table keeps an even 48 rows, plus the newest.

    The estimates are thirds, made by one correctly rounded division each, and Wynn's
    table is sums and quotients of them, so its answer is the same to the bit on every
    machine - which estimates made by ``sin``, the C library's, would not be.
    """
    table = Table()
    for i in range(50):
        table.append((-1) ** i * (1 + i % 7) / 3)
    assert table.qelg() == (0.6089525190686361, DBL_MAX)  # no three results yet: no error
    assert table.n == 49 and table.nres == 1


@pytest.mark.parametrize(
    ("p", "ndf", "expected"),
    [
        (0.5, 0, 0.0),  # no degrees of freedom
        (0.95, 3, 7.814727903253585),  # Wilson and Hilferty's start
        (0.9999, 1, 15.136705226623144),  # its far-tail correction
        (0.9, 0.2, 0.5323091074776573),  # the iterated start for few degrees of freedom
        (1e-10, 1, 1.5707963268578145e-20),  # the small-p start, too small to refine
        (0.3, 1, 0.14847186183268968),  # the small-p start, refined
        (0.6827, 1, 1.000043427117358),
        (0.5, 50, 49.33493673397557),
        (0.05, 2, 0.10258658877520474),
        (0.99, 10, 23.209251158948373),
        (0.3, 1e3, 976.0735912577578),  # a start good enough that the steps go on
        (0.9, 1e6, 1001812.8153153562),
    ],
)
def test_the_chisquare_quantile_has_roots_digits(p: float, ndf: float,
                                                   expected: float) -> None:  # fmt: skip
    """``TMath::ChisquareQuantile``, the numbers ROOT 6.40 prints for these."""
    assert chisquare_quantile(p, ndf) == pytest.approx(expected, rel=1e-14, abs=1e-300)
