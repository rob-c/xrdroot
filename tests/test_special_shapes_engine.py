"""The three shapes in ``xrdroot.function.special`` that no other test reaches, against ROOT.

``TMath::Gaus``, ``TMath::BreitWigner`` and ``ROOT::Math::ChebyshevN`` are the three that are
written out by hand rather than handed to a library, and each has a detail only ROOT's own
answer shows: a Gaussian is **zero** past 39 widths and ``1e30`` for no width at all, a
Breit-Wigner is normalised so that its integral is one whatever its width, and a Chebyshev series
is summed by Clenshaw's recurrence, which is not the same arithmetic as evaluating each
polynomial.

Every number here was read off the ROOT 6.40.04 oracle with ``%.17g`` for the same arguments.
Where the last bit differs between this machine's ``exp`` and the oracle's - the register says
which - the comparison is by a relative tolerance with the oracle's value quoted beside it.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.function.special import breit_wigner, chebyshev, gaus

#: How close a number ROOT printed is: the two machines' ``exp`` differ in the last place, so
#: bit-exactness is only claimed where the value is a quotient of exacts (``1``, ``0``, ``1e30``).
REL = 1e-15


def test_a_gaussian_is_the_oracle_s_own_value():
    """``TMath::Gaus(x, mean, sigma)`` is ``exp(-arg^2/2)`` for ``arg = (x - mean)/sigma``: 1 at
    the mean, 0.60653065971263342 one width out and 0.1353352832366127 two widths out at a
    half-width of 0.5 - the oracle's three numbers."""
    assert gaus(0.0, 0.0, 1.0) == 1.0
    assert gaus(1.0, 0.0, 1.0) == pytest.approx(0.60653065971263342, rel=REL)
    assert gaus(2.0, 1.0, 0.5) == pytest.approx(0.1353352832366127, rel=REL)


def test_a_gaussian_with_its_norm_flag_divided_by_the_root_of_two_pi_and_the_width():
    """A Gaussian ROOT normalises is divided by ``sqrt(2 pi) sigma``, so at the mean it is
    0.3989422804014327 rather than 1 - what a density needs to integrate to one."""
    assert gaus(0.0, 0.0, 1.0, 1.0) == pytest.approx(0.3989422804014327, rel=REL)
    assert gaus(0.0, 0.0, 2.0, 1.0) == pytest.approx(0.3989422804014327 / 2.0, rel=REL)


def test_a_gaussian_is_zero_past_thirty_nine_widths_and_enormous_for_no_width():
    """ROOT cuts a Gaussian at 39 widths - where ``exp`` has long underflowed - and answers
    ``1e30`` for a width of zero rather than dividing by it. Both are ROOT's own answers, and
    neither is what the arithmetic alone would give."""
    assert gaus(39.0, 0.0, 1.0) == 0.0
    assert gaus(39.5, 0.0, 1.0) == 0.0  # past the cut, and still zero rather than a tiny number
    assert gaus(0.0, 0.0, 0.0) == 1e30
    assert np.isfinite(gaus(38.0, 0.0, 1.0))  # inside the cut it is still a number


def test_a_breit_wigner_is_normalised_so_that_it_integrates_to_one():
    """``TMath::BreitWigner(x, mean, gamma)`` is ``gamma/((x-mean)^2 + gamma^2/4)/(2 pi)``:
    2/pi = 0.63661977236758138 at the mean, one gamma out 0.12732395447351627, and a wider
    resonance lower at its own mean - the oracle's three numbers."""
    assert breit_wigner(0.0, 0.0, 1.0) == pytest.approx(0.63661977236758138, rel=REL)
    assert breit_wigner(1.0, 0.0, 1.0) == pytest.approx(0.12732395447351627, rel=REL)
    assert breit_wigner(0.0, 2.0, 3.0) == pytest.approx(0.076394372684109757, rel=REL)


def test_a_chebyshev_series_sums_the_same_way_rootesum_s_it():
    """``ROOT::Math::ChebyshevN`` is ``sum c_i T_i(x)`` over the coefficients: at x = 0.5 the
    oracle answers 1, 2, 0.5, -3.5 and -3 for one, two, three, four and six coefficients - which
    is Clenshaw's recurrence's arithmetic, not a polynomial expanded term by term."""
    assert chebyshev(0.5, 1.0) == pytest.approx(1.0, rel=REL)
    assert chebyshev(0.5, 1.0, 2.0) == pytest.approx(2.0, rel=REL)
    assert chebyshev(0.5, 1.0, 2.0, 3.0) == pytest.approx(0.5, rel=REL)
    assert chebyshev(0.5, 1.0, 2.0, 3.0, 4.0) == pytest.approx(-3.5, rel=REL)
    assert chebyshev(0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0) == pytest.approx(-3.0, rel=REL)


def test_a_series_sums_every_point_it_is_given_at_once():
    """A formula's values are an array, so the series is summed over one: at x = 0 and x = 0.5
    three coefficients give -2 and 0.5, the same numbers the single-value calls give."""
    found = chebyshev(np.array([0.0, 0.5]), 1.0, 2.0, 3.0)
    assert found.tolist() == pytest.approx([-2.0, 0.5], rel=REL)
    assert chebyshev(np.array([0.0, 0.5]), 1.0).tolist() == [1.0, 1.0]  # one term is a constant
    assert chebyshev(np.array([0.0, 0.5]), 1.0, 2.0).tolist() == [1.0, 2.0]  # two terms are a line
