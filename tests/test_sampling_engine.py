"""``TF1::GetRandom``: the CDF table ROOT builds, the parabola it solves, and the draws it makes.

ROOT does not invert a function: it cuts the range into ``fNpx`` steps, integrates each,
accumulates, takes the integral over each step to be a parabola fixed by the whole step and its
first half, and for every ``Rndm()`` finds the step and solves that parabola. So a draw is fully
determined by the table and the numbers the generator hands over - which is what makes it testable
without a file, a seed or a real ``TRandom``: a stub generator answering fixed numbers gives fixed
draws.

Every expectation below is arithmetic done here - the CDF of ``f(x) = x`` on 0..1 is ``x**2``, so
half the mass is below ``sqrt(0.5)`` - rather than a number copied from a run. The batch path
below uses a real ``TRandom3`` with a fixed seed and asserts a property.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot import UnsupportedFeatureError
from xrdroot.function.sampling import Table, draw, invert, table


def _integrate(low: float, high: float, _eps: float) -> float:
    """``f(x) = x`` integrated exactly, which is what ``TF1::Integral`` answers for it."""
    return 0.5 * (high * high - low * low)


def _reciprocal(low: float, high: float, _eps: float) -> float:
    """``f(x) = 1/x**2`` integrated exactly, for a table cut in ``log10(x)``."""
    return -1.0 / high + 1.0 / low


class Generator:
    """A ``TRandom`` as the one-draw-per-value path uses it: its numbers, cycling."""

    def __init__(self, numbers: list[float]) -> None:
        self.numbers = list(numbers)
        self.asked = 0

    def rndm(self, count: int) -> np.ndarray:
        """``count`` numbers, cycling the list so a rejection loop cannot run it dry."""
        taken = [self.numbers[(self.asked + at) % len(self.numbers)] for at in range(count)]
        self.asked += count
        return np.array(taken, dtype=np.float64)


def _linear(npx: int = 100) -> Table:
    """The table for ``f(x) = x`` on 0..1: its CDF is ``x**2``, so the median is ``sqrt(0.5)``."""
    return table(_integrate, 0.0, 1.0, npx)


def test_the_table_is_the_cumulative_integral_divided_by_the_total():
    """``ComputeCdfTable`` adds each step's integral in order and divides by the whole, so the
    entries run from 0 to 1 - a quarter of the mass below 0.5 for a linear function."""
    found = table(_integrate, 0.0, 1.0, 2)
    assert found.integral.tolist() == pytest.approx([0.0, 0.25, 1.0])
    assert (found.low, found.high, found.logarithmic) == (0.0, 1.0, False)
    assert found.alpha.tolist() == pytest.approx([0.0, 0.5])  # the steps' lower ends


def test_inverting_the_half_of_the_mass_finds_the_median_root_would():
    """The parabola solved for half the mass is ``sqrt(0.5)`` for ``f(x) = x`` - within 0.1 % of the
    range, which is what a hundred-step table costs."""
    assert invert(_linear(), np.array([0.5]))[0] == pytest.approx(np.sqrt(0.5), abs=1e-3)
    assert invert(_linear(), np.array([0.25]))[0] == pytest.approx(0.5, abs=1e-3)


def test_a_draw_is_the_inverse_of_the_numbers_the_generator_gives():
    """``GetRandom`` with no range is one ``Rndm()`` per number, inverted - so three generator
    values become the three quantiles they stand for, in order."""
    numbers = np.array([0.25, 0.5, 0.75])
    drawn = draw(_linear(), 3, Generator(numbers.tolist()), None)
    assert drawn.tolist() == pytest.approx(np.sqrt(numbers), abs=1e-3)


def test_a_single_draw_is_one_number_and_not_a_list():
    """``n=None`` is ROOT's ``GetRandom()``: one number, a float, not a one-element array."""
    single = draw(_linear(), None, Generator([0.5]), None)
    assert isinstance(single, float)
    assert single == pytest.approx(np.sqrt(0.5), abs=1e-3)


def test_a_function_over_a_wide_range_is_drawn_on_the_logarithmic_table():
    """A range starting above zero and spanning more than ``npx`` decades is cut in ``log10(x)``,
    and the draw undoes the logarithm - so a draw from ``1/x**2`` on 1..100 lands inside it."""
    wide = table(_reciprocal, 1.0, 100.0, 10)
    assert wide.logarithmic is True
    drawn = draw(wide, 5, Generator([0.1, 0.3, 0.5, 0.7, 0.9]), None)
    assert np.all(drawn >= 1.0) and np.all(drawn <= 100.0)


def test_a_range_of_a_logarithmic_table_is_refused_rather_than_answered_with_a_log():
    """ROOT's ``GetRandom(xmin, xmax)`` does not undo the logarithm its table was built in, so for
    such a function it would hand back ``log10(x)``; the refusal says so in a sentence."""
    wide = table(_reciprocal, 1.0, 100.0, 10)
    with pytest.raises(UnsupportedFeatureError, match="does not undo the logarithm"):
        draw(wide, 2, Generator([0.5, 0.6]), (2.0, 3.0))


def test_a_range_keeps_every_draw_inside_it():
    """``GetRandom(xmin, xmax)`` draws ``Uniform(pmin, pmax)`` between the table's entries either
    side of the range and draws again until the answer is inside, so nothing lands outside.

    This path asks the generator for its draws in batches (``TRandom._sequential``), so it takes a
    real ``TRandom3`` with a fixed seed rather than the one-method stub above."""
    from xrdroot.random import TRandom3

    drawn = draw(_linear(), 200, TRandom3(12345), (0.2, 0.4))
    assert len(drawn) == 200
    assert np.all(drawn >= 0.2) and np.all(drawn <= 0.4)


def test_a_function_that_integrates_to_nothing_has_no_distribution_to_draw_from():
    """A table needs mass to divide by, and a function whose integral over the range is zero is
    refused - ROOT's own ``GetRandom`` answers NaN for one rather than raising."""
    with pytest.raises(ValueError, match="integrates to zero over its range"):
        table(lambda low, high, _eps: 0.0, 0.0, 1.0, 10)
