"""The y extent of a histogram's frame: ``THistPainter::PaintInit``'s, linear and logarithmic.

The lowest and highest bins with room above - from zero, or below a
negative lowest - and on a log scale from half the lowest positive bin to
twice the highest; a histogram's own minimum and maximum over them, and
ROOT's answers to a minimum at or above the maximum.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.canvas import frame


def test_bins_on_a_log_scale_run_from_half_the_lowest_positive_to_twice_the_highest() -> None:
    assert frame._histogram_y(np.array([0.0, 2.0, 8.0]), True) == (1.0, 16.0)
    assert frame._histogram_y(np.array([-1.0, 0.0]), True) == (0.1, 0.2)
    assert frame._histogram_y(np.array([np.nan]), True) == (0.1, 10.0)


def test_a_log_frame_takes_the_histograms_own_ends_or_falls_back() -> None:
    values = np.array([0.0, 2.0, 8.0])
    assert frame._log_y(values, -1111, -1111) == (1.0, 16.0)
    assert frame._log_y(values, 0.5, 4.0) == (0.5, 4.0)
    assert frame._log_y(values, 10.0, 4.0) == pytest.approx((0.004, 4.0))  # a thousandth
    assert frame._log_y(values, -1111, -1.0) == (1.0, 16.0)
    assert frame._log_y(np.array([-1.0, 0.0]), -1111, -1111) == (0.1, 0.2)


def test_a_minimum_at_or_above_the_maximum_is_doubled_away_from_zero() -> None:
    assert frame._crossed(0.0, 1.0) == (0.0, 1.0)
    assert frame._crossed(3.0, 2.0) == (0.0, 4.0)
    assert frame._crossed(-1.0, -2.0) == (-2.0, 0.0)
    assert frame._crossed(0.0, 0.0) == (0.0, 1.0)
