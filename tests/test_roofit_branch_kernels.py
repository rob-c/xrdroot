"""VDT's ``fast_log`` and ``fast_exp`` as RooFit's kernels call them, one number at a time and
a column at a time: the digits ``vdt::fast_log`` and ``vdt::fast_exp`` give in ROOT 6.40, their
ends of range, and a NaN passed through."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot.roofit import kernels

#: ``vdt::fast_log`` of each, as ROOT's own VDT headers compute it.
LOGS = {
    0.3: -1.2039728043259361,
    1.0: 0.0,
    2.5: 0.9162907318741551,
    1e-300: -690.7755278982137,
    7.25e10: 25.00685239880704,
    0.70710678: -0.3465735919580042,
    0.7072: -0.34644176765870327,
}
#: ``vdt::fast_exp`` of each.
EXPS = {-709.0: 0.0, 709.0: math.inf, 3.3: 27.112638920657883, -2.75: 0.06392786120670757,
        0.0: 1.0}  # fmt: skip


@pytest.mark.parametrize(("x", "expected"), LOGS.items())
def test_fast_log_of_a_number_and_of_a_column_is_vdts(x: float, expected: float) -> None:
    """Either side of the square root of a half, tiny and huge: the same bits both ways."""
    assert kernels.fast_log(x) == expected
    assert kernels.fast_log(np.array([x])).tolist() == [expected]


def test_fast_log_is_infinite_past_its_range_and_a_nan_below_zero() -> None:
    """VDT's limits: above ``1e307`` infinity, below zero not a number - scalar or not."""
    assert kernels.fast_log(2e307) == math.inf
    assert math.isnan(kernels.fast_log(-1.0))
    found = kernels.fast_log(np.array([2e307, -1.0, 2.5]))
    assert found[0] == math.inf and math.isnan(found[1]) and found[2] == LOGS[2.5]


@pytest.mark.parametrize(("x", "expected"), EXPS.items())
def test_fast_exp_of_a_number_and_of_a_column_is_vdts(x: float, expected: float) -> None:
    """Beyond its limits nothing or infinity; inside, VDT's digits either way."""
    assert kernels.fast_exp(x) == expected
    assert kernels.fast_exp(np.array([x])).tolist() == [expected]


def test_fast_exp_passes_a_nan_through() -> None:
    assert math.isnan(kernels.fast_exp(math.nan))
