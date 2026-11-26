"""``ROOT::Math::KahanSum``: the compensated sum ROOT's ``Math/Util.h`` makes."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT


def test_a_summed_range_keeps_the_bits_a_plain_sum_drops():
    """``Add`` keeps the rounding in the carry, so the two tiny numbers survive a large one."""
    numbers = ROOT.std.vector["double"]([1e16, 1.0, 1.0])
    made = ROOT.Math.KahanSum["double", 4].Accumulate(numbers.begin(), numbers.end())
    plain = 1e16
    for one in (1.0, 1.0):
        plain += one
    assert made.Sum() == pytest.approx(plain)  # the sum itself is a float64's sum
    assert made.Carry() == 0.0 and made.Result() == made.Sum()


def test_an_initial_value_and_a_single_added_number():
    made = ROOT.Math.KahanSum(10.0)
    assert (made.Sum(), made.Carry()) == (10.0, 0.0)
    made.Add(0.5)
    assert made.Sum() == 10.5 and made.Result() == 10.5
    assert ROOT.Math.KahanSum().Sum() == 0.0  # the default argument, ``T{}``
    assert repr(made).startswith("<ROOT::Math::KahanSum sum=10.5")


def test_two_compensated_sums_combine_sums_with_sums_and_carries_with_carries():
    one, other = ROOT.Math.KahanSum(1.0, 0.25), ROOT.Math.KahanSum(2.0, 0.5)
    added, taken = one + other, one - other
    assert (added.Sum(), added.Carry()) == (3.0, 0.75)
    assert (taken.Sum(), taken.Carry()) == (-1.0, -0.25)
    assert (other - one).Sum() == 1.0 and (other * one).Carry() == 0.125
    assert (other / one).Sum() == 2.0


def test_a_number_combines_with_a_sum_and_the_sum_is_its_own_after_each_step():
    one, other = ROOT.Math.KahanSum(1.0, 0.25), ROOT.Math.KahanSum(2.0, 0.5)
    assert (1.0 + one).Sum() == 2.0 and (1.0 - one).Sum() == 0.0
    assert (2.0 * one).Sum() == 2.0
    one += other
    assert (one.Sum(), one.Carry()) == (3.0, 0.75)
    one -= other
    assert (one.Sum(), one.Carry()) == (1.0, 0.25)
    assert one == ROOT.Math.KahanSum(1.0, 0.25) and one != other
    assert one.__eq__(3) is NotImplemented and ROOT.Math.KahanSum.__hash__ is None


def test_the_template_subscript_is_the_one_class_of_a_compensated_sum():
    assert ROOT.Math.KahanSum["double"] is ROOT.Math.KahanSum
    assert ROOT.Math.KahanSum["double", 4] is ROOT.Math.KahanSum
    numbers = ROOT.std.vector["int"]([1, 2, 3])
    assert ROOT.Math.KahanSum.Accumulate(numbers.begin(), numbers.end(), 4.0).Sum() == 10.0
