"""Exact C-library functions keep their bits when NumPy drives the element loop."""
from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot.random import libm


@pytest.mark.parametrize('name,values', [
    ('exp', [-746., -745., -30., -0., 0., .5, 1., 20., 700.]),
    ('log', [1e-300, .1, .3, .5, 1., 2., 10., 1e100, math.inf]),
    ('sin', [-1e100, -10., -math.pi, -0., 0., .1, math.pi, 10., 1e100]),
    ('atan', [-math.inf, -1e100, -1., -0., 0., .1, 1., 1e100, math.inf]),
])
def test_exact_unary_batches_match_each_c_library_call_bit_for_bit(name, values):
    exact, fast = getattr(math, name), getattr(np, name)
    values = np.asarray(values).reshape(3, 3).T
    expected = np.asarray([[exact(x) for x in row] for row in values.tolist()])
    actual = libm._each(exact, fast, values)
    assert actual.shape == values.shape
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_exact_power_batches_broadcast_with_the_c_librarys_rounding():
    bases = np.asarray([.1, .3, .5, 1., 1.1, 2., 10.])
    powers = np.asarray([[-1.5], [.5], [1.5], [2.5]])
    expected = np.asarray([[math.pow(b, p) for b in bases] for p in powers[:, 0]])
    actual = libm._each_pair(math.pow, np.power, bases, powers)
    assert actual.shape == expected.shape
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize('name,values', [
    ('exp', [-1000., 0., 1000., math.inf, math.nan]),
    ('log', [0., -1., math.inf, math.nan, 1.]),
    ('sin', [0., math.inf, -math.inf, math.nan, math.pi]),
])
def test_domain_errors_in_a_batch_keep_the_original_ieee_fallback(name, values):
    exact, fast = getattr(math, name), getattr(np, name)
    # A one-element call takes the original path, including its NaN sign and payload.
    expected = np.asarray([libm._each(exact, fast, np.asarray([x]))[0] for x in values])
    actual = libm._each(exact, fast, np.asarray(values))
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_power_domain_errors_and_overflows_keep_the_original_ieee_fallback():
    base = np.asarray([0., -0., -1., 10., math.inf, math.nan])
    power = np.asarray([-1., -3., .5, 400., 1., 1.])
    expected = np.asarray([libm._each_pair(math.pow, np.power, b, p)
                           for b, p in zip(base, power, strict=True)])
    actual = libm._each_pair(math.pow, np.power, base, power)
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_a_users_unary_callable_is_called_once_per_element_even_on_domain_errors():
    called = []

    def exact(x):
        called.append(x)
        return math.log(x)

    actual = libm._each(exact, np.log, [.5, -1., .25])
    assert called == [.5, -1., .25]
    assert actual[0] == math.log(.5) and math.isnan(actual[1]) and actual[2] == math.log(.25)
    assert np.array_equal(libm._each(abs, np.abs, [-2., 3.]), [2., 3.])


def test_a_users_binary_callable_keeps_its_call_order_and_side_effects():
    called = []

    def exact(a, b):
        called.append((a, b))
        return math.pow(a, b)

    actual = libm._each_pair(exact, np.power, [.5, 0., 2.], [-1., -1., 2.])
    assert called == [(.5, -1.), (0., -1.), (2., 2.)]
    assert actual.tolist() == [2., math.inf, 4.]
