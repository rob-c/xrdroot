"""Scalar Gamma kernels retain the array path's bits and exceptional cases."""

import math

import numpy as np
import pytest

from xrdroot.roofit.pdfs import gamma


@pytest.mark.parametrize("shape", [0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
@pytest.mark.parametrize("x", [0.0, 0.001, 0.5, 2.0, 10.0, 100.0])
def test_scalar_gamma_values_keep_the_array_kernels_bits(x, shape):
    expected = gamma._kernel(np.asarray([x]), shape, 1.5, 0.0)
    actual = np.asarray([gamma._kernel(x, shape, 1.5, 0.0)])
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize(
    "args",
    [
        (np.float32(1), 2.0, 1.0, 0.0),
        (np.asarray([1.0]), 2.0, 1.0, 0.0),
        (math.inf, 2.0, 1.0, 0.0),
        (1.0, math.nan, 1.0, 0.0),
        (1.0, 0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0, 0.0),
        (-1.0, 1.0, 1.0, 0.0),
        (0.0, 1.0, 1e-320, 0.0),
        (1.0, 1.0, 1e-320, 0.0),
        (1e308, 1.0, 1e-10, 0.0),
        (5e-324, 2.0, 1e308, 0.0),
        (1.7e308, 1e305, 1.0, 0.0),
    ],
)
def test_unsupported_scalar_gamma_regions_keep_the_array_evaluator(args):
    assert gamma._single(*args) is None


def test_a_gamma_shape_overflow_is_not_hidden_at_the_origin(monkeypatch):
    def overflow(value):
        raise OverflowError("lgamma unavailable")

    monkeypatch.setattr(gamma.libm, "lgamma", overflow)
    with pytest.raises(OverflowError, match="lgamma unavailable"):
        gamma._kernel(0.0, 1e308, 1.0, 0.0)


def test_the_c_library_can_return_infinity_for_a_boundary_shape():
    assert gamma._kernel(0.0, 1e308, 1.0, 0.0) == 0.0


def test_tiny_scales_keep_numpy_overflow_warnings_at_the_origin():
    with pytest.warns(RuntimeWarning) as warnings:
        result = gamma._kernel(0.0, 1.0, 1e-320, 0.0)
    assert math.isinf(result)
    assert any("overflow" in str(w.message) for w in warnings)
    assert any("invalid" in str(w.message) for w in warnings)


def test_integer_and_numpy_double_inputs_keep_double_arithmetic():
    actual = gamma._kernel(2, np.float64(3), 2, 0)
    expected = gamma._kernel(np.asarray([2]), 3.0, 2.0, 0.0)[0]
    assert np.asarray(actual).view(np.uint64) == np.asarray(expected).view(np.uint64)
