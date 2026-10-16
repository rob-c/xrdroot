"""A general constraint does not change its neighbors' arithmetic or diagnostics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from test_roofit_scalar_likelihood import _counting
from xrdroot.roofit import kernels
from xrdroot.roofit.fitting.scalar import constraint_sum
from xrdroot.roofit.pdf import RooAbsPdf
from xrdroot.roofit.pdfs.gamma import RooGamma
from xrdroot.roofit.variables import RooConstVar


class _Constant(RooAbsPdf):
    def __init__(self, value):
        super().__init__("custom", "")
        self.number = value
        self.calls = []

    def selfNormalized(self):
        return True

    def compute(self, ctx):
        self.calls.append((dict(ctx), kernels.active()))
        return self.number


def _same(nll):
    expected = nll._general_constraints()
    actual = nll._constraint_sum()
    assert np.asarray(actual).view(np.uint64) == np.asarray(expected).view(np.uint64)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_a_gamma_keeps_its_normalization_beside_scalar_constraints(position, monkeypatch):
    nll, _, efficiency, _ = _counting()
    gaussian = nll.constraints[0]
    gamma = RooGamma("gamma", "", efficiency, RooConstVar("shape", "", 2.3),
                     RooConstVar("scale", "", .7), RooConstVar("origin", "", -.1))
    nll.constraints = [gaussian, gaussian]
    nll.constraints.insert(position, gamma)
    general = nll._constraint_value
    calls = []

    def record(constraint):
        calls.append(constraint)
        return general(constraint)

    monkeypatch.setattr(nll, "_constraint_value", record)
    for value in [.8, 1., 1.2]:
        efficiency.setVal(value)
        expected = nll._general_constraints()
        calls.clear()
        assert nll._constraint_sum() == expected
        assert calls == [gamma]


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("value", [0., -.5, math.nan, math.inf, .37])
def test_custom_constraints_keep_one_call_and_the_general_log_rules(position, value):
    nll, *_ = _counting()
    custom = _Constant(value)
    nll.constraints = [nll.constraints[0]] * 2
    nll.constraints.insert(position, custom)
    _same(nll)
    assert custom.calls == [(nll._global_values, True)] * 2


@pytest.mark.parametrize("error", [ValueError, TypeError, OverflowError, ZeroDivisionError])
def test_a_custom_constraint_error_propagates_without_retrying(error):
    nll, *_ = _counting()
    calls = []

    def general(constraint):
        calls.append(constraint)
        raise error("custom constraint failure")

    custom = _Constant(.5)
    nll.constraints.append(custom)
    with pytest.raises(error, match="custom constraint failure"):
        constraint_sum(nll, general)
    assert calls == [custom] and custom.calls == []


def test_later_parameter_and_global_changes_keep_the_general_sum():
    nll, _, efficiency, _ = _counting()
    custom = _Constant(.25)
    nll.constraints.append(custom)
    for value in [.7, 1., 1.3]:
        efficiency.setVal(value)
        nll._global_values["nominal"] = value + .03
        custom.number = value / 2.
        _same(nll)
    assert len(custom.calls) == 6


def test_an_unused_huge_global_keeps_the_general_evaluator():
    nll, *_ = _counting()
    nll._global_values["unused"] = 10 ** 400
    assert constraint_sum(nll) is None
    _same(nll)


@pytest.mark.parametrize("value", [.9, np.float32(.9), np.asarray(.9)])
def test_custom_constraints_can_change_globals_for_later_terms(value):
    nll, *_ = _counting()

    class Updating(_Constant):
        def compute(self, ctx):
            nll._global_values["nominal"] = value
            return super().compute(ctx)

    custom = Updating(.3)
    nll.constraints.insert(0, custom)
    before = dict(nll._global_values)
    expected = nll._general_constraints()
    nll._global_values = before
    assert nll._constraint_sum() == expected
    assert len(custom.calls) == 2


def test_an_invalid_global_changed_by_a_callback_does_not_repeat_the_callback():
    nll, *_ = _counting()

    class Updating(_Constant):
        def compute(self, ctx):
            nll._global_values["nominal"] = np.asarray([.9, .9])
            return super().compute(ctx)

    custom = Updating(.3)
    nll.constraints.insert(0, custom)
    with pytest.raises(TypeError, match="arrays"):
        nll._constraint_sum()
    assert len(custom.calls) == 1
