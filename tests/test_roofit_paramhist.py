"""``RooParamHistFunc``: a histogram's contents scaled by one gamma parameter per bin."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT


@pytest.fixture
def histogram():
    x = ROOT.RooRealVar("x", "x", 0.0, 4.0)
    x.setBins(4)
    data = ROOT.RooDataHist("dh", "dh", ROOT.RooArgList(x), x)
    for value, weight in ((0.5, 10.0), (1.5, 20.0), (2.5, 30.0), (3.5, 40.0)):
        x.setVal(value)
        data.set(ROOT.RooArgSet(x), weight)
    x.setVal(0.5)
    return x, data


def test_a_bin_is_its_content_times_its_own_gamma(histogram):
    x, data = histogram
    function = ROOT.RooParamHistFunc("p", "p", data, x)
    assert [g.GetName() for g in function.binParameters()] == [
        f"p_gamma_bin_{index}" for index in range(4)
    ]
    x.setVal(2.5)
    assert function.getVal() == 30.0
    function.binParameters()[2].setVal(1.5)
    assert function.getVal() == pytest.approx(45.0)
    assert function.binContents().tolist() == [10.0, 20.0, 30.0, 40.0]
    assert [one.GetName() for one in function.xList()] == ["x"]
    assert "4 bins" in repr(function)


def test_a_point_outside_every_bin_is_zero(histogram):
    x, data = histogram
    function = ROOT.RooParamHistFunc("p", "p", data, x)
    # asked directly, without the variable's own range clamping the value first
    outside = function.compute({"x": np.array([-1.0, 2.5, 9.0])})
    assert list(outside) == [0.0, 30.0, 0.0]


def test_a_second_function_can_share_the_first_ones_gammas(histogram):
    x, data = histogram
    first = ROOT.RooParamHistFunc("p", "p", data, x)
    second = ROOT.RooParamHistFunc("p2", "p2", data, x, first, True)
    assert second.binParameters()[1] is first.binParameters()[1]
    x.setVal(1.5)
    first.binParameters()[1].setVal(2.0)
    assert second.getVal() == pytest.approx(40.0)


def test_a_histogram_of_another_size_is_refused(histogram):
    x, data = histogram
    first = ROOT.RooParamHistFunc("p", "p", data, x)
    other = ROOT.RooRealVar("y", "y", 0.0, 2.0)
    other.setBins(2)
    smaller = ROOT.RooDataHist("dh2", "dh2", ROOT.RooArgList(other), other)
    with pytest.raises(ValueError, match="they have to match"):
        ROOT.RooParamHistFunc("p3", "p3", smaller, other, first, True)


def test_the_constraint_ties_each_gamma_to_its_own_bin(histogram):
    x, data = histogram
    function = ROOT.RooParamHistFunc("p", "p", data, x)
    constraint = ROOT.RooHistConstraint("hc", "hc", function)
    assert len(constraint.binParameters()) == 4
    x.setVal(0.5)
    at_nominal = constraint.getVal()
    assert 0.0 < at_nominal < 1.0
    function.binParameters()[0].setVal(1.2)
    assert constraint.getVal() != at_nominal
    assert "4 bins" in repr(constraint)


def test_a_shared_set_of_gammas_is_constrained_once(histogram):
    x, data = histogram
    first = ROOT.RooParamHistFunc("p", "p", data, x)
    second = ROOT.RooParamHistFunc("p2", "p2", data, x, first, True)
    constraint = ROOT.RooHistConstraint("hc", "hc", {first, second})
    assert len(constraint.binParameters()) == 4


def test_a_very_large_bin_is_constrained_the_gaussian_way(histogram):
    x, data = histogram
    function = ROOT.RooParamHistFunc("p", "p", data, x)
    gaussian = ROOT.RooHistConstraint("hc", "hc", function, 5)
    x.setVal(0.5)
    assert 0.0 < gaussian.getVal() <= 1.0


def test_a_constraint_with_nothing_to_constrain_says_so():
    with pytest.raises(ValueError, match="at least one RooParamHistFunc"):
        ROOT.RooHistConstraint("hc", "hc", [])


def test_an_empty_bin_constrains_nothing(histogram):
    x, data = histogram
    function = ROOT.RooParamHistFunc("p", "p", data, x)
    function._weights[1] = 0.0  # one empty bin, as a sparse histogram has
    constraint = ROOT.RooHistConstraint("hc", "hc", function)
    x.setVal(0.5)
    finite = constraint.getVal()
    assert 0.0 < finite < 1.0
