"""Constraint normalization keeps its scalar arithmetic for multi-event and binned fits."""
from __future__ import annotations

import math

import numpy as np
import pytest

from test_roofit_scalar_likelihood import _counting
from xrdroot.roofit.data.datahist import RooDataHist
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.fitting.nll import RooNLLVar
from xrdroot.roofit.fitting.scalar import constraint_sum
from xrdroot.roofit.pdf import RooAbsPdf
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.pdfs.histpdf import RooHistFunc
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.pdfs.realsum import RooRealSumPdf
from xrdroot.roofit.variables import RooRealVar


def _same(nll):
    actual = constraint_sum(nll)
    assert actual is not None
    expected = nll._general_constraints()
    assert np.asarray(actual).view(np.uint64) == np.asarray(expected).view(np.uint64)
    assert nll._constraint_sum() == actual


def test_constraint_sums_are_independent_of_the_number_of_data_rows():
    nll, signal, efficiency, _ = _counting()
    x = nll.pdf.pdfs[0].x
    for entries in [1, 2, 7]:
        data = RooDataSet("data", "", [x])
        data.add_columns({"x": np.full(entries, 150.)})
        nll.setData(data)
        assert nll.data.numEntries() == entries
        _same(nll)
        signal.setVal(60.)
        efficiency.setVal(.95)
        _same(nll)


def test_a_binned_channel_keeps_scalar_constraint_normalization():
    x = RooRealVar("x", "", 0., 2.)
    x.setBins(2)
    template = RooDataHist("template", "", [x])
    template.set(0, 10.)
    template.set(1, 20.)
    function = RooHistFunc("function", "", [x], template)
    scale = RooRealVar("scale", "", 1., 0., 3.)
    channel = RooRealSumPdf("channel", "", [function], [scale], True)
    channel.setAttribute("BinnedLikelihood")
    nominal = RooRealVar("nominal", "", 1., 0., 3.)
    constraint = RooGaussian("constraint", "", nominal, scale, 0.1)
    model = RooProdPdf("model", "", [channel, constraint])
    nll = RooNLLVar(model, template, constraints=[constraint],
                    constrained=frozenset(["nominal"]), global_values={"nominal": 1.01})
    assert nll.binned
    _same(nll)
    for value in [.9, 1.2, 1.5]:
        scale.setVal(value)
        _same(nll)
        actual = nll.getVal()
        expected = nll.channel(model, None) + nll._general_constraints()
        assert np.asarray(actual).view(np.uint64) == np.asarray(expected).view(np.uint64)


def test_an_empty_constraint_sum_is_exactly_zero():
    nll, *_ = _counting()
    nll.constraints = []
    _same(nll)
    assert constraint_sum(nll) == 0.


def test_unknown_constraint_subclasses_keep_their_own_compute():
    class Gaussian(RooGaussian):
        calls = 0

        def compute(self, ctx):
            self.calls += 1
            return super().compute(ctx)

    nll, *_ = _counting()
    old = nll.constraints[0]
    constraint = Gaussian("custom", "", old.x, old.mean, old.sigma)
    nll.constraints = [constraint]
    assert constraint_sum(nll) is None and constraint.calls == 0
    expected = nll._general_constraints()
    constraint.calls = 0
    assert nll._constraint_sum() == expected and constraint.calls == 1


@pytest.mark.parametrize("value", [0., -.5, math.nan, math.inf])
def test_invalid_or_nonfinite_custom_constraints_keep_general_log_rules(value):
    class Constant(RooAbsPdf):
        calls = 0

        def selfNormalized(self):
            return True

        def compute(self, ctx):
            self.calls += 1
            return value

    nll, *_ = _counting()
    constraint = Constant("constant", "")
    nll.constraints = [constraint]
    assert constraint_sum(nll) is None and constraint.calls == 0
    expected = nll._general_constraints()
    constraint.calls = 0
    actual = nll._constraint_sum()
    assert math.isnan(expected) == math.isnan(actual)
    if not math.isnan(actual):
        assert actual == expected
    assert constraint.calls == 1


def test_a_zero_gaussian_constraint_density_uses_the_general_logarithm():
    nll, *_ = _counting()
    old = nll.constraints[0]
    nll.constraints = [RooGaussian("zero", "", old.x, old.mean, .0001)]
    nll._global_values["nominal"] = 2.
    assert constraint_sum(nll) is None
    assert nll._constraint_sum() == nll._general_constraints() == math.inf


def test_global_arrays_keep_the_general_scalar_conversion_error():
    nll, *_ = _counting()
    nll._global_values["nominal"] = np.asarray([1., 1.])
    assert constraint_sum(nll) is None
    with pytest.raises(TypeError, match="arrays"):
        nll._constraint_sum()
