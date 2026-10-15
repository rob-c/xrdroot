"""Counting studies keep the same kernel arithmetic for one event and a batch."""
from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot.roofit import evalerrors, kernels
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.fitting.nll import _log_terms
from xrdroot.roofit.nanpack import unpack
from xrdroot.roofit.pdf import RooAbsPdf
from xrdroot.roofit.pdfs.addpdf import RooAddPdf
from xrdroot.roofit.pdfs.shapes import RooPoisson
from xrdroot.roofit.variables import RooRealVar


@pytest.mark.parametrize('count,mean', [
    (0, .1), (.5, 1), (1., 20), (160., 150), (299., 300), (-1, 2),
    (1, 0), (1, -1), (math.inf, 1), (1, math.inf), (math.nan, 1),
    (1, math.nan), (1e308, 1e308),
])
@pytest.mark.parametrize('unrounded,protect', [(False, False), (True, False),
                                               (False, True), (True, True)])
def test_one_count_uses_the_same_poisson_kernel_as_a_batch(count, mean, unrounded, protect):
    x, mu = RooRealVar('x', '', 0, 300), RooRealVar('mu', '', 150, 0, 300)
    pdf = RooPoisson('p', '', x, mu, unrounded)
    pdf.protectNegativeMean(protect)
    with np.errstate(all='ignore'), kernels.likelihood():
        batch = np.asarray(pdf.compute({'x': np.full(2, count), 'mu': np.full(2, mean)}))
        for shape in [(), (1,), (1, 1)]:
            one = np.asarray(pdf.compute({'x': np.full(shape, count), 'mu': np.full(shape, mean)}))
            assert one.shape == shape
            assert np.array_equal(one.reshape(-1).view(np.uint64), batch[:1].view(np.uint64))


@pytest.mark.parametrize('probability', [.1, .3, 1., 3., 1e-200])
def test_an_event_of_zero_weight_does_not_change_a_valid_likelihood(probability):
    weights = np.asarray([1., .25, -.5, 1e100, -1e100, 1e-100])
    original, bad = _log_terms(np.asarray(probability), weights)
    padded, padded_bad = _log_terms(np.asarray(probability), np.append(weights, 0.))
    assert bad == padded_bad == 0.
    actual = np.asarray([original.total, original.carry]).view(np.uint64)
    expected = np.asarray([padded.total, padded.carry]).view(np.uint64)
    assert np.array_equal(actual, expected)


class _NegativeDensity(RooAbsPdf):
    """A user PDF owns its normalization but can still fail at a parameter point."""

    def __init__(self, x):
        super().__init__("negative", "")
        self.x = self._proxy("x", x)

    def selfNormalized(self):
        return True

    def compute(self, ctx):
        return np.zeros_like(self.x.compute(ctx)) - 0.5


@pytest.mark.parametrize('mixture', [False, True])
def test_a_self_normalizing_pdf_reports_its_invalid_density(mixture):
    x = RooRealVar("x", "", 0, -1, 1)
    pdf = _NegativeDensity(x)
    if mixture:
        pdf = RooAddPdf("mixture", "", [pdf], [])
    data = RooDataSet("data", "", [x])
    data.add(RooArgSet(x))
    nll = pdf.createNLL(data)
    evalerrors.clear()
    evalerrors.collecting(True)
    try:
        value = nll.getVal()
        assert math.isnan(value) and unpack(value) == 0.5
        assert evalerrors.count() == 1
        message = evalerrors.text(1)
        if mixture:
            assert "RooAddPdf::mixture[ [%] * negative ]" in message
            assert "!pdfs=(negative = -0.5)" in message
        else:
            assert "_NegativeDensity::negative" in message
        assert "top-level p.d.f not greater than zero" in message
    finally:
        evalerrors.collecting(False)
        evalerrors.clear()
