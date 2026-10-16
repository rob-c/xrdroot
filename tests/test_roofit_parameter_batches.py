"""Parameter batches retain scalar likelihood values and reject custom evaluation paths."""

import numpy as np
import pytest

from xrdroot.roofit import evalerrors
from xrdroot.roofit.data.datahist import RooDataHist
from xrdroot.roofit.fitting.nll import RooNLLVar
from xrdroot.roofit.fitting.parameterbatch import evaluate
from xrdroot.roofit.functions import RooProduct
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.pdfs.histfactory import ParamHistFunc
from xrdroot.roofit.pdfs.histpdf import RooHistFunc
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.pdfs.realsum import RooRealSumPdf
from xrdroot.roofit.pdfs.shapes import RooPoisson
from xrdroot.roofit.variables import RooConstVar, RooRealVar


def model(constraints=True):
    x = RooRealVar("x", "", 0.0, 2.0)
    x.setBins(2)
    data = RooDataHist("data", "", [x])
    data.set(0, 10.0)
    data.set(1, 20.0)
    scale = RooRealVar("scale", "", 1.0, 0.1, 3.0)
    gamma = RooRealVar("gamma", "", 1.0, 0.1, 2.0)
    alpha = RooRealVar("alpha", "", 0.0, -5.0, 5.0)
    hist = RooHistFunc("hist", "", [x], data)
    parameters = ParamHistFunc("parameters", "", [x], [gamma, RooConstVar("one", "", 1.0)])
    function = RooProduct("shape", "", [hist, parameters])
    channel = RooRealSumPdf("channel", "", [function], [scale], True)
    channel.setAttribute("BinnedLikelihood")
    channel.setForceNumInt(True)
    nominal = RooRealVar("nominal", "", 0.0, -5.0, 5.0)
    nominal.setConstant()
    observed = RooRealVar("observed", "", 100.0, 0.0, 1e30)
    observed.setConstant()
    gaussian = RooGaussian("gaussian", "", alpha, nominal, 1.0)
    mean = RooProduct("mean", "", [gamma, RooConstVar("tau", "", 100.0)])
    poisson = RooPoisson("poisson", "", observed, mean)
    poisson.setNoRounding(True)
    terms = [gaussian, poisson] if constraints else []
    pdf = RooProdPdf("pdf", "", [channel, *terms])
    nll = RooNLLVar(pdf, data, constraints=terms, constrained=frozenset(["nominal", "observed"]))
    return nll, [scale, gamma, alpha] if constraints else [scale, gamma]


def scalar(nll, parameters, points):
    found = []
    for point in points:
        for parameter, value in zip(parameters, point, strict=True):
            parameter.setVal(float(value))
        found.append(nll.getVal())
    return np.asarray(found)


@pytest.mark.parametrize("constraints", [False, True])
@pytest.mark.parametrize("squared", [False, True])
def test_batched_values_match_the_scalar_evaluator_bit_for_bit(constraints, squared):
    nll, parameters = model(constraints)
    nll.applyWeightSquared(squared)
    rng = np.random.default_rng(761)
    points = rng.uniform(
        [p.getMin() for p in parameters],
        [p.getMax() for p in parameters],
        size=(100, len(parameters)),
    )
    actual = evaluate(nll, parameters, points)
    assert actual is not None
    expected = scalar(nll, parameters, points)
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_parameter_batches_clip_with_the_scalar_setter_rules():
    nll, parameters = model()
    points = np.asarray([[-1.0, 0.5, -10.0], [10.0, 5.0, 10.0], [1.0, 1.0, 0.0]])
    actual = evaluate(nll, parameters, points)
    assert actual is not None
    expected = scalar(nll, parameters, points)
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_evaluating_a_parameter_batch_does_not_set_physical_parameters():
    nll, parameters = model()
    before = [p.getVal() for p in parameters]
    assert evaluate(nll, parameters, np.asarray([[2.0, 1.5, 3.0]])) is not None
    assert [p.getVal() for p in parameters] == before


def test_copied_parameter_bindings_use_the_scalar_evaluator():
    nll, parameters = model()
    parameters[0] = RooRealVar(parameters[0])
    assert evaluate(nll, parameters, np.asarray([[2.0, 1.5, 3.0]])) is None


def test_instance_callbacks_are_not_evaluated_by_the_batch_path(monkeypatch):
    nll, parameters = model()
    monkeypatch.setattr(parameters[0], "getVal", lambda: pytest.fail("custom callback ran"))
    assert evaluate(nll, parameters, np.asarray([[2.0, 1.5, 3.0]])) is None


def test_collecting_diagnostics_uses_the_ordinary_evaluator():
    nll, parameters = model()
    evalerrors.collecting(True)
    try:
        assert evaluate(nll, parameters, np.asarray([[2.0, 1.5, 3.0]])) is None
    finally:
        evalerrors.collecting(False)


def test_a_parameter_outside_the_model_uses_the_ordinary_binding():
    nll, parameters = model(False)
    parameters.append(RooRealVar("unused", "", 0.0, -1.0, 1.0))
    assert evaluate(nll, parameters, np.asarray([[2.0, 1.5, 0.0]])) is None
