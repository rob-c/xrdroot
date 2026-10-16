"""Batch bindings preserve values, warning order, counters and the final parameter state."""

import numpy as np
import pytest

from test_roofit_parameter_batches import model
from xrdroot.roofit.pdfs.basic import RooUniform
from xrdroot.roostats import bayesbatch, bayesscan
from xrdroot.roostats.bayesfuncs import Functor, Likelihood, Posterior, PosteriorFunction


def like_for(nll, parameters, prior=True):
    pdf = RooUniform("prior", "", [parameters[0]]) if prior else None
    return Likelihood(
        Functor(nll, parameters), Functor(pdf, parameters) if pdf else None, nll.getVal()
    )


def reset(like, parameters):
    for parameter, value in zip(parameters, [1.0, 1.0, 0.0], strict=True):
        parameter.setVal(value)
    like.nll.calls = 0
    if like.prior is not None:
        like.prior.calls = 0
    like.max = 0.0


@pytest.mark.parametrize("prior", [False, True])
def test_a_binding_keeps_values_counters_and_final_parameters(prior):
    nll, parameters = model()
    like = like_for(nll, parameters, prior)
    points = np.asarray([[1.0, 1.0, 0.0], [2.0, 0.5, 1.0], [0.5, 2.0, -3.0]])
    expected = np.asarray([like(point) for point in points])
    peak = like.max
    final = [p.getVal() for p in parameters]
    reset(like, parameters)
    actual = bayesbatch.integrand(like)(points)
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert like.nll.calls == len(points)
    if prior:
        assert like.prior.calls == len(points)
    assert like.max == peak
    assert [p.getVal() for p in parameters] == final


def test_chunked_parameter_points_keep_their_scalar_values(monkeypatch):
    nll, parameters = model()
    like = like_for(nll, parameters)
    rng = np.random.default_rng(100)
    points = rng.uniform([0.1, 0.1, -5.0], [3.0, 2.0, 5.0], size=(12, 3))
    expected = np.asarray([like(point) for point in points])
    reset(like, parameters)
    monkeypatch.setattr(bayesbatch, "_MAX_CELLS", 12)
    actual = bayesbatch.integrand(like)(points)
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert like.nll.calls == like.prior.calls == 12


def test_large_likelihoods_keep_their_scalar_warning_order(capsys):
    nll, parameters = model()
    like = like_for(nll, parameters, False)
    like.offset = 100.0
    points = np.asarray([[3.0, 2.0, 5.0], [2.0, 1.0, 3.0], [1.0, 1.0, 0.0]])
    capsys.readouterr()
    expected = np.asarray([like(point) for point in points])
    messages = capsys.readouterr().out
    assert "Huge likelihood" in messages
    reset(like, parameters)
    actual = bayesbatch.integrand(like)(points)
    assert capsys.readouterr().out == messages
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert like.nll.calls == 3


def test_extra_posterior_coordinates_set_the_parameter_of_interest():
    nll, parameters = model()
    scale, gamma, alpha = parameters
    like = Likelihood(Functor(nll, [gamma, alpha]), None, nll.getVal())
    points = np.asarray([[0.5, 1.0, 2.0], [2.0, -1.0, 0.7]])
    expected = []
    for point in points:
        scale.setVal(point[2])
        expected.append(like(point[:2]))
    final = [p.getVal() for p in parameters]
    reset(like, parameters)
    actual = bayesbatch._LikelihoodBatch(like, [gamma, alpha, scale], 100)(points)
    assert np.array_equal(actual.view(np.uint64), np.asarray(expected).view(np.uint64))
    assert [p.getVal() for p in parameters] == final
    assert like.nll.calls == 2


def posterior_state(post, parameters):
    return (
        post.like.nll.calls,
        post.like.prior.calls if post.like.prior is not None else None,
        [p.getVal() for p in parameters],
        post.error,
        post.like.max,
    )


@pytest.mark.parametrize("with_prior", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_independent_posterior_integrals_keep_results_errors_and_warnings(
    monkeypatch, capsys, with_prior, chunked
):
    nll, parameters = model()
    scale, gamma, alpha = parameters
    prior = RooUniform("prior", "", [scale]) if with_prior else None
    post = PosteriorFunction(nll, scale, [gamma, alpha], prior, 1.0, nll.getVal(), calls=300)
    exact = Posterior("posterior", post, scale)
    low, high, bins = 0.1, 3.0, 4
    dx = (high - low) / bins
    xs = [low + dx * i for i in range(bins + 1)]
    capsys.readouterr()
    with monkeypatch.context() as scalar_context:
        scalar_context.setattr(bayesbatch, "integrand", lambda *args: None)
        expected = np.asarray([post(x) for x in xs])
    messages = capsys.readouterr().out
    state = posterior_state(post, parameters)
    reset(post.like, parameters)
    post.error = 0.0
    if chunked:
        monkeypatch.setattr(bayesscan, "_WORK_SLOTS", 100020)
    actual = bayesscan.sample(exact, scale, low, high, bins)
    assert actual is not None
    assert np.array_equal(np.asarray(actual).view(np.uint64), expected.view(np.uint64))
    assert posterior_state(post, parameters) == state
    assert capsys.readouterr().out == messages
