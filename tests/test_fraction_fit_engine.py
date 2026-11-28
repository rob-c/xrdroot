"""``TFractionFitter``: the fractions of some templates in some data, by Barlow and Beeston's fit.

ROOT fits a data histogram as fractions of Monte Carlo templates, and the fit varies **both** within
their statistics: the likelihood is Barlow and Beeston's (Comp. Phys. Comm. 77 (1993) 219), and each
bin's per-template factor is solved for analytically rather than handed to Minuit, which leaves one
free parameter per template - the fraction of the data it accounts for. The engine takes plain
numbers (a bin's contents, one row per template), so it is testable without a histogram or a file,
which is what keeps ``xrdroot.fit.fraction`` inside the ``hist`` component's own isolation scope.

What the fit answers with is **not** the weighted sum of the templates: the per-bin solve's factors
are what the data prefers, and those are the smeared templates :meth:`FractionFit.predictions` hands
back - which is why a data set that *is* a mixture comes back a little redistributed
(266.4 against 250.0 in the first bin below) and the fit still recovers the mixture exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.fit.fraction import ERRORDEF, STEP, FractionFit, _ln

#: Two templates over four bins, the second's shape the first's reversed.
T1 = np.array([100.0, 200.0, 300.0, 400.0])
T2 = np.array([300.0, 200.0, 100.0, 0.0])
#: A quarter of the first and three quarters of the second, in counts.
MIXED = 0.25 * T1 + 0.75 * T2


def _mixed() -> FractionFit:
    """The two-template fit of the mixture above."""
    return FractionFit(MIXED, [T1, T2])


def test_one_parameter_per_template_and_the_integrals_the_fractions_are_measured_against():
    """A fit has one free parameter per template; ``integral`` is the data's total and
    ``integrals``/``measured`` each template's - the second with ROOT's per-bin weights in it."""
    found = _mixed()
    assert found.npar == 2
    assert found.integral == pytest.approx(700.0)  # 250 + 200 + 150 + 100
    assert found.integrals.tolist() == [1000.0, 600.0]
    assert found.measured.tolist() == [1000.0, 600.0]  # unweighted, so the two agree


def test_a_parameter_is_a_fraction_of_the_data_over_the_template_s_own_integral():
    """ROOT's ``fFractions`` rule: the parameter times the data's integral over the template's, so
    0.25 of the first template and 0.75 of the second read 0.175 and 0.875 here."""
    assert _mixed().fractions([0.25, 0.75]).tolist() == pytest.approx([0.175, 0.875])


def test_a_weight_changes_what_a_parameter_is_a_fraction_of_but_not_a_template_s_integral():
    """``SetWeight`` gives a template a per-bin factor: ``measured`` takes it and ``integrals`` does
    not, so the same parameter reads a different fraction."""
    found = FractionFit(
        np.array([10.0, 20.0]),
        [np.array([5.0, 5.0]), np.array([5.0, 5.0])],
        weights=[np.array([2.0, 2.0]), np.array([1.0, 1.0])],
    )
    assert found.integrals.tolist() == [10.0, 10.0]
    assert found.measured.tolist() == [20.0, 10.0]
    assert found.fractions([0.5, 0.5]).tolist() == pytest.approx([0.75, 1.5])


def test_the_likelihood_is_lowest_at_the_mixture_the_data_was_made_of():
    """The objective is the negative log-likelihood, so the truth must sit **below** an equal
    split - which is what a fit can find and a wrong answer cannot."""
    found = _mixed()
    assert found.objective([0.25, 0.75]) < found.objective([0.5, 0.5])
    assert found.objective([0.25, 0.75]) == pytest.approx(-10205.348054562444)


def test_the_prediction_smears_the_templates_and_their_fractions_add_up_to_it():
    """``GetPlot``'s total is the fractions' weighted sum of the **smeared** templates, not of the
    templates themselves: the first bin predicts 266.4 where the data holds 250."""
    total, fitted = _mixed().predictions([0.25, 0.75])
    assert total.tolist() == pytest.approx(
        [266.3978673082837, 205.69325401647802, 143.81773974361224, 74.46808510638299]
    )
    fractions = _mixed().fractions([0.25, 0.75])
    assert np.allclose((fractions[:, None] * fitted).sum(axis=0), total)
    assert fitted.shape == (2, 4)  # one smeared row per template
    assert fitted[1][3] == 0.0  # the second template holds nothing in the last bin


def test_the_fit_recovers_the_mixture_the_data_was_made_of():
    """Minuit on the profile likelihood from equal fractions finds 0.25 and 0.75 within its
    errors - and the errors are the size a 700-count sample gives, not a token number."""
    found = _mixed()
    result = found.solve()
    assert result.valid
    assert found.fractions(result.parameters).tolist() == pytest.approx([0.25, 0.75], abs=0.02)
    assert result.errors.tolist() == pytest.approx([0.03630515327536817, 0.04780206993867323])
    assert result.parameter_names == ("frac0", "frac1")  # ROOT names them as it does


def test_one_template_alone_leaves_the_other_s_fraction_at_zero():
    """A data set that *is* a template is fitted by that template alone, so the second's fraction is
    zero and the goodness of fit is zero too."""
    found = FractionFit(T1, [T1, T2])
    result = found.solve()
    fractions = found.fractions(result.parameters)
    assert fractions[0] == pytest.approx(1.0, abs=1e-4)
    assert fractions[1] == pytest.approx(0.0, abs=1e-4)
    assert found.chisquare(*found.predictions(result.parameters)) == pytest.approx(0.0, abs=1e-5)


def test_the_goodness_of_fit_is_twice_the_log_likelihood_ratio():
    """``ComputeChisquareLambda``: -2 ln(L of the fit / L of the data as its own model), which is 0
    for a perfect fit and 12.02 for the mixture above."""
    found = _mixed()
    total, fitted = found.predictions([0.25, 0.75])
    assert found.chisquare(total, fitted) == pytest.approx(12.018496759435948)


def test_a_bin_adds_nothing_to_the_likelihood_where_its_value_is_zero():
    """``_ln`` is ``counts * ln(value) - value`` per bin, and ROOT sums **nothing** for a zero value
    rather than -inf, which is how a template empty in a bin is fitted."""
    assert _ln(np.array([0.0, 2.0]), np.array([3.0, 4.0])).tolist() == pytest.approx(
        [0.0, 4.0 * np.log(2.0) - 2.0]
    )
    empty = FractionFit(np.array([0.0, 5.0]), [np.array([0.0, 5.0]), np.array([1.0, 0.0])])
    assert np.isfinite(empty.objective([0.5, 0.5]))  # a bin the data leaves empty is no -inf


def test_the_fits_two_constants_are_roots():
    """A **negative** log-likelihood has Minuit's error definition 0.5 rather than 1, so one
    standard deviation is an increase of one half; the step from equal fractions is ROOT's
    0.01."""
    assert ERRORDEF == 0.5
    assert STEP == 0.01


def test_newton_gives_up_after_the_turns_it_is_allowed_and_leaves_the_bin_unsolved():
    """``FindPrediction`` may only take so many Newton steps in a bin before it warns and leaves
    it, so a bin handed one turn comes back at its first step - not the answer the full solve
    reaches, and the turns after the first never run."""
    found = _mixed()
    strengths = found.weights * found.fractions([0.25, 0.75])[:, None]
    top = strengths.max(axis=0)
    one = found._newton(found.data, found.templates, strengths, top, turns=1)
    many = found._newton(found.data, found.templates, strengths, top)
    assert np.all(one != many)  # one step is not the answer the full solve reaches
    assert np.all(np.isfinite(one))
