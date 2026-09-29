"""RooStats' corners that only a bad minimum or a failed search reaches, as ROOT meets them.

A likelihood ratio below zero is inside any interval; a ratio that is not a
profile has no minimiser for contours; an expected p-value whose second
root is not found is -1; a calculator without a sampler tells no test
statistic its observables.
"""

from __future__ import annotations

from typing import Any

import xrdroot.pyroot as ROOT
from xrdroot.roostats import asympformulae
from xrdroot.roostats.likelihoodinterval import LikelihoodInterval


class _Ratio:
    """A likelihood ratio of one free parameter, ``mu``, fixed at a value - and no ``nll``."""

    def __init__(self, value: float) -> None:
        self.mu = ROOT.RooRealVar("mu", "mu", 1, 0, 5)
        self.nu = ROOT.RooRealVar("nu", "nu", 1, 0, 5)
        self.value = value

    def getVariables(self) -> Any:
        return ROOT.RooArgSet(self.mu, self.nu)

    def getVal(self) -> float:
        return self.value


def test_a_ratio_below_zero_is_inside_its_warning_held_back(capsys: Any) -> None:
    """ROOT warns, but with every message below FATAL killed meanwhile: nothing is said."""
    ratio = _Ratio(-0.5)
    interval = LikelihoodInterval("li", ratio, ROOT.RooArgSet(ratio.mu))
    interval.SetConfidenceLevel(0.68)
    assert interval.IsInInterval(ROOT.RooArgSet(ratio.mu)) is True
    assert capsys.readouterr().out == ""


def test_contours_of_a_ratio_without_a_likelihood_are_refused(capsys: Any) -> None:
    ratio = _Ratio(0.1)
    interval = LikelihoodInterval("li", ratio, ROOT.RooArgSet(ratio.mu, ratio.nu))
    assert interval.GetContourPoints(ratio.mu, ratio.nu, [], [], 4) == 0
    assert "Error returned creating minimizer for likelihood function" in capsys.readouterr().out


def test_an_expected_p_value_whose_second_root_fails_is_minus_one(monkeypatch: Any,
                                                                   capsys: Any) -> None:
    from xrdroot.numerics import rootfinder

    answers = iter([(True, 1.0), (False, 0.0)])
    monkeypatch.setattr(rootfinder, "brent_root_finder", lambda *args: next(answers))
    assert asympformulae.expected_p_values(0.1, 0.5, 1.0, True, one_sided=False) == -1.0
    assert "Error finding expected p-values - return -1" in capsys.readouterr().out


def test_a_calculator_without_a_sampler_tells_no_statistic() -> None:
    from xrdroot.roostats.calculators import FrequentistCalculator

    calc = FrequentistCalculator.__new__(FrequentistCalculator)
    calc._sampler = None
    calc._statistic_observables(ROOT.RooArgSet(), ROOT.RooArgSet())
