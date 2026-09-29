"""``ROOT::Math::MinimizerOptions``' process-wide defaults as the engine keeps them, and the C
library's ``lgamma`` the likelihood kernels call: the minimizer and its algorithm settled the
first time they are asked for, as a ROOT session settles them, and ``std::lgamma`` element by
element with its poles infinite."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest

from xrdroot.fit import defaults
from xrdroot.random import libm


@pytest.fixture(autouse=True)
def _fresh_defaults() -> Iterator[None]:
    """Each test starts from a new session's defaults, and leaves the process's as they were."""
    saved = dict(defaults.DEFAULTS)
    defaults.DEFAULTS.update(Minimizer="", Algorithm="Migrad")
    yield
    defaults.DEFAULTS.clear()
    defaults.DEFAULTS.update(saved)


def test_a_default_is_read_from_the_one_table() -> None:
    """``DefaultTolerance()`` and ``DefaultStrategy()`` are a new session's."""
    assert defaults.default("Tolerance") == 0.01
    assert defaults.default("Strategy") == 1


def test_the_algorithm_asked_for_first_forgets_migrad_for_good() -> None:
    """With no minimizer chosen yet, asking for the algorithm empties it, and it stays empty."""
    assert defaults.minimizer_algo() == ""
    assert defaults.minimizer_type() == "Minuit2"
    assert defaults.minimizer_algo() == ""


def test_the_algorithm_asked_for_after_the_minimizer_is_migrad() -> None:
    """Once the minimizer is Minuit2, Migrad is the algorithm and stays so."""
    assert defaults.minimizer_type() == "Minuit2"
    assert defaults.minimizer_type() == "Minuit2"  # asked again: already set
    assert defaults.minimizer_algo() == "Migrad"


def test_setting_minuit_without_an_algorithm_brings_migrad_back() -> None:
    """``SetDefaultMinimizer("Minuit")`` after Migrad was forgotten makes it the algorithm."""
    defaults.minimizer_algo()
    defaults.set_minimizer("Minuit")
    assert defaults.DEFAULTS["Minimizer"] == "Minuit"
    assert defaults.minimizer_algo() == "Migrad"


def test_setting_a_minimizer_and_algorithm_keeps_both() -> None:
    """A named algorithm is kept, and a minimizer that is not Minuit's leaves an empty one."""
    defaults.set_minimizer("Minuit2", "Simplex")
    assert (defaults.minimizer_type(), defaults.minimizer_algo()) == ("Minuit2", "Simplex")
    defaults.set_minimizer(None, "")
    assert defaults.minimizer_algo() == "Migrad"  # Minuit2's, when emptied
    defaults.set_minimizer("GSLMultiMin", "")
    assert defaults.DEFAULTS["Algorithm"] == ""


def test_lgamma_is_the_c_librarys_element_by_element() -> None:
    """``std::lgamma`` of a number, of an array, and infinite at the poles as C has it."""
    assert libm.lgamma(5.0) == pytest.approx(math.log(24.0), rel=1e-15)
    found = libm.lgamma(np.array([1.0, 2.0, 0.5]))
    assert found.shape == (3,)
    assert found[:2].tolist() == [0.0, 0.0]
    assert found[2] == pytest.approx(0.5 * math.log(math.pi), rel=1e-15)
    assert libm.lgamma(0.0) == math.inf
    assert libm.lgamma(-2.0) == math.inf
