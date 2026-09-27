"""``RooRandom``: RooFit's own ``TRandom3``, drawing ROOT's numbers to the bit.

Every expected draw came from ROOT 6.40's ``RooRandom::randomGenerator()``
in a fresh process, called in the same order, so a macro that generates
events here gets the events ROOT gives it.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.random import TRandom3
from xrdroot.roofit import rng
from xrdroot.roofit.rng import Generator, RooRandom


@pytest.fixture(autouse=True)
def fresh_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts as a fresh process does, with no generator made yet."""
    monkeypatch.setattr(rng, "_GENERATOR", [])


def test_the_generator_is_made_once_at_the_default_seed() -> None:
    """ROOT seeds RooFit's generator with 4357, apart from ``gRandom``."""
    first = RooRandom.randomGenerator()
    assert first is rng.generator()
    assert first.GetSeed() == 4357


def test_each_of_roots_draws_comes_out_as_root_draws_it() -> None:
    """The sequence, drawn through every method in turn, is ROOT's to the bit."""
    r = RooRandom.randomGenerator()
    drawn = [
        r.Rndm(),
        r.Uniform(2.0),
        r.Uniform(1.0, 3.0),
        r.Gaus(1.0, 2.0),
        r.Poisson(3.5),
        r.Integer(10),
        r.Exp(2.0),
        r.Landau(1.0, 0.5),
        r.BreitWigner(1.0, 0.5),
        r.Binomial(10, 0.3),
        RooRandom.uniform(),
        RooRandom.integer(7),
        RooRandom.gaussian(),
    ]
    assert drawn == [
        0.999741748906672,
        0.32581975078210235,
        1.5652356105856597,
        2.5635925123910157,
        6,
        6,
        2.3063209812835783,
        3.9551010442761845,
        1.0154701439475784,
        5,
        0.8983048577792943,
        4,
        2.060902142594318,
    ]


def test_a_reseeded_generator_fills_an_array_as_root_does() -> None:
    """``RndmArray(n, a)`` writes into the buffer it is given, and returns the draws too."""
    r = RooRandom.randomGenerator()
    r.SetSeed(42)
    out = np.zeros(3)
    found = r.RndmArray(3, out)
    expected = [0.37454011430963874, 0.7965429842006415, 0.9507143115624785]
    assert list(out) == expected
    assert list(found) == expected
    r.SetSeed(42)
    assert list(r.RndmArray(3)) == expected


def test_the_lower_case_names_draw_from_the_same_engine() -> None:
    """Code written against the engine can take a generator in its place."""
    r = Generator()
    r.SetSeed(42)
    assert r.rndm() == 0.37454011430963874
    assert list(r.rndm(2)) == [0.7965429842006415, 0.9507143115624785]
    r.SetSeed(42)
    assert r.gaus(0.0, 1.0) == 1.406914031598717


def test_a_generator_passed_explicitly_is_drawn_from_instead_of_roofits() -> None:
    """``RooRandom::uniform(engine)`` leaves RooFit's own sequence where it was."""
    mine = Generator(TRandom3(7))
    assert mine.Rndm() == 0.07630829117260873
    assert RooRandom.uniform(mine) == 0.22733907494693995
    assert RooRandom.randomGenerator().Rndm() == 0.999741748906672


def test_setting_the_generator_takes_a_generator_an_engine_or_a_pyroot_wrapper() -> None:
    """A PyROOT ``TRandom3`` holds its engine as ``_xrd``, and the engine itself is taken too."""
    given = Generator(TRandom3(7))
    RooRandom.setRandomGenerator(given)
    assert RooRandom.randomGenerator() is given

    RooRandom.setRandomGenerator(TRandom3(7))
    assert RooRandom.uniform() == 0.07630829117260873

    class Wrapper:
        _xrd = TRandom3(7)

    RooRandom.setRandomGenerator(Wrapper())
    assert RooRandom.randomGenerator().engine is Wrapper._xrd
    assert RooRandom.uniform() == 0.07630829117260873
