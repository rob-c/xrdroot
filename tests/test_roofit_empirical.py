"""ROOT's cached accept/reject sampler returns its trials and preserves its random stream."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xrdroot.random import TRandom3, libm
from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.generation import empirical
from xrdroot.roofit.generation.empirical import EmpiricalSampler
from xrdroot.roofit.rng import Generator
from xrdroot.roofit.variables import RooRealVar

#: Native proposals, resampling ratios and subsequent random draws for six densities.
CASES = json.loads((Path(__file__).parent / "data/roofit-empirical-6.40.json").read_text())


def density(kind):
    return (
        lambda p: 1 + 0.2 * p[:, 0],
        lambda p: libm.exp(4 * p[:, 0]),
        lambda p: 0.1 + (p[:, 0] < 0.1),
        lambda p: np.where(p[:, 0] == 5, 1.0, 0.25),
        lambda p: 1 + p[:, 0] * p[:, 0] + p[:, 1] * p[:, 1],
        lambda p: libm.exp(p[:, 0]) * (1 + 0.2 * (p[:, 1] == 5)),
    )[kind]


def variables(kind):
    x = RooRealVar("x", "x", 0.0, 1.0)
    y = RooRealVar("y", "y", -1.0, 1.0)
    cat = RooCategory("cat", "cat", {"Z": 5, "A": -2})
    found = [cat] if kind == 3 else [x]
    if kind == 4:
        found.append(y)
    if kind == 5:
        found.append(cat)
    return found


@pytest.mark.parametrize("case", CASES)
def test_cached_trials_refills_maximum_updates_and_random_draws_match_root(case):
    kind = case["kind"]
    rng = Generator(TRandom3(case["seed"]))
    trials = case["trials"] * (2 if kind in (3, 5) else 1)
    sampler = EmpiricalSampler(density(kind), variables(kind), rng, trials)
    found = []
    for i in range(case["count"]):
        point = sampler.generate(case["count"] - i)
        found.append([*map(float, point), sampler.resample_ratio])
    # These are uniform proposal coordinates and integer category indices, so
    # no transcendental rounding enters the expected coordinates or RNG state.
    assert found == case["events"]
    assert [rng.Rndm() for _ in range(16)] == case["next"]


def test_a_negative_density_cannot_supply_a_sampling_efficiency():
    sampler = EmpiricalSampler(lambda p: -np.ones(len(p)), variables(0), Generator(TRandom3(7)), 5)
    with pytest.raises(ValueError, match="nonpositive density"):
        sampler.generate(2)


def test_an_empty_initial_trial_set_cannot_estimate_the_efficiency():
    sampler = EmpiricalSampler(lambda p: np.ones(len(p)), variables(0), Generator(TRandom3(7)), 0)
    with pytest.raises(ValueError, match="cannot estimate an efficiency"):
        sampler.generate(2)


def test_trial_cache_eviction_retains_the_accumulated_integral_and_maximum(monkeypatch):
    # Exercise the same bounded-cache behavior with six proposals rather than
    # allocating ROOT's million-entry production cache in a unit test.
    monkeypatch.setattr(empirical, "MAX_CACHE", 5)
    sampler = EmpiricalSampler(lambda p: np.ones(len(p)), variables(0), Generator(TRandom3(7)), 7)
    point = sampler.generate(1)
    assert 0 <= point[0] <= 1
    assert sampler.total == 7 and sampler.weight_sum == 7 and sampler.maximum == 1.05
    assert len(sampler.points) == 1 and sampler.used == 1


def test_an_accepted_event_survives_eviction_after_a_long_cache_scan(monkeypatch):
    monkeypatch.setattr(empirical, "MAX_CACHE", 5)
    sampler = EmpiricalSampler(lambda p: np.ones(len(p)), variables(0), Generator(TRandom3(7)), 5)
    for _ in range(20):
        point = sampler.generate(20)
        assert 0 <= point[0] <= 1
    assert sampler.total > 5
