"""Profiles: histograms whose bins are the mean of something.

``tprofile.root`` is ROOT's, from go-hep: ``p1d`` is the mean of ``pz``
against ``px`` in a hundred bins and ``p2d`` against ``px`` and ``py`` in
forty by forty. The bin contents and errors checked here are the ones go-hep
checks the same file against, which came from uproot reading it - a reader
that shares no code with this one or with go-hep.

No file holds a ``TProfile3D``, so the one here is made by ``crafted.py``
from the class's declared members.
"""

from __future__ import annotations

import copy
import math
import pathlib
import sys

import numpy as np
import pytest

from crafted import craft
from xrdroot import Histogram, Profile, UnsupportedFeatureError, open_root

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def profiles():
    with open_root(str(DATA / "tprofile.root")) as handle:
        yield handle["p1d"], handle["p2d"]


def close(got: float, want: float) -> bool:
    return abs(got - want) <= 1e-12 * max(abs(want), 1.0)


#: Bin, entries, mean and error of ``p1d``, as uproot 5.7.6 reads them.
P1D = [
    (4, 2, 14.092398166656494, 0.25067697947563844),
    (5, 1, 13.228717803955078, 0),
    (7, 5, 13.223546028137207, 0.59328800344402866),
    (8, 5, 12.63072395324707, 0.36053058682843542),
    (9, 4, 12.230674505233765, 0.50937958607513056),
    (50, 806, 1.1207986534346643, 0.057260034472679973),
    (51, 782, 0.97998454005147739, 0.050277432103469626),
    (52, 836, 1.0887467147092864, 0.049020705724325105),
    (97, 0, 0, 0),
    (100, 0, 0, 0),
]

#: The same for ``p2d``, by ROOT's bin numbers along x and y.
P2D = [
    (14, 40, 1, 17.491657257080078, 0),
    (21, 21, 163, 0.028568437753818027, 0.0014118505025769237),
    (21, 22, 149, 0.10637074051387357, 0.0030085371542827887),
    (1, 1, 0, 0, 0),
    (40, 40, 0, 0, 0),
]


def test_a_profile_is_read_as_a_profile_of_means(profiles):
    p1d, p2d = profiles
    assert isinstance(p1d, Profile) and isinstance(p1d, Histogram)
    assert (p1d.kind, p1d.classname, p1d.shape, p2d.shape) == ("MEAN", "TProfile", (100,), (40, 40))
    assert (p1d.error_mode, p1d.weighted, p1d.name) == ("", False, "p1d")
    assert repr(p1d) == "<TProfile 'p1d' of 100 bins, 24999 entries>"


def test_the_bins_of_a_profile_are_the_means_root_gives(profiles):
    p1d, _ = profiles
    for number, entries, mean, error in P1D:
        assert p1d.bin_entries()[number - 1] == entries
        assert close(p1d.values()[number - 1], mean)
        assert close(p1d.errors()[number - 1], error)
        assert close(p1d.variances()[number - 1], error**2)


def test_the_bins_of_a_two_dimensional_profile_are_indexed_x_first(profiles):
    _, p2d = profiles
    for ix, iy, entries, mean, error in P2D:
        assert p2d.bin_entries()[ix - 1, iy - 1] == entries
        assert close(p2d.values()[ix - 1, iy - 1], mean)
        assert close(p2d.errors()[ix - 1, iy - 1], error)


def test_the_error_modes_are_root_s_four(profiles):
    p1d, _ = profiles
    at = 6  # ROOT's bin 7, five entries
    n = p1d.bin_entries()[at]
    spread = math.sqrt(abs(p1d._sumw2()[at + 1] / n - p1d.values()[at] ** 2))
    assert close(p1d.spread()[at], spread)
    assert close(p1d.errors(error_mode="s")[at], spread)
    assert close(p1d.errors(error_mode="S")[at], spread)  # ROOT takes either case
    assert close(p1d.errors(error_mode="i")[at], spread / math.sqrt(n))
    assert close(p1d.errors(error_mode="g")[at], 1 / math.sqrt(n))
    one = 4  # ROOT's bin 5, a single entry and so no spread at all
    assert p1d.errors(error_mode="s")[one] == 0
    assert close(p1d.errors(error_mode="i")[one], 1 / math.sqrt(12))
    assert p1d.errors(error_mode="g")[96] == 0  # an empty bin has no error at all
    assert p1d.errors(error_mode="i")[96] == 0
    with pytest.raises(ValueError, match="not one of ROOT's"):
        p1d.errors(error_mode="x")


def test_a_profile_saved_with_an_error_mode_uses_it(profiles):
    p1d, _ = profiles
    members = copy.deepcopy(p1d.members)
    members["fErrorMode"] = 1
    spread = Profile("TProfile", members)
    assert spread.error_mode == "s"
    assert close(spread.errors()[6], p1d.spread()[6])


def test_the_flow_bins_of_a_profile_are_there_when_asked_for(profiles):
    p1d, _ = profiles
    assert p1d.values(flow=True).shape == (102,)
    assert p1d.values(flow=True)[101] == 18.250492095947266  # one fill in the overflow
    assert p1d.sums(flow=True)[101] == 18.250492095947266


def test_a_profile_filled_without_weights_counts_its_fills(profiles):
    p1d, _ = profiles
    assert np.array_equal(p1d.counts(), p1d.bin_entries())


def weighted(profile: Profile) -> Profile:
    """The same profile as though every fill had been of weight two."""
    members = copy.deepcopy(profile.members)
    members["fBinSumw2"] = np.asarray(members["fBinEntries"]) * 4.0
    members["fBinEntries"] = np.asarray(members["fBinEntries"]) * 2.0
    inner = members["TH1D"]
    inner["TArrayD"] = np.asarray(inner["TArrayD"]) * 2.0
    inner["TH1"]["fSumw2"] = np.asarray(inner["TH1"]["fSumw2"]) * 2.0
    return Profile("TProfile", members)


def test_a_weighted_profile_counts_the_entries_its_weights_are_worth(profiles):
    p1d, _ = profiles
    heavy = weighted(p1d)
    assert heavy.weighted
    assert np.allclose(heavy.values(), p1d.values())
    assert np.allclose(heavy.counts(), p1d.bin_entries())  # (2n)**2 / 4n == n
    assert np.allclose(heavy.errors(), p1d.errors())


def test_a_profile_becomes_a_hist_of_mean_storage(profiles):
    p1d, _ = profiles
    made = p1d.to_hist()
    view = made.view(flow=True)
    assert view["count"][7] == 5 and close(view["value"][7], p1d.values()[6])
    assert close(made.values()[6], p1d.values()[6])
    heavy = weighted(p1d).to_hist().view(flow=True)
    assert heavy["sum_of_weights"][7] == 10 and heavy["sum_of_weights_squared"][7] == 20


def test_a_profile_without_hist_installed_says_how_to_get_it(profiles, monkeypatch):
    monkeypatch.setitem(sys.modules, "hist", None)
    with pytest.raises(UnsupportedFeatureError, match="pip install hist"):
        profiles[0].to_hist()


def test_a_sum_or_density_of_means_is_refused(profiles):
    with pytest.raises(UnsupportedFeatureError, match="a sum of means is not a thing"):
        profiles[0].sum()
    with pytest.raises(UnsupportedFeatureError, match="no density"):
        profiles[0].density()


def test_a_profile_draws_its_means_in_characters(profiles):
    lines = profiles[0].text().splitlines()
    assert len(lines) == 100
    assert lines[6].rstrip().endswith("13.2235")


def test_a_profile_without_its_weights_is_refused(profiles):
    members = copy.deepcopy(profiles[0].members)
    members["fBinEntries"] = np.zeros(3)
    with pytest.raises(UnsupportedFeatureError, match="keeps the weights of only 3"):
        Profile("TProfile", members)


def test_a_profile_without_its_sums_of_squares_has_no_spread(profiles):
    members = copy.deepcopy(profiles[0].members)
    members["TH1D"]["TH1"]["fSumw2"] = np.zeros(0)
    bare = Profile("TProfile", members)
    assert close(bare.values()[6], profiles[0].values()[6])  # the means are still there
    with pytest.raises(UnsupportedFeatureError, match="no sum of the squares"):
        bare.errors()


def test_a_three_dimensional_profile_is_read_from_its_declared_layout(tmp_path):
    base = Histogram.new("p3", ([0, 1, 2], [0, 1], [0, 1, 2, 3]), np.zeros((2, 1, 3)))
    bins = np.zeros(4 * 3 * 5)
    entries = np.zeros(4 * 3 * 5)
    squares = np.zeros(4 * 3 * 5)
    cell = 1 + 4 * (1 + 3 * 2)  # x bin 1, y bin 1, z bin 2, counting the flow
    bins[cell], entries[cell], squares[cell] = 12.0, 3.0, 50.0
    th3 = base.members["TH3"]
    th3["TH1"]["fSumw2"] = squares
    members = {
        "TH3D": {"TH3": th3, "TArrayD": bins},
        "fBinEntries": entries,
        "fErrorMode": 0,
        "fBinSumw2": [],
    }
    path = craft(tmp_path / "p3.root", [("TProfile3D", "p3", members)])
    with open_root(str(path)) as handle:
        p3 = handle["p3"]
    assert isinstance(p3, Profile) and len(p3.axes) == 3
    assert p3.values()[0, 0, 1] == 4.0
    assert close(p3.spread()[0, 0, 1], math.sqrt(50 / 3 - 16))
    assert p3.values().sum() == 4.0
