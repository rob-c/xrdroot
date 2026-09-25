"""Booking and filling histograms, profiles and efficiencies the way ROOT does.

The strongest check is ROOT's own files. ``tefficiency.root`` and
``tprofile.root`` were made by go-hep's ``gen-teff.go`` and
``gen-tprofile.go``, ROOT macros that fill from ``gRandom`` - a ``TRandom3``
at its default seed - and ``xrdroot.random.TRandom3``, which
``support.root_uniforms`` draws from, gives those same numbers bit for bit.
So the macros can be run again here, entry for entry, and what comes out
compared with what ROOT wrote: every bin, every square of weights,
every running sum and the count of entries, to the last bit. Where no macro
exists the expected numbers are worked out by hand from ROOT's source.
"""

from __future__ import annotations

import array
import pathlib

import numpy as np
import pytest

from support import plain, root_rannor, root_uniforms
from xrdroot import Efficiency, Histogram, Profile, open_root
from xrdroot.efficiency import USE_WEIGHTS

DATA = pathlib.Path(__file__).parent / "data"

#: How an object is drawn, which a style a ROOT session set decides and
#: nothing about what was filled does.
DRAWING = {"TAttAxis", "TAttLine", "TAttFill", "TAttMarker", "fUniqueID", "fBits"}


def physics(value):
    """Every member but the drawing ones, as plain lists, Histograms opened up."""
    if isinstance(value, Histogram):
        return physics(value.members)
    if isinstance(value, dict):
        return {key: physics(item) for key, item in value.items() if key not in DRAWING}
    return plain(value)


@pytest.fixture(scope="module")
def efficiencies():
    with open_root(str(DATA / "tefficiency.root")) as handle:
        yield {name: handle[name] for name in ("eff1", "eff2", "eff3")}


@pytest.fixture(scope="module")
def profiles():
    with open_root(str(DATA / "tprofile.root")) as handle:
        yield handle["p1d"], handle["p2d"]


# -- ROOT's macros, run again ---------------------------------------------------


def test_an_efficiency_filled_here_holds_what_root_filled_it_with(efficiencies):
    drawn = root_uniforms(4000).reshape(1000, 4)
    passed = 2 * drawn[:, 0] <= 1
    x, y, z = 10 * drawn[:, 1], 20 * drawn[:, 2], 30 * drawn[:, 3]
    made = {
        "eff1": Efficiency.book("eff1", (10, 0, 10), title="Eff1D"),
        "eff2": Efficiency.book("eff2", (10, 0, 10), (10, 0, 20), title="Eff2D"),
        "eff3": Efficiency.book("eff3", (10, 0, 10), (10, 0, 20), (10, 0, 30), title="Eff3D"),
    }
    made["eff1"].fill(passed, x)
    made["eff2"].fill(passed, x, y)
    made["eff3"].fill(passed, x, y, z)
    for name, efficiency in made.items():
        assert physics(efficiency.members) == physics(efficiencies[name].members), name


def test_a_profile_filled_here_holds_what_root_filled_it_with(profiles):
    px, py = root_rannor(25000)
    pz = px * px + py * py  # in single precision, as the macro's Float_t has it
    p1d = Profile.book("p1d", (100, -4, 4), title="Profile of pz versus px", value_range=(0, 20))
    p1d.fill(px, pz, weight=1)
    title = "Profile of pz versus px and py"
    p2d = Profile.book("p2d", (40, -4, 4), (40, -4, 4), title=title, value_range=(0, 20))
    p2d.fill(px, py, pz, weight=1)
    assert physics(p1d.members) == physics(profiles[0].members)
    assert physics(p2d.members) == physics(profiles[1].members)


def test_the_statistics_of_a_refilled_profile_are_those_of_roots(profiles):
    px, py = root_rannor(25000)
    again = Profile.book("p1d", (100, -4, 4), value_range=(0, 20))
    again.fill(px, px * px + py * py)
    for axis in (0, 1):
        assert again.mean(axis) == profiles[0].mean(axis)
        assert again.std(axis) == profiles[0].std(axis)
    assert again.entries == 24999  # one of 25000 fell outside the range and never counted


# -- booking --------------------------------------------------------------------


def test_a_booked_histogram_is_empty_the_way_a_root_constructor_leaves_one():
    h = Histogram.book("h", (4, 0.0, 1.0), title="spectrum;p_{T};events")
    assert (h.classname, h.name, h.title) == ("TH1D", "h", "spectrum")
    assert (h.axes[0].title, h.axes[0].even, h.axes[0].nbins) == ("p_{T}", True, 4)
    assert h.values(flow=True).tolist() == [0.0] * 6
    assert h.entries == 0 and not h.weighted
    assert h.members["TH1"]["fStatOverflows"] == 2
    two = Histogram.book("h2", (2, 0, 1), [0, 1, 5], labels=["x", "y"], kind="S")
    assert two.classname == "TH2S" and two.members["TH2"]["fScalefactor"] == 1.0
    assert two.axes[1].edges().tolist() == [0, 1, 5] and not two.axes[1].even
    assert two.values().dtype == np.int16 and two.axes[1].title == "y"
    three = Histogram.book("h3", (1, 0, 1), (1, 0, 1), (1, 0, 1), kind="C")
    assert three.classname == "TH3C" and "TAtt3D" in three.members["TH3"]


def test_booking_refuses_what_is_not_an_axis():
    for axes, why in (
        (((0, 0, 1),), "nowhere to put anything"),
        (((3, 1, 1),), "is not a range"),
        (((3, 0, float("inf")),), "is not a range"),
        (([0, 2, 1],), "must be finite and increase"),
        (([1],), "at least two edges"),
        (((1, 0, 1),) * 4, "one, two and three"),
        ((), "one, two and three"),
    ):
        with pytest.raises(ValueError, match=why):
            Histogram.book("h", *axes)
    with pytest.raises(ValueError, match="kind='L' is not a storage"):
        Histogram.book("h", (1, 0, 1), kind="L")
    with pytest.raises(ValueError, match="error_option='x' is not one of ROOT's"):
        Profile.book("p", (1, 0, 1), error_option="x")


def test_a_tuple_of_three_is_a_count_and_two_ends_and_a_list_is_edges():
    assert Histogram.book("h", (2, 0, 4)).axes[0].edges().tolist() == [0, 2, 4]
    assert Histogram.book("h", [2, 3, 4]).axes[0].edges().tolist() == [2, 3, 4]
    assert Histogram.book("h", np.array([0.0, 1.0, 3.0])).axes[0].nbins == 2


def test_a_booked_profile_keeps_its_squares_from_the_start():
    p = Profile.book("p", (3, 0, 3), (2, 0, 2), error_option="S", value_range=(-1, 1))
    assert p.classname == "TProfile2D" and p.error_mode == "s"
    assert len(p.members["TH2D"]["TH2"]["TH1"]["fSumw2"]) == 20
    assert (p.members["fZmin"], p.members["fZmax"]) == (-1.0, 1.0)
    assert not p.weighted
    three = Profile.book("p3", (1, 0, 1), (1, 0, 1), (1, 0, 1))
    assert three.classname == "TProfile3D" and "fTsumwt2" in three.members


# -- filling --------------------------------------------------------------------


def test_every_fill_counts_but_only_those_on_the_axis_count_towards_the_moments():
    h = Histogram.book("h", (2, 0.0, 1.0))
    h.fill([-1.0, 0.0, 0.75, 1.0, np.nan])  # the upper edge, and NaN, are overflow
    assert h.values(flow=True).tolist() == [1, 1, 1, 2]
    assert h.entries == 5
    core = h.members["TH1"]
    assert (core["fTsumw"], core["fTsumw2"], core["fTsumwx"], core["fTsumwx2"]) == (
        2.0,
        2.0,
        0.75,
        0.5625,
    )


def test_a_number_fills_as_an_array_of_one_does():
    one, many = Histogram.book("a", (4, 0, 1)), Histogram.book("a", (4, 0, 1))
    one.fill(0.3, weight=2.0)
    many.fill([0.3], weight=[2.0])
    assert physics(one.members) == physics(many.members)
    two = Histogram.book("c", (2, 0, 1), (2, 0, 1))
    two.fill([0.1, 0.9], 0.6)  # one y for both x
    assert two.values().tolist() == [[0, 1], [0, 1]]


def test_an_even_axis_finds_a_bin_the_way_root_works_it_out():
    # 10 * 0.3 is 3.0000000000000004, so ROOT puts 0.3 in the fourth bin even
    # though the edge it would be compared with, 0.1 * 3, is above it.
    h = Histogram.book("h", (10, 0.0, 1.0))
    assert h.axes[0].edges()[3] > 0.3
    assert h.find_bin(0.3) == 4
    uneven = Histogram.book("u", [0.0, 0.1, 0.30000000000000004, 1.0])
    assert uneven.find_bin(0.3) == 2
    assert uneven.find_bin([0.1, -1, 1.0, np.nan]).tolist() == [2, 0, 4, 4]


def test_the_squares_of_the_weights_start_at_the_first_weight_that_is_not_one():
    h = Histogram.book("h", (2, 0, 2))
    h.fill([0.5, 0.5, 1.5], weight=1)
    assert not h.weighted
    h.fill([0.5, 1.5, 1.5], weight=[1.0, 3.0, 1.0])
    # What was filled before the three counts once per entry; after it, w**2.
    assert h.members["TH1"]["fSumw2"].tolist() == [0, 3, 11, 0]
    assert h.values(flow=True).tolist() == [0, 3, 5, 0]
    assert h.members["TH1"]["fTsumw2"] == 1 + 1 + 1 + 1 + 9 + 1


def test_squares_begun_on_an_empty_histogram_start_from_nothing():
    h = Histogram.book("h", (1, 0, 1))
    h.fill(0.5, weight=0.5)
    assert h.members["TH1"]["fSumw2"].tolist() == [0, 0.25, 0]


def test_integer_bins_saturate_and_take_whole_weights_as_root_does():
    small = Histogram.book("c", (1, 0, 1), kind="C")
    small.fill(np.full(300, 0.5))
    small.fill(-1.0, weight=2.9)  # Int_t(2.9) is 2
    assert small.values(flow=True).tolist() == [2, 127, 0]
    small.fill(np.full(3, 0.5), weight=-100)
    assert small.values().tolist() == [-127]
    # Steps of both signs saturate in turn: 100, then 127, then 27.
    mixed = Histogram.book("m", (1, 0, 1), kind="C")
    mixed.fill([0.5, 0.5, 0.5], weight=[100, 100, -100])
    assert mixed.values().tolist() == [27]
    for kind, top in (("S", 32767), ("I", 2147483647)):
        wide = Histogram.book("w", (1, 0, 1), kind=kind)
        wide.fill(0.5, weight=1e10)
        assert wide.values().tolist() == [top]
    nan = Histogram.book("n", (1, 0, 1), kind="I")
    nan.fill(0.5, weight=np.nan)
    assert nan.values().tolist() == [0]


def test_a_float_histogram_adds_in_single_precision():
    h = Histogram.book("f", (1, 0, 1), kind="F")
    h.fill(0.5, weight=1e8)
    h.fill(0.5)  # a float cannot tell 1e8 from 1e8 + 1
    assert h.values().tolist() == [1e8]
    assert h.values().dtype == np.float32


def test_filling_takes_one_coordinate_per_axis():
    with pytest.raises(ValueError, match="takes one coordinate per axis, not 2"):
        Histogram.book("h", (1, 0, 1)).fill(0.5, 0.5)
    with pytest.raises(ValueError, match="then the value averaged: 2 arrays or numbers, not 1"):
        Profile.book("p", (1, 0, 1)).fill(0.5)
    with pytest.raises(ValueError, match="one coordinate per axis, not 2"):
        Efficiency.book("e", (1, 0, 1)).fill(True, 0.5, 0.5)


def test_a_histogram_read_from_a_file_fills_in_place():
    with open_root(str(DATA / "gauss-h1.root")) as handle:
        h1d, h1f = handle["h1d"], handle["h1f"]
    before = h1d.entries, h1d.values(flow=True).copy()
    h1d.fill(0.0)
    at = h1d.find_bin(0.0)
    assert h1d.entries == before[0] + 1
    assert h1d.values(flow=True)[at] == before[1][at] + 1
    h1f.fill(0.0, weight=2.0)  # its bins came big-endian and read-only from the file
    assert h1f.values().dtype == np.float32 and h1f.weighted


# -- profiles -------------------------------------------------------------------


def test_a_profile_drops_a_value_outside_its_range_before_counting_it():
    p = Profile.book("p", (1, 0, 1), value_range=(0, 1))
    p.fill([0.5, 0.5, 0.5, 0.5], [0.25, 2.0, -1.0, np.nan])
    assert p.entries == 1 and p.values().tolist() == [0.25]
    open_ended = Profile.book("q", (1, 0, 1))
    open_ended.fill([0.5, 0.5], [2.0, np.nan])
    assert open_ended.entries == 2  # with no range, a NaN is filled like anything else


def test_a_weighted_profile_keeps_the_squares_of_its_weights_from_the_first_heavy_fill():
    p = Profile.book("p", (2, 0, 2))
    p.fill([0.5, 1.5], [1.0, 2.0])
    p.fill([0.5, 0.5], [3.0, 5.0], weight=[1.0, 2.0])
    assert p.members["fBinSumw2"].tolist() == [0, 1 + 1 + 4, 1, 0]
    assert p.bin_entries(flow=True).tolist() == [0, 4, 1, 0]
    assert p.sums().tolist() == [1 + 3 + 10, 2]
    assert p.members["TH1D"]["TH1"]["fSumw2"].tolist() == [0, 1 + 9 + 50, 4, 0]
    assert p.members["fTsumwy"] == 1 + 2 + 3 + 10
    assert p.members["fTsumwy2"] == 1 + 4 + 9 + 50
    assert p.weighted


# -- efficiencies ---------------------------------------------------------------


def test_an_efficiency_is_booked_with_roots_names_and_settings():
    eff = Efficiency.book("trigger", (4, 0, 100), title="Trigger;p_{T}")
    assert (eff.name, eff.title) == ("trigger", "Trigger")
    assert (eff.total.name, eff.total.title) == ("trigger_total", "Trigger (total)")
    assert (eff.passed.name, eff.passed.title) == ("trigger_passed", "Trigger (passed)")
    assert eff.total.axes[0].title == "p_{T}"
    assert (eff.method, eff.level) == ("clopper-pearson", 0.682689492137)


def test_an_efficiency_filled_with_weights_says_so():
    eff = Efficiency.book("e", (2, 0, 2))
    eff.fill([True, False], [0.5, 0.5])
    eff.fill([True, True], [1.5, 1.5], weight=[0.5, 1.0])
    assert eff.members["TNamed"]["fBits"] & USE_WEIGHTS
    assert eff.total.members["TH1"]["fSumw2"].tolist() == [0, 2, 1.25, 0]
    assert eff.passed.values().tolist() == [1, 1.5]


def test_an_efficiency_is_made_from_two_histograms_when_one_is_part_of_the_other():
    total, passed = Histogram.book("all", (2, 0, 2)), Histogram.book("good", (2, 0, 2))
    total.fill([0.5, 0.5, 1.5])
    passed.fill([0.5])
    eff = Efficiency.from_histograms(passed, total)
    assert (eff.name, eff.title) == ("all_clone", "efficiency")
    assert (eff.total.name, eff.passed.name) == ("all_clone_total", "all_clone_passed")
    assert eff.values().tolist() == [0.5, 0.0]
    assert not eff.members["TNamed"]["fBits"] & USE_WEIGHTS
    assert passed.name == "good"  # copied, not taken
    heavy = passed.copy()
    heavy.scale(0.5)
    weighted = Efficiency.from_histograms(heavy, total, name="w")
    assert weighted.name == "w" and weighted.members["TNamed"]["fBits"] & USE_WEIGHTS


def test_an_efficiency_refuses_histograms_that_cannot_be_one():
    total, passed = Histogram.book("all", (2, 0, 2)), Histogram.book("good", (2, 0, 2))
    passed.fill([0.5, 5.0])
    with pytest.raises(ValueError, match="'good' holds more than 'all' in 2 bins"):
        Efficiency.from_histograms(passed, total)
    with pytest.raises(ValueError, match="are binned differently"):
        Efficiency.from_histograms(Histogram.book("x", (3, 0, 2)), total)
    with pytest.raises(TypeError, match="of kind 'MEAN' is not one"):
        Efficiency.from_histograms(Profile.book("p", (2, 0, 2)), total)
    with pytest.raises(TypeError, match="a list of kind None"):
        Efficiency.from_histograms([], total)


def test_a_weighted_float_efficiency_is_judged_with_a_float_tolerance():
    total = Histogram.book("t", (1, 0, 1), kind="F")
    total.fill(0.5, weight=1.000001)  # the square is not the weight, but near enough for a float
    eff = Efficiency.from_histograms(total.copy(), total)
    assert not eff.members["TNamed"]["fBits"] & USE_WEIGHTS


def test_members_put_together_by_hand_fill_as_well_as_ones_read():
    h = Histogram.book("h", (2, 0, 2))
    h.members["TArrayD"] = array.array("d", [0, 1, 2, 0])
    h.members["TH1"]["fSumw2"] = [0.0, 1.0, 2.0, 0.0]
    h.members["TH1"]["fEntries"] = 3.0
    h.fill(0.5, weight=2.0)
    assert isinstance(h.members["TArrayD"], np.ndarray)
    assert h.values().tolist() == [3, 2] and h.variances().tolist() == [5, 2]
