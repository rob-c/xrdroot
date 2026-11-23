"""A profile filled an entry at a time, and the poly search its fill leans on.

ROOT's own ``hist/hist/test/test_tprofile2poly.cxx`` fills a ``TProfile2D`` and a
``TProfile2Poly`` a million times over from a loop, and the two histograms must
hold the same doubles to 1e-12. A loop that calls ``Fill`` costs the lookup of
every array and moment it writes, so a profile keeps what its fills go into
beside the wrapper (``_fill_target``), and these tests say that the fast path is
the engine's to the last bit - in every branch of it: the value range, the flow
bins, the squared weights that a first weight other than one starts, and a
profile whose arrays have been replaced since the target was read.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from xrdroot.polybins import inside


def members(wrapper: Any) -> dict[str, Any]:
    """Every member of a wrapped histogram, flattened to ``"TH1D.fEntries"`` names."""
    found: dict[str, Any] = {}

    def walk(row: dict[str, Any], above: str) -> None:
        for name, value in row.items():
            if isinstance(value, dict):
                walk(value, f"{above}{name}.")
            else:
                found[f"{above}{name}"] = value

    walk(wrapper._xrd.members, "")
    return found


def same_as_the_engine(fast: Any, plain: Any) -> list[str]:
    """The members the two histograms do not share, bit for bit."""
    mine, theirs = members(fast), members(plain)
    return [name for name, value in mine.items() if "fName" not in name and "fTitle" not in name
            and np.asarray(value).tobytes() != np.asarray(theirs[name]).tobytes()]  # fmt: skip


def entries(count: int) -> list[tuple[float, float, float, float]]:
    """``count`` entries of a ``(x, y, value, weight)`` table, deterministic and mixed."""
    made = []
    for at in range(count):
        x = math.sin(at * 0.37) * 3.0
        y = math.cos(at * 0.11) * 2.5
        value = math.sin(at * 1.7) * 20.0 + 20.0
        weight = 1.0 + (at % 7) * 0.5 - (at % 3)
        made.append((x, y, value, weight))
    return made


def pair(cls: type, args: tuple[Any, ...]) -> tuple[Any, Any]:
    """A histogram filled through the wrapper, and the same one filled through the engine."""
    return cls("fast", "fast", *args), cls("plain", "plain", *args)


@pytest.mark.parametrize("cls,args,plane", [
    (ROOT.TProfile, (10, -3.0, 3.0), lambda x, y: ([x], y)),
    (ROOT.TProfile2D, (10, -3.0, 3.0, 10, -3.0, 3.0), lambda x, y: ([x, y], 20.0)),
    (ROOT.TProfile3D, (4, -3.0, 3.0, 4, -3.0, 3.0, 4, -3.0, 3.0), lambda x, y: ([x, y, x], 20.0)),
])  # fmt: skip
def test_a_filled_profile_holds_the_doubles_the_engine_holds(cls: Any, args: Any,
                                                            plane: Any) -> None:  # fmt: skip
    fast, plain = pair(cls, args)
    for x, y, _value, weight in entries(500):
        coordinates, averaged = plane(x, y)
        given = [*coordinates, averaged]
        fast.Fill(*coordinates, averaged, weight)
        plain._xrd.fill_one(given, weight)
    expect((same_as_the_engine(fast, plain), []), (fast.GetEntries(), plain.GetEntries()))


def test_an_unweighted_loop_leaves_no_squared_weights_to_keep() -> None:
    fast, plain = pair(ROOT.TProfile2D, (10, -3.0, 3.0, 10, -3.0, 3.0))
    for x, y, value, _ in entries(200):
        fast.Fill(x, y, value)
        plain._xrd.fill_one([x, y, value], None)
    keep = np.asarray(fast._xrd.members["fBinSumw2"])
    expect((keep.size, 0), (same_as_the_engine(fast, plain), []))


def test_the_first_weight_that_is_not_one_starts_the_squared_weights() -> None:
    fast, plain = pair(ROOT.TProfile2D, (10, -3.0, 3.0, 10, -3.0, 3.0))
    fast.Fill(0.5, 0.5, 3.0)  # a weight of one: nothing to keep but the value
    plain._xrd.fill_one([0.5, 0.5, 3.0], None)
    expect((np.asarray(fast._xrd.members["fBinSumw2"]).size, 0))
    fast.Fill(0.5, 0.5, 4.0, 2.5)
    plain._xrd.fill_one([0.5, 0.5, 4.0], 2.5)
    kept = np.asarray(fast._xrd.members["fBinSumw2"])
    expect((kept.size, 144), (float(kept.sum()), 6.25 + 1.0), (same_as_the_engine(fast, plain), []))


def test_a_value_outside_the_range_averaged_is_counted_nowhere() -> None:
    fast, plain = pair(ROOT.TProfile2D, (10, -3.0, 3.0, 10, -3.0, 3.0, 0.0, 10.0))
    expect((fast.Fill(0.5, 0.5, 12.0), 78))  # outside 0..10: ROOT drops it
    plain._xrd.fill_one([0.5, 0.5, 12.0], None)
    expect((fast.GetEntries(), 0.0), (fast.GetBinContent(fast.FindBin(0.5, 0.5)), 0.0))
    expect((fast.Fill(0.5, 0.5, 2.0), 78), (fast.GetEntries(), 1.0))
    plain._xrd.fill_one([0.5, 0.5, 2.0], None)
    expect((same_as_the_engine(fast, plain), []))


def test_a_fill_outside_the_axes_counts_an_entry_and_no_moments() -> None:
    fast, plain = pair(ROOT.TProfile2D, (10, -3.0, 3.0, 10, -3.0, 3.0))
    expect((fast.Fill(9.0, 9.0, 2.0), 143))  # the region above and to the right
    plain._xrd.fill_one([9.0, 9.0, 2.0], None)
    moments = fast._xrd._moment_homes()
    expect((moments["fTsumw"]["fTsumw"], 0.0), (fast.GetEntries(), 1.0))
    expect((same_as_the_engine(fast, plain), []))


def test_a_profile_that_starts_stopping_its_squares_is_read_afresh() -> None:
    """``Sumw2(false)`` replaces ``fBinSumw2``: the next fill must land in the new one."""
    fast, plain = pair(ROOT.TProfile2D, (10, -3.0, 3.0, 10, -3.0, 3.0))
    fast.Fill(0.5, 0.5, 3.0, 2.0)
    plain._xrd.fill_one([0.5, 0.5, 3.0], 2.0)
    fast.Sumw2(False)
    plain._xrd._drop_sumw2()
    fast.Fill(0.5, 0.5, 4.0)  # a weight of one: they stay dropped, and the fill still lands
    plain._xrd.fill_one([0.5, 0.5, 4.0], None)
    expect((np.asarray(fast._xrd.members["fBinSumw2"]).size, 0))
    expect((fast.GetBinEntries(fast.FindBin(0.5, 0.5)), 3.0))  # 2 and then 1
    expect((same_as_the_engine(fast, plain), []))


def test_a_reset_profile_fills_from_nothing_again() -> None:
    fast, plain = pair(ROOT.TProfile, (10, -3.0, 3.0))
    for x, _y, value, weight in entries(50):
        fast.Fill(x, value, weight)
        plain._xrd.fill_one([x, value], weight)
    fast.Reset()
    plain._xrd.reset()
    for x, _y, value, weight in entries(50):
        fast.Fill(x, value, weight)
        plain._xrd.fill_one([x, value], weight)
    expect((same_as_the_engine(fast, plain), []), (fast.GetEntries(), 50.0))


def test_a_TProfile2D_counts_the_cells_of_fBinEntries_as_its_bins() -> None:
    found = ROOT.TProfile2D("h", "h", 1, 0.0, 1.0, 1, 0.0, 1.0)
    found.SetCanExtend(ROOT.TH1.kAllAxes)
    made = found.Clone()
    found.Fill(0.5, 0.5, 0.0)
    made.Fill(1.5, 0.5, 0.0)
    listing = ROOT.TList()
    listing.Add(made)
    found.Merge(listing)
    expect((found.GetNumberOfBins(), 9.0), (found.GetEntries(), 2.0))
    expect((sum(found.GetBinEntries(at) for at in range(int(found.GetNumberOfBins()))), 2.0))


def scanned(bins: Any, x: float, y: float) -> int:
    """The bin a scan of every polygon in the order they were added finds, or -5 for none."""
    for at, polygons in enumerate(bins.polygons):
        if any(inside(px, py, x, y) for px, py in polygons):
            return at + 1
    return -5


def test_a_poly_bin_is_found_where_a_scan_of_every_bin_finds_it() -> None:
    """The search grid and its boxes choose no bin a scan from the first would not."""
    poly = ROOT.TH2Poly("poly", "poly", -10.0, 10.0, -10.0, 10.0)
    for at in range(-10, 10, 2):
        poly.AddBin(at, at + 4, at + 2, at + 6)  # rectangles off the diagonal, overlapping
    bins = poly._xrd_bins
    for x in np.arange(-9.5, 9.5, 0.7):
        for y in np.arange(-9.5, 9.5, 0.7):
            expect((bins.find(float(x), float(y)), scanned(bins, float(x), float(y))))


def test_a_poly_fill_outside_the_axes_goes_to_the_region_round_them() -> None:
    poly = ROOT.TProfile2Poly("poly", "poly", 0.0, 1.0, 0.0, 1.0)
    poly.AddBin(0.0, 0.0, 1.0, 1.0)
    expect((poly.FindBin(0.5, 0.5), 1), (poly.FindBin(-1.0, 0.5), -4),
           (poly.FindBin(0.5, -1.0), -8), (poly.FindBin(-1.0, -1.0), -7),
           (poly.FindBin(2.0, 2.0), -3), (poly.FindBin(0.5, 2.0), -2),
           (poly.FindBin(2.0, 0.5), -6))  # fmt: skip


def test_a_point_on_the_edge_of_a_box_is_the_bin_the_crossing_count_finds() -> None:
    """A box rejects only a point no crossing count can call inside, edges included."""
    poly = ROOT.TH2Poly("poly", "poly", 0.0, 1.0, 0.0, 1.0)
    poly.AddBin(0.0, 0.0, 0.5, 0.5)
    poly.AddBin(0.5, 0.5, 1.0, 1.0)
    bins = poly._xrd_bins
    for x, y in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.0, 0.5), (0.5, 0.0), (1.0, 0.5),
                 (0.25, 0.25), (0.75, 0.75)]:  # fmt: skip
        found = bins.find(x, y)
        expect((found, scanned(bins, x, y) if found > 0 or found == -5 else found))
