"""``TH1::FindFirstBinAbove`` and ``FindLastBinAbove``, by ROOT's own search.

ROOT's answer is a *bin number along one axis*, found by walking that axis's
bins in one direction and, at each of them, every bin of the other axes: a
histogram of three axes answers ``4`` for the x bin whose content passes the
threshold anywhere in its plane. The numbers below are worked out from
ROOT's ``hist/hist/src/TH1.cxx``, and ROOT's skipped ``firstbin``/``lastbin``
rules are the ones ``gtest/hist/hist/test_TH1_FindFirstBinAbove`` asserts.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh

#: What the two searches are asked about, in ``TH1``'s own terms: a *strictly* greater
#: threshold, so a bin exactly at it is not above.
THRESHOLD = 100.0


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def four_bins():
    """A ``TH1D`` of four bins, whose contents the tests set by hand."""
    return ROOT.TH1D("h1", "h1", 4, 0, 4)


def spike_3d():
    """A ``TH3D`` of 10 bins an axis whose only content above the threshold is set by hand:
    ``2 * THRESHOLD`` at ``(4, 5, 6)`` and ``1.5 * THRESHOLD`` at ``(14, 15, 16)``, as the
    gtest fills it, with every other bin filled below the threshold."""
    h = ROOT.TH3D("h3", "h3", 20, -5, 5, 20, -5, 5, 20, -5, 5)
    rng = ROOT.TRandomMixMax()
    for _ in range(1000):
        h.Fill(rng.Gaus(0, 2), rng.Gaus(-3, 1.5), rng.Gaus(2, 2.5))
    h.SetBinContent(4, 5, 6, 2.0 * THRESHOLD)
    h.SetBinContent(14, 15, 16, 1.5 * THRESHOLD)
    return h


def test_find_first_bin_above_walks_the_named_axis_only():
    h = spike_3d()
    expect(
        (h.FindFirstBinAbove(THRESHOLD, 1, 1, 10), 4),
        (h.FindFirstBinAbove(THRESHOLD, 1, 11, 20), 14),
        (h.FindFirstBinAbove(THRESHOLD, 2), 5),
        (h.FindFirstBinAbove(THRESHOLD, 3, 11, 20), 16),
    )


def test_find_last_bin_above_is_the_same_search_from_the_top():
    h = spike_3d()
    expect(
        (h.FindLastBinAbove(THRESHOLD, 1), 14),
        (h.FindLastBinAbove(THRESHOLD, 2, 1, 10), 5),
        (h.FindLastBinAbove(THRESHOLD, 3), 16),
        (h.FindLastBinAbove(THRESHOLD, 3, 1, 10), 6),
    )


def test_a_projection_answers_too():
    h = spike_3d()
    h2 = h.Project3D("ZY")
    h1 = h.Project3D("X")
    expect(
        (h2.FindFirstBinAbove(THRESHOLD, 1), 5),
        (h2.FindFirstBinAbove(THRESHOLD, 1, 13, 18), 15),
        (h2.FindLastBinAbove(THRESHOLD, 2), 16),
        (h1.FindFirstBinAbove(THRESHOLD, 1, 14, 16), 14),
        (h1.FindLastBinAbove(THRESHOLD, 1, 3, 8), 4),
    )


def test_no_bin_above_the_threshold_is_minus_one():
    h = four_bins()
    h.SetBinContent(2, THRESHOLD)
    expect(
        (h.FindFirstBinAbove(THRESHOLD), -1),
        (h.FindLastBinAbove(THRESHOLD), -1),
        (h.FindFirstBinAbove(THRESHOLD - 1), 2),
        (h.FindFirstBinAbove(THRESHOLD, 1, 3, 4), -1),
        (h.FindLastBinAbove(THRESHOLD, 1, 0, 1), -1),
    )


def test_a_first_bin_below_one_is_the_first_bin():
    h = four_bins()
    h.SetBinContent(1, 200.0)
    expect(
        (h.FindFirstBinAbove(THRESHOLD, 1, 0), 1),
        (h.FindFirstBinAbove(THRESHOLD, 1, -7), 1),
        (h.FindLastBinAbove(THRESHOLD, 1, 0), 1),
        # A *last* bin below one is not the last bin, as it is in ROOT: the search is empty.
        (h.FindFirstBinAbove(THRESHOLD, 1, 1, 0), -1),
    )


def test_a_last_bin_past_the_end_of_the_axis_is_the_axis_end():
    h = four_bins()
    h.SetBinContent(4, 200.0)
    expect(
        (h.FindFirstBinAbove(THRESHOLD, 1, 1, 99), 4),
        (h.FindLastBinAbove(THRESHOLD, 1, 1, 99), 4),
        (h.FindFirstBinAbove(THRESHOLD, 1, 1, -1), 4),
    )


def test_an_axis_the_histogram_has_not_is_x_with_root_warning(capsys):
    h = four_bins()
    h.SetBinContent(3, 200.0)
    expect(
        (h.FindFirstBinAbove(THRESHOLD, 2), 3),
        (h.FindLastBinAbove(THRESHOLD, 0), 3),
        (h.FindFirstBinAbove(THRESHOLD, 4), 3),
    )
    printed = capsys.readouterr().err
    expect(
        (printed.count("Warning in <TH1::FindFirstBinAbove>: Invalid axis number : "), 3),
        # ROOT's FindLastBinAbove names FindFirstBinAbove in its own message.
        ("Invalid axis number : 0, axis x assumed" in printed, True),
    )


def test_a_two_dimensional_histogram_walks_the_other_axis_at_each_bin():
    h = ROOT.TH2D("h2", "h2", 4, 0, 4, 3, 0, 3)
    h.SetBinContent(3, 2, 200.0)  # x bin 3, y bin 2
    expect(
        (h.FindFirstBinAbove(THRESHOLD, 1), 3),
        (h.FindFirstBinAbove(THRESHOLD, 2), 2),
        (h.FindLastBinAbove(THRESHOLD, 2), 2),
        (h.FindFirstBinAbove(THRESHOLD, 2, 3, 3), -1),
    )


def test_a_bin_exactly_at_the_threshold_is_not_above_it():
    h = four_bins()
    h.SetBinContent(2, THRESHOLD)
    h.SetBinContent(3, np.nextafter(THRESHOLD, np.inf))
    expect(
        (h.FindFirstBinAbove(THRESHOLD), 3),
        (h.FindLastBinAbove(THRESHOLD), 3),
    )
