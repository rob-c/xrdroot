"""A colour or style given by its name, as PyROOT takes one: ``SetLineColor("kBlue")``."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.core.colors import named


def test_a_name_plus_or_minus_a_number_is_the_enumerator_moved_by_it() -> None:
    """``kRed+2`` is 634 and ``kAzure - 9`` 851, as ROOT's headers make them."""
    assert named("kBlue") == 600
    assert named("kRed+2") == 634
    assert named(" kAzure - 9 ") == 851
    assert named("kDashed") == 2
    assert named("kFullCircle") == 20


def test_a_name_ROOT_has_no_enumerator_for_is_refused() -> None:
    """A misspelt colour is an error, not black."""
    with pytest.raises(ValueError, match="not the name of one of ROOT's colours or styles"):
        named("kBlu")
    with pytest.raises(ValueError, match="not the name"):
        named("blue")


def test_a_histogram_takes_its_line_colour_by_name() -> None:
    """rf305 sets a TH2's colour as ``SetLineColor("kBlue")``; ROOT's PyROOT takes it."""
    h = ROOT.TH2D("h_named_colour", "h", 2, 0, 1, 2, 0, 1)
    h.SetLineColor("kBlue")
    assert h.GetLineColor() == 600
