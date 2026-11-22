"""``EIOFeatures`` and ``TBasket``'s bits: which IO features a file of ROOT 6.40 records.

ROOT declares the enum twice - ``ROOT::EIOFeatures``, the 6.34 spelling 6.40 keeps but no
longer sets, and ``ROOT::Experimental::EIOFeatures``, the one a current writer sets - and a
basket's header has ``TBasket::kIOBitCount`` bits for them. The numbers are ROOT 6.40's own,
read off its ``TIOFeatures.IOBits`` test: ``kIOBitCount`` is 1, so the one supported bit is
set and the 6.34 enum's ``kSupported`` is 0.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from xrdroot.pyroot.core import tiofeatures
from xrdroot.pyroot.core.tiofeatures import (
    EIOFeatures,
    EIOUnsupportedFeatures,
    ExperimentalEIOFeatures,
)


def test_the_bits_root_64_records_are_the_ones_its_own_test_names():
    expect(
        (ROOT.EIOFeatures.kSupported, 0),
        (ROOT.Experimental.EIOFeatures.kSupported, 1),
        (ROOT.Experimental.EIOUnsupportedFeatures.kUnsupported, 0),
        (ROOT.TBasket.kIOBitCount, 1),
        (ROOT.TBasket.EIOBits.kSupported, 1),
    )


def test_the_namespace_has_only_the_two_enums_root_has_at_its_top():
    """``ROOT::EIOUnsupportedFeatures`` and ``ROOT::ExperimentalEIOFeatures`` are not ROOT's
    names - measured against ROOT 6.40, which has neither - so they are not the namespace's."""
    with pytest.raises(AttributeError):
        ROOT.EIOUnsupportedFeatures  # noqa: B018 - the refusal is the assertion
    expect(
        (tiofeatures.__all__, ["EIOFeatures"]),
        (EIOUnsupportedFeatures.kUnsupported, 0),
        (ExperimentalEIOFeatures.kSupported, 1),
    )


def test_the_two_enums_of_that_name_are_different_ones():
    """``ROOT::EIOFeatures`` is not ``ROOT::Experimental::EIOFeatures``: 6.40 writes the
    second and keeps the first for reading a file of 6.34."""
    expect(
        (EIOFeatures is ExperimentalEIOFeatures, False),
        (ROOT.Experimental.EIOFeatures is EIOFeatures, False),
    )
