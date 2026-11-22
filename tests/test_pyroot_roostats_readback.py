"""A RooStats result read from a ROOT file, answering the numbers ROOT answers for it.

``hypotestinv-6.21.root`` is ROOT's own input for
``roofit/roostats/test/testHypoTestInvResult.cxx``: one
``RooStats::HypoTestInverterResult`` of six scanned values, a
``RooStats::HypoTestResult`` at each of them, and the sampling distributions
the p-values were counted from. Every number below is what ROOT 6.40.04
answers for that file - the gtest's seven expectations among them - so a
member left unread, or a p-value reckoned again instead of taken from the
file, fails here rather than at the last digit somewhere else.
"""

from __future__ import annotations

import math
import pathlib

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from refmachine import roots
from xrdroot.pyroot.roostats.readback import (
    hypotest_from,
    inverter_result_from,
    sampling_distribution_from,
)
from xrdroot.roostats.inverterresult import HypoTestInverterResult as EngineResult

#: ROOT's file of one scan, written by ROOT 6.21 beside its own test of reading it.
SCAN = pathlib.Path(__file__).parent / "data" / "hypotestinv-6.21.root"


def test_a_stored_scan_answers_the_upper_limit_and_its_error_root_answers() -> None:
    result = ROOT.TFile(str(SCAN)).GetObject("result")
    expect(
        (type(result), EngineResult),
        (result.GetName(), "result_SigXsecOverSM"),
        (result.ArraySize(), 6),
        (result.ConfidenceLevel(), 0.95),
        (result.UpperLimit(), roots(2.4613464754748358, rel=1e-14)),
        # An error ROOT fits for, so it is the gtest's 1e-5 and not a last-bit comparison.
        (result.UpperLimitEstimatedError(), pytest.approx(0.059684301, abs=1e-5)),
    )


def test_a_stored_scan_answers_the_expected_limits_from_the_toys_it_kept() -> None:
    result = ROOT.TFile(str(SCAN)).GetObject("result")
    expect(
        (result.GetExpectedUpperLimit(0), roots(1.6098842702856895, rel=1e-13)),
        (result.GetExpectedUpperLimit(1), roots(2.0901937952513951, rel=1e-13)),
        (result.GetExpectedUpperLimit(2), roots(2.794445618740776, rel=1e-13)),
    )


def test_a_point_of_a_stored_scan_keeps_the_p_values_the_file_stored() -> None:
    point = ROOT.TFile(str(SCAN)).GetObject("result").GetResult(1)
    expect(
        (point.GetName(), "HypoTestCalculator_result"),
        (point.NullPValue(), 0.747),
        (point.AlternatePValue(), 0.912),
        (point.GetBackGroundIsAlt(), True),
        (point.GetNullDistribution().GetSize(), 1000),
        (point.GetAltDistribution().GetSize(), 500),
        (point.CLs(), roots(0.81907894736842102, rel=1e-15)),
        (point.CLsError(), roots(0.018886307765386367, rel=1e-13)),
    )


def test_a_sampling_distribution_keeps_its_values_its_weights_and_its_variable() -> None:
    made = sampling_distribution_from(
        {
            "TNamed": {"fName": "dist", "fTitle": "a distribution"},
            "fSamplingDist": [1.5, 2.5],
            "fSampleWeights": [1.0, 0.5],
            "fVarName": "TS0",
        }
    )
    empty = sampling_distribution_from({})
    expect(
        (made.GetName(), "dist"),
        (made.GetTitle(), "a distribution"),
        (made.GetSamplingDistribution(), [1.5, 2.5]),
        (made.GetSampleWeights(), [1.0, 0.5]),
        (made.GetVarName(), "TS0"),
        (empty.GetSize(), 0),
        (empty.GetVarName(), ""),
    )


def test_a_test_of_no_distributions_stored_has_no_distributions_to_reckon_from() -> None:
    made = hypotest_from(
        {"fNullPValue": 0.25, "fAlternatePValue": 0.5, "fNullDistr": None, "fAltDistr": None}
    )
    bare = hypotest_from({})
    expect(
        (made.NullPValue(), 0.25),
        (made.AlternatePValue(), 0.5),
        (made.GetNullDistribution(), None),
        (made.GetAltDistribution(), None),
        (made.CLs(), 2.0),
        (made.CLsError(), 0.0),
        (bare.GetName(), ""),
        (bare.NullPValueError(), 0.0),
        (bare.HasTestStatisticData(), False),
        (bare.GetPValueIsRightTail(), True),
        (bare.GetBackGroundIsAlt(), False),
    )


def test_a_scan_with_no_members_at_all_has_the_settings_root_makes_one_with() -> None:
    made = inverter_result_from({})
    expect(
        (made.ArraySize(), 0),
        (made.ConfidenceLevel(), 0.95),
        (made.GetInterpolationOption(), EngineResult.kLinear),
        (made.IsOneSided(), True),
        (made.IsTwoSided(), False),
        (made.UpperLimit(), math.inf),
        (made.UpperLimitEstimatedError(), 0.0),
    )
