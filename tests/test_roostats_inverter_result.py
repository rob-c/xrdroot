"""HypoTestInverterResult on points given by hand: its accessors, merges and limit searches.

Asymptotic-style results - p-values alone - at chosen values of ``mu`` let
each branch of the interpolation be reached: a curve that never crosses, one
that peaks inside the scan, too few points, a point asked for that is not
there. The numbers follow from the straight lines between the points.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit.messages import service
from xrdroot.roostats import inverterlimits as limits
from xrdroot.roostats.hypotest import HypoTestResult
from xrdroot.roostats.inverterresult import HypoTestInverterResult


@pytest.fixture(autouse=True)
def _quiet() -> Any:
    service().reset()
    yield
    service().reset()


def scan(points: dict[float, float], cls: bool = False) -> HypoTestInverterResult:
    """A result whose ``CLs+b`` at each ``mu`` is the value given - ``CLb`` one half."""
    mu = ROOT.RooRealVar("mu", "mu", 1, 0, 10)
    made = HypoTestInverterResult("r", mu, 0.95)
    made.UseCLs(cls)
    for x, y in points.items():
        made.Add(x, point(y))
    return made


def point(y: float) -> HypoTestResult:
    """A test with the background as alternate, as the inverter makes its points'."""
    made = HypoTestResult("p", y, 0.5)
    made.SetBackgroundAsAlt(True)
    return made


def test_the_accessors_give_each_points_numbers_and_refuse_a_missing_one(capsys: Any) -> None:
    r = scan({1.0: 0.4, 2.0: 0.2})
    assert (r.ArraySize(), r.GetXValue(1), r.GetYValue(1), r.CLsplusb(1)) == (2, 2.0, 0.2, 0.2)
    assert (r.CLb(0), r.CLs(0), r.GetLastXValue(), r.GetLastYValue()) == (0.5, 0.8, 2.0, 0.2)
    assert (r.CLbError(0), r.CLsplusbError(0), r.CLsError(0), r.GetYError(0)) == (0, 0, 0, 0)
    assert (r.GetLastYError(), r.GetLastResult().NullPValue()) == (0.0, 0.2)
    assert (r.GetXValue(5), r.GetYValue(5), r.GetResult(-1)) == (-999.0, -999.0, None)
    assert "Problem: You are asking for an impossible array index value" in (
        capsys.readouterr().out)  # fmt: skip
    assert (r.GetBackgroundTestStatDist(7), r.GetSignalAndBackgroundTestStatDist(7)) == (None, None)
    r.SetTestSize(0.1)
    r.SetCLsCleanupThreshold(0.01)
    r.SetInterpolationOption(r.kSpline)
    assert (r.ConfidenceLevel(), r._cleanup, r.GetInterpolationOption()) == (0.9, 0.01, 1)
    assert (r.IsOneSided(), r.IsTwoSided()) == (True, False)


def test_a_point_added_again_is_merged_and_another_scan_merged_in(capsys: Any) -> None:
    r = scan({1.0: 0.4})
    r.Add(1.0 + 1e-14, point(0.3))
    assert r.ArraySize() == 1 and r.FindIndex(1.0) == 0 and r.FindIndex(3.0) == -1
    other = scan({1.0: 0.2, 5.0: 0.01})
    other.SetName("o")
    assert r.Add(other) and r.ArraySize() == 2
    assert r.Add(scan({}))
    out = capsys.readouterr().out
    assert "HypoTestInverterResult::Add - merging result from o in r" in out
    assert "HypoTestInverterResult::Add  - new number of points is 2" in out
    copy = r.Clone("c")
    assert copy.GetName() == "c" and copy.ArraySize() == 2 and copy._results[0] is not r._results[0]


def test_the_straight_lines_are_tgraphs_eval_extrapolated_at_the_ends() -> None:
    xs, ys = [1.0, 2.0, 4.0], [0.4, 0.2, 0.1]
    assert limits.graph_eval([], [], 1.0) == 0.0 and limits.graph_eval([1.0], [3.0], 9) == 3.0
    assert limits.graph_eval(xs, ys, 2.0) == 0.2
    assert limits.graph_eval(xs, ys, 3.0) == pytest.approx(0.15)
    assert limits.graph_eval(xs, ys, 0.0) == pytest.approx(0.6)
    assert limits.graph_eval(xs, ys, 6.0) == pytest.approx(0.0)
    assert limits.graph_eval([1.0, 1.0], [0.3, 0.2], 2.0) == 0.2


def test_the_upper_limit_crosses_the_size_and_the_lower_is_the_range_below_the_peak() -> None:
    r = scan({1.0: 0.5, 2.0: 0.1, 3.0: 0.01})
    assert r.UpperLimit() == pytest.approx(2.0 + 0.05 / 0.09, rel=1e-5)
    assert r.LowerLimit() == 0.0
    peaked = scan({1.0: 0.02, 2.0: 0.5, 3.0: 0.01})
    assert peaked.LowerLimit() == pytest.approx(1.0 + 0.03 / 0.48, rel=1e-5)
    assert peaked.UpperLimit() == pytest.approx(2.0 + 0.45 / 0.49, rel=1e-5)
    assert scan({1.0: 0.5, 2.0: 0.9}).UpperLimit() == 10.0
    high = scan({1.0: 0.9, 2.0: 0.5})
    assert (high.LowerLimit(), high.UpperLimit()) == (0.0, 10.0)


def test_a_curve_below_the_size_is_searched_on_the_side_of_its_peak() -> None:
    early = scan({1.0: 0.04, 2.0: 0.03, 3.0: 0.02})
    assert early.UpperLimit() == 0.0 and early._lower == 0.0
    late = scan({1.0: 0.02, 2.0: 0.03, 3.0: 0.04})
    assert late.LowerLimit() == 10.0 and late._upper == 10.0


def test_too_few_points_give_the_ends_of_the_range(capsys: Any) -> None:
    one = scan({1.0: 0.2})
    assert (one.UpperLimit(), one.LowerLimit()) == (10.0, 0.0)
    assert ("HypoTestInverterResult::FindInterpolatedLimit - not enough points to get the "
            "inverted interval - return 0") in capsys.readouterr().out
    assert one.UpperLimitEstimatedError() == 0.0
    assert HypoTestInverterResult("e").LowerLimitEstimatedError() == 0.0
    xs, ys = [1.0], [0.3]
    assert limits.graph_x(one, xs, ys, 0.05, False)[0] == 0.3
    assert limits.graph_x(one, [], [], 0.05, False)[0] == 0.0
    assert "GetGraphX - need at least 2 points for interpolation (n=1)" in capsys.readouterr().out


def test_the_nearest_point_is_found_by_error_or_by_position() -> None:
    r = scan({1.0: 0.5, 2.0: 0.1, 3.0: 0.01})
    assert limits.closest_point_index(r, 0.05) == 2
    assert [limits.closest_point_index(r, 0.05, m, 2.5) for m in (1, 2, 3)] == [2, 1, 2]
    assert limits.closest_point_index(r, 0.05, 1, 0.5) == 0
    assert limits.closest_point_index(r, 0.05, 1, 3.5) == 2
    r._interpolate = [False, False]
    assert (r.UpperLimit(), r.LowerLimit()) == (3.0, 3.0)
    r._fitted = [True, True]
    assert r.UpperLimit() == 3.0


def test_a_spline_is_refused_and_a_failed_root_is_infinite(capsys: Any) -> None:
    from xrdroot.errors import UnsupportedFeatureError

    r = scan({1.0: 0.5, 2.0: 0.1, 3.0: 0.01})
    r.SetInterpolationOption(r.kSpline)
    with pytest.raises(UnsupportedFeatureError, match="spline option is not here yet"):
        r.UpperLimit()
    flat = scan({1.0: 0.5, 2.0: 0.05, 3.0: 0.05})
    assert limits.graph_x(flat, [1.0, 2.0, 3.0], [0.5, 0.01, 0.06], 0.05, False, 1.0, 1.5)[0] == (
        math.inf)  # fmt: skip
    assert "HypoTestInverterResult - interpolation failed for interval [1,1.5 ]" in (
        capsys.readouterr().out)  # fmt: skip


def test_asymptotic_points_give_expected_limits_from_their_bands(capsys: Any) -> None:
    """No toys: each point's expected p-values from -5 to 5 sigma, and the limit of each."""
    r = scan({1.0: 0.5, 2.0: 0.1, 3.0: 0.01}, cls=True)
    dist = r.GetExpectedPValueDist(0)
    assert dist.GetName() == "Asymptotic expected values" and dist.GetSize() == 11
    median = r.GetExpectedUpperLimit(0)
    assert 1.0 < median < 3.0 and r.GetExpectedUpperLimit(-1) < median
    assert r.GetUpperLimitDistribution().GetSize() == 11
    assert "set a minimum size of 10" not in capsys.readouterr().out
    assert scan({}).GetExpectedUpperLimit(0) == 0.0 and scan({}).GetExpectedLowerLimit() == 1.0
    assert scan({1.0: 0.5}).GetUpperLimitDistribution() is None
    assert "GetLimitDistribution not  enough points -  return 0" in capsys.readouterr().out
    assert scan({1.0: 0.5}).GetExpectedUpperLimit(0) == 0.0


def test_an_expected_p_value_that_cannot_be_found_leaves_its_point_out(capsys: Any) -> None:
    r = scan({1.0: 1.0, 2.0: 0.1, 3.0: 0.01}, cls=True)
    r._two_sided = True
    assert r.GetExpectedPValueDist(0) is None
    r._results[0]._null = r._results[0]._alt = "toys"  # read as toys, but none to read
    r._results[0].GetNullDistribution = lambda: None
    r._results[0].GetAltDistribution = lambda: "alt"
    r._results[1] = r._results[0]
    assert r.GetExpectedUpperLimit(0, "P") == 0.0
    out = capsys.readouterr().out
    assert "cannot compute expected p value distribution for point, x = 1 skip it" in out
    assert "HypoTestInverterResult - cannot compute limits , not enough points, n =  1" in out


def test_the_quantiles_are_tmaths_of_types_seven_and_one() -> None:
    values = [3.0, 1.0, 2.0, 4.0]
    assert limits.quantile(values, 0.5) == 2.5 and limits.quantile(values, 0.0) == 1.0
    assert limits.quantile(values, 1.0) == 4.0
    assert [limits.quantile(values, p, 1) for p in (0.25, 0.3, 1.0)] == [1.0, 2.0, 4.0]
    assert limits.quantile([5.0], 0.0, 1) == 5.0


def test_the_exclusion_cleanup_drops_points_the_curve_should_not_have(capsys: Any) -> None:
    """A CLs that rises again, or is negative, goes - and the point after it goes unchecked, as
    RooStats' loop steps past it."""
    r = scan({0.0: 0.5, 1.0: 0.25, 2.0: 0.3, 3.0: 0.2, 4.0: 0.15, 5.0: 0.1}, cls=True)
    removed = r.ExclusionCleanup()
    assert removed == 1 and r._x == [0.0, 1.0, 3.0, 4.0, 5.0]
    assert not r._fitted[0]
    r = scan({0.0: 0.5, 1.0: 0.25}, cls=True)
    r._results[1] = point(-1.0)
    assert r.ExclusionCleanup() == 1
    assert scan({}).ExclusionCleanup() == 0
    two = scan({0.0: 1.0, 1.0: 0.25}, cls=True)
    two._two_sided = True
    assert two.ExclusionCleanup() == 0


def test_a_curve_crossing_twice_is_searched_again_on_the_near_side() -> None:
    """The whole range's root refined within the points on the limit's side - and, for a curve
    starting above the target, from the parameter's lower end."""
    xs, ys = [1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 0.01, 0.01, 0.01, 0.5]
    r = scan(dict(zip(xs, ys, strict=False)))
    lower = limits.graph_x(r, xs, ys, 0.05, True)[0]
    upper = limits.graph_x(r, xs, ys, 0.05, False)[0]
    assert lower == pytest.approx(1.0 + 0.45 / 0.49, rel=1e-5)
    assert upper == pytest.approx(4.0 + 0.04 / 0.49, rel=1e-5)
    high = [0.5, 0.2, 0.01]
    assert limits.graph_x(r, [1.0, 2.0, 3.0], high, 0.05, True)[1] == 0.0


def test_the_band_quantiles_of_toys_are_tmaths() -> None:
    values = [float(i) for i in range(11)]
    assert limits._band_quantiles(values, False)[2] == 5.0
    assert limits._band_quantiles(values, True) == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_a_point_with_no_expected_band_is_left_out_of_the_limit_distribution() -> None:
    r = scan({1.0: 1.0, 2.0: 0.1, 3.0: 0.01}, cls=True)
    r._two_sided = True
    assert r.GetUpperLimitDistribution().GetSize() == 10
