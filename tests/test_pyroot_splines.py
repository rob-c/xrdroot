"""``TSpline5``: ROOT's quintic spline, and the C function ``SaveAs`` writes of it."""

from __future__ import annotations

import math

import numpy as np
import pytest

import xrdroot.pyroot as ROOT

#: ``hist/test/test_spline.cxx``'s table: eleven samples of ``sqrt(2) x^2`` over ``0 .. 1``.
SAMPLES = 11
XS = [i * 0.1 for i in range(SAMPLES)]
YS = [math.sqrt(2.0) * x**2 for x in XS]
#: ROOT's own tolerance in that test.
TOLERANCE = 0.000001
#: The first derivative of the quadratic ``sqrt(2) x^2``, and its second divided by two.
SLOPE = math.sqrt(2.0) * 2.0
SECOND = math.sqrt(2.0)


def quadratic() -> ROOT.TSpline5:
    """The spline the ROOT test builds: evenly spread knots over 0 .. 1."""
    return ROOT.TSpline5("splinePrecisionTest", 0, 1, YS, SAMPLES)


def coefficients(spline: ROOT.TSpline5, i: int) -> list[float]:
    """``GetCoeff`` of knot ``i``, into seven cells as C++ passes them."""
    cells = [np.zeros(1) for _ in range(7)]
    spline.GetCoeff(i, *cells)
    return [float(cell[0]) for cell in cells]


def test_the_spline_gives_back_every_sample_of_the_quadratic_within_roots_tolerance():
    """The point of ``hist/test/test_spline.cxx``: a quintic spline through a quadratic's
    samples is that quadratic, to the 1e-6 the test allows."""
    spline = quadratic()
    for x, y in zip(XS, YS, strict=True):
        assert spline.Eval(x) == pytest.approx(y, abs=TOLERANCE)


def test_the_knots_carry_the_quadratic_s_derivatives():
    """``GetCoeff``: the value at each knot, its first derivative in ``b`` and half the second
    in ``c``, with ``d``, ``e`` and ``f`` left at the rounding of the fit."""
    spline = quadratic()
    for i, x in enumerate(XS):
        knot_x, knot_y, b, c, d, e, f = coefficients(spline, i)
        assert knot_x == x
        assert knot_y == YS[i]
        assert b == pytest.approx(SLOPE * x, abs=1e-11)
        assert c == pytest.approx(SECOND, abs=1e-10)
        assert max(abs(d), abs(e), abs(f)) < 1e-10


def test_the_derivative_is_the_quadratic_s_own():
    """``Derivative`` reads the same polynomial ``Eval`` reads."""
    spline = quadratic()
    for x in XS:
        assert spline.Derivative(x) == pytest.approx(SLOPE * x, abs=TOLERANCE)


def test_a_knot_is_handed_back_into_the_cells_a_macro_gives():
    """``GetKnot(i, x, y)``: ``x`` and ``y`` take the knot's abscissa and value."""
    y_into, x_into = np.zeros(1), np.zeros(1)
    quadratic().GetKnot(4, x_into, y_into)
    assert (float(x_into[0]), float(y_into[0])) == (XS[4], YS[4])


def test_the_table_it_was_built_from_is_reported_as_roots_does():
    spline = quadratic()
    assert (spline.GetNp(), spline.GetDelta()) == (SAMPLES, pytest.approx(0.1))
    assert (spline.GetXmin(), spline.GetXmax()) == (0.0, 1.0)
    assert spline.GetNpx() == 100
    spline.SetNpx(50)
    assert spline.GetNpx() == 50


def test_a_table_of_abscissas_is_searched_for_the_knot_a_value_falls_in():
    """Knots ROOT cannot divide into - unevenly spread ones - are found by a binary search,
    which is what an ``x`` between two of them exercises."""
    knots = [0.0, 1.0, 2.0, 5.0, 10.0]
    values = [math.sin(x) for x in knots]
    spline = ROOT.TSpline5("uneven", knots, values, len(knots))
    assert spline.GetDelta() == -1.0
    for x, y in zip(knots, values, strict=True):
        assert spline.Eval(x) == pytest.approx(y, abs=1e-12)
    assert spline.FindX(1.5) == 1
    assert spline.FindX(8.0) == 3


def test_outside_the_knots_the_end_polynomial_is_extrapolated():
    """``FindX`` hands back the first knot below and the last above, and ``Eval`` then runs
    that knot's polynomial - ROOT's own extrapolation, which it calls badly wrong at times."""
    spline = quadratic()
    first = coefficients(spline, 0)
    last = coefficients(spline, SAMPLES - 1)
    below, above = -0.5, 1.5
    assert spline.Eval(below) == pytest.approx(
        first[1] + (below - first[0]) * (first[2] + (below - first[0]) * first[3]), abs=1e-12
    )
    dx = above - last[0]
    assert spline.Eval(above) == pytest.approx(
        last[1] + dx * (last[2] + dx * (last[3] + dx * last[4])), abs=1e-12
    )


@pytest.mark.parametrize(("option", "size"), [("b1", 12), ("e1", 12), ("b1e1", 13)])
def test_a_first_derivative_option_pins_the_slope_on_the_knots_it_adds(option: str, size: int):
    """``b1``/``e1`` put one or two knots on the end abscissa carrying the first derivative
    asked for, and the spline still runs through every sample."""
    ends = [SLOPE * 0.0, SLOPE * 1.0, SECOND, SECOND]
    spline = ROOT.TSpline5("ended", 0, 1, YS, SAMPLES, option, *ends)
    assert spline.GetNp() == size
    for x, y in zip(XS, YS, strict=True):
        assert spline.Eval(x) == pytest.approx(y, abs=TOLERANCE)


@pytest.mark.parametrize(("option", "size", "shared", "at"), [
    ("b2", 13, (0, 1, 2), 0),
    ("b1b2", 13, (0, 1, 2), 0),
    ("e2", 13, (-3, -2, -1), -1),
    ("e1e2", 13, (-3, -2, -1), -1),
    ("b2e2", 15, (0, 1, 2), 0),
])
def test_a_second_derivative_option_pins_a_triple_knot(option: str, size: int,
                                                       shared: tuple[int, ...], at: int):
    """``b2``/``e2`` put three knots on the end abscissa, one of them carrying the second
    derivative asked for, which is why the end sample is the one that stays exactly right."""
    ends = [SLOPE * 0.0, SLOPE * 1.0, SECOND, SECOND]
    spline = ROOT.TSpline5("ended", 0, 1, YS, SAMPLES, option, *ends)
    assert spline.GetNp() == size
    assert [coefficients(spline, i)[0] for i in shared] == [XS[at]] * 3
    assert spline.Eval(XS[at]) == pytest.approx(YS[at], abs=TOLERANCE)


def test_a_triple_knot_without_a_slope_takes_roots_finite_difference():
    """With ``b2`` alone the slope at that end is the difference of the two knots next to it,
    which ROOT itself calls not recommended: the spline is then ROOT's own and not the
    quadratic, and 0.044508609829041847 at x = 0.2 is the value ROOT's ``Eval`` gives back."""
    ends = [SLOPE * 0.0, SLOPE * 1.0, SECOND, SECOND]
    spline = ROOT.TSpline5("ended", 0, 1, YS, SAMPLES, "b2", *ends)
    assert spline.Eval(0.2) == pytest.approx(0.044508609829041847, rel=1e-12)


def test_a_spline_of_three_knots_and_a_short_one_are_the_tables_they_are():
    """Two knots leave no interval to solve for, and three leave the linear system with one
    unknown: both are splines ROOT builds, and both interpolate their samples."""
    for knots, values in (([0.0, 1.0], [1.0, 3.0]), ([0.0, 1.0, 2.0], [0.0, 2.0, 1.0])):
        spline = ROOT.TSpline5("short", knots, values, len(knots))
        for x, y in zip(knots, values, strict=True):
            assert spline.Eval(x) == pytest.approx(y, abs=1e-12)


def test_four_knots_still_solve_the_system_though_they_need_no_back_substitution():
    knots, values = [0.0, 1.0, 2.0, 4.0], [0.0, 1.0, 4.0, 16.0]
    spline = ROOT.TSpline5("four", knots, values, len(knots))
    for x, y in zip(knots, values, strict=True):
        assert spline.Eval(x) == pytest.approx(y, abs=1e-12)


def test_setpoint_moves_a_knot_and_setpointcoeff_sets_its_coefficients():
    """Both leave the coefficients where ROOT leaves them, which is where they were set."""
    spline = quadratic()
    spline.SetPoint(2, 0.25, 7.0)
    assert coefficients(spline, 2)[:2] == [0.25, 7.0]
    spline.SetPoint(SAMPLES, 0.0, 0.0)
    spline.SetPointCoeff(1, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert coefficients(spline, 1)[2:] == [1.0, 2.0, 3.0, 4.0, 5.0]
    spline.SetPointCoeff(SAMPLES, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert coefficients(spline, 1)[2:] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_saveas_writes_the_function_root_writes(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """``SaveAs`` writes a C function of ``x``, named for the file up to its first dot, whose
    coefficients are the spline's own - ROOT's text, line for line."""
    monkeypatch.chdir(tmp_path)
    quadratic().SaveAs("splinePrecisionTest.C")
    text = (tmp_path / "splinePrecisionTest.C").read_text(encoding="ascii")
    lines = text.splitlines()
    assert lines[0] == "double splinePrecisionTest(double x) {"
    assert lines[1] == "   const int fNp = 11, fKstep = 1;"
    assert lines[2] == "   const double fDelta = 0.10000000000000001, fXmin = 0, fXmax = 1;"
    assert lines[3] == (
        "   const double fX[11] = { 0, 0.10000000000000001, 0.20000000000000001,"
        " 0.30000000000000004, 0.40000000000000002,"
    )
    assert lines[-1] == "}"
    assert "   return (fY[klow]+dx*(fB[klow]+" in text
    for name in ("fX", "fY", "fB", "fC", "fD", "fE", "fF"):
        assert f"   const double {name}[11] = {{" in text


def test_the_macro_saveas_writes_is_the_spline_when_the_interpreter_runs_it(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """What ``hist/test/test_spline.cxx`` does: read the file back, append the function's name
    and run it through ``gInterpreter``, which hands back the function itself."""
    monkeypatch.chdir(tmp_path)
    quadratic().SaveAs("splinePrecisionTest.C")
    code = (tmp_path / "splinePrecisionTest.C").read_text(encoding="ascii") + "splinePrecisionTest;"
    spline = ROOT.gInterpreter.ProcessLine(code)
    for x, y in zip(XS, YS, strict=True):
        assert spline(x) == pytest.approx(y, abs=TOLERANCE)
