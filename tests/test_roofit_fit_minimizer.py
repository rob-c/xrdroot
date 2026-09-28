"""RooFit's minimizer - Minuit set up and driven as RooFit drives it - held to ROOT 6.40.04.

The fits here were run in ROOT through PyROOT with the same data - drawn
after ``RooRandom::randomGenerator()->SetSeed(4357)`` or typed in - and
the same starting points. What RooFit prints on the way (the evaluation
errors it walls off, Minuit2's summary, MINOS's banners) is ROOT's text;
where a number is Minuit's, it agrees to Minuit's tolerance.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from refmachine import ROOTS_MACHINE, roots
from xrdroot.roofit.fitting.minimizer import RooMinimizer, as_set, cov_quality, first_step
from xrdroot.roofit.fitting.minimizer import _status as status_of
from xrdroot.roofit.pdfs.basic import RooGaussian, RooPolynomial
from xrdroot.roofit.printing import PRECISION
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar

#: The line a walled-off point's warning starts with, and how it goes on.
WALL = "RooAbsMinimizerFcn: Minimized function has error status.\n"
RETURNING = (
    "Returning maximum FCN so far ({}) to force MIGRAD to back out of this region. Error log "
)
NEGATIVE_NORM = (
    "     p.d.f normalization integral is zero or negative @ numerator=g=1, "
    "denominator=g_Int[x]=-0.00671192\n"
)
NAN_TOP = (
    "     getLogVal() top-level p.d.f evaluates to NaN @ numerator=g=1, "
    "denominator=g_Int[x]=-0.00671192\n"
)

#: What ROOT printed when MIGRAD first stepped a Gaussian's width below zero.
NEGATIVE_WIDTH = (
    WALL
    + RETURNING.format("1079.95")
    + "follows.\nParameter values: \tm=0\ts=-0.00267767\n"
    + "RooFit::Detail::RooNormalizedPdf::g_over_g_Int[x][ numerator=g denominator=g_Int[x] ]\n"
    + NEGATIVE_NORM * 10
    + NAN_TOP * 2
    + "    ... (remaining 10 messages suppressed)\n\n"
)

#: What ROOT printed at the start of a line fitted where it is negative for some events.
LINE_NAN = (
    WALL
    + RETURNING.format("-inf")
    + "follows.\nParameter values: \ta=1.2\n"
    + "RooFit::Detail::RooNormalizedPdf::p_over_p_Int[x][ numerator=p denominator=p_Int[x] ]\n"
    + "     getLogVal() top-level p.d.f evaluates to NaN @ numerator=p=1, "
    + "denominator=p_Int[x]=2\n\n"
)


def _gauss(n: int = 50, width: tuple[float, float, float] = (2, 0.1, 10)) -> tuple[Any, ...]:
    """A Gaussian and ``n`` events of it, drawn with ROOT's seed."""
    generator().SetSeed(4357)
    x = RooRealVar("x", "x", -10, 10)
    m = RooRealVar("m", "m", 0, -5, 5)
    s = RooRealVar("s", "s", *width)
    g = RooGaussian("g", "g", x, m, s)
    return g, g.generate([x], n), m, s


def _line(start: float) -> tuple[Any, Any, Any]:
    """``1 + a x`` on [-1, 1], 20 events of it with a = 0.8, and ``a`` set to ``start``."""
    generator().SetSeed(4357)
    x = RooRealVar("x", "x", -1, 1)
    a = RooRealVar("a", "a", 0.8, -5, 5)
    p = RooPolynomial("p", "p", x, [a])
    data = p.generate([x], 20)
    a.setVal(start)
    a.setError(3)
    return p, data, a


def test_a_point_where_the_likelihood_cannot_be_had_is_walled_off_as_roofit_walls_it(
    capsys: Any,
) -> None:
    """A step to a negative width makes the normalisation negative: RooFit says why, hands
    MIGRAD the worst value so far plus the badness, and the fit carries on to ROOT's minimum."""
    g, data, m, s = _gauss(10, (0.5, -3, 10))
    s.setVal(0.02)
    s.setError(1)
    capsys.readouterr()
    result = g.fitTo(data, Save=True, PrintLevel=-1)
    out = capsys.readouterr().out
    assert out.count(WALL) == 1
    assert NEGATIVE_WIDTH in out
    assert result.numInvalidNLL() == 1
    assert result.minNll() == pytest.approx(1.89007, abs=5e-6)
    assert m.getVal() == pytest.approx(0.03510715381667138, rel=1e-9)
    assert s.getVal() == pytest.approx(0.29250302960642977, rel=1e-9)


def test_a_fit_that_starts_where_nothing_can_be_had_backs_out_to_roots_minimum(
    capsys: Any,
) -> None:
    """Every point is bad until MIGRAD backs out: the worst so far is then nothing, and the
    fit still finds the parameter ROOT finds after nine bad points."""
    p, data, a = _line(1.2)
    capsys.readouterr()
    result = p.fitTo(data, Save=True, PrintLevel=-1)
    out = capsys.readouterr().out
    assert out.count(WALL) == result.numInvalidNLL() == 9
    assert LINE_NAN in out
    assert a.getVal() == pytest.approx(0.6442657844265741, rel=1e-9)
    assert a.getError() == pytest.approx(0.30692319873640106, rel=1e-6)


def test_the_minimum_after_bad_points_is_the_likelihood_without_minuits_offset(
    capsys: Any,
) -> None:
    """ROOT offsets the values Minuit sees once a valid one follows only invalid ones, but the
    saved minimum is the likelihood's own: 12.2898, not -0.0654213."""
    p, data, _ = _line(1.2)
    result = p.fitTo(data, Save=True, PrintLevel=-1)
    capsys.readouterr()
    result.Print()
    assert "minimized FCN value: 12.2898, estimated distance" in capsys.readouterr().out


@pytest.mark.xfail(strict=True, reason="a point evaluated twice is logged twice, not once")
def test_a_point_minuit_asks_for_again_has_no_new_errors_to_log(capsys: Any) -> None:
    """ROOT does not evaluate the likelihood again at the point it has just evaluated, so the
    sixth bad point's log is empty: eight NaN lines for nine bad points."""
    p, data, _ = _line(1.2)
    capsys.readouterr()
    p.fitTo(data, Save=True, PrintLevel=-1)
    assert capsys.readouterr().out.count("top-level p.d.f evaluates to NaN") == 8


def test_without_the_wall_bad_points_go_to_minuit_as_they_are(capsys: Any) -> None:
    """``EvalErrorWall(false)`` hands Minuit the NaN itself: RooFit says the error is ignored,
    and MIGRAD fails with ROOT's status."""
    p, data, a = _line(1.2)
    capsys.readouterr()
    result = p.fitTo(data, Save=True, PrintLevel=-1, EvalErrorWall=False)
    out = capsys.readouterr().out
    ignored = "RooAbsMinimizerFcn: Minimized function has error status but is ignored.\n"
    assert (
        ignored + "Parameter values: \ta=1.2\n"
        "RooFit::Detail::RooNormalizedPdf::p_over_p_Int[x][ numerator=p denominator=p_Int[x] ]\n"
        "     getLogVal() top-level p.d.f evaluates to NaN @ numerator=p=1, denominator=p_Int[x]=2"
        "\n\n"
    ) in out
    assert out.count(ignored) == result.numInvalidNLL() == 40
    assert (result.covQual(), a.getVal(), a.getError()) == (0, 1.2, 0.0)
    labels = [result.statusLabelHistory(i) for i in range(result.numStatusHistory())]
    assert (labels, result.statusCodeHistory(0)) == (["MINIMIZE", "HESSE"], -1)


@pytest.mark.xfail(strict=True, reason="HESSE's failure is always 100, and not reported")
def test_a_hesse_that_cannot_invert_says_so_and_scores_as_minuit2_scores_it(capsys: Any) -> None:
    """ROOT reports ``calculateHessErrors() Error when calculating Hessian`` and adds 100 times
    Minuit2's HESSE code - 2, the matrix could not be inverted - to the status: 203."""
    p, data, _ = _line(1.2)
    capsys.readouterr()
    result = p.fitTo(data, Save=True, PrintLevel=-1, EvalErrorWall=False)
    out = capsys.readouterr().out
    assert result.statusCodeHistory(1) == result.status() == 203
    assert (
        "[#0] ERROR:Minimization -- RooMinimizer::calculateHessErrors() Error when calculating"
        in out
    )


def test_print_eval_errors_counts_or_silences_what_is_logged(capsys: Any) -> None:
    """``PrintEvalErrors(0)`` says only how many errors each object had; ``-1`` says nothing,
    and the fit is the same either way."""
    p, data, a = _line(1.2)
    capsys.readouterr()
    p.fitTo(data, PrintLevel=-1, PrintEvalErrors=0)
    counted = capsys.readouterr().out
    assert (
        WALL + RETURNING.format("-inf") + "follows.\nParameter values: \ta=1.2\n"
        "RooFit::Detail::RooNormalizedPdf::p_over_p_Int[x][ numerator=p denominator=p_Int[x] ]"
        " has 1 errors\n\n"
    ) in counted
    first = a.getVal()
    a.setVal(1.2)
    a.setError(3)
    p.fitTo(data, PrintLevel=-1, PrintEvalErrors=-1)
    assert "error status" not in capsys.readouterr().out
    assert a.getVal() == first


def _minimizer(start: tuple[float, float] = (0.0, 2.0)) -> tuple[RooMinimizer, Any, Any, Any]:
    """A minimizer of the likelihood of 50 Gaussian events, the parameters at ``start``."""
    g, data, m, s = _gauss()
    nll = g.createNLL(data)
    m.setVal(start[0])
    s.setVal(start[1])
    return RooMinimizer(nll), nll, m, s


def test_hesse_minos_and_save_before_migrad_are_refused_as_roofit_refuses_them(
    capsys: Any,
) -> None:
    """Asking for errors before there is a minimum is a mistake RooFit names, with status -1."""
    minimizer, _, _, _ = _minimizer()
    capsys.readouterr()
    assert minimizer.hesse() == -1
    assert minimizer.minos() == -1
    assert minimizer.save() is None
    assert minimizer.evalCounter() == 0
    assert capsys.readouterr().out == (
        "[#0] WARNING:Minimization -- RooMinimizer::hesse: Error, run Migrad before Hesse!\n"
        "[#0] WARNING:Minimization -- RooMinimizer::minos: Error, run Migrad before Minos!\n"
        "[#0] WARNING:Minimization -- RooMinimizer::save: Error, run minimization before!\n"
    )


def test_migrad_prints_minuit2s_summary(capsys: Any) -> None:
    """At print level 1 Minuit2 says what it was asked and what it found - the lines ROOT
    prints, with Minuit's numbers."""
    minimizer, _, _, _ = _minimizer()
    capsys.readouterr()
    assert minimizer.migrad() == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        "[#1] INFO:Minimization -- [fitFCN] No discrete parameters, performing continuous "
        "minimization only",
        "Minuit2Minimizer: Minimize with max-calls 1000 convergence for edm < 1 strategy 1",
        "Minuit2Minimizer : Valid minimum - status = 0",
    ]
    fval, edm = (float(line.split("=")[1]) for line in lines[3:5])
    assert fval == pytest.approx(102.310351911713894, abs=1e-9)
    assert edm == pytest.approx(1.79401040347328976e-08, rel=1e-4)
    assert lines[5:] == [
        "Nfcn  = 31",
        "m\t  = 0.0482195\t +/-  0.264684\t(limited)",
        "s\t  = 1.87247\t +/-  0.187172\t(limited)",
    ]


def test_minos_prints_its_banners_and_gives_asymmetric_errors(capsys: Any) -> None:
    """MINOS of s alone: ROOT's banners and errors, and only s has them in the saved table."""
    minimizer, _, m, s = _minimizer()
    minimizer.setPrintLevel(-1)
    minimizer.migrad()
    minimizer.setPrintLevel(1)
    assert (minimizer.hesse(), minimizer.minos([s])) == (0, 0)
    stars = "*" * 102
    assert capsys.readouterr().out.endswith(
        f"{stars}\nMinuit2Minimizer::GetMinosError - Run MINOS LOWER error for parameter #1 : s "
        f"using max-calls 1000, tolerance 1\n{stars}\nMinuit2Minimizer::GetMinosError - Run MINOS"
        " UPPER error for parameter #1 : s using max-calls 1000, tolerance 1\n"
        "Minos: Lower error for parameter s  :  -0.172696\n"
        "Minos: Upper error for parameter s  :  0.204115\n"
    )
    assert (s.getAsymErrorLo(), s.getAsymErrorHi()) == pytest.approx((-0.172696, 0.204115), 1e-5)
    assert not m.hasAsymError()
    result = minimizer.save("myname", "mytitle")
    assert (result.GetName(), result.GetTitle()) == ("myname", "mytitle")
    result.Print("v")
    assert (
        "                     m    0.0000e+00    4.8220e-02         +/-  2.65e-01  <none>\n"
        "                     s    2.0000e+00    1.8725e+00 (+2.04e-01,-1.73e-01)  <none>\n"
    ) in capsys.readouterr().out


@pytest.mark.xfail(strict=True, reason="MINOS of no fitted parameter is not recorded")
def test_minos_of_parameters_that_are_not_fitted_is_still_recorded() -> None:
    """ROOT saves ``MINOS=0`` in the history even when none of the parameters asked for floats."""
    minimizer, _, _, _ = _minimizer()
    minimizer.setPrintLevel(-1)
    minimizer.migrad()
    assert minimizer.minos([RooRealVar("x", "x", 0)]) == 0
    result = minimizer.save()
    labels = [result.statusLabelHistory(i) for i in range(result.numStatusHistory())]
    assert labels == ["MIGRAD", "MINOS"]


def test_a_parameter_made_constant_after_the_minimizer_is_fixed_in_minuit(capsys: Any) -> None:
    """RooFit reads the parameters' constness when it runs, not when it is made: the parameter
    is fixed, printed as fixed, and keeps its error."""
    minimizer, _, m, s = _minimizer()
    minimizer.setPrintLevel(-1)
    minimizer.migrad()
    minimizer.hesse()
    m.setConstant(True)
    minimizer.setPrintLevel(1)
    capsys.readouterr()
    assert minimizer.migrad() == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2:] == ["m\t  = 0.0482195\t (fixed)", "s\t  = 1.87249\t +/-  0.187177\t(limited)"]
    assert m.getVal() == pytest.approx(0.048219535850741575, rel=1e-9)
    assert s.getVal() == pytest.approx(1.8724922220616362, rel=1e-9)
    assert s.getError() == pytest.approx(0.18717709190222154, rel=1e-8)
    assert m.getError() == pytest.approx(0.264684, rel=1e-5)
    minimizer.minos()
    assert not m.hasAsymError() and s.hasAsymError()
    result = minimizer.save()
    assert result.constPars().names() == ["m"]
    assert result.floatParsFinal().names() == ["s"]


def test_a_function_with_nothing_to_vary_is_not_minimised(capsys: Any) -> None:
    """With every parameter constant there is nothing for Minuit to do; RooFit says so."""
    minimizer, _, m, s = _minimizer()
    m.setConstant(True)
    s.setConstant(True)
    alone = RooMinimizer(minimizer.function)
    capsys.readouterr()
    assert alone.migrad() == -1
    assert alone.getNPar() == 0
    assert capsys.readouterr().out.startswith(
        "[#0] ERROR:Minimization -- RooMinimizer::fitFCN(): FCN function has zero parameters\n"
    )


@pytest.mark.xfail(strict=True, reason="the error level line is printed by fitTo, not here")
def test_a_minimizer_says_which_error_level_its_function_takes(capsys: Any) -> None:
    """ROOT's ``RooMinimizer`` constructor itself prints where its error level comes from, so a
    minimizer made by hand says it too."""
    g, data, _, _ = _gauss()
    nll = g.createNLL(data)
    capsys.readouterr()
    RooMinimizer(nll)
    assert capsys.readouterr().out == (
        "[#1] INFO:Fitting -- RooAddition::defaultErrorLevel(nll_g_over_g_Int[x]_gData) "
        "Summation contains a RooNLLVar, using its error level\n"
    )


#: From (m, s) = (0.3, 1.5) with no errors: status, m, s, their errors, and the calls made.
RUNS = {
    "simplex": (5, 0.034211820293225635, 1.905862136731493, 0.22566006842502503, 0.22711522307),
    "improve": (0, 0.04766557861201023, 1.8696943635300594, 0.2642917104346336, 0.18653675164),
    "seek": (0, 0.04766557861201023, 1.8696943635300594, 0.2642917104346336, 0.18653675164),
}
CALLS = {"simplex": 15, "improve": 32, "seek": 32}


@pytest.mark.parametrize("name", sorted(RUNS))
def test_each_algorithm_finds_what_roots_finds_from_the_same_start(name: str) -> None:
    """SIMPLEX, IMPROVE and SEEK from a fresh minimizer: ROOT's values, errors and calls, and
    the algorithm's name in the history."""
    minimizer, _, m, s = _minimizer((0.3, 1.5))
    minimizer.setPrintLevel(-1)
    status, *expected = RUNS[name]
    assert getattr(minimizer, name)() == status
    assert [m.getVal(), s.getVal(), m.getError(), s.getError()] == pytest.approx(expected, 1e-8)
    assert minimizer.evalCounter() == CALLS[name]
    assert minimizer.history == [(name.upper(), status)]


def test_minimize_takes_minuit2_with_migrad_or_simplex() -> None:
    """``minimize("Minuit2", "simplex")`` is SIMPLEX, anything else MIGRAD, both saved as
    ``MINIMIZE`` as in ROOT."""
    minimizer, _, m, s = _minimizer((0.3, 1.5))
    minimizer.setPrintLevel(-1)
    assert minimizer.minimize("Minuit2", "simplex") == 5
    assert (m.getVal(), s.getVal()) == pytest.approx((0.034211820293225635, 1.905862136731493))
    minimizer, _, m, s = _minimizer((0.3, 1.5))
    minimizer.setPrintLevel(-1)
    assert minimizer.minimize("Minuit2", "migrad") == 0
    assert (m.getVal(), s.getVal()) == pytest.approx((0.04766557861201023, 1.8696943635300594))
    assert minimizer.history == [("MINIMIZE", 0)]
    assert minimizer.minimizer_type == "Minuit2"


def test_migrad_and_hesse_count_roots_calls() -> None:
    """The number of likelihood evaluations is how one compares fits' cost: ROOT's 42 after
    MIGRAD and HESSE, 105 after MINOS too. MINOS searches for each crossing until it is within a
    tolerance, and off ROOT's machine - a likelihood an ulp away, Minuit2 built for arm64 with
    its multiplies and adds fused - one search can end a call sooner or later: 104 is seen."""
    minimizer, _, _, _ = _minimizer((0.3, 1.5))
    minimizer.setPrintLevel(-1)
    minimizer.migrad()
    minimizer.hesse()
    assert minimizer.evalCounter() == 42
    minimizer.minos()
    assert minimizer.evalCounter() == roots(105, abs=4)


def test_strategy_tolerance_and_error_level_change_the_fit_as_in_root() -> None:
    """An error level of 2 doubles the errors' square; strategy 0 and a looser tolerance stop
    MIGRAD where ROOT's stops."""
    minimizer, _, m, s = _minimizer((0.3, 1.5))
    minimizer.setPrintLevel(-1)
    minimizer.setStrategy(0)
    minimizer.setEps(0.1)
    minimizer.setErrorLevel(2.0)
    assert minimizer.migrad() == 0
    expected = [0.04766557053043234, 1.8696943602819212, 0.4613926323396632, 0.33712019362123746]
    assert [m.getVal(), s.getVal(), m.getError(), s.getError()] == pytest.approx(expected, 1e-8)


def test_offsetting_changes_nothing_minuit_finds() -> None:
    """Offsetting subtracts the first value from every other, for precision: the minimum is
    ROOT's, to Minuit's tolerance."""
    minimizer, _, m, s = _minimizer((0.3, 1.5))
    minimizer.setPrintLevel(-1)
    minimizer.setOffsetting(True)
    assert minimizer.migrad() == 0
    assert (m.getVal(), s.getVal()) == pytest.approx((0.04766614084649667, 1.8696945685485702))


@pytest.mark.xfail(strict=True, reason="the likelihood's own value keeps the offset")
def test_an_offset_likelihood_still_has_its_own_value() -> None:
    """ROOT's offset is the minimizer's business: the likelihood's ``getVal`` is 102.31 with
    offsetting on, and after it is turned off."""
    minimizer, nll, _, _ = _minimizer((0.3, 1.5))
    minimizer.setPrintLevel(-1)
    minimizer.setOffsetting(True)
    minimizer.migrad()
    assert nll.getVal() == pytest.approx(102.31046564177177, abs=1e-8)
    minimizer.setOffsetting(False)
    assert nll.getVal() == pytest.approx(102.31046564177177, abs=1e-8)


@pytest.mark.xfail(strict=True, reason="the calls are counted per run, not since construction")
def test_the_call_count_adds_up_over_runs() -> None:
    """ROOT counts the likelihood's evaluations since the minimizer was made, over every run."""
    minimizer, _, _, _ = _minimizer((0.3, 1.5))
    minimizer.setPrintLevel(-1)
    minimizer.migrad()
    first = minimizer.evalCounter()
    minimizer.migrad()
    assert minimizer.evalCounter() > first


@pytest.fixture
def _cout_precision() -> Iterator[None]:
    """A verbose fit leaves ``std::cout`` at four digits, as ROOT's does, for the rest of the
    process: put it back for the tests that follow."""
    before = PRECISION[0]
    yield
    PRECISION[0] = before


@pytest.mark.usefixtures("_cout_precision")
def test_a_verbose_fit_says_each_parameter_it_moves_and_each_value_it_finds(capsys: Any) -> None:
    """``Verbose()`` shows the first steps RooFit picked and every point Minuit tried, in
    ROOT's text - its precision dropping to four digits after the first value."""
    g, data, m, s = _gauss(20)
    m.setVal(0.2)
    s.setVal(1.5)
    capsys.readouterr()
    g.fitTo(data, PrintLevel=-1, Verbose=True, Hesse=False)
    out = capsys.readouterr().out
    warning = (
        "[#0] WARNING:Minimization -- RooAbsMinimizerFcn::synchronize: WARNING: no initial "
        "error estimate available for {}: using {}\n"
    )
    assert warning.format("m", "1") + warning.format("s", "0.7") in out
    if ROOTS_MACHINE:
        assert (
            "m=0.2, \nprevFCN = 38.79413316  m=0.2101, \nprevFCN = 38.83576049  m=0.1899, \n"
        ) in out
        assert out.endswith("prevFCN = 37.76774481  m=-0.2628, s=1.598, ")
        assert out.count("prevFCN") == 31
    # Elsewhere a parameter can come back from Minuit2's sine transform an ulp off and be said
    # to have moved, and the walk down can take other steps; the first steps in m, and what the
    # likelihood was there, are the same.
    assert _moved(out, "m")[:3] == ["0.2", "0.2101", "0.1899"]
    assert _found(out)[:2] == [38.79413316, 38.83576049]
    assert (m.getVal(), s.getVal()) == pytest.approx((-0.26275685043154534, 1.5982264153830419))


def _moved(out: str, name: str) -> list[str]:
    """The values a verbose fit said ``name`` was moved to, call after call."""
    return re.findall(rf"(?m)(?:^|\s){name}=([^,\s]+),", out)


def _found(out: str) -> list[float]:
    """The likelihood's values a verbose fit said it found, call after call."""
    return [float(chunk.split()[0]) for chunk in out.split("\nprevFCN = ")[1:]]


#: A parameter's range and value, and the first step RooFit gives it without an error.
STEPS = [
    ((-5, 5, 0.0), 1.0),
    ((-5, 0.5, 0.3), 0.1),
    ((-0.2, 5, 0.1), 0.15),
    ((-5, 5, 5.0), 1.0),
    ((-5, 5, -5.0), 1.0),
]


@pytest.mark.parametrize(("where", "step"), STEPS)
def test_a_parameter_without_an_error_starts_with_roofits_step(
    where: tuple[float, float, float], step: float
) -> None:
    """A tenth of the range, halved distance to an end that is nearer than two steps, as
    ROOT's ``synchronizeParameterSettings`` says in its warnings."""
    par = RooRealVar("p", "p", where[2], where[0], where[1])
    assert first_step(par) == pytest.approx(step)
    par.setError(0.25)
    assert first_step(par) == 0.25


def test_a_parameter_without_limits_steps_by_one() -> None:
    """With an end missing there is no range to take a tenth of: the step is one."""
    par = RooRealVar("p", "p", 0.3, -float("inf"), float("inf"))
    assert first_step(par) == 1.0


@pytest.mark.usefixtures("_cout_precision")
def test_a_fit_at_a_parameter_limit_starts_inside_it_as_minuit_does(capsys: Any) -> None:
    """Minuit2 moves a value on a limit in by a tenth of its step before the first call: ROOT's
    first points are 4.9 and -4.9, not the limits."""
    for value, start in ((5.0, "m=4.9, "), (-5.0, "m=-4.9, ")):
        g, data, m, s = _gauss(20)
        m.setVal(value)
        s.setVal(1.5)
        minimizer = RooMinimizer(g.createNLL(data))
        minimizer.setPrintLevel(-1)
        minimizer.setVerbose(True)
        minimizer.setMaxFunctionCalls(1)
        capsys.readouterr()
        minimizer.migrad()
        out = capsys.readouterr().out
        if ROOTS_MACHINE:
            assert start + "\nprevFCN = " in out
        # s, which it did not move, may come back from Minuit2's sine transform an ulp off
        # and be said to have moved too - but the first m it is given is 4.9, not the limit.
        assert _moved(out, "m")[0] == start[2:-2]


def test_minuit2s_status_is_the_last_of_its_checks_that_fails() -> None:
    """``ExamineMinimum``'s codes: 1 made positive, 2 HESSE failed, 3 above the EDM, 4 the call
    limit, 5 not positive-definite, and 6 for an invalid minimum none of them explains."""
    flags = {
        "has_posdef_covar": True,
        "has_made_posdef_covar": False,
        "hesse_failed": False,
        "is_above_max_edm": False,
        "has_reached_call_limit": False,
        "is_valid": True,
    }
    assert status_of(SimpleNamespace(**flags)) == 0
    assert status_of(SimpleNamespace(**{**flags, "has_posdef_covar": False})) == 5
    for name, code in (
        ("has_made_posdef_covar", 1),
        ("hesse_failed", 2),
        ("is_above_max_edm", 3),
        ("has_reached_call_limit", 4),
    ):
        assert status_of(SimpleNamespace(**{**flags, name: True})) == code
    assert status_of(SimpleNamespace(**{**flags, "is_valid": False})) == 6


def test_the_covariance_quality_is_minuit2s_best_claim_for_its_matrix() -> None:
    """``CovMatrixStatus``: 3 accurate, 2 forced positive, 1 approximate, 0 there but not
    positive, -1 none at all or no minimum."""
    names = ("has_accurate_covar", "has_made_posdef_covar", "has_posdef_covar", "has_covariance")
    for index, code in enumerate((3, 2, 1, 0)):
        assert cov_quality(SimpleNamespace(**{n: n == names[index] for n in names})) == code
    assert cov_quality(SimpleNamespace(**dict.fromkeys(names, False))) == -1
    assert cov_quality(None) == -1


def test_any_function_of_parameters_can_be_minimised() -> None:
    """A minimizer takes any function, not only a likelihood; a value past 1e30 counts as one
    that cannot be had, and the error level is the one it is given - 1 for a chi-square-like
    sum, as ROOT takes it, finds ROOT's error."""
    from xrdroot.roofit.functions import RooAddition, RooFormulaVar

    a = RooRealVar("a", "a", 2.9, -10, 10)
    a.setError(0.5)
    formula = RooFormulaVar("f", "(a-1.5)*(a-1.5)+1e31*(a>3)", [a])
    minimizer = RooMinimizer(RooAddition("sum", "sum", [formula]))
    minimizer.setPrintLevel(-1)
    minimizer.setErrorLevel(1.0)
    assert minimizer.migrad() == 0
    assert a.getVal() == pytest.approx(1.5043769834367244, rel=1e-8)
    assert a.getError() == pytest.approx(0.9983291283403735, rel=1e-6)
    assert minimizer.save().minNll() == pytest.approx(1.915798400536005e-05, rel=1e-5)
    a.setVal(3.5)
    before = minimizer.invalid
    minimizer.migrad()
    assert minimizer.invalid > before


def test_the_settings_roofit_takes_for_its_own_reasons_change_nothing() -> None:
    """Iterations, constant-term optimisation and profiling are kept for the calls' sake."""
    minimizer, _, _, _ = _minimizer()
    minimizer.setMaxIterations(10)
    minimizer.optimizeConst(2)
    minimizer.setProfile(True)
    minimizer.setMinimizerType("")
    assert minimizer.minimizer_type == "Minuit2"
    assert (minimizer.GetName(), minimizer.ClassName()) == ("RooMinimizer", "RooMinimizer")
    assert minimizer.fitter() is minimizer
    assert minimizer.setPrintLevel(-1) == -1
    minimizer.migrad()
    assert minimizer.lastMinuitFit().GetName() == "nll_g_over_g_Int[x]_gData"
    assert as_set([RooRealVar("p", "p", 1)]).names() == ["p"]


def test_an_invalid_minimum_prints_its_value_at_six_digits(capsys: Any) -> None:
    """Minuit2 prints an invalid minimum without raising the stream's precision."""
    g, data, _, _ = _gauss()
    capsys.readouterr()
    g.fitTo(data, MaxCalls=5, Hesse=False)
    out = capsys.readouterr().out
    assert "Invalid minimum - status = 4\nFVAL  = 102.533\nEdm   = 0.239047\nNfcn  = 9\n" in out


@pytest.mark.xfail(strict=True, reason="SIMPLEX does not print Minuit2's first line")
def test_simplex_says_what_it_was_asked_as_migrad_does(capsys: Any) -> None:
    """``Minuit2Minimizer::Minimize`` prints its settings for every algorithm."""
    g, data, _, _ = _gauss()
    capsys.readouterr()
    g.fitTo(data, Strategy=2, Minimizer=("Minuit2", "simplex"))
    out = capsys.readouterr().out
    assert (
        "Minuit2Minimizer: Minimize with max-calls 1000 convergence for edm < 1 strategy 2" in out
    )


#: rf612's messages, as ROOT printed them for the fit below and the curve after it.
RF612 = [
    "[#0] ERROR:Minimization -- RooMinimizer: all function calls during minimization gave "
    "invalid NLL values!",
    "[#0] ERROR:Minimization -- RooMinimizer::calculateHessErrors() Error when calculating "
    "Hessian",
    "[#0] ERROR:Minimization -- RooMinimizer: all function calls during minimization gave "
    "invalid NLL values!",
    "[#0] ERROR:Eval -- RooAbsReal::logEvalError(pol3) evaluation error, ",
    " origin       : RooPolynomial::pol3[ x=x coefList=(a1,a2,a3) ]",
    " message      : p.d.f normalization integral is zero or negative: -2220.000000",
    " server values: x=x=0, coefList=(a1 = 10 +/- 0,a2 = -1 +/- 0,a3 = 0.01)",
]


def test_a_fit_with_no_valid_value_and_a_curve_that_cannot_be_normalised_say_so_as_root(
    capsys: Any,
) -> None:
    """rf612: without recovery every call is invalid - RooFit says so after MIGRAD and after
    HESSE - and the curve of the unnormalisable density logs its error once. With recovery the
    Minuit2 lines appear at ``PrintLevel(0)``, and the minimum carries the offset back."""
    from xrdroot.roofit.cmdargs import RooCmdArg
    from xrdroot.roofit.plot.frame import make_frame

    x = RooRealVar("x", "x", -15, 15)
    a1, a2 = RooRealVar("a1", "a1", -0.5, -10.0, 20.0), RooRealVar("a2", "a2", 0.2, -10.0, 20.0)
    pdf = RooPolynomial("pol3", "pol3", x, [a1, a2, RooRealVar("a3", "a3", 0.01)])
    generator().SetSeed(4357)
    data = pdf.generate([x], 1000)

    def fit(strength: float, level: int) -> Any:
        a1.setVal(10.0)
        a2.setVal(-1.0)
        options = [RooCmdArg("RecoverFromUndefinedRegions", strength),
                   RooCmdArg("PrintEvalErrors", -1), RooCmdArg("PrintLevel", level)]  # fmt: skip
        return pdf.fitTo(data, RooCmdArg("Save"), *options)

    capsys.readouterr()
    bad = fit(0.0, -1)
    pdf.plotOn(make_frame(x, (), {}))
    lines = [one for one in capsys.readouterr().out.splitlines() if "INFO" not in one]
    assert lines == RF612
    assert (bad.minNll(), bad.status(), bad.numInvalidNLL()) == (0.0, 302, 23)
    good = fit(1.0, 0)
    printed = capsys.readouterr().out
    assert "Minuit2Minimizer: Minimize with max-calls 1000 convergence for edm < 1 strategy 1" in (
        printed
    )
    assert "a1\t  = -0.579502\t +/-  0.0614758\t(limited)" in printed
    assert (good.minNll(), good.numInvalidNLL()) == (roots(2959.918384170729, rel=1e-9), 64)


def test_describing_a_failure_reports_no_errors_of_its_own(capsys: Any) -> None:
    """rf506: the values a failure's log lists are evaluated quietly, not logged again."""
    from xrdroot.roofit import evalerrors

    x = RooRealVar("x", "x", 0, 1)
    p = RooPolynomial("p", "p", x, [RooRealVar("a", "a", -5.0)])
    capsys.readouterr()
    evalerrors.collecting(True)
    try:
        evalerrors.record("k", lambda: str(p.value({}, {"x"})), "m", lambda: "")
    finally:
        evalerrors.collecting(False)
        evalerrors.clear()
    assert capsys.readouterr().out == ""
    p.value({}, {"x"})
    assert "normalization integral is zero or negative: -1.500000" in capsys.readouterr().out
