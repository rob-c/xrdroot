"""RooFit's fit results, held to what ROOT 6.40.04 printed for the same fits.

Each fit here was run in ROOT through PyROOT after
``RooRandom::randomGenerator()->SetSeed(4357)``, on data generated from the
same model: the printed tables are ROOT's to the character, and the numbers
behind them agree to Minuit's tolerance.
"""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.roofit.fitting.result import RooFitResult
from xrdroot.roofit.pdfs.addpdf import RooAddPdf
from xrdroot.roofit.pdfs.basic import RooExponential, RooGaussian
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar

#: ``model.fitTo(data, Save(), PrintLevel(-1), Minos(true))`` of a Gaussian plus an exponential.
#: (``\x20`` is a space ROOT prints at the end of a line.)
MINOS_VERBOSE = """
  RooFitResult: minimized FCN value: 531.687, estimated distance to minimum: 0.000294427
                covariance matrix quality: Full, accurate covariance matrix
                Status : MINIMIZE=0 HESSE=0 MINOS=0\x20

    Floating Parameter  InitialValue    FinalValue (+HiError,-LoError)    GblCorr.
  --------------------  ------------  ----------------------------------  --------
                     c   -1.0000e-01   -5.2136e-02 (+2.24e-02,-2.45e-02)  <none>
                     f    7.0000e-01    6.2159e-01 (+5.75e-02,-5.82e-02)  <none>
                     m    0.0000e+00    8.7821e-02 (+2.18e-01,-2.21e-01)  <none>
                     s    2.0000e+00    1.8693e+00 (+2.15e-01,-1.95e-01)  <none>

"""

MINOS_STANDARD = """
  RooFitResult: minimized FCN value: 531.687, estimated distance to minimum: 0.000294427
                covariance matrix quality: Full, accurate covariance matrix
                Status : MINIMIZE=0 HESSE=0 MINOS=0\x20

    Floating Parameter    FinalValue +/-  Error\x20\x20\x20
  --------------------  --------------------------
                     c   -5.2136e-02 +/-  2.33e-02
                     f    6.2159e-01 +/-  5.78e-02
                     m    8.7821e-02 +/-  2.19e-01
                     s    1.8693e+00 +/-  2.04e-01

"""

#: ROOT's covariance matrix of that fit, row by row, in the order c, f, m, s.
MINOS_COVARIANCE = [
    [5.428588305e-04, -3.937812903e-04, -1.235259404e-03, -1.172024851e-03],
    [-3.937812903e-04, 3.356526421e-03, 1.274395197e-03, 5.675532318e-03],
    [-1.235259404e-03, 1.274395197e-03, 4.811569844e-02, 6.184403959e-03],
    [-1.172024851e-03, 5.675532318e-03, 6.184403959e-03, 4.147033045e-02],
]


def _mixture() -> tuple[Any, Any, dict[str, Any]]:
    """The Gaussian-plus-exponential model, and 200 events of it drawn with ROOT's seed."""
    generator().SetSeed(4357)
    x = RooRealVar("x", "x", -10, 10)
    m = RooRealVar("m", "m", 0, -5, 5)
    s = RooRealVar("s", "s", 2, 0.1, 10)
    c = RooRealVar("c", "c", -0.1, -1, 0)
    f = RooRealVar("f", "f", 0.7, 0, 1)
    g = RooGaussian("g", "g", x, m, s)
    e = RooExponential("e", "e", x, c)
    model = RooAddPdf("model", "model", [g, e], [f])
    data = model.generate([x], 200)
    return model, data, {"x": x, "m": m, "s": s, "c": c, "f": f}


def _minos_fit() -> tuple[Any, dict[str, Any]]:
    model, data, v = _mixture()
    return model.fitTo(data, Save=True, PrintLevel=-1, Minos=True), v


def _gauss() -> tuple[Any, Any, dict[str, Any]]:
    """A Gaussian and 50 events of it, drawn with ROOT's seed."""
    generator().SetSeed(4357)
    x = RooRealVar("x", "x", -10, 10)
    m = RooRealVar("m", "m", 0, -5, 5)
    s = RooRealVar("s", "s", 2, 0.1, 10)
    g = RooGaussian("g", "g", x, m, s)
    return g, g.generate([x], 50), {"x": x, "m": m, "s": s}


def _gauss_fit() -> tuple[Any, dict[str, Any]]:
    g, data, v = _gauss()
    return g.fitTo(data, Save=True, PrintLevel=-1), v


def test_a_minos_fit_prints_roots_tables_standard_and_verbose(capsys: Any) -> None:
    """The table is what people read a fit by, so it must be ROOT's to the character."""
    result, _ = _minos_fit()
    capsys.readouterr()
    result.Print("v")
    assert capsys.readouterr().out == MINOS_VERBOSE
    result.Print()
    assert capsys.readouterr().out == MINOS_STANDARD


def _history(result: Any) -> list[tuple[str, int]]:
    return [
        (result.statusLabelHistory(i), result.statusCodeHistory(i))
        for i in range(result.numStatusHistory())
    ]


def _matrix(matrix: Any, size: int) -> list[list[float]]:
    return [[matrix(i, j) for j in range(size)] for i in range(size)]


def test_a_fit_result_holds_roots_minimum_and_status_history() -> None:
    """Scripts read the numbers, not the table: each accessor must say what ROOT's says."""
    result, _ = _minos_fit()
    assert (result.status(), result.covQual(), result.numInvalidNLL()) == (0, 3, 0)
    assert (result.minNll(), result.edm()) == pytest.approx(
        (531.6869729962241, 0.00029442734112466887), rel=1e-6
    )
    assert result.minNll() == pytest.approx(531.6869729962241, abs=1e-9)
    assert _history(result) == [("MINIMIZE", 0), ("HESSE", 0), ("MINOS", 0)]


def test_a_fit_result_holds_roots_covariance_and_starting_values() -> None:
    """The covariance is ROOT's to Minuit's tolerance, and the starting values are kept."""
    result, _ = _minos_fit()
    found = _matrix(result.covarianceMatrix(), 4)
    for row, expected in zip(found, MINOS_COVARIANCE):
        assert row == pytest.approx(expected, rel=1e-6)
    init = result.floatParsInit()
    assert (init.names(), [p.getVal() for p in init]) == (["c", "f", "m", "s"], [-0.1, 0.7, 0, 2])
    assert (result.params().names(), len(result.constPars())) == (["c", "f", "m", "s"], 0)


def test_global_correlations_are_printed_once_they_have_been_asked_for(capsys: Any) -> None:
    """ROOT makes the global correlations on first use, and only then prints them."""
    result, _ = _minos_fit()
    found = result.globalCorr()
    assert found.names() == ["GC[c]", "GC[f]", "GC[m]", "GC[s]"]
    expected = [0.37512667207823963, 0.5130965525661303, 0.2550469174157545, 0.49840131353143047]
    assert [one.getVal() for one in found] == pytest.approx(expected, rel=1e-6)
    assert result.globalCorr("m") == pytest.approx(0.2550469174157545, rel=1e-6)
    capsys.readouterr()
    result.Print("v")
    table = capsys.readouterr().out
    row = "                     c   -1.0000e-01   -5.2136e-02 (+2.24e-02,-2.45e-02)  0.375127\n"
    assert row in table
    assert table.count("<none>") == 0


def test_a_correlation_is_had_by_name_or_variable_and_a_row_of_them_by_one_name() -> None:
    """``correlation(a, b)`` is a number, ``correlation(a)`` ROOT's row of ``C[a,b]`` values."""
    result, v = _minos_fit()
    assert result.correlation("m", "s") == pytest.approx(0.13844768694715523, rel=1e-6)
    assert result.correlation(v["m"], v["f"]) == pytest.approx(0.1002803160022339, rel=1e-6)
    row = result.correlation("m")
    assert row.names() == ["C[m,c]", "C[m,f]", "C[m,m]", "C[m,s]"]
    assert [one.getVal() for one in row] == pytest.approx(
        [-0.241697032424935, 0.1002803160022339, 1.0, 0.13844768694715523], rel=1e-6
    )


@pytest.mark.xfail(strict=True, reason="correlationMatrix() makes the global correlations show")
def test_a_correlation_matrix_or_pair_does_not_make_the_global_correlations_show(
    capsys: Any,
) -> None:
    """ROOT prints ``<none>`` after ``correlationMatrix()`` or ``correlation(a, b)``: only
    ``globalCorr()`` and ``correlation(a)`` fill the global correlations."""
    result, _ = _gauss_fit()
    result.correlationMatrix()
    result.correlation("m", "s")
    capsys.readouterr()
    result.Print("v")
    assert capsys.readouterr().out.count("<none>") == 2


def test_the_conditional_covariance_fixes_the_other_parameters_as_root_does() -> None:
    """The conditional covariance is the Schur complement: the others held at their values."""
    result, v = _minos_fit()
    found = result.conditionalCovarianceMatrix([v["m"], v["s"]])
    expected = [0.04525828512, 0.002922972317, 0.002922972317, 0.03135771787]
    assert [found(i, j) for i in range(2) for j in range(2)] == pytest.approx(expected, rel=1e-6)
    alone = _gauss_fit()[0].conditionalCovarianceMatrix([RooRealVar("s", "s", 0)])
    assert alone(0, 0) == pytest.approx(0.035061623004236174, rel=1e-6)


@pytest.mark.xfail(strict=True, reason="reducedCovarianceMatrix returns the conditional one")
def test_the_reduced_covariance_is_the_plain_sub_matrix_in_the_order_asked() -> None:
    """ROOT's ``reducedCovarianceMatrix`` is a sub-matrix of V, rows in the order given."""
    result, v = _minos_fit()
    found = result.reducedCovarianceMatrix([v["s"], v["m"]])
    expected = [0.04147033044708756, 0.006184403959136655, 0.006184403959136655, 0.0481156984441]
    assert [found(i, j) for i in range(2) for j in range(2)] == pytest.approx(expected, rel=1e-6)


#: ``g.fitTo(data, Save())`` with the mean constant, printed verbose.
CONSTANT_VERBOSE = """
  RooFitResult: minimized FCN value: 102.327, estimated distance to minimum: 0.000218214
                covariance matrix quality: Full, accurate covariance matrix
                Status : MINIMIZE=0 HESSE=0\x20

    Constant Parameter    Value\x20\x20\x20\x20\x20
  --------------------  ------------
                     m    0.0000e+00

    Floating Parameter  InitialValue    FinalValue +/-  Error     GblCorr.
  --------------------  ------------  --------------------------  --------
                     s    2.0000e+00    1.8692e+00 +/-  1.86e-01  <none>

"""

#: What the table says for each ``covQual``, as ``setCovQual`` sets it.
QUALITIES = [
    (-1, "Unknown, matrix was externally provided"),
    (0, "Not calculated at all"),
    (1, "Approximation only, not accurate"),
    (2, "Full matrix, but forced positive-definite"),
    (3, "Full, accurate covariance matrix"),
]


def test_constant_parameters_are_listed_first_in_the_verbose_table(capsys: Any) -> None:
    """A parameter held constant is part of the fit's record, in its own table, as in ROOT."""
    g, data, v = _gauss()
    v["m"].setConstant(True)
    result = g.fitTo(data, Save=True, PrintLevel=-1)
    capsys.readouterr()
    result.Print("v")
    assert capsys.readouterr().out == CONSTANT_VERBOSE
    assert result.constPars().names() == ["m"]
    assert result.floatParsFinal().names() == ["s"]


@pytest.mark.parametrize(("quality", "words"), QUALITIES)
def test_each_covariance_quality_is_named_in_roots_words(
    quality: int, words: str, capsys: Any
) -> None:
    """The quality line is how one tells a forced or missing matrix from a good one."""
    result, _ = _gauss_fit()
    result.setCovQual(quality)
    assert result.covQual() == quality
    capsys.readouterr()
    result.Print()
    assert f"                covariance matrix quality: {words}\n" in capsys.readouterr().out


def test_a_fit_that_runs_out_of_calls_is_saved_with_its_starting_values_as_root_saves_it(
    capsys: Any,
) -> None:
    """``MaxCalls(5)`` stops MIGRAD before it moves; ROOT records status -1 and an
    approximate matrix, and leaves the parameters where they started."""
    g, data, v = _gauss()
    result = g.fitTo(data, Save=True, MaxCalls=5, Hesse=False, PrintLevel=-1)
    assert (result.status(), result.covQual()) == (-1, 1)
    assert (v["m"].getVal(), v["s"].getVal()) == (0.0, 2.0)
    assert v["m"].getError() == pytest.approx(0.28269399379713744, rel=1e-9)
    assert v["s"].getError() == pytest.approx(0.21808037373914957, rel=1e-9)
    capsys.readouterr()
    result.Print("v")
    out = capsys.readouterr().out
    assert "estimated distance to minimum: 0.239047\n" in out
    assert "                Status : MINIMIZE=-1 \n" in out
    assert "                     m    0.0000e+00    0.0000e+00 +/-  2.83e-01  <none>\n" in out


def test_randomized_parameters_are_drawn_from_the_covariance_with_roofits_generator() -> None:
    """``randomizePars`` feeds toy studies: with ROOT's seed it must draw ROOT's values."""
    result, _ = _minos_fit()
    generator().SetSeed(99)
    drawn = result.randomizePars()
    assert drawn.names() == ["c", "f", "m", "s"]
    expected = [-0.060392437856223655, 0.6194467349996472, -0.3612498528356794, 1.8267273058517386]
    assert [p.getVal() for p in drawn] == pytest.approx(expected, rel=1e-7)
    assert result.floatParsFinal()[0].getVal() == pytest.approx(-5.2136e-02, abs=1e-6)


def test_the_hesse_pdf_is_a_gaussian_in_the_parameters_asked_for(capsys: Any) -> None:
    """``createHessePdf`` of some or all parameters: a Gaussian centred on the fitted values,
    peaking at one there, named as ROOT names it."""
    result, v = _gauss_fit()
    part = result.createHessePdf([v["s"], RooRealVar("other", "other", 0)])
    whole = result.createHessePdf([v["s"], v["m"]])
    assert part.getVal() == whole.getVal() == 1.0
    capsys.readouterr()
    part.Print()
    whole.Print()
    assert capsys.readouterr().out == (
        "RooMultiVarGaussian::pdf_fitresult_g_gData[ x=(s) mu=(s_centralvalue) ] = 1\n"
        "RooMultiVarGaussian::pdf_fitresult_g_gData[ x=(m,s) mu=(m_centralvalue,s_centralvalue) ]"
        " = 1\n"
    )


def test_a_result_answers_to_its_name_and_class_and_an_empty_one_has_no_correlations() -> None:
    """A result made by hand is an empty record: nothing floats, nothing is correlated.

    (ROOT crashes printing one, so only its names are compared.)"""
    result = RooFitResult("r", "a result")
    result.SetName("renamed")
    assert (result.GetName(), result.GetTitle(), result.ClassName()) == (
        "renamed",
        "a result",
        "RooFitResult",
    )
    assert len(result.globalCorr()) == 0
    assert (result.printName(), result.printTitle(), result.printClassName()) == (
        "renamed",
        "a result",
        "RooFitResult",
    )
    assert result.printValue() == "0"
    fitted, _ = _gauss_fit()
    assert (fitted.GetName(), fitted.GetTitle()) == (
        "fitresult_g_gData",
        "Result of fit of p.d.f. g to dataset gData",
    )
