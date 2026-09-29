"""The function discriminant: its formula, its fitters, every analysis, and its weight file."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from tmvafitsupport import MULTICLASS, REGRESSION, multiclass_loader, reader, regression_loader
from tmvasupport import classify, loader, session
from xrdroot.tmva import TMVAError
from xrdroot.tmva.fdaformula import FormulaError, compile_fda, tformula_text

__all__ = ["session"]

FDA = ROOT.TMVA.Types.kFDA
#: A linear function of the four variables, and its parameters' ranges.
LINEAR = "Formula=(0)+(1)*x0+(2)*x1+(3)*x2+(4)*x3:ParRanges=(-1,1);(-1,1);(-1,1);(-1,1);(-1,1)"
#: A Monte Carlo fit small enough to be quick.
MC = "!H:!V:FitMethod=MC:SampleSize=300"
#: A genetic fit small enough to be quick.
GA = "!H:!V:FitMethod=GA:PopSize=20:Steps=3:Cycles=1"


def _book(options: str, silent: str = "Silent") -> None:
    factory = ROOT.TMVA.Factory("job", f"!V:{silent}:AnalysisType=Classification")
    factory.BookMethod(loader(), FDA, "FDA", options)


def test_a_monte_carlo_fit_is_trained_and_read_back_by_a_reader(session, capsys):
    classify([(FDA, "FDA_MC", f"{MC}:{LINEAR}")])
    printed = capsys.readouterr().out
    assert 'TFormula-compatible formula string: "[0]+[1]*[5]' in printed
    assert "Value of estimator at minimum" in printed
    made = reader("FDA_MC")
    assert 'User-defined formula string       : "(0)+(1)*x0' in capsys.readouterr().out
    assert np.isfinite(made.EvaluateMVA([1.0, 1.0, 1.0, 1.0], "FDA_MC"))


def test_a_genetic_fit_of_a_formula_of_functions_is_trained(session, capsys):
    formula = "Formula=(0)+(1)*exp(-x0^2)+(2)*sin(x1)+abs(x2):ParRanges=(-1,1);(0,2);(-1,1);"
    classify([(FDA, "FDA_GA", f"{GA}:{formula}")])
    assert 'Results for parameter fit using "GA" fitter' in capsys.readouterr().out


def test_a_regression_is_fitted_to_its_target_and_read_back(session):
    options = f"{MC}:Formula=(0)+(1)*x0+(2)*x1*x1:ParRanges=(-10,10);(0,20);(0,10)"
    classify([(FDA, "FDA_R", options)], options=REGRESSION, data=regression_loader())
    made = reader("FDA_R", ("var1", "var2"))
    assert np.isfinite(made.EvaluateRegression(0, [1.0, 1.0], "FDA_R"))


def test_several_classes_each_have_their_function_and_their_softmax_is_read_back(session):
    options = f"{MC}:Formula=(0)+(1)*x0:ParRanges=(-1,1);(-1,1)"
    classify([(FDA, "FDA_M", options)], options=MULTICLASS, data=multiclass_loader())
    outputs = reader("FDA_M").EvaluateMulticlass([1.0, 0.0, 0.0, 0.0], "FDA_M")
    assert sum(outputs) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    ("options", "said"),
    [
        ("Formula=(0)+(1)*x0:ParRanges=(0,1)(0,1)", "Mismatch in parameter string"),
        ("Formula=(0):ParRanges=(1,0)", "max > min in interval"),
        ("Formula=(0)+(5):ParRanges=(0,1)", 'expression: "\\(5\\)"'),
        ("Formula=(0)*x7:ParRanges=(0,1)", 'expression: "x7"'),
        ("Formula=(0)+*:ParRanges=(0,1)", "could not be properly compiled"),
        ("Formula=(0)<x0:ParRanges=(0,1)", "is not arithmetic"),
    ],
)
def test_a_formula_or_ranges_fda_cannot_use_is_refused(session, options, said):
    with pytest.raises(TMVAError, match=said):
        _book(f"FitMethod=MC:{options}")


@pytest.mark.parametrize("options", ["FitMethod=MINUIT", "FitMethod=MC:Converger=MINUIT"])
def test_a_fitter_or_converger_xrdroot_does_not_have_is_refused_by_name(session, options):
    with pytest.raises(TMVAError, match="xrdroot does not have"):
        _book(f"{options}:Formula=(0):ParRanges=(0,1)")


@pytest.mark.parametrize(
    "text",
    [
        "'a'",
        "sin(a)[c]",
        "a[b]",
        "[0.5]",
        "a[0.5]",
        "[0]//[1]",
        "~[0]",
        "[0]([1])",
        "sin(x=[0])",
        "foo([0])",
        "[0] if [1] else [2]",
    ],
)
def test_arithmetic_tformula_does_not_know_is_refused_by_name(text):
    with pytest.raises(FormulaError, match="is not arithmetic"):
        compile_fda(text)


def test_a_formula_evaluates_its_signs_functions_and_operators_for_many_sets_at_once():
    run = compile_fda(tformula_text("-(0)+(+x0)%2+pow(x1,2)-TMath::Max(x0,x1)/1", 1, 2))
    found = run([np.array([[1.0], [2.0]]), np.array([3.0, 4.0]), np.array([1.0, 2.0])])
    assert found.shape == (2, 2)
    assert found[0].tolist() == pytest.approx([-1 + 1 + 1 - 3, -1 + 0 + 4 - 4])


def test_a_range_s_numbers_are_read_as_far_as_they_go_and_nothing_is_read_as_zero(session, capsys):
    _book("FitMethod=MC:Formula=(0)+(1):ParRanges=(abc,1);(-2.5e0junk,)", "!Silent")
    printed = capsys.readouterr().out
    assert "parameter 0 : [0,1]" in printed and "parameter 1 : [-2.5,0]" in printed


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_a_class_without_weight_is_refused_when_the_fit_starts(session):
    made = loader(split="SplitMode=Random:NormMode=None:!V")
    made.SetBackgroundWeightExpression("0")
    factory = ROOT.TMVA.Factory("job", "!V:Silent:AnalysisType=Classification")
    factory.BookMethod(made, FDA, "FDA", f"{MC}:Formula=(0):ParRanges=(0,1)")
    with pytest.raises(TMVAError, match="Troubles in sum of weights"):
        factory.TrainAllMethods()
