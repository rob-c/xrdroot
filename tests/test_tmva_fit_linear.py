"""The linear discriminants, LD and Fisher: classification, regression, and their weight files."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from tmvafitsupport import REGRESSION, SPLIT, reader, regression_loader
from tmvasupport import classify, gaussian, session
from xrdroot.tmva import TMVAError

__all__ = ["session"]

LD, FISHER = ROOT.TMVA.Types.kLD, ROOT.TMVA.Types.kFisher


def _scaled(expressions: tuple[str, ...]):
    """A loader of the given expressions of the variables, over the usual signal and background."""
    made = ROOT.TMVA.DataLoader("dataset")
    for expression in expressions:
        made.AddVariable(expression, "F")
    made.AddSignalTree(gaussian("TreeS", 1.0, 1))
    made.AddBackgroundTree(gaussian("TreeB", -1.0, 2))
    made.PrepareTrainingAndTestTree("", SPLIT)
    return made


def test_ld_and_fisher_both_ways_are_read_back_ranking_signal_above_background(session):
    factory = classify(
        [
            (LD, "LD", "!H:!V:VarTransform=N"),
            (FISHER, "Fisher", "!H:!V"),
            (FISHER, "Mahalanobis", "!H:!V:Method=Mahalanobis"),
        ]
    )
    for title in ("LD", "Fisher", "Mahalanobis"):
        made = reader(title, silent=True)
        high = made.EvaluateMVA([2.0, 2.0, 2.0, 2.0], title)
        low = made.EvaluateMVA([-2.0, -2.0, -2.0, -2.0], title)
        assert high > low
        assert factory.GetMethod("dataset", title).GetMethodTypeName() in ("LD", "Fisher")


def test_the_coefficients_of_transformed_variables_say_so(session, capsys):
    classify([(FISHER, "Fisher", "!H:!V:VarTransform=D,N")])
    printed = capsys.readouterr().out
    assert "NOTE: The coefficients must be applied to TRANFORMED variables" in printed
    assert "  -- " in printed


def test_ld_regresses_a_target_and_can_leave_out_negative_weights(session):
    options = "!H:!V:IgnoreNegWeightsInTraining"
    classify([(LD, "LD", options)], options=REGRESSION, data=regression_loader())
    found = reader("LD", ("var1", "var2"), silent=True).EvaluateRegression(0, [1.0, 2.0], "LD")
    assert found == pytest.approx(10 + 5 * 4, abs=15)


def test_a_nearly_singular_matrix_is_warned_of(session, capsys):
    classify([(LD, "LD", "!H:!V")], data=_scaled(("var1*1e-7", "var2*1e-7", "var3*1e-7", "var4")))
    assert "matrix is almost singular" in capsys.readouterr().out


@pytest.mark.parametrize("kind", [LD, FISHER])
def test_a_singular_matrix_is_refused(session, kind):
    with pytest.raises(TMVAError, match="matrix is singular"):
        classify([(kind, "Linear", "!H:!V")], data=_scaled(("var1", "var2", "2*var1")))
