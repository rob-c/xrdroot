"""The variable transformations and their handler: made, prepared, applied and taken back."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from tmvasupport import classify, loader, regression_tree, session
from xrdroot.tmva import TMVAError, eigen
from xrdroot.tmva.dataset import DataSetInfo, Events
from xrdroot.tmva.handler import TransformationHandler, parse_definition
from xrdroot.tmva.log import Logger
from xrdroot.tmva.transforms import Normalize, Transform, make_transform
from xrdroot.tmva.variables import VariableInfo

__all__ = ["session"]


def _regression(transformations: str = "I;N") -> None:
    """A regression of ``fvalue`` on two variables, by k-nearest neighbours, normalised."""
    data = ROOT.TMVA.DataLoader("dataset")
    data.AddVariable("var1", "F")
    data.AddVariable("var2", "F")
    data.AddTarget("fvalue")
    data.AddRegressionTree(regression_tree(count=120))
    data.PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V")
    classify(
        [(ROOT.TMVA.Types.kKNN, "KNN", "!H:!V:nkNN=10:VarTransform=N,D")],
        data=data,
        options=f"!V:!Silent:AnalysisType=Regression:Transformations={transformations}",
    )


def test_a_regression_normalises_its_target_and_takes_its_output_back(session, capsys):
    _regression()
    printed = capsys.readouterr().out
    assert "Input : target 'fvalue' <---> Output : target 'fvalue'" in printed
    assert 'Create Transformation "N" with events from all classes.' in printed


def test_a_transformation_made_from_one_class_says_which(session, capsys):
    classify([(ROOT.TMVA.Types.kLD, "LD", "!H:!V:VarTransform=D_Signal+P_Background")], output="")
    printed = capsys.readouterr().out
    assert 'Create Transformation "D" with reference class Signal=(0)' in printed
    assert 'Create Transformation "P" with reference class Background=(1)' in printed


def test_a_transformation_made_from_a_class_not_known_is_refused(session):
    with pytest.raises(TMVAError, match="Class Nothing not known for variable transformation"):
        classify([(ROOT.TMVA.Types.kLD, "LD", "!H:!V:VarTransform=D_Nothing")], output="")


def test_a_transformation_not_known_is_refused(session):
    with pytest.raises(TMVAError, match="Variable transform 'X' unknown"):
        classify([(ROOT.TMVA.Types.kLD, "LD", "!H:!V:VarTransform=X")], output="")


def test_no_transformation_is_made_of_none(session):
    handler = TransformationHandler(loader().GetDataSetInfo(), "LD")
    handler.create("None", Logger("Factory"))
    handler.create("", Logger("Factory"))
    assert handler.transforms == [] and handler.name == ""


def test_a_definition_is_split_at_its_commas_and_pluses_outside_brackets():
    assert parse_definition("G(_V0_,_V1_),,D_Signal+N_AllClasses") == [
        ("G", -1),
        ("D", "Signal"),
        ("N", -1),
    ]


def _events(count: int = 50, nclasses: int = 2) -> Events:
    rng = np.random.default_rng(5)
    values = rng.normal(0, 1, (count, 2))
    return Events(
        values, values[:, :1] * 3, np.zeros((count, 0)), np.arange(count) % nclasses, np.ones(count)
    )


def test_a_handler_applies_its_chain_to_bare_values_and_inverts_its_targets(session):
    info = DataSetInfo("dataset")
    info.AddClass("Regression")
    info.variables = [VariableInfo("x"), VariableInfo("y")]
    info.targets = [VariableInfo("t")]
    handler = TransformationHandler(info, "Test")
    handler.create("I,N", Logger("Factory"))
    events = _events(nclasses=1)
    made = handler.prepare(events)
    assert np.allclose(handler.apply_values(events.values), made.values)
    back = handler.inverse_targets(made.targets)
    assert np.allclose(back, events.targets, atol=1e-5)
    assert handler.name == "Id_Norm"


def test_a_normalisation_maps_the_targets_onto_minus_one_to_one():
    transform = Normalize()
    events = _events()
    transform.prepare(events, 2)
    mapped = transform.apply_targets(events.targets, 2)
    assert np.isclose(mapped.min(), -1.0) and np.isclose(mapped.max(), 1.0)
    assert np.allclose(transform.inverse_targets(mapped, 2), events.targets, atol=1e-5)
    assert len(transform.params) == 3


def test_the_plain_transformation_fits_nothing_and_leaves_the_values():
    transform = Transform()
    transform.prepare(_events(), 1)
    assert transform.params == [None] and transform.announce() is None
    assert transform.apply(np.ones((2, 2))).sum() == 4.0
    assert isinstance(make_transform("Ident"), Transform)


def test_the_eigenvalues_of_a_diagonal_matrix_are_its_diagonal_largest_first():
    values, vectors = eigen.symmetric(np.diag([1.0, 3.0, 2.0]))
    assert list(values) == [3.0, 2.0, 1.0]
    assert np.allclose(np.abs(vectors), [[0, 0, 1], [1, 0, 0], [0, 1, 0]])


def test_an_empty_matrix_has_no_eigenvalues():
    values, vectors = eigen.symmetric(np.zeros((0, 0)))
    assert values.shape == (0,) and vectors.shape == (0, 0)


def test_a_matrix_too_large_to_hold_is_given_up_on_after_thirty_iterations():
    with np.errstate(all="ignore"):
        values, _ = eigen.symmetric([[1e308, 1e308], [1e308, -1e308]])
    assert np.isnan(values).all()


def test_the_inverse_square_root_of_a_covariance_whitens_it():
    covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    root = eigen.inverse_square_root(covariance)
    assert np.allclose(root @ covariance @ root.T, np.eye(2))
