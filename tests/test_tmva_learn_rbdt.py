"""``SaveXGBoost`` and ``RBDT``: an XGBoost model saved in a ROOT file, read back, evaluated."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from tmvasupport import session
from xrdroot.tmva import rbdt

__all__ = ["session"]

xgboost = pytest.importorskip("xgboost")
Experimental = ROOT.TMVA.Experimental


def sample(classes: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Events of three variables, one of them missing now and then, and their classes."""
    rng = np.random.default_rng(8)
    values = rng.normal(size=(120, 3)).astype(np.float32)
    labels = (values[:, 0] + values[:, 1] > 0).astype(int) + (values[:, 2] > 1) * (classes - 2)
    values[::7, 1] = np.nan
    return values, labels


def test_a_binary_classifier_answers_as_xgboost_does(session):
    values, labels = sample()
    model = xgboost.XGBClassifier(n_estimators=4, max_depth=2).fit(values, labels)
    Experimental.SaveXGBoost(model, "model", "models.root", 3)
    found = Experimental.RBDT("model", "models.root").Compute(values)
    assert found.shape == (120, 1)
    assert np.allclose(found[:, 0], model.predict_proba(values)[:, 1], atol=1e-6)


def test_a_multiclass_classifier_answers_every_class_as_xgboost_does(session):
    values, labels = sample(3)
    model = xgboost.XGBClassifier(n_estimators=3, max_depth=2).fit(values, labels)
    Experimental.SaveXGBoost(model, "multi", "models.root", 3)
    found = Experimental.RBDT("multi", "models.root").Compute(values)
    assert np.allclose(found, model.predict_proba(values), atol=1e-6)


def test_a_regression_saved_beside_another_model_answers_as_xgboost_does(session):
    values, labels = sample()
    target = values[:, 0] * 2.0 + labels
    classifier = xgboost.XGBClassifier(n_estimators=2, max_depth=2).fit(values, labels)
    Experimental.SaveXGBoost(classifier, "first", "models.root")
    model = xgboost.XGBRegressor(n_estimators=4, max_depth=3).fit(values, target)
    Experimental.SaveXGBoost(model, "second", "models.root", 3)
    read = Experimental.RBDT("second", "models.root")
    assert np.allclose(read.Compute(values)[:, 0], model.predict(values), atol=1e-4)
    one = read.Compute([0.1, 0.2, 0.3])
    assert list(one) == pytest.approx(
        [float(model.predict(np.array([[0.1, 0.2, 0.3]]))[0])], abs=1e-4
    )
    tensor = Experimental.RTensor["float"](values[:2].ravel(), [2, 3])
    assert read.Compute(tensor).GetShape() == [2, 1]


def test_the_base_scores_are_a_list_of_one_per_class_or_a_single_number():
    config = {"learner": {"learner_model_param": {"base_score": "[5E-1,2E-1]"}}}
    assert rbdt._base_scores(config) == [0.5, 0.2]
    assert rbdt._base_scores({"learner": {"learner_model_param": {"base_score": "3"}}}) == [3.0]
    assert rbdt._base_scores({"learner": {"learner_model_param": {}}}) == [0.5]
