"""``TMVA::Experimental``'s reader and scaler: a weight file asked about events, columns scaled."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from test_tmva_learn_bdt import train_one
from tmvasupport import session, weights
from xrdroot.tmva.log import CONFIG

__all__ = ["session"]

Experimental = ROOT.TMVA.Experimental


def test_a_reader_answers_one_event_with_a_vector_and_a_tensor_of_them_with_a_tensor(session):
    method = train_one(ROOT.TMVA.Types.kBDT, "BDT", "NTrees=3")
    CONFIG.silent = False
    reader = Experimental.RReader(weights("BDT"))
    assert CONFIG.silent is False
    assert list(reader.GetVariableNames()) == ["var1", "var2", "var3", "var4"]
    one = reader.Compute([0.5, 0.5, 0.5, 0.5])
    assert len(one) == 1 and one[0] == pytest.approx(method.evaluate(np.full((1, 4), 0.5))[0])
    table = Experimental.RTensor["float"](np.zeros(8, dtype=np.float32), [2, 4])
    assert reader.Compute(table).GetShape() == [2]
    compute = Experimental.Compute[4, "float"](reader)
    found = compute(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    assert found.dtype == np.float32 and len(found) == 2


def test_a_reader_of_a_multiclass_method_answers_every_class(session):
    train_one(ROOT.TMVA.Types.kBDT, "BDTG", "NTrees=2:BoostType=Grad", "Multiclass")
    reader = Experimental.RReader(weights("BDTG"))
    assert sum(reader.Compute([0.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0, abs=1e-6)
    table = Experimental.RTensor["float"](np.zeros(8, dtype=np.float32), [2, 4])
    assert reader.Compute(table).GetShape() == [2, 3]


def test_a_reader_of_a_regression_answers_its_target(session):
    train_one(ROOT.TMVA.Types.kBDT, "BDTR", "NTrees=2:BoostType=Grad", "Regression")
    reader = Experimental.RReader(weights("BDTR"))
    assert len(reader.Compute([1.0, 2.0])) == 1


def test_a_scaler_learns_each_columns_mean_and_deviation_and_scales_by_them():
    table = Experimental.RTensor["float"](np.array([1, 10, 3, 30], dtype=np.float32), [2, 2])
    scaler = Experimental.RStandardScaler["float"]()
    scaler.Fit(table)
    assert list(scaler.GetMeans()) == [2.0, 20.0]
    assert list(scaler.GetStds()) == pytest.approx([np.sqrt(2), np.sqrt(200)])
    scaled = scaler.Compute(table)
    assert type(scaled) is type(table) and scaled(1, 1) == pytest.approx(1 / np.sqrt(2))
    assert list(scaler.Compute([2.0, 20.0])) == [0.0, 0.0]


def test_a_scaler_refuses_a_tensor_that_is_not_a_table():
    scaler = Experimental.RStandardScaler["double"]()
    flat = Experimental.RTensor["double"]([3])
    with pytest.raises(RuntimeError, match=r"Can only fit to input tensor of rank 2\."):
        scaler.Fit(flat)
    with pytest.raises(RuntimeError, match=r"Can only compute output for input tensor of rank 2\."):
        scaler.Compute(flat)


def test_a_scaler_saved_to_a_file_is_read_back_beside_another_saved_there(session):
    table = Experimental.RTensor["float"](np.array([1, 10, 3, 30], dtype=np.float32), [2, 2])
    first, second = Experimental.RStandardScaler["float"](), Experimental.RStandardScaler["float"]()
    first.Fit(table)
    second.Fit(Experimental.RTensor["float"](np.array([0, 0, 2, 4], dtype=np.float32), [2, 2]))
    first.Save("first", "scalers.root")
    second.Save("second", "scalers.root")
    back = Experimental.RStandardScaler["float"]("first", "scalers.root")
    assert list(back.GetMeans()) == [2.0, 20.0]
    assert list(Experimental.RStandardScaler["float"]("second", "scalers.root").GetMeans()) == [
        1.0,
        2.0,
    ]
