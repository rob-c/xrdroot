"""``ROOT.Experimental.ML.RDataLoader``: a frame's entries as batches."""

from __future__ import annotations

from typing import Any

import numpy as np

import xrdroot.pyroot as ROOT

RDataLoader = ROOT.Experimental.ML.RDataLoader


class Frame:
    """What the loader asks of an ``RDataFrame``: its columns' names, and their values."""

    def __init__(self, **columns: Any) -> None:
        self.columns = columns

    def GetColumnNames(self) -> list[str]:
        return list(self.columns)

    def AsNumpy(self, names: list[str]) -> dict[str, Any]:
        return {name: self.columns[name] for name in names}


def test_a_frame_is_batched_in_order_with_its_short_last_batch_kept_when_asked():
    frame = ROOT.RDataFrame(10).Define("x", "(float)rdfentry_").Define("y", "x*2")
    loader = RDataLoader(frame, 4, shuffle=False, drop_remainder=False)
    training, testing = loader.train_test_split(0.2)
    batches = list(training.as_numpy())
    assert [len(batch) for batch in batches] == [4, 4]
    assert batches[0][:, 1].tolist() == [0.0, 2.0, 4.0, 6.0]
    assert [len(batch) for batch in testing.as_numpy()] == [2]
    assert training.columns == ["x", "y"]


def test_the_short_last_batch_is_dropped_and_the_order_shuffled_by_default():
    loader = RDataLoader(Frame(x=np.arange(10.0)), 3, set_seed=4)
    training, testing = loader.train_test_split()
    first = np.concatenate(list(training.as_numpy()))[:, 0]
    assert len(first) == 9 and sorted(first) != list(first)
    assert list(testing.as_numpy()) == []


def test_targets_and_weights_come_as_batches_of_their_own_beside_the_inputs():
    frame = Frame(x=np.arange(4.0), t=np.ones(4), u=np.zeros(4), w=np.full(4, 0.5))
    loader = RDataLoader(frame, 2, target=["t", "u"], weights="w", shuffle=False)
    inputs, targets, weights = next(loader.train_test_split()[0].as_numpy())
    assert inputs.shape == (2, 1) and targets.shape == (2, 2) and weights.tolist() == [[0.5], [0.5]]
    only = RDataLoader(Frame(x=np.arange(4.0), t=np.ones(4)), 2, target="t", shuffle=False)
    inputs, targets = next(only.train_test_split()[0].as_numpy())
    assert targets.tolist() == [[1.0], [1.0]]


def test_a_vector_column_is_spread_over_columns_padded_with_zeros_or_cut_short():
    frame = Frame(v=[[1.0], [1.0, 2.0, 3.0, 4.0]], x=np.array([5.0, 6.0]))
    loader = RDataLoader(frame, 2, max_vec_sizes={"v": 3}, shuffle=False)
    assert loader.columns == ["v_0", "v_1", "v_2", "x"]
    assert loader.inputs.tolist() == [[1.0, 0.0, 0.0, 5.0], [1.0, 2.0, 3.0, 6.0]]


def test_a_frame_of_nothing_but_targets_has_no_inputs_to_batch():
    loader = RDataLoader(Frame(t=np.ones(3)), 2, target="t")
    assert loader.inputs.shape == (0, 0)
    assert list(loader.train_test_split(0.5)[0].as_numpy()) == []
