"""``ROOT.Experimental.ML.RDataLoader``: a frame's entries as batches."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

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


def _flat(batches: Any) -> list[float]:
    return [float(v) for batch in batches for v in np.ravel(batch[0])]


def test_a_list_of_frames_read_lazily_is_split_as_one_long_frame():
    loader = RDataLoader(
        [Frame(x=np.arange(5.0)), Frame(x=np.arange(5.0, 10.0))], 3, shuffle=False,
        drop_remainder=False)
    training, testing = loader.train_test_split(0.4)
    assert _flat((b,) for b in training.as_numpy()) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert _flat((b,) for b in testing.as_numpy()) == [6.0, 7.0, 8.0, 9.0]
    assert (training.num_batches, training.last_batch_no_of_rows) == (2, 0)
    assert (testing.num_batches, testing.last_batch_no_of_rows) == (2, 1)


def test_a_list_of_frames_loaded_eagerly_keeps_each_frames_share_for_validation():
    loader = RDataLoader(
        [Frame(x=np.arange(5.0)), Frame(x=np.arange(5.0, 10.0))], 3, shuffle=False,
        drop_remainder=False, load_eager=True)
    training, testing = loader.train_test_split(0.4)
    assert _flat((b,) for b in training.as_numpy()) == [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
    assert _flat((b,) for b in testing.as_numpy()) == [3.0, 4.0, 8.0, 9.0]


def test_the_loader_unsplit_batches_every_entry_and_counts_its_whole_batches():
    loader = RDataLoader(Frame(x=np.arange(7.0)), 3, shuffle=False)
    assert _flat((b,) for b in loader.as_numpy()) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert (loader.num_batches, loader.last_batch_no_of_rows) == (2, 0)


def test_a_shuffled_split_holds_back_different_entries_for_different_seeds():
    def held(seed: int) -> list[float]:
        loader = RDataLoader(Frame(x=np.arange(20.0)), 4, shuffle=True, set_seed=seed)
        return sorted(_flat((b,) for b in loader.train_test_split(0.5)[1].as_numpy()))

    assert held(42) == held(42) and held(42) != held(43)


def _balanced(sampling: str, ratio: float, **options: Any) -> tuple[list[float], list[float]]:
    major, minor = Frame(x=np.arange(0.0, 40.0, 2.0)), Frame(x=np.arange(1.0, 10.0, 2.0))
    loader = RDataLoader(
        [minor, major], 2, shuffle=False, drop_remainder=False, load_eager=True,
        sampling_type=sampling, sampling_ratio=ratio, **options)
    training, testing = loader.train_test_split(0.4)
    return _flat((b,) for b in training.as_numpy()), _flat((b,) for b in testing.as_numpy())


def test_undersampling_cuts_the_larger_frame_to_the_ratio_and_puts_it_first():
    training, testing = _balanced("undersampling", 0.5)
    assert training == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0]
    assert testing == [24.0, 26.0, 28.0, 30.0, 7.0, 9.0]


def test_undersampling_with_replacement_repeats_the_majority_when_it_runs_short():
    training, _ = _balanced("undersampling", 0.2, replacement=True)
    assert training[:15] == [*np.arange(0.0, 24.0, 2.0), 0.0, 2.0, 4.0]


def test_undersampling_too_far_without_replacement_is_refused():
    with pytest.raises(ValueError, match="The sampling_ratio is too low"):
        _balanced("undersampling", 0.2)


def test_oversampling_repeats_the_smaller_frame_up_to_the_ratio_and_puts_it_first():
    training, testing = _balanced("oversampling", 0.5)
    assert training == [1.0, 3.0, 5.0, 1.0, 3.0, 5.0, *np.arange(0.0, 24.0, 2.0)]
    assert testing[:4] == [7.0, 9.0, 7.0, 9.0]


def test_sampling_needs_two_frames_and_one_of_its_two_ways():
    with pytest.raises(ValueError, match="neither 'undersampling' nor 'oversampling'"):
        RDataLoader([Frame(x=np.ones(2)), Frame(x=np.ones(3))], 2, sampling_type="smote")
    with pytest.raises(ValueError, match="was given 1"):
        RDataLoader(Frame(x=np.ones(2)), 2, sampling_type="oversampling")

