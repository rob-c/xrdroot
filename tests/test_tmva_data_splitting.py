"""How a data set is split into training and testing, renormalised and mixed, as TMVA does it."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from tmvasupport import gaussian, session
from xrdroot.tmva import TMVAError
from xrdroot.tmva.dataset import DataSetInfo, Events
from xrdroot.tmva.shuffle import RandomGenerator, draw_below
from xrdroot.tmva.splitting import _interleaved, _numbers

__all__ = ["session"]


def _split(
    options: str, signal: int = 40, background: int = 40, cut: str = "", **types: Any
) -> Any:
    """The data set of two variables over a signal and a background tree, split by ``options``."""
    made = ROOT.TMVA.DataLoader("dataset")
    made.AddVariable("var1", "F")
    made.AddVariable("var2", "F")
    kind = types.get("signal_type", "Training and Testing")
    made.AddTree(gaussian("TreeS", 1.0, 1, signal), "Signal", 1.0, "", kind)
    for number, (count, what) in enumerate(types.get("extra", [])):
        made.AddTree(gaussian(f"TreeX{number}", 1.0, 10 + number, count), "Signal", 1.0, "", what)
    made.AddTree(gaussian("TreeB", -1.0, 2, background), "Background")
    made.PrepareTrainingAndTestTree(cut, options)
    return made, made.dataset()


def _counts(dataset: Any, number: int) -> tuple[int, int]:
    return int(np.sum(dataset.train.classes == number)), int(np.sum(dataset.test.classes == number))


def test_a_fraction_to_train_on_is_that_share_of_the_classs_events(session):
    _, dataset = _split("TrainTestSplit_Signal=0.25:!V")
    assert _counts(dataset, 0) == (10, 30) and _counts(dataset, 1) == (20, 20)


def test_only_a_number_to_train_on_tests_on_the_rest(session):
    _, dataset = _split("nTrain_Signal=15:!V")
    assert _counts(dataset, 0) == (15, 25)


def test_only_a_number_to_test_on_trains_on_the_rest(session):
    _, dataset = _split("nTest_Signal=15:!V")
    assert _counts(dataset, 0) == (25, 15)


def test_more_events_asked_for_than_there_are_is_refused(session):
    with pytest.raises(TMVAError, match="More events requested than available"):
        _split("nTrain_Signal=50:!V")


def test_events_given_for_training_and_testing_keep_what_they_were_given_for(session):
    _, dataset = _split("!V", signal=10, signal_type="Training", extra=[(30, "Test")])
    assert _counts(dataset, 0) == (10, 30)


def test_a_short_training_sample_is_topped_up_from_the_unassigned_events(session):
    _, dataset = _split("!V", signal=10, signal_type="Training", extra=[(30, "Test"), (4, 2)])
    assert _counts(dataset, 0) == (14, 30)


def test_a_short_testing_sample_is_topped_up_from_the_unassigned_events(session):
    _, dataset = _split("!V", signal=30, signal_type="Training", extra=[(10, "Test"), (4, 2)])
    assert _counts(dataset, 0) == (30, 14)


def test_the_alternate_split_deals_the_events_out_in_turn(session):
    _, dataset = _split("SplitMode=Alternate:MixMode=Block:nTrain_Signal=5:!V")
    signal = dataset.train.values[dataset.train.classes == 0]
    assert len(signal) == 5 and _counts(dataset, 0) == (5, 35)


def test_a_block_split_keeps_the_first_events_asked_for(session):
    _, dataset = _split("SplitMode=Block:nTrain_Signal=5:nTest_Signal=5:NormMode=None:!V")
    assert _counts(dataset, 0) == (5, 5)


def test_a_random_split_drops_its_surplus_at_random(session):
    _, dataset = _split("nTrain_Signal=5:nTest_Signal=5:!V")
    assert _counts(dataset, 0) == (5, 5)


def test_a_preselection_scales_the_numbers_asked_for_when_so_opted(session, capsys):
    _, dataset = _split("nTrain_Signal=20:nTest_Signal=10:ScaleWithPreselEff:!V", cut="var1>0")
    printed = capsys.readouterr().out
    assert "scaling the number of requested training/testing events" in printed
    assert "preselection efficiency)" in printed and "a scaling factor has" in printed
    assert _counts(dataset, 0)[0] < 20


def test_a_preselection_otherwise_counts_the_events_after_it(session, capsys):
    _split("!V", cut="var1>0")
    assert "to be the number of events AFTER your preselection cuts" in capsys.readouterr().out


def test_the_alternate_mix_interleaves_the_classes(session):
    _, dataset = _split("SplitMode=Block:MixMode=Alternate:!V", signal=20, background=40)
    assert list(dataset.train.classes[:6]) == [0, 1, 0, 1, 0, 1]
    assert list(dataset.train.classes[-4:]) == [1, 1, 1, 1]


def test_the_alternate_insertion_puts_what_is_left_at_the_end():
    assert _interleaved([0, 0, 0, 0], [1, 1, 1], 1) == [0, 1, 0, 1, 0, 1, 0]
    assert _interleaved([0], [2, 2], 2) == [0, 2, 2]


def test_nothing_asked_for_with_unequal_given_samples_tops_up_the_smaller():
    assert _numbers(0, 0, (10, 30, 4)) == (14, 30, 14, 30)
    assert _numbers(0, 0, (30, 10, 4)) == (30, 14, 30, 14)


def test_an_unknown_normalisation_is_refused(session):
    with pytest.raises(TMVAError, match="Unknown NormMode: SOMETIMES"):
        _split("NormMode=Sometimes:!V")


def test_without_correlations_the_matrices_are_worked_out_but_not_printed(session, capsys):
    made, _ = _split("!Correlations:!V")
    assert "Correlation matrix" not in capsys.readouterr().out
    assert set(made.info.correlations) == {"Signal", "Background"}


def test_without_calculated_correlations_there_are_none(session):
    made, _ = _split("!CalcCorrelations:!V")
    assert made.info.correlations == {}


def test_a_generator_draws_whole_numbers_and_a_uniform_one():
    generator = RandomGenerator(100)
    assert 0 <= generator() < 2**32 and 0.0 < generator.rndm() < 1.0
    assert draw_below(generator, 1) == 0 and 0 <= draw_below(generator, 5) < 5


def test_a_dataset_info_names_itself_and_tells_the_signal():
    info = DataSetInfo("dataset")
    info.AddClass("Background")
    info.AddClass("Signal")
    assert info.GetName() == "dataset" and info.IsSignal(1) and not info.IsSignal(0)
    assert info.GetNClasses() == 2 and info.GetClassNameMaxLength() == 10


def test_no_samples_joined_are_one_empty_sample():
    joined = Events.joined([], 3, 1, 0)
    assert len(joined) == 0 and joined.values.shape == (0, 3)


def test_an_odd_event_left_over_by_the_alternate_split_goes_to_training(session):
    _, dataset = _split("SplitMode=Alternate:MixMode=Block:nTrain_Signal=30:!V", signal=41)
    assert _counts(dataset, 0) == (21, 11)
