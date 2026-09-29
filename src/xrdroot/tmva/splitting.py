"""Which events train and which test: ``DataSetFactory::MixEvents`` and ``RenormEvents``.

The options are ``PrepareTrainingAndTestTree``'s - ``nTrain_Signal=1000``,
``SplitMode=Random``, ``NormMode=NumEvents`` - and the algorithm is TMVA's,
step for step and draw for draw: each class's unassigned events shuffled,
the numbers asked for worked out by TMVA's three cases, the events handed
out, any surplus dropped at random, the training weights renormalised, and
the classes mixed. Every shuffle is :func:`~.shuffle.shuffle` with the one
generator TMVA seeds with ``SplitSeed``, in TMVA's order, so the samples are
TMVA's own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import MAX_TREE_TYPE, TESTING, TRAINING, DataSet, DataSetInfo, Events
from .log import Logger
from .options import Options
from .reading import ClassCounts
from .shuffle import RandomGenerator, shuffle

__all__ = ["SplitSpec", "mix_events", "split_spec"]


@dataclass
class SplitSpec:
    """The options of ``PrepareTrainingAndTestTree`` that say how to split."""

    split_mode: str = "RANDOM"
    mix_mode: str = "RANDOM"
    seed: int = 100
    norm_mode: str = "EQUALNUMEVENTS"
    scale_with_presel: bool = False
    correlations: bool = True
    compute_correlations: bool = True


def split_spec(dsi: DataSetInfo, counts: list[ClassCounts]) -> SplitSpec:
    """``DataSetFactory::InitOptions``: the split options, and each class's numbers asked for."""
    options = Options(dsi.split_options)
    split = options.text_of("SplitMode", "Random").upper()
    mix = options.text_of("MixMode", "SameAsSplitMode").upper()
    for info, count in zip(dsi.classes, counts):
        count.train_requested = options.integer(f"nTrain_{info.name}", 0)
        count.test_requested = options.integer(f"nTest_{info.name}", 0)
        count.split_requested = options.number(f"TrainTestSplit_{info.name}", 0.0)
    return SplitSpec(
        split,
        split if mix == "SAMEASSPLITMODE" else mix,
        options.integer("SplitSeed", 100),
        options.text_of("NormMode", "EqualNumEvents").upper(),
        options.flag("ScaleWithPreselEff", False),
        options.flag("Correlations", True),
        options.flag("CalcCorrelations", True),
    )


def _requested(count: ClassCounts, available: int, spec: SplitSpec) -> tuple[int, int]:
    """The training and testing numbers asked for, after ``TrainTestSplit_`` and scaling."""
    if 0.0 < count.split_requested < 1.0:
        count.train_requested = int(count.split_requested * available)
        count.test_requested = 0
    scale = count.cut_scaling() if spec.scale_with_presel else 1.0
    return int(count.train_requested * scale), int(count.test_requested * scale)


def _numbers(train: int, test: int, avail: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """TMVA's cases: how many to use for training and testing, and the numbers now asked for."""
    have_train, have_test, undefined = avail
    everything = have_train + have_test + undefined
    if train == 0 and test == 0:
        if undefined >= abs(have_train - have_test):
            use_train = use_test = everything // 2
        else:
            use_train, use_test = have_train, have_test
            if have_train < have_test:
                use_train += undefined
            else:
                use_test += undefined
        return use_train, use_test, use_train, use_test
    if test == 0:
        use_train = max(train, have_train)
        return use_train, everything - use_train, train, everything - use_train
    if train == 0:
        use_test = max(test, have_test)
        return everything - use_test, use_test, everything - use_test, test
    need_train, need_test = max(train - have_train, 0), max(test - have_test, 0)
    free = max(undefined - need_train - need_test, 0)
    use_train = max(train, have_train) + free // 2
    return use_train, everything - use_train, train, test


def _alternate(train: Events, test: Events, undefined: Events, wanted: int) -> list[Events]:
    """``SplitMode=Alternate``: the unassigned events dealt out, one to each in turn."""
    to_train, to_test = [], []
    count, position = len(train), 0
    while position < len(undefined):
        count += 1
        if count <= wanted:
            to_train.append(position)
            position += 1
        if position < len(undefined):
            to_test.append(position)
            position += 1
    return [
        Events.joined([train, undefined.take(np.array(to_train, dtype=int))], *_shape(train)),
        Events.joined([test, undefined.take(np.array(to_test, dtype=int))], *_shape(test)),
    ]


def _shape(events: Events) -> tuple[int, int, int]:
    return events.values.shape[1], events.targets.shape[1], events.spectators.shape[1]


def _trimmed(events: Events, wanted: int, spec: SplitSpec, rng: Any) -> Events:
    """A sample cut down to the number asked for: at random, or from the end."""
    size = len(events)
    if size <= wanted:
        return events
    if "RANDOM" not in spec.split_mode:
        return events.take(np.arange(wanted))
    indices = list(range(size))
    shuffle(indices, rng)
    keep = np.ones(size, dtype=bool)
    keep[indices[: size - wanted]] = False
    return events.take(keep)


def _say_scaling(count: ClassCounts, spec: SplitSpec, name: str) -> None:
    """What ``MixEvents`` says of a preselection that kept fewer than all of a class's events."""
    if count.cut_scaling() >= 1:
        return
    if spec.scale_with_presel:
        what = "scaling the number of requested training/testing events\n to be scaled by the "
        what += "preselection efficiency"
    else:
        what = "interpreting the requested number of training/testing events\n to be the number "
        what += "of events AFTER your preselection cuts"
    Logger("DataSetFactory").info(f"Dataset[{name}] :  you have opted for {what}")


def _gathered(count: ClassCounts, kind: int, shape: tuple[int, int, int]) -> Events:
    return Events.joined(count.events.get(kind, []), *shape)


def _assign(
    count: ClassCounts, spec: SplitSpec, rng: Any, shape: tuple[int, int, int], name: str
) -> list[Events]:
    """One class's training and testing samples, as ``MixEvents``'s loop over classes makes them."""
    train, test = _gathered(count, TRAINING, shape), _gathered(count, TESTING, shape)
    undefined = _gathered(count, MAX_TREE_TYPE, shape)
    avail = (len(train), len(test), len(undefined))
    wanted = _requested(count, sum(avail), spec)
    _say_scaling(count, spec, name)
    use_train, use_test, want_train, want_test = _numbers(*wanted, avail)
    if spec.split_mode == "ALTERNATE":
        train, test = _alternate(train, test, undefined, want_train)
    else:
        to_train = max(use_train - avail[0], 0)
        to_test = max(use_test - avail[1], 0)
        if to_train + to_test > avail[2]:
            raise Logger("DataSetFactory").fatal(
                f"Dataset[{name}] : More events requested than available!"
            )
        train = Events.joined([train, undefined.take(np.arange(to_train))], *shape)
        rest = undefined.take(np.arange(to_train, to_train + to_test))
        test = Events.joined([test, rest], *shape)
    scale = float(np.float32(count.cut_scaling()))
    if spec.scale_with_presel and scale < 1:
        Logger("DataSetFactory").info(
            f" ( {count.train_requested} * {scale:g} preselection efficiency)"
        )
    else:
        Logger("DataSetFactory").info("")
    return [_trimmed(train, want_train, spec, rng), _trimmed(test, want_test, spec, rng)]


def _shuffled_undefined(count: ClassCounts, rng: Any, shape: tuple[int, int, int]) -> None:
    undefined = _gathered(count, MAX_TREE_TYPE, shape)
    if len(undefined):
        order = list(range(len(undefined)))
        shuffle(order, rng)
        count.events[MAX_TREE_TYPE] = [undefined.take(np.array(order, dtype=int))]


def mix_events(dsi: DataSetInfo, counts: list[ClassCounts], spec: SplitSpec) -> DataSet:
    """``DataSetFactory::MixEvents``: the data set, split, renormalised and mixed."""
    from .renorm import renormalise

    rng = RandomGenerator(spec.seed)
    shape = (dsi.GetNVariables(), dsi.GetNTargets(), dsi.GetNSpectators())
    if "RANDOM" in spec.split_mode:
        for count in counts:
            _shuffled_undefined(count, rng, shape)
    samples = [_assign(count, spec, rng, shape, dsi.name) for count in counts]
    renormalise(dsi, samples, counts, spec.norm_mode)
    train = _mixed([sample[0] for sample in samples], spec, shape)
    test = _mixed([sample[1] for sample in samples], spec, shape)
    if spec.mix_mode == "RANDOM":
        train, test = _random_order(train, rng), _random_order(test, rng)
    _check_sizes(dsi.name, train, test)
    return DataSet(train, test)


def _check_sizes(name: str, train: Events, test: Events) -> None:
    """A data set with nothing to train on stops the job; one with nothing to test on only warns."""
    log = Logger("DataSetFactory")
    if not len(train):
        raise log.fatal(
            f"Dataset {name} does not have any training events,"
            " I better stop here and let you fix that one first "
        )
    if not len(test):
        log.error(
            f"Dataset {name} does not have any testing events, guess that will cause"
            " problems later..but for now, I continue "
        )


def _random_order(events: Events, rng: Any) -> Events:
    order = list(range(len(events)))
    shuffle(order, rng)
    return events.take(np.array(order, dtype=int))


def _mixed(parts: list[Events], spec: SplitSpec, shape: tuple[int, int, int]) -> Events:
    """The classes put together: one after another, or interleaved for ``MixMode=Alternate``."""
    if spec.mix_mode != "ALTERNATE" or len(parts) < 2:
        return Events.joined(parts, *shape)
    order: list[tuple[int, int]] = [(0, row) for row in range(len(parts[0]))]
    for number in range(1, len(parts)):
        order = _interleaved(order, [(number, row) for row in range(len(parts[number]))], number)
    pieces = [parts[number].take(np.array([row], dtype=int)) for number, row in order]
    return Events.joined(pieces, *shape)


def _interleaved(target: list[Any], extra: list[Any], number: int) -> list[Any]:
    """``MixMode=Alternate``'s insertion: every ``number + 1``-th place, the rest at the end."""
    position = -1
    for index, item in enumerate(extra):
        if len(target) - position < number + 1:
            return target + extra[index:]
        position += number + 1
        target.insert(position, item)
    return target
