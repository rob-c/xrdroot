"""``DataSetFactory::RenormEvents``: the training weights renormalised, and the numbers printed.

``NormMode=NumEvents`` scales each class's training weights so that they sum
to its number of training events; ``EqualNumEvents`` - the default - so that
every class sums to the first class's number; ``None`` leaves them alone.
Test events are never touched. The table of how many events each class has
for training and testing is printed here, as TMVA prints it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import DataSetInfo, Events
from .log import Logger
from .reading import ClassCounts

__all__ = ["renormalise"]


def _sums(dsi: DataSetInfo, samples: list[list[Events]]) -> None:
    """The training and testing sums of signal and background weights, kept on ``dsi``."""
    signal = dsi.GetSignalClassIndex()
    sums = {"train_s": 0.0, "train_b": 0.0, "test_s": 0.0, "test_b": 0.0}
    for number, (train, test) in enumerate(samples):
        side = "s" if number == signal else "b"
        sums[f"train_{side}"] += float(np.sum(train.weights))
        sums[f"test_{side}"] += float(np.sum(test.weights))
    dsi.sums = sums


def _ratio(size: float, total: float) -> float:
    """``size / total`` as C++ divides floats: infinite, not an error, for a class of no weight."""
    return size / total if total else float("inf")


def _factors(samples: list[list[Events]], mode: str, dsi: DataSetInfo) -> list[float]:
    """Each class's renormalisation factor, or the refusal of a mode TMVA does not know."""
    sizes = [float(np.float32(len(train))) for train, _ in samples]
    totals = [float(np.sum(train.weights)) for train, _ in samples]
    if mode == "NUMEVENTS":
        return [_ratio(size, total) for size, total in zip(sizes, totals, strict=False)]
    if mode == "EQUALNUMEVENTS":
        _explain_equal(dsi)
        return [_ratio(sizes[0], total) for total in totals]
    raise Logger("DataSetFactory").fatal(
        f"Dataset[{dsi.name}] : <PrepareForTrainingAndTesting> Unknown NormMode: {mode}"
    )


def _explain_equal(dsi: DataSetInfo) -> None:
    log = Logger("DataSetFactory")
    prefix = f"Dataset[{dsi.name}] : "
    for line in (
        'Weight renormalisation mode: "EqualNumEvents": renormalises all event classes ...',
        " such that the effective (weighted) number of events in each class is the same ",
        " (and equals the number of events (entries) given for class=0 )",
        "... i.e. such that Sum[i=1..N_j]{w_i} = N_classA, j=classA, classB, ...",
        "... (note that N_j is the sum of TRAINING events",
        " ..... Testing events are not renormalised nor included in the renormalisation factor!)",
    ):
        log.info(prefix + line)


def _table(dsi: DataSetInfo, samples: list[list[Events]], counts: list[ClassCounts]) -> None:
    """The ``Number of training and testing events`` table."""
    log = Logger("DataSetFactory")
    width = dsi.GetClassNameMaxLength()
    log.info("Number of training and testing events")
    log.info("-" * 75)
    for info, (train, test), count in zip(dsi.classes, samples, counts, strict=False):
        name = info.name.ljust(width)
        log.info(f"{name} -- training events            : {len(train)}")
        log.info(f"{name} -- testing events             : {len(test)}")
        log.info(f"{name} -- training and testing events: {len(train) + len(test)}")
        if count.after < count.before:
            log.info(
                f"Dataset[{dsi.name}] : {name} -- due to the preselection a scaling factor has "
                f"been applied to the numbers of requested events: {count.cut_scaling():g}"
            )
    log.info("")


def renormalise(
    dsi: DataSetInfo, samples: list[list[Events]], counts: list[ClassCounts], mode: str
) -> None:
    """Scale each class's training weights in place, as ``NormMode`` asks, and say how many."""
    _sums(dsi, samples)
    dsi.normalization = mode
    if mode == "NONE":
        Logger("DataSetFactory").info(
            f"Dataset[{dsi.name}] : No weight renormalisation applied: "
            "use original global and event weights"
        )
        return
    for (train, _), factor in zip(samples, _factors(samples, mode, dsi), strict=False):
        train.weights = train.weights * factor
    _table(dsi, samples, counts)
    _sums(dsi, samples)


def summary(dsi: DataSetInfo, counts: list[ClassCounts]) -> None:
    """``Number of events in input trees``, and what each class's cut let through."""
    log = Logger("DataSetFactory")
    log.header(f"[{dsi.name}] : Number of events in input trees")
    if not any(info.cut for info in dsi.classes):
        return
    width = dsi.GetClassNameMaxLength()
    prefix = f"Dataset[{dsi.name}] :     "
    for info, count in zip(dsi.classes, counts, strict=False):
        name = info.name.ljust(width)
        log.info(f'{prefix}{name} requirement: "{info.cut}"')
        log.info(
            f"{prefix}{name}      -- number of events passed: {count.after:<5}"
            f"  / sum of weights: {_g(count.weight_after):<5}"
        )
        efficiency = count.weight_after / count.weight_before if count.weight_before else 0.0
        log.info(f"{prefix}{name}      -- efficiency             : {_g(efficiency):<6}")


def _g(value: Any) -> str:
    """A number as a C++ stream writes it by default."""
    return format(float(value), "g")
