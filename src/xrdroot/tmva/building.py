"""``DataSetFactory::CreateDataSet``: from a loader's trees to its training and test samples.

The events are read (:mod:`.reading`), counted, split and renormalised
(:mod:`.splitting`, :mod:`.renorm`), the variables' ranges taken from the
training sample, and each class's correlation matrix worked out and printed
as ``DataSetInfo::PrintCorrelationMatrix`` prints it - all in TMVA's order,
so that the lines come out as TMVA's do.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import DataSet, DataSetInfo, Events
from .log import Logger
from .reading import TreeInput, read_class
from .renorm import summary
from .splitting import mix_events, split_spec

__all__ = ["correlation_matrix", "create_dataset", "formatted_matrix"]


def correlation_matrix(events: Events) -> Any:
    """``CalcCorrelationMatrix``: the weighted correlations of the variables, 1 on the diagonal."""
    weights = events.weights
    total = float(np.sum(weights))
    first = weights @ events.values
    second = (events.values * weights[:, None]).T @ events.values
    covariance = second / total - np.outer(first, first) / (total * total)
    diagonal = np.diag(covariance)
    scale = np.outer(diagonal, diagonal)
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(scale > 0, covariance / np.sqrt(np.where(scale > 0, scale, 1.0)), 0.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def formatted_matrix(matrix: Any, labels: list[str]) -> list[str]:
    """``Tools::FormattedOutput``: a matrix with its labels, three decimals and a sign."""
    widths = [max(len(label), 7) for label in labels]
    widest = max([7, *widths])
    rule = "-" * (widest + 1 + sum(width + 1 for width in widths))
    lines = [
        rule,
        " " * (widest + 1) + "".join(l.rjust(w + 1) for l, w in zip(labels, widths, strict=False)),
    ]
    for row, label in enumerate(labels):
        cells = "".join(
            format(matrix[row, col], "+1.3f").rjust(w + 1) for col, w in enumerate(widths)
        )
        lines.append(label.rjust(widest) + ":" + cells)
    lines.append(rule)
    return lines


def _set_ranges(dsi: DataSetInfo, train: Events) -> None:
    """``CalcMinMax``: every variable's, target's and spectator's range over the training sample."""
    log = Logger("DataSetFactory")
    for infos, values in (
        (dsi.variables, train.values),
        (dsi.targets, train.targets),
        (dsi.spectators, train.spectators),
    ):
        for index, info in enumerate(infos):
            column = values[:, index]
            info.minimum, info.maximum = float(column.min()), float(column.max())
            if infos is dsi.variables and abs(info.maximum - info.minimum) <= 1.17549435e-38:
                log.warning(
                    f"Dataset[{dsi.name}] : Variable {info.expression} is constant. "
                    "Please remove the variable."
                )


def create_dataset(dsi: DataSetInfo, inputs: list[TreeInput]) -> DataSet:
    """The data set of ``dsi``'s declarations over ``inputs``, as TMVA builds and describes it."""
    log = Logger("DataSetFactory")
    Logger("DataSetInfo").info(f"Rebuilding Dataset {dsi.name}")
    counts = [read_class(dsi, inputs, number, log) for number in range(len(dsi.classes))]
    summary(dsi, counts)
    spec = split_spec(dsi, counts)
    dataset = mix_events(dsi, counts, spec)
    if len(dataset.train) > 1 and spec.compute_correlations:
        _set_ranges(dsi, dataset.train)
        for info in dsi.classes:
            matrix = correlation_matrix(dataset.train.of_class(info.number))
            dsi.correlations[info.name] = matrix
            if spec.correlations:
                _print_matrix(dsi, info.name, matrix)
        log.header(f"[{dsi.name}] :  ")
        log.info("")
    return dataset


def _print_matrix(dsi: DataSetInfo, name: str, matrix: Any) -> None:
    info = Logger("DataSetInfo")
    info.header(f"Correlation matrix ({name}):")
    for line in formatted_matrix(matrix, dsi.GetListOfVariables()):
        info.info(line)
