"""``EvaluateAllMethods`` for regressions and multiclass classifiers: TMVA's lines and tables."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .multieval import confusion, performance_graphs, response_histograms, roc_summary
from .regeval import RegressionStats, test_regression

if TYPE_CHECKING:
    from .factory import Booked
    from .log import Logger

__all__ = ["EvaluatingModes"]

#: The rule of the regression tables.
REG_LINE = "-" * 98
#: The rule of the multiclass tables.
MULTI_LINE = "-" * 103
#: The background efficiencies the confusion matrices are given at.
LEVELS = (0.01, 0.10, 0.30)


class EvaluatingModes:
    """The regression and multiclass halves of ``EvaluateAllMethods``."""

    log: Logger

    def _write_evaluation(self, item: Booked, test: Any, before: Any = (), after: Any = ()) -> None:
        raise NotImplementedError

    # -- regression -----------------------------------------------------------------------

    def _test_regression(self, item: Booked, which: str) -> RegressionStats:
        """``TestRegression``: the lines it prints, and its numbers for one sample."""
        dataset = item.loader.dataset()
        events = dataset.test if which == "test" else dataset.train
        values = item.test_values if which == "test" else item.train_values
        method = item.method
        method.log.info("Calculate regression for all events")
        start = time.perf_counter()
        found = test_regression(np.asarray(values)[:, 0], events.targets[:, 0], events.weights)
        method.log.info(
            f"Elapsed time for evaluation of {len(events)} events: "
            f"{time.perf_counter() - start:.3g} sec       "
        )
        return found

    def _evaluate_regression(self, item: Booked) -> dict[str, Any]:
        method = item.method
        self.log.info(f"Evaluate regression method: {method.name}")
        self.log.info("TestRegression (testing)")
        test = self._test_regression(item, "test")
        self.log.info("TestRegression (training)")
        train = self._test_regression(item, "train")
        dataset = item.loader.dataset()
        if method.handler.transforms:
            method.handler.print_stats(method.handler.apply(dataset.test))
        self._write_evaluation(
            item,
            dataset.test,
            item.extra.get("test_results", ()),
            item.extra.get("train_results", ()),
        )
        return {"name": method.name, "test": test, "train": train}

    def _regression_tables(self, dataset: str, rows: list[dict[str, Any]]) -> None:
        """The two tables a regression ends with, ranked by the test sample's mean deviation."""
        ranked = sorted(rows, key=lambda row: row["test"].dev)
        log = self.log
        log.info("")
        log.info("Evaluation results ranked by smallest RMS on test sample:")
        log.info('("Bias" quotes the mean deviation of the regression from true target.')
        log.info(' "MutInf" is the "Mutual Information" between regression and target.')
        log.info(' Indicated by "_T" are the corresponding "truncated" quantities ob-')
        log.info(" tained when removing events deviating more than 2sigma from average.)")
        log.info(REG_LINE)
        log.info(REG_LINE)
        for row in ranked:
            log.info(_regression_row(dataset, row["name"], row["test"]))
        log.info(REG_LINE)
        log.info("")
        log.info("Evaluation results ranked by smallest RMS on training sample:")
        log.info("(overtraining check)")
        log.info(REG_LINE)
        log.info(
            "DataSet Name:         MVA Method:        <Bias>   <Bias_T>    RMS    RMS_T  |  "
            "MutInf MutInf_T"
        )
        log.info(REG_LINE)
        for row in ranked:
            log.info(_regression_row(dataset, row["name"], row["train"]))
        log.info(REG_LINE)
        log.info("")

    # -- multiclass -----------------------------------------------------------------------

    def _evaluate_multiclass(self, item: Booked) -> dict[str, Any]:
        method = item.method
        self.log.info(f"Evaluate multiclass classification method: {method.name}")
        dataset = item.loader.dataset()
        test, train = dataset.test, dataset.train
        output_test, output_train = np.asarray(item.test_values), np.asarray(item.train_values)
        storage = list(item.extra.get("test_results", ()))
        # TMVA's TestMulticlass makes the "_Train" histograms again, from the testing sample.
        prefix = f"{method.testvar}_Train"
        method.log.info("Creating multiclass response histograms...")
        method.log.info("Creating multiclass performance histograms...")
        method.log.info("Creating multiclass response histograms...")
        storage += response_histograms(prefix, method.dsi, test, output_test)
        method.log.info("Creating multiclass performance histograms...")
        storage += performance_graphs(prefix, method.dsi, test, output_test)
        nclasses = method.dsi.GetNClasses()
        matrices = {
            (kind, level): confusion(output, events, nclasses, level)
            for kind, output, events in (
                ("test", output_test, test),
                ("train", output_train, train),
            )
            for level in LEVELS
        }
        if method.handler.transforms:
            method.handler.print_stats(method.handler.apply(test))
        self._write_evaluation(item, test, storage, item.extra.get("train_results", ()))
        summaries = [
            (roc_summary(output_test, test, k), roc_summary(output_train, train, k))
            for k in range(nclasses)
        ]
        return {
            "name": method.name,
            "matrices": matrices,
            "summaries": summaries,
            "classes": [cls.name for cls in method.dsi.classes],
        }

    def _multiclass_tables(self, dataset: str, rows: list[dict[str, Any]]) -> None:
        """The per-class performance table and every method's confusion matrices."""
        log = self.log
        log.info("")
        log.info("1-vs-rest performance metrics per class")
        log.info(MULTI_LINE)
        log.info("")
        log.info("Considers the listed class as signal and the other classes")
        log.info("as background, reporting the resulting binary performance.")
        log.info("A score of 0.820 (0.850) means 0.820 was acheived on the")
        log.info("test set and 0.850 on the training set.")
        log.info("")
        log.info(
            "".join(
                f"{t:<15}"
                for t in (
                    "Dataset",
                    "MVA Method",
                    "ROC AUC",
                    "Sig eff@B=0.01",
                    "Sig eff@B=0.10",
                    "Sig eff@B=0.30",
                )
            )
        )
        log.info(
            "".join(
                f"{t:<15}"
                for t in (
                    "Name:",
                    "/ Class:",
                    "test  (train)",
                    "test  (train)",
                    "test  (train)",
                    "test  (train)",
                )
            )
        )
        for row in rows:
            log.info("")
            log.info(f"{dataset:<15}{row['name']:<15}")
            log.info("------------------------------")
            for name, (test, train) in zip(row["classes"], row["summaries"]):
                cells = [f"{a:5.3f} ({b:5.3f})" for a, b in zip(test, train)]
                log.info("".join(f"{t:<15}" for t in ("", name, *cells)))
        log.info("")
        log.info(MULTI_LINE)
        log.info("")
        _confusion_tables(log, rows)


def _regression_row(dataset: str, name: str, stats: RegressionStats) -> str:
    numbers = "".join(_g(value) for value in (stats.bias, stats.bias_t, stats.rms, stats.rms_t))
    return f"{dataset:<20} {name:<15}:{numbers}  |  {stats.minf:#5.3f}  {stats.minf_t:#5.3f}"


def _g(value: float) -> str:
    """``%#9.3g``: three significant figures, the point and trailing zeros kept."""
    return format(value, "#9.3g")


def _confusion_tables(log: Logger, rows: list[dict[str, Any]]) -> None:
    log.info("")
    log.info("Confusion matrices for all methods")
    log.info(MULTI_LINE)
    log.info("")
    log.info("Does a binary comparison between the two classes given by a ")
    log.info("particular row-column combination. In each case, the class ")
    log.info("given by the row is considered signal while the class given ")
    log.info("by the column index is considered background.")
    log.info("")
    for row in rows:
        names = row["classes"]
        log.info(f"=== Showing confusion matrix for method : {row['name']:<15}")
        for level, label in zip(LEVELS, ("0.01", "0.10", "0.30")):
            log.info(f"(Signal Efficiency for Background Efficiency {label}%)")
            log.info("---------------------------------------------------")
            # TMVA hands its printer the testing matrix as the training one and the other way
            # round, so what it heads "test (train)" is the training sample's, then the testing's.
            test, train = row["matrices"][("train", level)], row["matrices"][("test", level)]
            log.info(f" {' ':<14}" + "".join(f" {name:<14}" for name in names))
            log.info(f" {' ':<14}" + "".join(f" {' test (train)':<14}" for _ in names))
            for i, own in enumerate(names):
                cells = [
                    f" {'-':<14}"
                    if i == j
                    else f" {f'{test[i, j]:<5.3f} ({train[i, j]:<5.3f})':<14}"
                    for j in range(len(names))
                ]
                log.info(f" {own:<14}" + "".join(cells))
            log.info("")
    log.info(MULTI_LINE)
    log.info("")
