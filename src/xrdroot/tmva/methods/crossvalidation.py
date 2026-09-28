"""``CrossValidation``: the method a cross-validation leaves behind, one trained method per fold.

Each event is answered by the method of the fold it was held out of - the
fold its ``SplitExpr`` names, or, straight after the cross-validation, the
fold it was drawn into - or, with ``OutputEnsembling=Avg``, by the average
of all of them. The folds' methods are read from their own weight files,
``<job>_<method>_fold<i>.weights.xml``, as TMVA reads them; the method's own
weight file only names them, so a reader needs the fold files beside it.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..dataset import Events
from ..method import CLASSIFICATION, MULTICLASS, REGRESSION, Method
from ..xmlfile import Node

__all__ = ["MethodCrossValidation"]


class MethodCrossValidation(Method):
    """``TMVA::MethodCrossValidation``."""

    type_name = "CrossValidation"
    analyses = frozenset({CLASSIFICATION, REGRESSION, MULTICLASS})
    defaults = {
        "EncapsulatedMethodName": "",
        "EncapsulatedMethodTypeName": "",
        "NumFolds": 2,
        "OutputEnsembling": "None",
        "SplitExpr": "",
    }

    def process_options(self) -> None:
        self.folds: list[Method] = []
        #: Each training event's fold, in the order of the recombined sample.
        self.event_folds: Any = None
        self._settings(
            self.opt("EncapsulatedMethodName"),
            self.opt("EncapsulatedMethodTypeName"),
            self.opt("NumFolds"),
            self.opt("OutputEnsembling"),
            self.opt("SplitExpr"),
        )

    def _settings(self, name: Any, kind: Any, folds: Any, ensembling: Any, split: Any) -> None:
        self.inner_name, self.inner_type = str(name), str(kind)
        self.nfolds, self.ensembling, self.split = int(folds), str(ensembling), str(split)

    def fold_file(self, fold: int) -> str:
        """``GetWeightFileNameForFold``: the fold's weight file, beside this method's."""
        here = os.path.dirname(self.source or self.weight_file)
        base = f"{self.job}_{self.inner_name}_fold{fold + 1}.weights.xml"
        return f"{here}/{base}" if here else base

    def _read_folds(self) -> None:
        from ..weightfile import read_method

        self.folds = []
        for fold in range(self.nfolds):
            path = self.fold_file(fold)
            self.log.info(f"Reading weightfile: {path}")
            self.folds.append(read_method(path, self.dsi, self.job, self.log))

    def booked(self) -> None:
        self._read_folds()

    def train(self, events: Events) -> None:
        """Nothing: the folds' methods were trained by the cross-validation."""

    def _event_folds(self, events: Events) -> Any:
        if self.split:
            from ..cvresults import _named_split

            found = _named_split(
                self.split, self.dsi.spectators, self.nfolds, np.asarray(events.spectators)
            )
            return np.broadcast_to(found, (len(events),))
        if self.event_folds is not None and len(self.event_folds) == len(events):
            return self.event_folds
        raise self.log.fatal(
            "MethodCrossValidation supports XML reading only for deterministic splitting !"
        )

    def mva(self, events: Events) -> Any:
        answers = [fold.mva(events) for fold in self.folds]
        if self.ensembling == "Avg":
            return np.mean(np.asarray(answers, dtype=np.float64), axis=0)
        if self.ensembling != "None":
            raise self.log.fatal(f"Ensembling type {self.ensembling} unknown")
        chosen = self._event_folds(events)
        stacked = np.asarray(answers, dtype=np.float64)
        return stacked[chosen, np.arange(len(events))]

    def evaluate(self, values: Any) -> Any:
        raise self.log.fatal("CrossValidation answers events, which carry their spectators")

    def write_weight_file(self) -> None:
        super().write_weight_file()
        for text in ("MakeClassSpecificHeader", "MakeClassSpecific"):
            self.log.warning(f"{text} not implemented for CrossValidation")

    def add_weights(self, node: Node) -> None:
        node.add(
            "Weights",
            JobName=self.job,
            SplitExpr=self.split,
            NumFolds=self.nfolds,
            EncapsulatedMethodName=self.inner_name,
            EncapsulatedMethodTypeName=self.inner_type,
            OutputEnsembling=self.ensembling,
        )

    def read_weights(self, node: Any) -> None:
        self.job = str(node.get("JobName", self.job))
        self._settings(
            node.get("EncapsulatedMethodName"),
            node.get("EncapsulatedMethodTypeName"),
            node.get("NumFolds", 2),
            node.get("OutputEnsembling", "None"),
            node.get("SplitExpr", ""),
        )
        self._read_folds()
        if not self.split:
            raise self.log.fatal(
                "MethodCrossValidation supports XML reading only for deterministic splitting !"
            )
