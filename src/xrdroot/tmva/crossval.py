"""``TMVA::CrossValidation``: every booked method trained and tested fold by fold, then as one.

The loader's training sample is cut into ``NumFolds`` folds - by
``SplitExpr`` of the spectators, or by TMVA's shuffled draw - and each
method is trained on all folds but one and tested on that one, by a Factory
of its own that writes no file, as ``<title>_fold<i>``. Then the folds are
put back together, and each method is booked once more as a
``CrossValidation`` method answering each event with the fold it was held
out of, and trained, tested and evaluated by the Factory writing the output
file, over the whole sample for both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .cvresults import CrossValidationFoldResult, CrossValidationResult, MethodInfo, split_folds
from .dataset import DataSet, Events
from .factory import Factory, _method_name
from .log import Logger
from .options import Options

if TYPE_CHECKING:
    from .factory import Booked
    from .loader import DataLoader

__all__ = ["CrossValidation"]

#: The banner's rule.
RULE = "========================================"


class FoldLoader:
    """A loader whose data set is one fold's: what a Factory needs of a loader, and no more."""

    def __init__(self, loader: Any, dataset: DataSet) -> None:
        self.loader, self.info, self._dataset = loader, loader.info, dataset

    def GetName(self) -> str:
        return str(self.loader.GetName())

    def dataset(self) -> DataSet:
        return self._dataset


def _joined(events: Events, parts: list[Any]) -> Events:
    return events.take(np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64))


def _factory_options(options: Options) -> str:
    """The options of the Factories the cross-validation trains with: its own, passed on."""
    factory = [f"AnalysisType={options.text_of('AnalysisType', 'Auto')}", "!DrawProgressBar"]
    for flag, default in (("V", False), ("Correlations", False), ("ROC", True), ("Silent", False)):
        factory.append(flag if options.flag(flag, default) else f"!{flag}")
    if options.given("Transformations"):
        factory.append(f"Transformations={options.text_of('Transformations')}")
    return ":".join(factory)


class CrossValidation:
    """``CrossValidation(job, loader, outputFile, options)`` - or ``(job, loader, options)``."""

    def __init__(self, job: Any, loader: Any, *args: Any) -> None:
        target, text = (None, args[0]) if len(args) == 1 else (*args, None, "")[:2]
        if isinstance(target, str):
            target, text = None, target
        self.job, self.loader, self.output_file = str(job), loader, target
        options = Options(text or "")
        self.log = Logger("CrossValidation")
        self.nfolds = options.integer("NumFolds", 2)
        self.split_type = options.text_of("SplitType", "Random")
        self.split_expr = options.text_of("SplitExpr", "")
        self.ensembling = options.text_of("OutputEnsembling", "None")
        if self.split_type != "Deterministic" and self.split_expr:
            raise self.log.fatal("SplitExpr can only be used with Deterministic Splitting")
        self.factory_options = _factory_options(options)
        self.fold_factory = Factory(self.job, self.factory_options)
        self.factory = (
            Factory(self.job, target, self.factory_options)
            if target is not None
            else Factory(self.job, self.factory_options)
        )
        self.methods: list[MethodInfo] = []
        self.results: list[CrossValidationResult] = []

    # -- booking and asking ----------------------------------------------------------------

    def BookMethod(self, method: Any, title: Any, options: Any = "") -> None:
        self.methods.append(
            MethodInfo(
                MethodName=_method_name(method), MethodTitle=str(title), MethodOptions=str(options)
            )
        )

    def GetMethods(self) -> list[MethodInfo]:
        return self.methods

    def GetResults(self) -> list[CrossValidationResult]:
        if not self.results:
            raise self.log.fatal("No cross-validation results available")
        return self.results

    def GetNumFolds(self) -> int:
        return self.nfolds

    def GetFactory(self) -> Factory:
        return self.factory

    def GetDataLoader(self) -> Any:
        return self.loader

    def SetNumFolds(self, folds: int) -> None:
        self.nfolds = int(folds)

    def SetSplitExpr(self, expression: Any) -> None:
        self.split_expr = str(expression)

    def _folds(self) -> list[Any]:
        dataset = self.loader.dataset()
        return split_folds(
            dataset.train,
            self.loader.info.spectators,
            self.nfolds,
            self.split_expr,
            self.split_type == "RandomStratified",
        )

    # -- the folds ---------------------------------------------------------------------------

    def _fold_result(self, fold: int, title: str) -> CrossValidationFoldResult:
        """What ``ProcessFold`` keeps of a fold: its evaluation's numbers and its ROC curve."""
        made = CrossValidationFoldResult(fold)
        factory = self.fold_factory
        rows = [row for rows in factory.last_rows.values() for row in rows if row["name"] == title]
        if not rows or "roc" not in rows[0]:
            return made
        row = rows[0]
        made.roc_integral = factory.GetROCIntegral(self.loader.GetName(), title)
        graph = factory.GetROCCurve(self.loader.GetName(), title, True)
        graph.SetLineColor(fold + 1)
        graph.SetLineWidth(2)
        graph.SetTitle(title)
        made.roc, made.sig, made.sep, made.eff_area = graph, row["sig"], row["sep"], row["area"]
        for level, suffix in ((0.01, "01"), (0.10, "10"), (0.30, "30")):
            setattr(made, f"eff{suffix}", row[f"eff{level}"])
            setattr(made, f"train_eff{suffix}", row[f"train{level}"])
        return made

    def _process_fold(self, fold: int, info: MethodInfo, folds: list[Any]) -> Any:
        """``ProcessFold``: the method trained without fold ``fold``, and tested on it."""
        train = self.loader.dataset().train
        others = [folds[i] for i in range(self.nfolds) if i != fold]
        dataset = DataSet(_joined(train, others), _joined(train, [folds[fold]]))
        title = f"{info['MethodTitle']}_fold{fold + 1}"
        factory = self.fold_factory
        held_out = cast("DataLoader", FoldLoader(self.loader, dataset))
        factory.BookMethod(held_out, info["MethodName"], title, info["MethodOptions"])
        item = factory._booked(self.loader.GetName(), title)
        if item is not None:
            factory._train_one(item)
        factory.TestAllMethods()
        factory.EvaluateAllMethods()
        result = self._fold_result(fold, title)
        factory.DeleteAllMethods()
        return result

    def _banner(self, text: str) -> None:
        for line in ("", "", RULE, text, RULE, ""):
            self.log.info(line)

    def _cross_validate(
        self, info: MethodInfo, folds: list[Any], recombined: DataLoader, mapping: Any
    ) -> Booked:
        """A method through its folds, then booked as a ``CrossValidation`` of the whole sample.

        A ``CrossValidation`` method can do every analysis, so its booking is never refused.
        """
        self._banner(f"Processing folds for method {info['MethodTitle']}")
        result = CrossValidationResult(self.nfolds)
        for fold in range(self.nfolds):
            result.Fill(self._process_fold(fold, info, folds))
        self.results.append(result)
        options = (
            f"SplitExpr={self.split_expr}:NumFolds={self.nfolds}"
            f":EncapsulatedMethodName={info['MethodTitle']}"
            f":EncapsulatedMethodTypeName={info['MethodName']}"
            f":OutputEnsembling={self.ensembling}"
        )
        self.factory.BookMethod(recombined, "CrossValidation", info["MethodTitle"], options)
        item = self.factory.booked[self.loader.GetName()][-1]
        item.method.event_folds = mapping  # type: ignore[attr-defined]
        return item

    def Evaluate(self) -> None:
        """Every method through its folds, then all of them over the whole recombined sample."""
        folds = self._folds()
        train = self.loader.dataset().train
        whole = _joined(train, folds)
        mapping = np.concatenate([np.full(len(members), i) for i, members in enumerate(folds)])
        # A fold's loader is what the Factory asks of a DataLoader, and no more.
        recombined = cast("DataLoader", FoldLoader(self.loader, DataSet(whole, whole)))
        booked = [self._cross_validate(info, folds, recombined, mapping) for info in self.methods]
        self._banner("Folds processed for all methods, evaluating.")
        for item in booked:
            if self.output_file is not None:
                self.factory._write_data_information(recombined)
            self.factory._train_one(item)
        self.factory.TestAllMethods()
        self.factory.EvaluateAllMethods()
        self.log.info("Evaluation done.")
