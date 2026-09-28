"""What ``TMVA::CrossValidation`` hands back: each fold's result, and a method's over its folds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..pyroot.core.objects import templated
from .log import Logger
from .tools import CxxVector

__all__ = ["CrossValidationFoldResult", "CrossValidationResult", "MethodInfo", "split_folds"]


@dataclass
class CrossValidationFoldResult:
    """``CrossValidationFoldResult``: one fold's numbers, and its ROC curve."""

    fold: int
    roc_integral: float = 0.0
    roc: Any = None
    sig: float = 0.0
    sep: float = 0.0
    eff01: float = 0.0
    eff10: float = 0.0
    eff30: float = 0.0
    eff_area: float = 0.0
    train_eff01: float = 0.0
    train_eff10: float = 0.0
    train_eff30: float = 0.0


@dataclass
class CrossValidationResult:
    """``CrossValidationResult``: a method's fold results, and their averages."""

    folds: int
    results: list[CrossValidationFoldResult] = field(default_factory=list)

    def Fill(self, result: CrossValidationFoldResult) -> None:
        self.results.append(result)

    def _values(self, name: str) -> CxxVector:
        return CxxVector(float(getattr(result, name)) for result in self.results)

    def GetROCValues(self) -> CxxVector:
        return self._values("roc_integral")

    def GetSigValues(self) -> CxxVector:
        return self._values("sig")

    def GetSepValues(self) -> CxxVector:
        return self._values("sep")

    def GetEff01Values(self) -> CxxVector:
        return self._values("eff01")

    def GetEff10Values(self) -> CxxVector:
        return self._values("eff10")

    def GetEff30Values(self) -> CxxVector:
        return self._values("eff30")

    def GetEffAreaValues(self) -> CxxVector:
        return self._values("eff_area")

    def GetTrainEff01Values(self) -> CxxVector:
        return self._values("train_eff01")

    def GetTrainEff10Values(self) -> CxxVector:
        return self._values("train_eff10")

    def GetTrainEff30Values(self) -> CxxVector:
        return self._values("train_eff30")

    def GetROCAverage(self) -> float:
        return float(np.mean(self.GetROCValues())) if self.results else 0.0

    def GetROCStandardDeviation(self) -> float:
        """The folds' ROC integrals' spread, ``sqrt(sum (x - mean)^2 / (n - 1))`` as TMVA has it."""
        values = np.asarray(self.GetROCValues())
        if len(values) < 2:
            return 0.0
        return float(np.sqrt(np.sum((values - values.mean()) ** 2) / (len(values) - 1)))

    def GetROCCurves(self, fLegend: bool = True) -> dict[int, Any]:
        return {result.fold: result.roc for result in self.results}

    def GetNumFolds(self) -> int:
        return self.folds

    def Print(self) -> None:
        """``CrossValidationResult::Print``: each fold's ROC integral, and their average."""
        log = Logger("CrossValidation")
        log.info("")
        for result in self.results:
            log.info(f"Fold {result.fold} ROC-Int : {result.roc_integral:.4f}")
        log.info("------------------------")
        log.info(f"Average ROC-Int : {self.GetROCAverage():.4f}")
        log.info(f"Std-Dev ROC-Int : {self.GetROCStandardDeviation():.4f}")

    def DrawAvgROCCurve(self, *_: Any) -> None:
        """Nothing in batch mode, which is the only mode there is."""


class MethodInfo(dict):  # type: ignore[type-arg]
    """``OptionMap`` of a booked method: ``GetValue<TString>("MethodName")``."""

    @templated
    def GetValue(self, key: Any) -> Any:
        return self[str(key)]


def _named_split(expression: str, spectators: list[Any], folds: int, values: Any) -> Any:
    """``CvSplitKFoldsExpr::Eval`` of every event: its fold from its spectators, rounded."""
    import re

    from .fdaformula import FormulaError, compile_fda

    names: list[str] = []

    def index(match: Any) -> str:
        name = match.group(1)
        if name not in names:
            names.append(name)
        return f"[{names.index(name)}]"

    text = re.sub(r"\[([A-Za-z_][A-Za-z0-9_]*)\]", index, expression)
    try:
        function = compile_fda(text)
    except FormulaError as why:
        raise Logger("CvSplit").fatal(
            f'Split expression "{expression}" is not a valid TFormula.'
        ) from why
    columns = []
    for name in names:
        if name in ("NumFolds", "numFolds"):
            columns.append(float(folds))
            continue
        found = [
            i
            for i, info in enumerate(spectators)
            if name in (info.internal, info.label, info.expression)
        ]
        if not found:
            raise Logger("CvSplit").fatal(f'Spectator "{name}" not found.')
        columns.append(values[:, found[0]])
    return np.rint(np.asarray(function(columns), dtype=np.float64)).astype(np.int64)


def split_folds(
    events: Any,
    spectators: list[Any],
    folds: int,
    expression: str,
    stratified: bool,
    seed: int = 100,
) -> list[Any]:
    """``CvSplitKFolds::SplitSets``: each fold's events, in the order TMVA puts them in it.

    By expression, each event goes to the fold it names, in the sample's order;
    otherwise the fold numbers ``0, 1, ..., 0, 1, ...`` are shuffled as TMVA's
    ``RandomGenerator<TRandom3>(seed)`` shuffles them - per class, after the
    class's events are shuffled too, for ``RandomStratified``.
    """
    from .shuffle import RandomGenerator, shuffle

    count = len(events)
    made: list[list[int]] = [[] for _ in range(folds)]
    if expression:
        found = np.broadcast_to(
            _named_split(expression, spectators, folds, np.asarray(events.spectators)), (count,)
        )
        if np.any((found < 0) | (found >= folds)):
            raise Logger("CvSplit").fatal(
                "Output of splitExpr should be a non-negativeinteger between 0 and numFolds-1 "
                "inclusive."
            )
        for index, fold in enumerate(found):
            made[int(fold)].append(index)
        return [np.asarray(members, dtype=np.int64) for members in made]
    groups = [list(range(count))]
    if stratified:
        groups = [
            [int(i) for i in np.flatnonzero(events.classes == cls)]
            for cls in range(int(events.classes.max()) + 1 if count else 0)
        ]
        for members in groups:
            shuffle(members, RandomGenerator(seed))
    for members in groups:
        mapping = [i % folds for i in range(len(members))]
        shuffle(mapping, RandomGenerator(seed))
        for event, fold in zip(members, mapping):
            made[fold].append(event)
    return [np.asarray(members, dtype=np.int64) for members in made]
