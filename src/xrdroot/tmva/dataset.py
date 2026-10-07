"""``DataSetInfo`` and ``DataSet``: what a ``DataLoader`` declares, and the events it makes of it.

The declarations - variables, targets, spectators, the classes and their
weights and cuts - are a :class:`DataSetInfo`, as in TMVA; the events read
from the trees are a :class:`DataSet` of two :class:`Events`, the training
and the test sample, each a set of NumPy arrays with a row per event. The
values are TMVA's ``Float_t`` - single precision, held here as doubles of
single-precision values - so that every sum over them is the sum TMVA takes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .log import Logger
from .variables import ClassInfo, VariableInfo

__all__ = [
    "TRAINING",
    "TESTING",
    "MAX_TREE_TYPE",
    "DataSet",
    "DataSetInfo",
    "Events",
]

#: ``Types::ETreeType``.
TRAINING, TESTING, MAX_TREE_TYPE = 0, 1, 2


def as_float(values: Any) -> Any:
    """Values as ``Float_t`` holds them: rounded to single precision, kept as doubles."""
    return np.asarray(values, dtype=np.float64).astype(np.float32).astype(np.float64)


class Labels(list[str]):
    """Names in a list a macro asks the ``size()`` of, as of the vector C++ hands it."""

    def size(self) -> int:
        return len(self)


@dataclass
class Events:
    """A sample of events: their values, targets, spectators, classes and weights, a row each."""

    values: Any
    targets: Any
    spectators: Any
    classes: Any
    weights: Any

    def __len__(self) -> int:
        return len(self.classes)

    def of_class(self, number: int) -> Events:
        """The events of one class."""
        return self.take(self.classes == number)

    def take(self, which: Any) -> Events:
        """The events a mask or index array picks, in its order."""
        return Events(
            self.values[which],
            self.targets[which],
            self.spectators[which],
            self.classes[which],
            self.weights[which],
        )

    def with_values(self, values: Any) -> Events:
        """The same events with other values: what a transformation makes of them."""
        return Events(values, self.targets, self.spectators, self.classes, self.weights)

    @staticmethod
    def joined(parts: list[Events], nvar: int, ntgt: int, nspec: int) -> Events:
        """Several samples one after another; none at all is an empty one."""
        if not parts:
            return Events(
                np.zeros((0, nvar)),
                np.zeros((0, ntgt)),
                np.zeros((0, nspec)),
                np.zeros(0, dtype=np.int64),
                np.zeros(0),
            )
        return Events(
            np.concatenate([part.values for part in parts]),
            np.concatenate([part.targets for part in parts]),
            np.concatenate([part.spectators for part in parts]),
            np.concatenate([part.classes for part in parts]),
            np.concatenate([part.weights for part in parts]),
        )


@dataclass
class DataSet:
    """The training and test samples of a data set."""

    train: Events
    test: Events

    def GetNTrainingEvents(self) -> int:
        return len(self.train)

    def GetNTestEvents(self) -> int:
        return len(self.test)

    def GetNEvents(self) -> int:
        return len(self.train)


@dataclass
class DataSetInfo:
    """``TMVA::DataSetInfo``: a data set's declarations, and what was worked out from its events."""

    name: str
    variables: list[VariableInfo] = field(default_factory=list)
    targets: list[VariableInfo] = field(default_factory=list)
    spectators: list[VariableInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    split_options: str = ""
    normalization: str = "NONE"
    correlations: dict[str, Any] = field(default_factory=dict)
    #: The training and testing sums of signal and background weights, once split.
    sums: dict[str, float] = field(default_factory=dict)
    log: Logger = field(default_factory=lambda: Logger("DataSetInfo"))

    def GetName(self) -> str:
        return self.name

    def AddClass(self, name: str) -> ClassInfo:
        """The class called ``name``, added - and said so - the first time it is named."""
        for known in self.classes:
            if known.name == name:
                return known
        made = ClassInfo(name, len(self.classes))
        self.classes.append(made)
        self.log.header(f'[{self.name}] : Added class "{name}"')
        return made

    def GetClassInfo(self, which: Any) -> ClassInfo | None:
        if isinstance(which, int):
            return self.classes[which] if 0 <= which < len(self.classes) else None
        return next((known for known in self.classes if known.name == str(which)), None)

    def GetNClasses(self) -> int:
        return len(self.classes)

    def GetNVariables(self) -> int:
        return len(self.variables)

    def GetNTargets(self) -> int:
        return len(self.targets)

    def GetNSpectators(self) -> int:
        return len(self.spectators)

    def GetVariableInfos(self) -> list[VariableInfo]:
        return self.variables

    def GetVariableInfo(self, index: int) -> VariableInfo:
        return self.variables[index]

    def GetListOfVariables(self) -> list[str]:
        """The variables' labels, as the ``std::vector<TString>`` ROOT hands back."""
        return Labels(variable.label for variable in self.variables)

    def GetSignalClassIndex(self) -> int:
        signal = self.GetClassInfo("Signal")
        return signal.number if signal is not None else 0

    def IsSignal(self, number: int) -> bool:
        return number == self.GetSignalClassIndex()

    def GetClassNameMaxLength(self) -> int:
        return max((len(known.name) for known in self.classes), default=0)

    def SetWeightExpression(self, expression: Any, class_name: str = "") -> None:
        for known in self._named(class_name):
            known.weight = str(expression)

    def SetCut(self, cut: Any, class_name: str = "") -> None:
        for known in self._named(class_name):
            known.cut = str(cut)

    def AddCut(self, cut: Any, class_name: str = "") -> None:
        """A cut added to what a class has already - ``(old)&&(new)`` - as ``TCut +=`` adds it."""
        text = str(cut)
        if not text:
            return
        for known in self._named(class_name):
            known.cut = f"({known.cut})&&({text})" if known.cut else text

    def _named(self, class_name: str) -> list[ClassInfo]:
        """Every class for an empty name, else the one named - made if it is not known yet."""
        if not class_name:
            return list(self.classes)
        return [self.AddClass(class_name)]
