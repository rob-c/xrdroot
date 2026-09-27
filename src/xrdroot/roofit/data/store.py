"""``RooAbsData``: the events a model is fitted to - columns of values, and a weight for each.

A dataset is its variables - copies of the model's, sharing their ranges -
and one NumPy column per variable, a category's column holding its index,
and optionally a column of weights. Everything that reads a dataset reads
the columns, all events at once; ``get(i)`` is how a macro reads one event,
by setting the copies to its values.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..arg import RooAbsArg
from ..collections import RooArgSet, as_list
from ..printing import RooPrintable, g, kArgs, kClassName, kName, kValue

__all__ = ["RooAbsData", "copies_of"]


def copies_of(variables: Iterable[Any]) -> RooArgSet:
    """The dataset's own copies of ``variables``, which share their ranges."""
    made = RooArgSet()
    for one in variables:
        made.add(one.clone(one.GetName()) if one.isFundamental() else one)
    return made


class RooAbsData(RooPrintable):
    """Events: a column per variable, and a weight per event."""

    #: ``RooAbsData::ErrorType``: how data points' error bars are drawn.
    Poisson, SumW2, NONE, Expected, Auto = 0, 1, 2, 3, 4

    def __init__(self, name: Any = "", title: Any = "", variables: Any = ()) -> None:
        self._name = str(name)
        self._title = str(title)
        self._vars = copies_of(as_list(variables))
        self._columns: dict[str, np.ndarray[Any, Any]] = {
            one.GetName(): np.zeros(0) for one in self._vars
        }
        self._weights: np.ndarray[Any, Any] | None = None
        self._sumw2: np.ndarray[Any, Any] | None = None
        self._global_observables: RooArgSet | None = None

    # -- TObject ------------------------------------------------------------------

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        return self._title

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def SetTitle(self, title: str) -> None:
        self._title = str(title)

    def ClassName(self) -> str:
        return type(self).__name__

    def InheritsFrom(self, name: Any) -> bool:
        wanted = name if isinstance(name, str) else getattr(name, "__name__", str(name))
        return any(klass.__name__ == wanted for klass in type(self).__mro__)

    # -- contents -----------------------------------------------------------------

    def numEntries(self) -> int:
        first = next(iter(self._columns.values()), None)
        return 0 if first is None else len(first)

    def __len__(self) -> int:
        return self.numEntries()

    def column(self, name: str) -> np.ndarray[Any, Any]:
        """All the values of the variable ``name``, one per event."""
        return self._columns[name]

    def columns(self) -> dict[str, np.ndarray[Any, Any]]:
        return dict(self._columns)

    def weights(self) -> np.ndarray[Any, Any]:
        """Each event's weight: one each, unless the data are weighted."""
        if self._weights is None:
            return np.ones(self.numEntries())
        return self._weights

    def weights_squared(self) -> np.ndarray[Any, Any]:
        if self._sumw2 is not None:
            return self._sumw2
        return self.weights() ** 2

    def isWeighted(self) -> bool:
        return self._weights is not None

    def isNonPoissonWeighted(self) -> bool:
        if self._weights is None:
            return False
        w = self._weights
        return bool(np.any((w != np.floor(w)) | (w < 0)))

    def get(self, index: Any = None) -> RooArgSet:
        """The variables - set to event ``index``'s values if one is named."""
        if index is not None:
            self._load(int(index))
        return self._vars

    def _load(self, index: int) -> None:
        self._current = index
        for one in self._vars:
            column = self._columns.get(one.GetName())
            if column is not None and index < len(column):
                one.load_value(column[index])

    def weight(self) -> float:
        index = getattr(self, "_current", 0)
        return float(self.weights()[index]) if self.numEntries() else 1.0

    def weightSquared(self) -> float:
        index = getattr(self, "_current", 0)
        return float(self.weights_squared()[index]) if self.numEntries() else 1.0

    def mask(self, cut: Any = None, rng: Any = None) -> np.ndarray[Any, Any]:
        """Which events pass the selection ``cut`` and lie in the range ``rng``."""
        from .selection import selected

        return selected(self, cut, rng)

    def sumEntries(self, cut: Any = None, rng: Any = None) -> float:
        keep = self.mask(cut, rng)
        return float(np.sum(self.weights()[keep]))

    def sumEntriesW2(self) -> float:
        return float(np.sum(self.weights_squared()))

    def variable(self, name: str) -> Any:
        return self._vars.find(name)

    def setGlobalObservables(self, globs: Any) -> None:
        self._global_observables = copies_of(as_list(globs))

    def getGlobalObservables(self) -> Any:
        return self._global_observables

    def changeObservableName(self, old: str, new: str) -> bool:
        found = self._vars.find(old)
        if found is None:
            return True
        found.SetName(new)
        self._columns[new] = self._columns.pop(old)
        return False

    def plotOn(self, frame: Any, *args: Any, **kwargs: Any) -> Any:
        from ..plot.data import plot_data

        return plot_data(self, frame, args, kwargs)

    # -- printing -----------------------------------------------------------------

    def printName(self) -> str:
        return self._name

    def printTitle(self) -> str:
        return self._title

    def printClassName(self) -> str:
        return self.ClassName()

    def defaultPrintContents(self, option: Any) -> int:
        return kName | kClassName | kArgs | kValue

    def printMultiline(self, contents: int, verbose: bool, indent: str) -> str:
        """The data store's lines: its name, how many events, and what variables."""
        text = f"{indent}DataStore {self._name} ({self._title})\n"
        text += f"{indent}  Contains {self.numEntries()} entries\n"
        if not verbose:
            return text + f"{indent}  Observables {self._vars.printValue()}\n"
        from ..printing import kExtras, kTitle, kVerbose

        text += f"{indent}  Observables: \n"
        return text + self._vars.printStream(kName | kValue | kExtras | kTitle, kVerbose, indent + "  ")

    def __repr__(self) -> str:
        return f"<{self.ClassName()}::{self._name} {self.numEntries()} entries>"


def value_text(value: Any) -> str:
    return g(value)


def is_arg(obj: Any) -> bool:
    return isinstance(obj, RooAbsArg)
