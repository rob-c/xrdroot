"""``RooCategory``: a discrete variable - labelled states, each with an index.

A category is a variable whose values are states such as ``"Plus"`` and
``"Minus"``, each with an integer index; a dataset keeps the index in its
column, and a formula reads ``c==c::Plus`` as a comparison of indices. A
state defined without an index gets one more than the largest so far, as
``RooAbsCategory::nextAvailableStateIndex`` gives it.
"""

from __future__ import annotations

from typing import Any

from .messages import ERROR, log
from .printing import kClassName, kName, kValue
from .real import Context, RooAbsReal

__all__ = ["RooAbsCategory", "RooCategory"]


class RooAbsCategory(RooAbsReal):
    """A discrete value: the index of one of its labelled states."""

    def __init__(self, name: Any = "", title: Any = "") -> None:
        super().__init__(name, title)
        self._states: dict[str, int] = {}
        self._index = 0

    def isCategory(self) -> bool:
        return True

    # -- states -------------------------------------------------------------------

    def defineType(self, label: str, index: Any = None) -> bool:
        """``defineType``: a new state; ``True`` - with an error - if its label or index is taken."""
        label = str(label)
        if ";" in label:
            log(self, ERROR, "InputArguments", f"RooCategory::defineType({self._name}): semicolons "
                "not allowed in label name")  # fmt: skip
            return True
        number = self._next() if index is None else int(index)
        if number in self._states.values():
            log(self, ERROR, "InputArguments", f"RooAbsCategory::defineState({self._name}): index "
                f"{number} already assigned")  # fmt: skip
            return True
        if label in self._states:
            log(self, ERROR, "InputArguments", f"RooAbsCategory::defineState({self._name}): label "
                f"{label} already assigned or not allowed")  # fmt: skip
            return True
        self._states[label] = number
        if len(self._states) == 1:
            self._index = number
        return False

    defineState = defineType

    def _next(self) -> int:
        return 1 + max(self._states.values()) if self._states else 0

    def defineTypes(self, labels: Any) -> None:
        for label in labels:
            self.defineType(label)

    def states(self) -> dict[str, int]:
        return dict(self._states)

    def hasLabel(self, label: str) -> bool:
        return str(label) in self._states

    def hasIndex(self, index: int) -> bool:
        return int(index) in self._states.values()

    def lookupIndex(self, label: str) -> int:
        return self._states.get(str(label), -2147483648)

    def lookupName(self, index: int) -> str:
        return next((label for label, number in self._states.items() if number == int(index)), "")

    def size(self) -> int:
        return len(self._states)

    numTypes = size

    def __iter__(self) -> Any:
        return iter(self._states.items())

    def __len__(self) -> int:
        return len(self._states)

    # -- the value ----------------------------------------------------------------

    def getIndex(self) -> int:
        return self._index

    getCurrentIndex = getIndex

    def getLabel(self) -> str:
        return self.lookupName(self._index)

    getCurrentLabel = getLabel

    def compute(self, ctx: Context) -> Any:
        return ctx.get(self._name, float(self._index))

    def getVal(self, nset: Any = None) -> float:
        return float(self._index)

    def printValue(self) -> str:
        return f"{self.getLabel()}(idx = {self._index})\n"

    def defaultPrintContents(self, option: Any) -> int:
        return kName | kClassName | kValue


class RooCategory(RooAbsCategory):
    """A category a macro sets: ``setLabel``, ``setIndex``, and a dataset's column of them."""

    def __init__(self, name: Any = "", title: Any = "", states: Any = None) -> None:
        super().__init__(name, title)
        for label, index in (states or {}).items():
            self.defineType(label, index)

    def isFundamental(self) -> bool:
        return True

    def isDerived(self) -> bool:
        return False

    def setIndex(self, index: Any, printError: bool = True) -> bool:
        if not self.hasIndex(int(index)):
            if printError:
                log(self, ERROR, "InputArguments", f"RooCategory: Trying to set invalid state "
                    f"{int(index)} for category {self._name}")  # fmt: skip
            return True
        self._index = int(index)
        return False

    def setLabel(self, label: str, printError: bool = True) -> bool:
        if str(label) not in self._states:
            if printError:
                log(self, ERROR, "InputArguments", f"Trying to set invalid state label '{label}' for "
                    f"category {self._name}")  # fmt: skip
            return True
        self._index = self._states[str(label)]
        return False

    def __setitem__(self, label: str, index: int) -> None:
        self.defineType(label, index)

    def __getitem__(self, label: str) -> int:
        return self._states[label]

    def setConstant(self, value: bool = True) -> None:
        self.setAttribute("Constant", bool(value))

    def setRange(self, name: str, labels: str) -> None:
        self._ranges = getattr(self, "_ranges", {})
        self._ranges[str(name)] = [one for one in str(labels).split(",") if one]

    def inRange(self, name: str) -> bool:
        return self.getLabel() in getattr(self, "_ranges", {}).get(str(name), [self.getLabel()])

    def hasRange(self, name: Any) -> bool:
        return not name or str(name) in getattr(self, "_ranges", {})

    # -- as a dataset keeps it ----------------------------------------------------

    def stored_value(self) -> float:
        return float(self._index)

    def load_value(self, value: Any) -> None:
        self._index = int(value)

    def can_hold(self, value: Any) -> bool:
        return int(value) in self._states.values()

    def value_text(self, value: Any) -> str:
        return f"{self.lookupName(int(value))}({int(value)})"

    def copy_value_from(self, other: Any) -> None:
        self._index = int(other.getIndex())

    def _copy_state(self, other: Any) -> None:
        self._states = dict(other._states)
