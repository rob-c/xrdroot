"""``RooCategory``: a discrete variable - labelled states, each with an index.

A category is a variable whose values are states such as ``"Plus"`` and
``"Minus"``, each with an integer index; a dataset keeps the index in its
column, and a formula reads ``c==c::Plus`` as a comparison of indices. A
state defined without an index gets one more than the largest so far, as
``RooAbsCategory::nextAvailableStateIndex`` gives it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .messages import ERROR, log
from .printing import kClassName, kName, kValue
from .real import Context, RooAbsReal

__all__ = ["RooAbsCategory", "RooBinningCategory", "RooCategory", "RooThresholdCategory"]


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
        """``defineType``: a new state; ``True`` - with an error - if its label or index is
        taken."""
        label = str(label)
        if ";" in label:
            log(
                self,
                ERROR,
                "InputArguments",
                f"RooCategory::defineType({self._name}): semicolons not allowed in label name",
            )
            return True
        number = self._next() if index is None else int(index)
        if number in self._states.values():
            log(
                self,
                ERROR,
                "InputArguments",
                f"RooAbsCategory::defineState({self._name}): index {number} already assigned",
            )
            return True
        if label in self._states:
            log(
                self,
                ERROR,
                "InputArguments",
                f"RooAbsCategory::defineState({self._name}): label "
                f"{label} already assigned or not allowed",
            )
            return True
        self._add_state(label, number)
        return False

    defineState = defineType

    def _add_state(self, label: str, number: int) -> None:
        """A state, unchecked: a multi-category's ``{a;b}`` has the semicolons users may not."""
        self._states[label] = number
        if len(self._states) == 1:
            self._index = number

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
        return iter([State(label, number) for label, number in self._states.items()])

    def begin(self) -> Any:
        """``begin()``: an iterator's first state - whose ``first`` is its label."""
        return next(iter(self))

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


class State(tuple):  # type: ignore[type-arg]
    """A state as C++ iterates a category: ``std::pair<std::string, int>`` - a tuple too."""

    def __new__(cls, label: str, index: int) -> State:
        return super().__new__(cls, (label, index))

    @property
    def first(self) -> str:
        return str(self[0])

    @property
    def second(self) -> int:
        return int(self[1])


class RooCategory(RooAbsCategory):
    """A category a macro sets: ``setLabel``, ``setIndex``, and a dataset's column of them."""

    def __init__(self, name: Any = "", title: Any = "", states: Any = None) -> None:
        super().__init__(name, title)
        #: The named ranges - sets of labels - shared with every copy, as RooFit shares them.
        self._ranges: dict[str, list[str]] = {}
        for label, index in (states or {}).items():
            self.defineType(label, index)

    def isFundamental(self) -> bool:
        return True

    def isDerived(self) -> bool:
        return False

    def setIndex(self, index: Any, printError: bool = True) -> bool:
        if not self.hasIndex(int(index)):
            if printError:
                log(
                    self,
                    ERROR,
                    "InputArguments",
                    f"RooCategory: Trying to set invalid state "
                    f"{int(index)} for category {self._name}",
                )
            return True
        self._index = int(index)
        return False

    def setLabel(self, label: str, printError: bool = True) -> bool:
        if str(label) not in self._states:
            if printError:
                log(
                    self,
                    ERROR,
                    "InputArguments",
                    f"Trying to set invalid state label '{label}' for category {self._name}",
                )
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
        self._ranges[str(name)] = [one for one in str(labels).split(",") if one]

    def addToRange(self, name: str, labels: str) -> None:
        self._ranges.setdefault(str(name), []).extend(one for one in str(labels).split(",") if one)

    def range_indices(self, name: str) -> list[int]:
        return [self.lookupIndex(label) for label in self._ranges.get(str(name), [])]

    def inRange(self, name: str) -> bool:
        return self.getLabel() in self._ranges.get(str(name), [self.getLabel()])

    def hasRange(self, name: Any) -> bool:
        return not name or str(name) in self._ranges

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


class _Derived(RooAbsCategory):
    """A category computed from other values - not set, but worked out, event by event."""

    def isFundamental(self) -> bool:
        return False

    def as_fundamental(self) -> RooCategory:
        """The category a dataset keeps a column of: the same states, set rather than computed."""
        made = RooCategory(self._name, self._title)
        for label, index in self._states.items():
            made._add_state(label, index)
        return made

    def getIndex(self) -> int:
        return int(np.asarray(self.compute({})))

    getCurrentIndex = getIndex

    def getLabel(self) -> str:
        return self.lookupName(self.getIndex())

    getCurrentLabel = getLabel


class RooThresholdCategory(_Derived):
    """``RooThresholdCategory``: the state of the first threshold a value is below, else the
    default."""

    def __init__(
        self, name: Any, title: Any, x: Any, defaultLabel: str, defaultIndex: int = 0
    ) -> None:
        super().__init__(name, title)
        self.x = self._proxy("inputVar", x)
        self.defineType(defaultLabel, defaultIndex)
        self._default = int(defaultIndex)
        self._thresholds: list[tuple[float, int]] = []

    def addThreshold(self, upperLimit: float, label: str, index: Any = None) -> bool:
        if not self.hasLabel(label):
            self.defineType(label, index)
        self._thresholds.append((float(upperLimit), self.lookupIndex(label)))
        self._thresholds.sort(key=lambda pair: pair[0])
        return False

    def compute(self, ctx: Context) -> Any:
        x = np.asarray(self.x.compute(ctx), dtype=np.float64)
        found = np.full(x.shape, float(self._default))
        for limit, index in reversed(self._thresholds):
            found = np.where(x < limit, float(index), found)
        return found if found.ndim else float(found)


class RooBinningCategory(_Derived):
    """``RooBinningCategory``: the number of the bin of a named binning a value falls in."""

    def __init__(
        self, name: Any, title: Any, x: Any, binningName: Any = None, catTypeName: str = ""
    ) -> None:
        super().__init__(name, title)
        self.x = self._proxy("inputVar", x)
        self._binning = binningName
        prefix = catTypeName or f"{x.GetName()}_{binningName or ''}_bin".replace("__", "_")
        for index in range(x.getBins(binningName)):
            self.defineType(f"{prefix}{index}", index)

    def compute(self, ctx: Context) -> Any:
        edges = self.x.getBinning(self._binning).array()
        x = np.asarray(self.x.compute(ctx), dtype=np.float64)
        found = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2).astype(
            np.float64
        )
        return found if found.ndim else float(found)


class RooMappedCategory(_Derived):
    """``RooMappedCategory``: another category's states mapped - by wildcard - onto new ones."""

    def __init__(
        self,
        name: Any,
        title: Any,
        input: Any,
        defaultLabel: str = "NotMapped",
        defaultIndex: Any = None,
    ) -> None:
        super().__init__(name, title)
        self.input = self._proxy("inputCat", input)
        self.defineType(defaultLabel, defaultIndex)
        self._default = self.lookupIndex(defaultLabel)
        self._rules: list[tuple[str, int]] = []

    def map(self, pattern: str, label: str, index: Any = None) -> bool:
        if not self.hasLabel(label):
            self.defineType(label, index)
        self._rules.append((str(pattern), self.lookupIndex(label)))
        return False

    def _target(self, source: str) -> int:
        import fnmatch

        return next(
            (index for pattern, index in self._rules if fnmatch.fnmatchcase(source, pattern)),
            self._default,
        )

    def compute(self, ctx: Context) -> Any:
        values = np.asarray(self.input.compute(ctx), dtype=np.float64)
        table = {
            float(index): float(self._target(label)) for label, index in self.input.states().items()
        }
        found = np.vectorize(
            lambda v: table.get(float(v), float(self._default)), otypes=[np.float64]
        )(values)
        return found if found.ndim else float(found)


class RooMultiCategory(_Derived):
    """``RooMultiCategory``: one state for each combination of several categories' states."""

    def __init__(self, name: Any, title: Any, inputs: Any) -> None:
        from .collections import as_list

        super().__init__(name, title)
        # a Python set has no order ROOT sees either: sorted by name, as ROOT's run
        # happened to take them
        ordered = sorted(as_list(inputs), key=lambda one: one.GetName())
        self.inputs = self._list_proxy("inputCats", ordered)
        import itertools

        lists = [list(one.states().items()) for one in ordered]
        for number, combination in enumerate(itertools.product(*reversed(lists))):
            labels = [label for label, _ in reversed(combination)]
            self._add_state("{" + ";".join(labels) + "}", number)

    def compute(self, ctx: Context) -> Any:
        found: Any = 0.0
        stride = 1
        for one in self.inputs:
            position = {float(index): float(i) for i, index in enumerate(one.states().values())}
            values = np.asarray(one.compute(ctx), dtype=np.float64)
            found = found + stride * np.vectorize(position.get, otypes=[np.float64])(values)
            stride *= len(position)
        found = np.asarray(found, dtype=np.float64)
        return found if found.ndim else float(found)


class RooSuperCategory(RooMultiCategory):
    """``RooSuperCategory``: a multi-category whose state can be set, setting its inputs'."""

    def setLabel(self, label: str, printError: bool = True) -> bool:
        if not self.hasLabel(label):
            return True
        labels = str(label).strip("{}").split(";")
        for one, part in zip(self.inputs, labels, strict=False):
            one.setLabel(part)
        return False
