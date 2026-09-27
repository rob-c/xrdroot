"""``RooRealVar`` and ``RooConstVar``: the variables and constants a model is made of.

A variable has a value, a range it is clipped to, an error once it has been
fitted, a default binning of 100 bins and any number of named ranges and
binnings, which it shares with its copies - a dataset's copy of ``x`` knows
the ``"signal"`` range defined on ``x`` - as ROOT's shared properties do.
"""

from __future__ import annotations

import math
from typing import Any

from .binning import RooAbsBinning, RooParamBinning, RooRangeBinning, RooUniformBinning
from .messages import INFO, WARNING, log
from .printing import g, kClassName, kExtras, kName, kValue
from .real import Context, RooAbsReal

__all__ = ["RooAbsRealLValue", "RooConstVar", "RooRealVar", "is_infinite"]

#: ``RooNumber::infinity()``.
INFINITY = math.inf


def is_infinite(value: float) -> bool:
    return math.isinf(value) or abs(value) >= 1e30


class RooAbsRealLValue(RooAbsReal):
    """A real value that can be set: what a dataset has columns of and a plot has an axis of."""

    def __init__(self, name: Any = "", title: Any = "", unit: str = "") -> None:
        super().__init__(name, title, unit)
        self._val = 0.0
        self._binning: RooAbsBinning = RooUniformBinning(-INFINITY, INFINITY, 100)
        #: The named binnings and ranges, shared with every copy.
        self._shared: dict[str, RooAbsBinning] = {}

    def isFundamental(self) -> bool:
        return True

    def isDerived(self) -> bool:
        return False

    def compute(self, ctx: Context) -> Any:
        return ctx.get(self._name, self._val)

    def getVal(self, nset: Any = None) -> float:
        return self._val

    def setVal(self, value: float) -> None:
        low, high = self.getMin(), self.getMax()
        self._val = min(max(float(value), low), high)

    def copy_value_from(self, other: Any) -> None:
        self._val = float(other.getVal())

    # -- as a dataset keeps it ----------------------------------------------------

    def stored_value(self) -> float:
        """What a dataset's column keeps of this variable: its value."""
        return self._val

    def load_value(self, value: Any) -> None:
        """Take a dataset's value, unclipped, as a data store sets it."""
        self._val = float(value)

    def can_hold(self, value: Any) -> bool:
        """``isValidReal``: whether ``value`` is inside the range."""
        return bool(self.getMin() <= value <= self.getMax())

    def value_text(self, value: Any) -> str:
        return g(value)

    # -- ranges and binnings ------------------------------------------------------

    def getBinning(
        self, name: Any = None, verbose: bool = True, createOnTheFly: bool = False
    ) -> Any:
        if not name:
            return self._binning
        found = self._shared.get(str(name))
        if found is None:
            if not createOnTheFly:
                return self._binning
            found = RooRangeBinning(self.getMin(), self.getMax(), str(name))
            self._shared[str(name)] = found
        return found

    def hasBinning(self, name: str) -> bool:
        return str(name) in self._shared

    def hasRange(self, name: Any) -> bool:
        return name is None or (bool(name) and str(name) in self._shared)

    def getBinningNames(self) -> list[str]:
        return ["", *sorted(self._shared)]

    def getMin(self, name: Any = None) -> Any:  # an array, for a range whose end is a column
        return self.getBinning(name).lowBound()

    def getMax(self, name: Any = None) -> Any:  # an array, for a range whose end is a column
        return self.getBinning(name).highBound()

    def getRange(self, name: Any = None) -> tuple[float, float]:
        return self.getMin(name), self.getMax(name)

    def hasMin(self, name: Any = None) -> bool:
        return not is_infinite(self.getMin(name))

    def hasMax(self, name: Any = None) -> bool:
        return not is_infinite(self.getMax(name))

    def inRange(self, value: float, name: Any = None) -> bool:
        return bool(self.getMin(name) <= float(value) <= self.getMax(name))

    def getBins(self, name: Any = None) -> int:
        return int(self.getBinning(name).numBins())

    def numBins(self, name: Any = None) -> int:
        return self.getBins(name)

    def frame(self, *args: Any, **kwargs: Any) -> Any:
        """A :class:`~xrdroot.roofit.plot.RooPlot` with this variable on its axis."""
        from .plot.frame import make_frame

        return make_frame(self, args, kwargs)

    # -- printing -----------------------------------------------------------------

    def printValue(self) -> str:
        return g(self.getVal())


class RooRealVar(RooAbsRealLValue):
    """A variable: a value in a range, with an error once fitted, constant or free."""

    def __init__(self, name: Any = "", title: Any = "", *args: Any) -> None:
        unit = str(args[-1]) if args and isinstance(args[-1], str) else ""
        numbers = [float(a) for a in args if not isinstance(a, str)]
        super().__init__(name, title, unit)
        self._error = -1.0
        self._asym = (1.0, -1.0)
        self._start(numbers)

    def _start(self, numbers: list[float]) -> None:
        """The three constructors: a constant, a range, or a value in a range."""
        if len(numbers) == 1:
            self._val = numbers[0]
            self.setConstant(True)
        elif len(numbers) == 2:
            low, high = numbers
            self._binning = RooUniformBinning(low, high, 100)
            self._val = _middle(low, high)
            self.setRange(low, high)
        elif len(numbers) == 3:
            self._binning = RooUniformBinning(numbers[1], numbers[2], 100)
            self.setRange(numbers[1], numbers[2])
            self._val = min(max(numbers[0], numbers[1]), numbers[2])

    def _copy_state(self, other: Any) -> None:
        self._binning = other._binning.clone()

    # -- ranges -------------------------------------------------------------------

    def setRange(self, *args: Any) -> None:
        """``setRange(min, max)``, or ``setRange(name, min, max)`` for a named range."""
        if args and isinstance(args[0], str):
            self._set_named_range(args[0], float(args[1]), float(args[2]))
            return
        if any(hasattr(one, "getVal") for one in args[:2]):
            self._param_range(*args[:2])
            return
        low, high = float(args[0]), float(args[1])
        if low > high:
            log(
                self,
                WARNING,
                "InputArguments",
                f"RooRealVar::setRange({self._name}): Proposed "
                "new fit max. smaller than min., setting max. to min.",
            )
            high = low
        self._binning.setRange(low, high)  # the value is left as it is: only setVal checks it

    def _param_range(self, low: Any, high: Any) -> None:
        """``setRange(tmin, tmax)``: ends that are functions, read whenever the range is asked
        for."""
        ends = [
            one if hasattr(one, "getVal") else RooConstVar(g(one), g(one), float(one))
            for one in (low, high)
        ]
        self._binning = RooParamBinning(ends[0], ends[1], self._binning.numBins())

    def _set_named_range(self, name: str, low: float, high: float) -> None:
        exists = name in self._shared
        binning = self.getBinning(name, createOnTheFly=True)
        binning.setRange(low, max(low, high))
        if not exists:
            log(
                self,
                INFO,
                "Eval",
                f"RooRealVar::setRange({self._name}) new range named "
                f"'{name}' created with bounds [{g(low)},{g(high)}]",
            )

    def setMin(self, *args: Any) -> None:
        name, value = (args[0], float(args[1])) if len(args) == 2 else (None, float(args[0]))
        if name:
            self._named(str(name)).setRange(value, max(value, self.getMax(name)))
            return
        if value > self.getMax():
            self._warn_end("setMin", "min. larger than max., setting min. to max.")
            value = self.getMax()
        self._binning.setRange(value, self.getMax())
        self._val = min(max(self._val, value), self.getMax())

    def setMax(self, *args: Any) -> None:
        name, value = (args[0], float(args[1])) if len(args) == 2 else (None, float(args[0]))
        if name:
            low = self.getMin(name)
            self._named(str(name)).setRange(low, max(low, value))
            return
        if value < self.getMin():
            self._warn_end("setMax", "max. smaller than min., setting max. to min.")
            value = self.getMin()
        self._binning.setRange(self.getMin(), value)
        self._val = min(max(self._val, self.getMin()), value)

    def _warn_end(self, method: str, text: str) -> None:
        log(
            self,
            WARNING,
            "InputArguments",
            f"RooRealVar::{method}({self._name}): Proposed new fit {text}",
        )

    def _named(self, name: str) -> Any:
        """``getBinning(name, true, true)``: the named range - made, and said, if it is new."""
        if name not in self._shared:
            log(
                self,
                INFO,
                "Eval",
                f"RooRealVar::getBinning({self._name}) new range named '{name}' "
                "created with default bounds",
            )
        return self.getBinning(name, createOnTheFly=True)

    def removeRange(self, name: Any = None) -> None:
        """``removeMin(name); removeMax(name)``: a named range stays, its ends at infinity."""
        if name:
            self._named(str(name)).setRange(-INFINITY, INFINITY)
        else:
            self._binning.setRange(-INFINITY, INFINITY)

    def removeMin(self, name: Any = None) -> None:
        self._binning.setRange(-INFINITY, self._binning.highBound())

    def removeMax(self, name: Any = None) -> None:
        self._binning.setRange(self._binning.lowBound(), INFINITY)

    def setBin(self, ibin: int, name: Any = None) -> None:
        """``setBin``: the value set to the centre of bin ``ibin`` of the binning ``name``."""
        from .messages import ERROR

        if not 0 <= int(ibin) < self.getBins(name):
            log(
                self,
                ERROR,
                "InputArguments",
                f"RooAbsRealLValue::setBin({self._name}) ERROR: bin index "
                f"{int(ibin)} is out of range (0,{self.getBins(name) - 1})",
            )
            return
        self.setVal(self.getBinning(name).binCenter(int(ibin)))

    def setBins(self, nbins: int, name: Any = None) -> None:
        if name:
            self._shared[str(name)] = RooUniformBinning(
                self.getMin(), self.getMax(), nbins, str(name)
            )
        else:
            self._binning = RooUniformBinning(self.getMin(), self.getMax(), nbins)

    def setBinning(self, binning: RooAbsBinning, name: Any = None) -> None:
        made = binning.clone(str(name) if name else "")
        if name:
            self._shared[str(name)] = made
        else:
            self._binning = made
            self._val = min(max(self._val, made.lowBound()), made.highBound())

    # -- errors and constancy -----------------------------------------------------

    def getError(self) -> float:
        return self._error if self._error >= 0 else 0.0

    def setError(self, value: float) -> None:
        self._error = float(value)

    def removeError(self) -> None:
        self._error = -1.0

    def hasError(self, allowZero: bool = True) -> bool:
        return self._error >= 0 if allowZero else self._error > 0

    def getAsymErrorLo(self) -> float:
        return self._asym[0] if self._asym[0] <= 0 else 0.0

    def getAsymErrorHi(self) -> float:
        return self._asym[1] if self._asym[1] >= 0 else 0.0

    def getErrorLo(self) -> float:
        return self._asym[0] if self._asym[0] <= 0 else -self._error

    def getErrorHi(self) -> float:
        return self._asym[1] if self._asym[1] >= 0 else self._error

    def setAsymError(self, low: float, high: float) -> None:
        self._asym = (float(low), float(high))

    def removeAsymError(self) -> None:
        self._asym = (1.0, -1.0)

    def hasAsymError(self, allowZero: bool = True) -> bool:
        low, high = self._asym
        return (high >= 0 and low <= 0) if allowZero else (high > 0 and low < 0)

    def setConstant(self, value: bool = True) -> None:
        self.setAttribute("Constant", bool(value))

    def copy_value_from(self, other: Any) -> None:
        self._val = float(other.getVal())
        if isinstance(other, RooRealVar):
            self._error = other._error
            self._asym = other._asym

    def format(self, *args: Any) -> str:
        """``format(sigDigits, options)`` or ``format(Format(...))``: the variable as text."""
        from .formatting import format_command, format_var

        if args and hasattr(args[0], "args"):
            return format_command(self, args[0])
        return format_var(self, int(args[0]) if args else 2, str(args[1]) if len(args) > 1 else "")

    # -- printing -----------------------------------------------------------------

    def defaultPrintContents(self, option: Any) -> int:
        if str(option or "") == "I":
            return kName | kClassName | kValue
        return kName | kClassName | kValue | kExtras

    def printValue(self) -> str:
        text = g(self._val)
        if self.hasError() and not self.hasAsymError():
            text += f" +/- {g(self.getError())}"
        elif self.hasAsymError():
            text += f" +/- ({g(self._asym[0])},{g(self._asym[1])})"
        return text

    def printExtras(self) -> str:
        text = "C " if self.isConstant() else ""
        text += " L(" + (g(self.getMin()) if self.hasMin() else "-INF")
        text += (" - " + g(self.getMax())) if self.hasMax() else " - +INF"
        text += ") "
        if self.getBins() != 100:
            text += f"B({self.getBins()}) "
        if self._unit:
            text += f"// [{self._unit}]"
        return text

    def printMultiline(self, contents: int, verbose: bool, indent: str) -> str:
        unit = f" {self._unit}" if self._unit else ""
        return super().printMultiline(contents, verbose, indent) + (
            f"{indent}--- RooRealVar ---\n{indent}  Error = {g(self.getError())}{unit}\n"
        )


def _middle(low: float, high: float) -> float:
    """Where a variable made from a range starts: its middle, or its one finite end."""
    if is_infinite(low):
        return 0.0 if is_infinite(high) else high
    return low if is_infinite(high) else 0.5 * (low + high)


class RooConstVar(RooAbsReal):
    """A constant: a value that never changes, named after it if made without a name."""

    def __init__(self, name: Any = "", title: Any = "", value: float = 0.0) -> None:
        super().__init__(name, title)
        self._val = float(value)

    def isFundamental(self) -> bool:
        return True

    def isConstant(self) -> bool:
        return True

    def compute(self, ctx: Context) -> Any:
        return self._val

    def getVal(self, nset: Any = None) -> float:
        return self._val

    def printValue(self) -> str:
        return g(self._val)

    def copy_value_from(self, other: Any) -> None:
        """A constant keeps its value."""
