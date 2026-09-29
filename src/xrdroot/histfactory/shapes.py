"""HistFactory's one-histogram uncertainties, the stat-error configuration, functions, Asimovs.

A ``ShapeSys`` is a constrained bin-by-bin variation read from a relative
error histogram; a ``ShapeFactor`` free bin-by-bin factors, perhaps from
an initial shape; a ``StatError`` the sample's share of the channel's
Monte Carlo statistical uncertainty, from its own errors or a histogram.
Each keeps its one histogram where ROOT keeps it: as the "high" one.
"""

from __future__ import annotations

from typing import Any

from ..roofit.printing import g
from .systematics import Constraint, HistFactoryError, _Histograms, _write

__all__ = [
    "Asimov",
    "PreprocessFunction",
    "ShapeFactor",
    "ShapeSys",
    "StatError",
    "StatErrorConfig",
]


class _OneHistogram(_Histograms):
    """An uncertainty with one histogram: its file, name and path."""

    def SetInputFile(self, value: str) -> None:
        self._set("high", "file", value)

    def SetHistoName(self, value: str) -> None:
        self._set("high", "name", value)

    def SetHistoPath(self, value: str) -> None:
        self._set("high", "path", value)

    def GetInputFile(self) -> str:
        return self._where["high"]["file"]

    def GetHistoName(self) -> str:
        return self._where["high"]["name"]

    def GetHistoPath(self) -> str:
        return self._where["high"]["path"]

    def GetErrorHist(self) -> Any:
        return self._hists["high"]

    def SetErrorHist(self, hist: Any) -> None:
        self._hists["high"] = hist


class ShapeSys(_OneHistogram):
    """A constrained bin-by-bin variation, its relative errors a histogram."""

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._constraint = Constraint.Gaussian

    def SetConstraintType(self, kind: Any) -> None:
        self._constraint = Constraint.GetType(kind) if isinstance(kind, str) else int(kind)

    def GetConstraintType(self) -> int:
        return self._constraint

    def text(self) -> str:
        high = self._where["high"]
        return (f"\t \t Name: {self._name}\t InputFile: {high['file']}\t HistoName: "
                f"{high['name']}\t HistoPath: {high['path']}\n")  # fmt: skip


class ShapeFactor(_OneHistogram):
    """Free bin-by-bin factors, from an initial shape if it has one, in their range."""

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._constant = self._has_shape = False
        self._val, self._low, self._high = 1.0, 0.0, 1000.0

    def _set(self, side: str, key: str, value: str) -> None:
        super()._set(side, key, value)
        self._has_shape = True

    def SetInitialShape(self, hist: Any) -> None:
        self._hists["high"] = hist

    def GetInitialShape(self) -> Any:
        return self._hists["high"]

    def HasInitialShape(self) -> bool:
        return self._has_shape

    def SetConstant(self, constant: bool) -> None:
        self._constant = bool(constant)

    def IsConstant(self) -> bool:
        return self._constant

    def SetVal(self, val: float) -> None:
        self._val = float(val)

    def SetLow(self, low: float) -> None:
        self._low = float(low)

    def SetHigh(self, high: float) -> None:
        self._high = float(high)

    def GetVal(self) -> float:
        return self._val

    def GetLow(self) -> float:
        return self._low

    def GetHigh(self) -> float:
        return self._high

    def text(self) -> str:
        high = self._where["high"]
        found = f"\t \t Name: {self._name}\n"
        if high["name"]:
            found += (f"\t \t  Shape Hist Name: {high['name']} Shape Hist Path Name: "
                      f"{high['path']} Shape Hist FileName: {high['file']}\n")  # fmt: skip
        found += f"\t \t Value: {g(self._val)}  L({g(self._low)} - {g(self._high)})\n"
        return found + ("\t \t ( Constant ): \n" if self._constant else "")


class StatError(_OneHistogram):
    """A sample's part in the channel's statistical uncertainty: its errors, or a histogram's."""

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._active = self._use_histo = False

    def Activate(self, active: bool = True) -> None:
        self._active = bool(active)

    def GetActivate(self) -> bool:
        return self._active

    def SetUseHisto(self, use: bool = True) -> None:
        self._use_histo = bool(use)

    def GetUseHisto(self) -> bool:
        return self._use_histo

    def text(self) -> str:
        high = self._where["high"]
        return (f"\t \t Activate: {int(self._active)}\t InputFile: {high['file']}\t HistoName: "
                f"{high['name']}\t histoPath: {high['path']}\n")  # fmt: skip


class StatErrorConfig:
    """A channel's statistical uncertainty: the threshold below which a bin's is dropped, and
    whether its constraints are Gaussian or Poisson."""

    def __init__(self) -> None:
        self._threshold = 0.05
        self._constraint = Constraint.Poisson

    def SetRelErrorThreshold(self, threshold: float) -> None:
        self._threshold = float(threshold)

    def GetRelErrorThreshold(self) -> float:
        return self._threshold

    def SetConstraintType(self, kind: int) -> None:
        self._constraint = int(kind)

    def GetConstraintType(self) -> int:
        return self._constraint

    def text(self) -> str:
        return (f"\t \t RelErrorThreshold: {g(self._threshold)}\t ConstraintType: "
                f"{Constraint.Name(self._constraint)}\n")  # fmt: skip

    def Print(self, stream: Any = None) -> None:
        _write(stream, self.text())


class PreprocessFunction:
    """A function of the model's parameters, made in the workspace before the model is."""

    def __init__(self, name: str = "", expression: str = "", dependents: str = "") -> None:
        self._name, self._expression, self._dependents = str(name), str(expression), str(dependents)

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def GetName(self) -> str:
        return self._name

    def SetExpression(self, expression: str) -> None:
        self._expression = str(expression)

    def GetExpression(self) -> str:
        return self._expression

    def SetDependents(self, dependents: str) -> None:
        self._dependents = str(dependents)

    def GetDependents(self) -> str:
        return self._dependents

    def GetCommand(self) -> str:
        """The factory's command that makes it: ``expr::name('expression',{dependents})``."""
        return f"expr::{self._name}('{self._expression}',{{{self._dependents}}})"

    def Print(self, stream: Any = None) -> None:
        _write(stream, f"\t \t Name: {self._name}\t \t Expression: {self._expression}\t \t "
               f"Dependents: {self._dependents}\n")  # fmt: skip


class Asimov:
    """An Asimov dataset to add: the parameters set and fixed before it is generated."""

    def __init__(self, name: str = "") -> None:
        self._name = str(name)
        self._fix: dict[str, bool] = {}
        self._values: dict[str, float] = {}

    def GetName(self) -> str:
        return self._name

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def SetFixedParam(self, param: str, constant: bool = True) -> None:
        self._fix[str(param)] = bool(constant)

    def SetParamValue(self, param: str, value: float) -> None:
        self._values[str(param)] = float(value)

    def GetParamsToFix(self) -> dict[str, bool]:
        return self._fix

    def GetParamsToSet(self) -> dict[str, float]:
        return self._values

    def ConfigureWorkspace(self, wspace: Any) -> None:
        """The parameters in the workspace set, then fixed, as the Asimov dataset wants them."""
        from ..roofit import cout

        for param, value in sorted(self._values.items()):
            var = wspace.var(param)
            if var is None:
                cout.write("Error: Trying to set variable: 0x0 to a specific value in creation "
                           f"of asimov dataset: {self._name} but this variable doesn't appear to "
                           "exist in the workspace\n")  # fmt: skip
                raise HistFactoryError(f"HistFactory - the workspace has no variable {param}")
            if not var.inRange(value, None):
                raise HistFactoryError(f"HistFactory - {g(value)} is outside the range of {param}")
            cout.write(f"Configuring Asimov Dataset: Setting {param} = {g(value)}\n")
            var.setVal(value)
        for param, constant in sorted(self._fix.items()):
            var = wspace.var(param)
            if var is None:
                raise HistFactoryError(f"HistFactory - the workspace has no variable {param}")
            cout.write(f"Configuring Asimov Dataset: Setting {param} to constant \n")
            var.setConstant(constant)
