"""HistFactory's systematics: the pieces of a sample's uncertainty, as the configuration has them.

Each is a name and its numbers or histograms - an ``OverallSys``'s low and
high normalisation, a ``HistoSys``'s low and high histograms, a
``ShapeSys``'s relative error histogram - printed as ROOT's ``Print``
prints them, tab by tab. A histogram is kept as a copy, as ``HistRef``
keeps one.
"""

from __future__ import annotations

from typing import Any

from ..roofit.printing import g

__all__ = [
    "Constraint",
    "HistFactoryError",
    "HistoFactor",
    "HistoSys",
    "NormFactor",
    "OverallSys",
]


class HistFactoryError(RuntimeError):
    """``hf_exc``: what HistFactory throws when the configuration cannot be built."""


class Constraint:
    """``Constraint::Type``: a Gaussian or a Poisson constraint, by number or by name."""

    Gaussian, Poisson = 0, 1

    @staticmethod
    def Name(kind: int) -> str:
        return {0: "Gaussian", 1: "Poisson"}.get(int(kind), "")

    @staticmethod
    def GetType(name: str) -> int:
        from ..roofit import cout

        if not name:
            cout.write("Error: Given empty name for ConstraintType\n")
            raise HistFactoryError("HistFactory - an empty name was given for a constraint type")
        if name in ("Gaussian", "Gauss"):
            return Constraint.Gaussian
        if name in ("Poisson", "Pois"):
            return Constraint.Poisson
        cout.write(f"Error: Unknown name given for Constraint Type: {name}\n")
        raise HistFactoryError(f"HistFactory - {name} is not a constraint type")


def _copied(hist: Any) -> Any:
    """``HistRef``'s copy: a clone in no directory."""
    if hist is None:
        return None
    made = hist.Clone()
    if hasattr(made, "SetDirectory"):
        made.SetDirectory(0)
    return made


class _Named:
    """A name, which every systematic has."""

    def __init__(self, name: str = "") -> None:
        self._name = str(name)

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def GetName(self) -> str:
        return self._name


class OverallSys(_Named):
    """A constrained systematic scaling the sample's normalisation between ``low`` and ``high``."""

    def __init__(self) -> None:
        super().__init__()
        self._low = self._high = 0.0

    def SetLow(self, low: float) -> None:
        self._low = float(low)

    def SetHigh(self, high: float) -> None:
        self._high = float(high)

    def GetLow(self) -> float:
        return self._low

    def GetHigh(self) -> float:
        return self._high

    def text(self) -> str:
        return f"\t \t Name: {self._name}\t Low: {g(self._low)}\t High: {g(self._high)}\n"

    def Print(self, stream: Any = None) -> None:
        _write(stream, self.text())


class NormFactor(OverallSys):
    """A free factor on the sample's normalisation: its value and range."""

    def __init__(self) -> None:
        super().__init__()
        self._val = self._low = self._high = 1.0

    def SetVal(self, val: float) -> None:
        self._val = float(val)

    def GetVal(self) -> float:
        return self._val

    def text(self) -> str:
        return (f"\t \t Name: {self._name}\t Val: {g(self._val)}\t Low: {g(self._low)}\t "
                f"High: {g(self._high)}\n")  # fmt: skip


class _Histograms(_Named):
    """``HistogramUncertaintyBase``: a low and a high histogram, each with where it is read from."""

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._where = {side: {"file": "", "name": "", "path": ""} for side in ("low", "high")}
        self._hists: dict[str, Any] = {"low": None, "high": None}

    def __deepcopy__(self, memo: Any) -> Any:
        made = type(self).__new__(type(self))
        rest = {k: v for k, v in self.__dict__.items() if k not in ("_where", "_hists")}
        made.__dict__.update(rest)
        made._where = {side: dict(where) for side, where in self._where.items()}
        made._hists = {side: _copied(h) for side, h in self._hists.items()}
        return made

    def _set(self, side: str, key: str, value: str) -> None:
        self._where[side][key] = str(value)

    def SetInputFileLow(self, value: str) -> None:
        self._set("low", "file", value)

    def SetInputFileHigh(self, value: str) -> None:
        self._set("high", "file", value)

    def SetHistoNameLow(self, value: str) -> None:
        self._set("low", "name", value)

    def SetHistoNameHigh(self, value: str) -> None:
        self._set("high", "name", value)

    def SetHistoPathLow(self, value: str) -> None:
        self._set("low", "path", value)

    def SetHistoPathHigh(self, value: str) -> None:
        self._set("high", "path", value)

    def GetInputFileLow(self) -> str:
        return self._where["low"]["file"]

    def GetInputFileHigh(self) -> str:
        return self._where["high"]["file"]

    def GetHistoNameLow(self) -> str:
        return self._where["low"]["name"]

    def GetHistoNameHigh(self) -> str:
        return self._where["high"]["name"]

    def GetHistoPathLow(self) -> str:
        return self._where["low"]["path"]

    def GetHistoPathHigh(self) -> str:
        return self._where["high"]["path"]

    def SetHistoLow(self, hist: Any) -> None:
        _detached(hist)
        self._hists["low"] = hist

    def SetHistoHigh(self, hist: Any) -> None:
        _detached(hist)
        self._hists["high"] = hist

    def GetHistoLow(self) -> Any:
        return self._hists["low"]

    def GetHistoHigh(self) -> Any:
        return self._hists["high"]

    def text(self) -> str:
        low, high = self._where["low"], self._where["high"]
        return (f"\t \t Name: {self._name}\t HistoFileLow: {low['file']}\t HistoNameLow: "
                f"{low['name']}\t HistoPathLow: {low['path']}\t HistoFileHigh: {high['file']}"
                f"\t HistoNameHigh: {high['name']}\t HistoPathHigh: {high['path']}\n")  # fmt: skip

    def Print(self, stream: Any = None) -> None:
        _write(stream, self.text())


def _detached(hist: Any) -> None:
    """``SetDirectory(nullptr)``: the histogram belongs to the systematic, not to a file."""
    if hasattr(hist, "SetDirectory"):
        hist.SetDirectory(0)


class HistoSys(_Histograms):
    """A constrained shape variation: the sample's histogram at minus and plus one sigma."""


class HistoFactor(_Histograms):
    """An unconstrained shape variation."""


def _write(stream: Any, text: str) -> None:
    """``os << ...``: to ``std::cout``, or a stream given."""
    if stream is None:
        from ..roofit import cout

        cout.write(text)
    else:
        stream.write(text)
