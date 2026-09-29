"""HistFactory's ``Data``, ``Sample`` and ``Channel``: what a measurement is made of.

A channel has its data and its samples, each sample its nominal histogram
and its systematics; each is given by value, as C++ copies it - a sample
added to a channel is a copy, its histograms clones - and read from its
file by ``CollectHistograms``, which says what it reads as ROOT says it.
"""

from __future__ import annotations

import copy
from typing import Any

from ..roofit.messages import ERROR, INFO, PROGRESS, log
from ..roofit.printing import address
from .shapes import ShapeFactor, ShapeSys, StatError
from .systematics import (
    Constraint,
    HistFactoryError,
    HistoFactor,
    HistoSys,
    NormFactor,
    OverallSys,
    _copied,
    _write,
)

__all__ = ["Data", "Sample"]


def _address(hist: Any) -> str:
    """``os << pointer``: a histogram's address, ``0x0`` for none."""
    from ..roofit.printing import NULL_POINTER

    return address(hist) if hist is not None else NULL_POINTER[0]


def _hf(level: int, text: str) -> None:
    log(None, level, "HistFactory", text)


class Data:
    """A channel's observed histogram: where it is read from, and the histogram once read."""

    def __init__(self, HistoName: str = "", InputFile: str = "", HistoPath: str = "") -> None:
        self._name = ""
        self._file, self._histo, self._path = str(InputFile), str(HistoName), str(HistoPath)
        self._hist: Any = None

    def __deepcopy__(self, memo: Any) -> Data:
        made = copy.copy(self)
        made._hist = _copied(self._hist)
        return made

    def GetName(self) -> str:
        return self._name

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def SetInputFile(self, value: str) -> None:
        self._file = str(value)

    def GetInputFile(self) -> str:
        return self._file

    def SetHistoName(self, value: str) -> None:
        self._histo = str(value)

    def GetHistoName(self) -> str:
        return self._histo

    def SetHistoPath(self, value: str) -> None:
        self._path = str(value)

    def GetHistoPath(self) -> str:
        return self._path

    def GetHisto(self) -> Any:
        return self._hist

    def SetHisto(self, hist: Any) -> None:
        self._hist = hist
        self._histo = hist.GetName()

    def text(self) -> str:
        return (f"\t \t InputFile: {self._file}\t HistoName: {self._histo}\t HistoPath: "
                f"{self._path}\t HistoAddress: {_address(self._hist)}\n")  # fmt: skip

    def Print(self, stream: Any = None) -> None:
        _write(stream, self.text())


def _histogram(files: dict[str, Any], InputFile: str, HistoPath: str, HistoName: str) -> Any:
    """``Channel::GetHistogram``: the histogram, read from its file - opened once - and cloned
    out of it, said as ROOT says it."""
    from ..pyroot.core import TFile

    _hf(PROGRESS, f"Getting histogram {InputFile}:{HistoPath}/{HistoName}")
    handle = files.get(InputFile)
    if handle is None:
        handle = TFile.Open(InputFile)
        if handle is None or not handle.IsOpen():
            _hf(ERROR, f"Error: Unable to open input file: {InputFile}")
            raise HistFactoryError(f"HistFactory - {InputFile} cannot be opened")
        files[InputFile] = handle
        _hf(INFO, f"Opened input file: {InputFile}: ")
    where = f"{HistoPath}/{HistoName}" if HistoPath else HistoName
    found = handle.Get(where)
    if found is None:
        _hf(ERROR, f"Histogram '{HistoName}' wasn't found in file '{InputFile}' in directory "
            f"'{HistoPath}'.")  # fmt: skip
        raise HistFactoryError(f"HistFactory - {InputFile} has no histogram {where}")
    made = found.Clone()
    made.SetDirectory(0)
    return made


class Sample:
    """A process in a channel: its nominal histogram, and the systematics that move it."""

    def __init__(self, Name: str = "", HistoName: str = "", InputFile: str = "",
                 HistoPath: str = "") -> None:  # fmt: skip
        self._name, self._histo = str(Name), str(HistoName)
        self._file, self._path = str(InputFile), str(HistoPath)
        self._channel = ""
        kinds = ("overall", "norm", "histosys", "histofactor", "shapesys", "shapefactor")
        self._lists: dict[str, list[Any]] = {key: [] for key in kinds}
        self._stat = StatError()
        self._normalize_by_theory = bool(Name)
        self._hist: Any = None

    def __deepcopy__(self, memo: Any) -> Sample:
        made = copy.copy(self)
        made._lists = {key: [copy.deepcopy(one, memo) for one in items]
                       for key, items in self._lists.items()}  # fmt: skip
        made._stat = copy.deepcopy(self._stat, memo)
        made._hist = _copied(self._hist)
        return made

    # -- names and histograms -----------------------------------------------------

    def GetName(self) -> str:
        return self._name

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def GetInputFile(self) -> str:
        return self._file

    def SetInputFile(self, value: str) -> None:
        self._file = str(value)

    def GetHistoName(self) -> str:
        return self._histo

    def SetHistoName(self, value: str) -> None:
        self._histo = str(value)

    def GetHistoPath(self) -> str:
        return self._path

    def SetHistoPath(self, value: str) -> None:
        self._path = str(value)

    def GetChannelName(self) -> str:
        return self._channel

    def SetChannelName(self, value: str) -> None:
        self._channel = str(value)

    def SetNormalizeByTheory(self, norm: bool) -> None:
        self._normalize_by_theory = bool(norm)

    def GetNormalizeByTheory(self) -> bool:
        return self._normalize_by_theory

    def GetHisto(self) -> Any:
        return self._hist

    def SetHisto(self, hist: Any) -> None:
        self._hist = hist
        self._histo = hist.GetName()

    def SetValue(self, value: float) -> None:
        """A counting sample: a one-bin histogram ``<name>_hist`` holding ``value``."""
        from ..pyroot.core import TH1F

        name = f"{self._name}_hist"
        hist = TH1F(name, name, 1, 0, 1)
        hist.SetDirectory(0)
        hist.SetBinContent(1, value)
        self.SetHisto(hist)

    # -- systematics --------------------------------------------------------------

    def ActivateStatError(self, HistoName: Any = None, InputFile: str = "",
                          HistoPath: str = "") -> None:  # fmt: skip
        """The sample's statistical errors in the channel's - its own, or a histogram's."""
        self._stat.Activate(True)
        self._stat.SetUseHisto(HistoName is not None)
        if HistoName is not None:
            self._stat.SetInputFile(InputFile)
            self._stat.SetHistoName(HistoName)
            self._stat.SetHistoPath(HistoPath)

    def AddOverallSys(self, name: Any, low: float = 0.0, high: float = 0.0) -> None:
        if isinstance(name, OverallSys):
            self._lists["overall"].append(copy.deepcopy(name))
            return
        sys = OverallSys()
        sys.SetName(name)
        sys.SetLow(low)
        sys.SetHigh(high)
        self._lists["overall"].append(sys)

    def AddNormFactor(self, name: Any, val: float = 1.0, low: float = 1.0,
                      high: float = 1.0) -> None:  # fmt: skip
        if isinstance(name, NormFactor):
            self._lists["norm"].append(copy.deepcopy(name))
            return
        norm = NormFactor()
        norm.SetName(name)
        norm.SetVal(val)
        norm.SetLow(low)
        norm.SetHigh(high)
        self._lists["norm"].append(norm)

    def _histograms(self, key: str, kind: Any, args: tuple[Any, ...]) -> None:
        if len(args) == 1:
            self._lists[key].append(copy.deepcopy(args[0]))
            return
        made = kind(args[0])
        made.SetHistoNameLow(args[1])
        made.SetInputFileLow(args[2])
        made.SetHistoPathLow(args[3])
        made.SetHistoNameHigh(args[4])
        made.SetInputFileHigh(args[5])
        made.SetHistoPathHigh(args[6])
        self._lists[key].append(made)

    def AddHistoSys(self, *args: Any) -> None:
        """``(name, low name, low file, low path, high name, high file, high path)``."""
        self._histograms("histosys", HistoSys, args)

    def AddHistoFactor(self, *args: Any) -> None:
        self._histograms("histofactor", HistoFactor, args)

    def AddShapeFactor(self, name: Any, val: Any = None, low: float = 0.0,
                       high: float = 1000.0) -> None:  # fmt: skip
        if isinstance(name, ShapeFactor):
            self._lists["shapefactor"].append(copy.deepcopy(name))
            return
        made = ShapeFactor(name)
        if val is not None:
            made.SetVal(val)
            made.SetLow(low)
            made.SetHigh(high)
        self._lists["shapefactor"].append(made)

    def AddShapeSys(self, name: Any, constraint: Any = Constraint.Gaussian, HistoName: str = "",
                    HistoFile: str = "", HistoPath: str = "") -> None:  # fmt: skip
        if isinstance(name, ShapeSys):
            self._lists["shapesys"].append(copy.deepcopy(name))
            return
        made = ShapeSys(name)
        made.SetConstraintType(constraint)
        made.SetHistoName(HistoName)
        made.SetHistoPath(HistoPath)
        made.SetInputFile(HistoFile)
        self._lists["shapesys"].append(made)

    def GetOverallSysList(self) -> list[Any]:
        return self._lists["overall"]

    def GetNormFactorList(self) -> list[Any]:
        return self._lists["norm"]

    def GetHistoSysList(self) -> list[Any]:
        return self._lists["histosys"]

    def GetHistoFactorList(self) -> list[Any]:
        return self._lists["histofactor"]

    def GetShapeSysList(self) -> list[Any]:
        return self._lists["shapesys"]

    def GetShapeFactorList(self) -> list[Any]:
        return self._lists["shapefactor"]

    def GetStatError(self) -> StatError:
        return self._stat

    def SetStatError(self, error: StatError) -> None:
        self._stat = copy.deepcopy(error)

    def HasStatError(self) -> bool:
        return False  # fStatErrorActivate, which nothing but the XML reader sets

    def text(self) -> str:
        """``Print``: the sample, where its histogram is, and its statistical error's."""
        found = (f"\t \t Name: {self._name}\t \t Channel: {self._channel}\t NormalizeByTheory: "
                 f"{'True' if self._normalize_by_theory else 'False'}\t StatErrorActivate: "
                 f"False\n\t \t \t \t \t InputFile: {self._file}\t HistName: {self._histo}\t "
                 f"HistoPath: {self._path}\t HistoAddress: {_address(self._hist)}\n")  # fmt: skip
        if self._stat.GetActivate():
            found += (f"\t \t \t StatError Activate: 1\t InputFile: {self._file}\t HistName: "
                      f"{self._stat.GetHistoName()}\t HistoPath: {self._stat.GetHistoPath()}\t "
                      f"HistoAddress: {_address(self._stat.GetErrorHist())}\n")  # fmt: skip
        return found

    def Print(self, stream: Any = None) -> None:
        _write(stream, self.text())
