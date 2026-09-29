"""HistFactory's ``Channel``: one region's data and samples, and reading their histograms.

``CollectHistograms`` reads every histogram the channel's configuration
names - the data, each sample's nominal, its statistical error histogram
and its systematics' - opening each file once, as ``Channel`` does;
``CheckHistograms`` says which is missing, and warns of negative bins.
"""

from __future__ import annotations

import copy
from typing import Any

from ..roofit import cout
from ..roofit.messages import ERROR, WARNING
from ..roofit.printing import g
from .model import Data, Sample, _hf, _histogram
from .shapes import StatErrorConfig
from .systematics import Constraint, _write

__all__ = ["Channel"]


class Channel:
    """A region of the measurement: its data, its samples, its statistical error configuration."""

    def __init__(self, Name: str = "", InputFile: str = "") -> None:
        self._name, self._file, self._path = str(Name), str(InputFile), ""
        self._data = Data()
        self._additional: list[Data] = []
        self._stat_config = StatErrorConfig()
        self._samples: list[Sample] = []

    def __deepcopy__(self, memo: Any) -> Channel:
        made = copy.copy(self)
        made._data = copy.deepcopy(self._data, memo)
        made._additional = [copy.deepcopy(one, memo) for one in self._additional]
        made._stat_config = copy.copy(self._stat_config)
        made._samples = [copy.deepcopy(one, memo) for one in self._samples]
        return made

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def GetName(self) -> str:
        return self._name

    def SetInputFile(self, value: str) -> None:
        self._file = str(value)

    def GetInputFile(self) -> str:
        return self._file

    def SetHistoPath(self, value: str) -> None:
        self._path = str(value)

    def GetHistoPath(self) -> str:
        return self._path

    def SetData(self, data: Any, InputFile: str = "", HistoPath: str = "") -> None:
        """The data: a ``Data``, a histogram's name and file, a histogram, or one count."""
        if isinstance(data, Data):
            self._data = copy.deepcopy(data)
        elif isinstance(data, str):
            self._data.SetHistoName(data)
            self._data.SetInputFile(InputFile)
            self._data.SetHistoPath(HistoPath)
        elif isinstance(data, (int, float)):
            from ..pyroot.core import TH1F

            name = f"{self._name}_data"
            hist = TH1F(name, name, 1, 0, 1)
            hist.SetBinContent(1, float(data))
            self._data.SetHisto(hist)
        else:
            self._data.SetHisto(data)

    def GetData(self) -> Data:
        return self._data

    def AddAdditionalData(self, data: Data) -> None:
        self._additional.append(copy.deepcopy(data))

    def GetAdditionalData(self) -> list[Data]:
        return self._additional

    def SetStatErrorConfig(self, threshold: Any, kind: Any = None) -> None:
        if isinstance(threshold, StatErrorConfig):
            self._stat_config = copy.copy(threshold)
            return
        self._stat_config.SetRelErrorThreshold(threshold)
        self._stat_config.SetConstraintType(Constraint.GetType(kind) if isinstance(kind, str)
                                            else int(kind))  # fmt: skip

    def GetStatErrorConfig(self) -> StatErrorConfig:
        return self._stat_config

    def AddSample(self, sample: Sample) -> None:
        """A copy of ``sample``, told it is this channel's."""
        made = copy.deepcopy(sample)
        made.SetChannelName(self._name)
        self._samples.append(made)

    def GetSamples(self) -> list[Sample]:
        return self._samples

    def text(self) -> str:
        found = (f"\t Channel Name: {self._name}\t InputFile: {self._file}\n\t Data:\n"
                 f"{self._data.text()}\t statErrorConfig:\n{self._stat_config.text()}")  # fmt: skip
        if self._samples:
            found += "\t Samples: \n" + "".join(one.text() for one in self._samples)
        return found + f"\t End of Channel {self._name}\n"

    def Print(self, stream: Any = None) -> None:
        _write(stream, self.text())

    # -- histograms ---------------------------------------------------------------

    def CollectHistograms(self) -> None:
        """Every histogram the configuration names, read."""
        files: dict[str, Any] = {}
        for data in [self._data, *self._additional]:
            if data.GetInputFile():
                data.SetHisto(_histogram(files, data.GetInputFile(), data.GetHistoPath(),
                                         data.GetHistoName()))  # fmt: skip
        for sample in self._samples:
            sample.SetHisto(_histogram(files, sample.GetInputFile(), sample.GetHistoPath(),
                                       sample.GetHistoName()))  # fmt: skip
            stat = sample.GetStatError()
            if stat.GetUseHisto():
                stat.SetErrorHist(_histogram(files, stat.GetInputFile(), stat.GetHistoPath(),
                                             stat.GetHistoName()))  # fmt: skip
            for sys in [*sample.GetHistoSysList(), *sample.GetHistoFactorList()]:
                sys.SetHistoLow(_histogram(files, sys.GetInputFileLow(), sys.GetHistoPathLow(),
                                           sys.GetHistoNameLow()))  # fmt: skip
                sys.SetHistoHigh(_histogram(files, sys.GetInputFileHigh(), sys.GetHistoPathHigh(),
                                            sys.GetHistoNameHigh()))  # fmt: skip
            for shape in sample.GetShapeSysList():
                shape.SetErrorHist(_histogram(files, shape.GetInputFile(), shape.GetHistoPath(),
                                              shape.GetHistoName()))  # fmt: skip
            for factor in sample.GetShapeFactorList():
                if factor.HasInitialShape():
                    factor.SetInitialShape(_histogram(files, factor.GetInputFile(),
                                                      factor.GetHistoPath(),
                                                      factor.GetHistoName()))  # fmt: skip

    def CheckHistograms(self) -> bool:
        """Whether every histogram is read - said, if one is not - and negative bins warned of."""
        if self._data.GetHisto() is None and self._data.GetInputFile():
            _hf(ERROR, f"Error: Data Histogram for channel {self._name} is nullptr.")
            return False
        for sample in self._samples:
            if not _sample_has_histograms(sample):
                return False
            _negative_bins(sample, self._name)
        return True


def _sample_has_histograms(sample: Sample) -> bool:
    """Whether the sample's histograms are all read: said by the first missing, if not."""
    name = sample.GetName()
    if sample.GetHisto() is None:
        _hf(ERROR, f"Error: Nominal Histogram for sample {name} is nullptr.")
        return False
    stat = sample.GetStatError()
    if stat.GetUseHisto() and stat.GetErrorHist() is None:
        _hf(ERROR, f"Error: Statistical Error Histogram for sample {name} is nullptr.")
        return False
    for sys in [*sample.GetHistoSysList(), *sample.GetHistoFactorList()]:
        for side, hist in (("Low", sys.GetHistoLow()), ("High", sys.GetHistoHigh())):
            if hist is None:
                _hf(ERROR, f"Error: HistoSyst {side} for Systematic {sys.GetName()} in sample "
                    f"{name} is nullptr.")  # fmt: skip
                return False
    for shape in sample.GetShapeSysList():
        if shape.GetErrorHist() is None:
            _hf(ERROR, f"Error: HistoSyst High for Systematic {shape.GetName()} in sample {name} "
                "is nullptr.")  # fmt: skip
            return False
    return True


def _negative_bins(sample: Sample, channel: str) -> None:
    """The warning of negative bins in the nominal histogram - the bins listed on ``cout``."""
    hist = sample.GetHisto()
    bad = [(i, hist.GetBinContent(i)) for i in range(1, hist.GetNbinsX() + 1)
           if hist.GetBinContent(i) < 0]  # fmt: skip
    if not bad:
        return
    _hf(WARNING, f"WARNING: Nominal Histogram {hist.GetName()} for Sample = {sample.GetName()} "
        f"in Channel = {channel} has negative entries in bin numbers = ")  # fmt: skip
    cout.write(" , ".join(f"{i} : {g(v)}" for i, v in bad) + "\n")
